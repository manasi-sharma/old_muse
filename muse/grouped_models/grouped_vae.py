"""
VAE,
"""
import random

import torch
from typing import Callable

from muse.experiments import logger
from muse.grouped_models.grouped_model import GroupedModel
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from muse.utils.general_utils import timeit
from muse.utils.torch_utils import broadcast_dims


class GroupedVAEModel(GroupedModel):
    required_models = [
        "posterior_input_selector",
        "prior_input_selector",
        "decoder_input_selector",
        "posterior",
        "prior",
        "decoder"
    ]

    latent_prior_parsed_kwargs = {}
    latent_posterior_parsed_kwargs = {}

    def _init_params_to_attrs(self, params: d):
        super(GroupedVAEModel, self)._init_params_to_attrs(params)

        # PARAMS
        self.beta = get_with_default(params, "beta", 1.0)
        self.beta_schedule = 1.  # default is 100% of beta
        # self.beta_info = get_with_default(params, "beta_info", 0.)

        # represents the minimum horizon to use for posterior & decoder sequences.
        self.horizon = params["horizon"]
        self.min_horizon = get_with_default(params, "min_horizon", self.horizon)
        assert 1 < self.min_horizon <= self.horizon, [self.min_horizon, self.horizon]
        self.latent_size = params["latent_size"]

        # TODO beta schedule
        # logger.debug("Beta = %f, custom scheduler = %s" % (self.beta, params.has_leaf_key("beta_schedule_fn")))
        # self.beta_schedule_fn = get_with_default(params, "beta_schedule_fn", default=lambda step: 1.)
        # takes in each of the inputs and returns embedding for obs, proprio

        # NAMES
        self.latent_name = get_with_default(params, "z", default="z")
        self.decoder_names = params["decoder_names"]

        # self.all_input_names = get_with_default(params, "all_inputs", default=self._env_spec.all_names)
        # self.include_goal_proprio = get_with_default(params, "include_goal_proprio", default=False)

        # (z) -> (sampled z)
        self.set_fn("latent_sample_fn", params["latent_sample_fn"], Callable[[d], d])

        # (z_prior, z_posterior, inputs, outputs) -> KL loss between z prior and posterior, for example
        self.set_fn("latent_dist_fn", params["latent_dist_fn"], Callable[[d, d, d, d], torch.Tensor])

        # (probabilistic output) -> sampled action
        self.set_fn("decoder_sample_fn", params["decoder_sample_fn"], Callable[[d], d])

        # (self, model_outs, ins, outs) ->
        self.set_fn("decoder_loss_fn", params["decoder_loss_fn"],
                    Callable[[__class__, d, d, d], torch.Tensor])

        if params.has_leaf_key("batch_mask_fn"):
            # takes inputs, outputs (B,) tensor mask
            self.batch_mask_fn = params["batch_mask_fn"]
        else:
            self.batch_mask_fn = None

        self._block_model_training_steps = get_with_default(params, "block_model_training_steps", 0)
        self._block_prior_training_steps = get_with_default(params, "block_prior_training_steps", 0)

        # in z_distance, pass in a fixed posterior. makes the prior have no regularizing effect.
        if self._block_model_training_steps > 0:
            logger.info(f"Blocking Model training for {self._block_model_training_steps} steps")
        if self._block_prior_training_steps > 0:
            assert self._block_model_training_steps <= self._block_prior_training_steps
            logger.info(f"Blocking Prior training for {self._block_prior_training_steps} steps")

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def _prepare_inputs(self, inputs, preproc=True, current_horizon=None) -> d:
        # accept tuple inputs
        if not isinstance(inputs, d):
            assert len(inputs) == len(self.all_input_names), [self.all_input_names, len(inputs)]
            inputs = d.from_dict({k: v for k, v in zip(self.all_input_names, inputs)})
        else:
            inputs = inputs.leaf_copy()

        # varying time horizon
        if current_horizon is not None:
            inputs.leaf_assert(lambda arr: arr.shape[1] >= current_horizon)
            # truncate to appropriate length
            inputs.leaf_modify(lambda arr: arr[:, :current_horizon])

        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=torch.float32)

        return self._preproc_fn(inputs) if preproc else inputs

    def compute_latent_prior(self, pl_inputs, model_outs, sample=True, **kwargs):
        # get the proposed latent (e.g., from start or start and goal)
        latent_propose_ins = self.prior_input_selector(pl_inputs)
        latent_propose_outs = self.prior(latent_propose_ins)
        model_outs.latent_prior = latent_propose_outs

        if sample:
            # sample from proposal otherwise, in decoder for example
            sampled_latent = self.latent_sample_fn(latent_propose_outs)
            model_outs.latent_prior_sample = sampled_latent

        return latent_propose_outs

    def compute_latent_posterior(self, pl_inputs, model_outs, sample=True, **kwargs):
        # get the recognized latent
        latent_recog_ins = self.posterior_input_selector(pl_inputs)
        latent_recog_outs = self.posterior(latent_recog_ins)
        model_outs.latent_posterior = latent_recog_outs

        if sample:
            # sample from recognition for training, for example
            sampled_latent = self.latent_sample_fn(latent_recog_outs)
            model_outs.latent_posterior_sample = sampled_latent

        return latent_recog_outs

    def compute_decoder_outs(self, pl_inputs, model_outs, latent_sample_name, sample=False, **kwargs):
        # add the sampled latent + inputs + encodings, optional sampling the output
        decoder_ins = self.decoder_input_selector(pl_inputs)
        decoder_outs = self.decoder(decoder_ins, **kwargs)
        model_outs[latent_sample_name + "_decoder"].combine(decoder_outs)

        if sample:
            decoder_sample = self.decoder_sample_fn(model_outs[latent_sample_name + "_decoder"])
            assert decoder_sample.has_leaf_keys(self.decoder_names)
            model_outs[latent_sample_name + "_decoder"].combine(
                decoder_sample)  # sampled action will be at the root level

        return model_outs[latent_sample_name + "_decoder"]

    def check_broadcast_latent(self, latent, current_horizon):
        assert current_horizon is not None
        assert len(latent.shape) > 1, f"Latent needs to be batched: {latent.shape}"
        assert latent.shape[-1] == self.latent_size, latent.shape
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(1)  # horizon dim
        return broadcast_dims(latent, dims=[-2], new_shape=[current_horizon])

    # run on a latent
    def forward(self, inputs, preproc=True, postproc=True, run_prepare=True,
                compute_posterior=False, compute_prior=True,
                decode_posterior=False, decode_random_latent=False, decode_prior=True,
                run_all=False,
                sample=False, current_horizon=None, batch_element_mask=None, model_outs=d(), **kwargs):
        """
        VAE forward:
        0. inputs are (B x H x ...)
        1. For posterior (full seq):
            a. inputs from posterior_input_selector(inputs, goals) -> (B x H x ...), run posterior : (B x H x PD)
        2. For prior (seq / name subset):
            a. inputs from prior_input_selector(inputs, goals) -> (B x ...), run prior:  (B x PD)
        3. Broadcast the latent, and run the decoder on the latent vector & potentially some subset of the inputs, depending on decoder_input_selector.

        :return model_outputs: (d)  (B x ...)
        - embeddings/...
        - latent_posterior(_sample)
        - latent_prior(_sample)
        - decoder_names

        """
        if run_all:
            compute_posterior = compute_prior = True
            decode_posterior = decode_prior = True

        ## flag checks ##
        compute_latent = compute_posterior or compute_prior

        # if plan was not passed in but we wanna decode, we need to compute the plan first
        if not model_outs.has_node_leaf_key('latent_posterior_sample'):
            assert compute_posterior or not decode_posterior
        if not model_outs.has_node_leaf_key('latent_prior_sample'):
            assert compute_prior or not decode_prior

        # running accumulation
        model_outs = model_outs.leaf_copy()  # outputs

        if run_prepare:
            inputs = self._prepare_inputs(inputs, preproc=preproc, current_horizon=current_horizon)

        # for example, computing goals
        pl_inputs = inputs.leaf_copy()
        # pl_inputs.leaf_apply(lambda arr: arr.shape).pprint()


        with timeit("vae_forward"):
            if batch_element_mask is not None:
                pl_inputs.leaf_modify(lambda arr: arr[batch_element_mask])

            # also accepts a tuple with a specific order

            if compute_latent:
                with timeit("vae_forward/prior"):
                    latent_prior_kwargs = self.parse_kwargs_for_method("latent_prior", kwargs)
                    self.compute_latent_prior(pl_inputs, model_outs, **latent_prior_kwargs)

                if compute_posterior:
                    with timeit("vae_forward/posterior"):
                        latent_post_kwargs = self.parse_kwargs_for_method("latent_posterior", kwargs)
                        self.compute_latent_posterior(pl_inputs, model_outs, **latent_post_kwargs)

            samples = []
            if decode_posterior:
                samples.append("latent_posterior")
            if decode_prior:
                samples.append("latent_prior")

            # per plan sample, generate the decoder output.
            for latent_sample_name in samples:
                # broadcasting plans (B, zdim...) -> (B, H, ...), and copying it to top level
                pl_inputs.combine(model_outs[latent_sample_name + "_sample"].leaf_apply(
                    lambda arr: self.check_broadcast_latent(arr, current_horizon)))
                with timeit("vae_forward/decoder"):
                    # add the sampled plan + inputs + encodings, optional sampling the output
                    self.compute_decoder_outs(pl_inputs, model_outs, latent_sample_name, sample=False, **kwargs)
                    # model_outs[latent_sample_name + "_decoder"].leaf_shapes().pprint()

            if decode_random_latent:

                key = None
                if model_outs.has_node_leaf_key('latent_posterior_sample'):
                    key = "latent_posterior"
                elif model_outs.has_node_leaf_key('latent_prior_sample'):
                    key = "latent_prior"
                assert key is not None, "latent sample must be computed in order to decode random latent distribution"

                old_latents = model_outs[f"{key}_sample"]
                rand_latents = d()
                for k, latent in old_latents.leaf_items():
                    # either B, H, D or B, D
                    latent = latent.detach()
                    rand_latent_mean = latent.mean(dim=tuple(range(len(latent.shape) - 1)), keepdim=True)
                    rand_latent_std = latent.std(dim=tuple(range(len(latent.shape) - 1)), keepdim=True)
                    rand_latents[k] = torch.randn_like(latent) * rand_latent_std + rand_latent_mean

                pl_inputs.combine(rand_latents.leaf_apply(
                    lambda arr: self.check_broadcast_latent(arr, current_horizon)))
                with timeit("vae_forward/random_latent_decoder"):
                    self.compute_decoder_outs(pl_inputs, model_outs, "latent_random", sample=False, **kwargs)

            # move this to "top level" to sample appropriate action
            if decode_posterior:
                model_outs.combine(model_outs["latent_posterior_decoder"]
            elif decode_prior:
                model_outs.combine(model_outs["latent_prior_decoder"]

            if sample:
                decoder_sample = self.decoder_sample_fn(model_outs)
                assert decoder_sample.has_leaf_keys(self.decoder_names)
                model_outs.combine(decoder_sample)  # sampled action will be at the root level

            self.compute_additional_forward(pl_inputs, model_outs)

            return self._postproc_fn(pl_inputs, model_outs) if postproc else model_outs

    def compute_additional_forward(self, pl_inputs, model_outs):
        pass

    def additional_losses(self, model_outs, inputs, outputs, i=0, writer=None, writer_prefix="", current_horizon=None, normalize=True):
        # returns losses, extra_scalars
        return d(), d()

    # don't call this and then do backprop!! graphs are not properly retained for some reason.
    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, normalize_inputs=True, ret_dict=False,
             always_run_all=False, randomize_horizon=True, do_forward=True, meta=d(), model_outs=d(), **kwargs):
        """
        :param inputs: (d)  (B x H x ...)
        :param outputs: (d)  (B x H x ...)
        :param i: (int) current step, used to scale beta
        :param writer: (SummaryWriter)
        :param writer_prefix: (str)
        :param training: (bool)
        :param ret_dict: (bool)
        :param randomize_horizon: (bool) choose between min_horizon and horizon for this batch

        :return loss: (torch.Tensor)  (1,)
        """

        if randomize_horizon:
            current_horizon = random.randint(self.min_horizon, self.horizon)
        else:
            current_horizon = self.horizon

        if self.batch_mask_fn is not None:
            with timeit("loss/batch_mask"):
                batch_mask = self.batch_mask_fn(inputs)
                inputs = inputs.leaf_apply(lambda arr: arr[batch_mask])
                outputs = outputs.leaf_apply(lambda arr: arr[batch_mask])
        else:
            batch_mask = None

        run_all = always_run_all or writer is not None

        if not do_forward:
            assert not model_outs.is_empty(), "No outputs were provided, but model is not gonna run!"

        if i >= self._block_model_training_steps:

            if do_forward:
                # be careful if do_forward=False that you put all the necessary model_outs.
                with timeit("loss/forward"):
                    # inputs.leaf_apply(lambda arr: arr.shape).pprint()
                    model_outs = self.forward(inputs, preproc=True, postproc=True, compute_posterior=True,
                                              compute_prior=True,
                                              decode_posterior=True, decode_prior=False, run_all=run_all,
                                              decode_random_latent=run_all, sample=False, meta=meta)

            with timeit("loss/decoder_and_latent_loss"):

                model_outs.combine(model_outs["latent_posterior_decoder"])
                decoder_loss = self.decoder_loss_fn(self, model_outs, inputs, outputs, i=i, writer=writer,
                                                    writer_prefix=writer_prefix + "posterior/", normalize=normalize_inputs)

                latent_dist_loss = self.latent_dist_fn(model_outs["latent_prior"], model_outs["latent_posterior"],
                                                       inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix, normalize=normalize_inputs)
        else:
            model_outs = d()
            # blocks training, effectively.
            decoder_loss = torch.zeros(1, device=self.device)
            latent_dist_loss = torch.zeros(1, device=self.device)

        with timeit("loss/additional_losses"):
            additional_losses, extra_scalars = self.additional_losses(model_outs, inputs, outputs, i=i, writer=writer,
                                                                      writer_prefix=writer_prefix,
                                                                      current_horizon=current_horizon, normalize=normalize_inputs)

        additional_losses['latent_distance'] = (self.latent_beta, latent_dist_loss)

        decoder_loss = decoder_loss.mean()
        latent_dist_loss = latent_dist_loss.mean()

        loss = decoder_loss
        if i >= self._block_prior_training_steps:
            loss = loss + self.latent_beta * latent_dist_loss
        if not additional_losses.is_empty():
            for key, (weight, added_loss) in additional_losses.leaf_items():
                avg = added_loss.mean()
                additional_losses[key] = (weight, avg)
                loss += weight * avg

        if run_all and i >= self._block_model_training_steps:
            model_outs.combine(model_outs["latent_prior_decoder"])
            decoder_prior_loss = self.decoder_loss_fn(self, model_outs, inputs, outputs, i=i, writer=writer,
                                                     writer_prefix=writer_prefix + "prior/", normalize=normalize_inputs)

        if writer is not None:
            with timeit("writer"):
                writer.add_scalar(writer_prefix + "loss", loss.item(), i)
                if i >= self._block_model_training_steps:
                    writer.add_scalar(writer_prefix + "decoder_loss", decoder_loss.item(), i)
                    writer.add_scalar(writer_prefix + "beta", self.latent_beta, i)
                    writer.add_scalar(writer_prefix + "prior/decoder_loss", decoder_prior_loss.mean().item(), i)
                    writer.add_scalar(writer_prefix + "posterior/decoder_loss", decoder_loss.mean().item(),
                                      i)  # extra but good for consistency
                if batch_mask is not None:
                    writer.add_scalar(writer_prefix + "batch_utilization",
                                      torch.count_nonzero(batch_mask) / len(batch_mask), i)
                for key, (weight, add_loss) in additional_losses.leaf_items():
                    writer.add_scalar(writer_prefix + key, add_loss.item(), i)
                for key, scalar in extra_scalars.leaf_items():
                    writer.add_scalar(writer_prefix + key, scalar, i)

        if ret_dict:
            dc = d(
                loss=loss[None],
                latent_dist_loss=latent_dist_loss[None],
                decoder_loss=decoder_loss[None],
            ) & additional_losses.leaf_apply(lambda vs: vs[1])
            if run_all and i >= self._block_model_training_steps:
                dc.decoder_prior_loss = decoder_prior_loss.mean()[None]

            dc.model_outs = model_outs
            return dc

        return loss

    def set_beta_schedule(self, bs: float):
        self.beta_schedule = bs

    @property
    def latent_beta(self):
        return self.beta_schedule * self.beta

    # MODELS
    @property
    def posterior_input_selector(self):
        return self._posterior_input_selector

    @property
    def prior_input_selector(self):
        return self._prior_input_selector

    @property
    def decoder_input_selector(self):
        return self._decoder_input_selector

    @property
    def posterior(self):
        return self._posterior

    @property
    def prior(self):
        return self._prior

    @property
    def decoder(self):
        return self._decoder

