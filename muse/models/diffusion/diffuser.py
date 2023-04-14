"""
Gaussian Diffusion layers

Adapted from Diffuser: https://github.com/jannerm/diffuser/blob/main/diffuser/models
"""
import numpy as np
import torch
from typing import List

from muse.experiments import logger
from muse.models.model import Model
from muse.utils.loss_utils import write_avg_per_last_dim
from muse.utils.general_utils import timeit
from muse.utils.torch_utils import combine_then_concatenate, unsqueeze_n, concatenate, \
    combine_after_dim, torch_disable_grad

from attrdict import AttrDict as d
from attrdict.utils import get_with_default


# def get_loss_weights(action_weight, discount, weights_dict):
#     '''
#         sets loss coefficients for trajectory
#         action_weight   : float
#             coefficient on first action loss
#         discount   : float
#             multiplies t^th timestep of trajectory loss by discount**t
#         weights_dict    : dict
#             { i: c } multiplies dimension i of observation loss by c
#     '''
#     self.action_weight = action_weight
#
#     dim_weights = torch.ones(self.input_size, dtype=torch.float32)
#
#     ## set loss coefficients for dimensions of observation
#     if weights_dict is None: weights_dict = {}
#     for ind, w in weights_dict.items():
#         dim_weights[self.cond_slice_dim + ind] *= w
#
#     ## decay loss with trajectory timestep: discount**t
#     discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
#     discounts = discounts / discounts.mean()
#     loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
#
#     ## manually set a0 weight
#     loss_weights[0, :self.cond_slice_dim] = action_weight
#     return loss_weights

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32, device=None):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype, device=device)


def apply_conditioning(x, conditions, dim_range=None):
    """ Sets conditions as states in the array

    conditions: [(time, state) ... ]
    """
    if dim_range is None:
        dim_range = slice(x.shape[2])  # all
    for t, val in conditions:
        x[:, t, dim_range] = val[..., dim_range].clone()
    return x


def apply_relative(x, h=0, dim_range=None):
    """ Sets dim_range of x to be relative to that at timestep h. if dim_range=None, does all
    """
    if dim_range is None:
        dim_range = slice(x.shape[2])
    base = x[:, h, dim_range].clone()
    x = x.clone()
    x[:, :, dim_range] -= base.unsqueeze(1)
    return x, base


def apply_rebase(x, base, dim_range, inverse=False):
    if dim_range is None:
        dim_range = slice(x.shape[2])

    x = x.clone()
    x[..., dim_range] += (1 - 2 * int(inverse)) * unsqueeze_n(base, len(x.shape) - len(base.shape), 1)
    return x


class DiffusionModel(Model):
    def _init_params_to_attrs(self, params):

        #     horizon, observation_dim, cond_slice_dim, n_timesteps=1000,
        #     clip_denoised=False, predict_epsilon=True,
        # ):
        self.inputs = params["model_inputs"]
        self.input_size = self.env_spec.dim(self.inputs)

        # these are the inputs to the conditioning portion of the model.
        self.extra_inputs = params["model_extra_inputs"]
        self.extra_input_size = self.env_spec.dim(self.extra_inputs)

        # self.output = str(params["model_output"])
        self.generator = (params["network"]).to_module_list(as_sequential=True).to(self.device)

        # if > 0, will use dynamics loss with this weighting
        self.dynamics_beta = get_with_default(params, "dynamics_beta", 0)
        if self.dynamics_beta > 0:
            self.state_names = params["state_names"]  # outputs of model
            # self.action_names = params["action_names"]  # other inputs
            # dynamics go inputs -> next states
            # assert set(self.state_names).isdisjoint(self.action_names)
            # assert set(self.state_names).union(self.action_names) == set(self.inputs), \
            #     [self.inputs, self.state_names, self.action_names]
            self.dynamics = (params["dynamics_network"]).to_module_list(as_sequential=True).to(self.device)
            logger.info(f"Diffusion using dynamics penalty (beta = {self.dynamics_beta})")

        self.num_diffusion_steps = int(params["num_diffusion_steps"])
        self.clip_denoised = get_with_default(params, "clip_denoised", False)
        self.predict_epsilon = get_with_default(params, "predict_epsilon", True)

        self._unnormalize_outs = get_with_default(params, "unnormalize_outs", True)

        self._compute_negatives = get_with_default(params, "compute_negatives", 0)  # > 0 means do negatives

        self._get_default_condition_fn = get_with_default(params, "get_default_condition_fn", lambda inputs, names: [])

        self.concat_dtype = get_with_default(params, "concat_dtype", torch.float32)

        self.cond_slice_dim = params << "condition_slice_dim"  # where to slice in flat x when applying conditioning

        self.relative_to_h = get_with_default(params, "relative_to_h", None)
        self.relative_slice_dim = params << "relative_slice_dim"  # where to use "relative" inputs in flat x (s0=0)
        if self.relative_to_h is not None:
            logger.debug(f"Using relative to h = {self.relative_to_h}")
            assert self.dynamics_beta == 0, "Dynamics not implemented for relative diffusion"

    def _init_setup(self):

        betas = cosine_beta_schedule(self.num_diffusion_steps, device=self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss uses single step diffusion, regresses onto noise
        self._loss_forward_kwargs['single_step'] = True
        self._loss_forward_kwargs['inf_steps'] = -1
        self._loss_forward_kwargs['do_dynamics'] = self.dynamics_beta > 0
        self._loss_forward_kwargs['compute_negatives'] = self._compute_negatives

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, extra=None):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.generator((x, cond, t, extra)))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, extra=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, extra=extra)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, extra=None, verbose=True, return_diffusion=False, do_cond_slice=True):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.cond_slice_dim if do_cond_slice else None)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, extra=extra)
            x = apply_conditioning(x, cond, self.cond_slice_dim if do_cond_slice else None)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=2)  # dim=1 before
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon, batch_size=None, extra=None, do_cond_slice=True, **kwargs):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        batch_size = batch_size or len(cond[0][1])
        shape = (batch_size, horizon, self.input_size)

        return self.p_sample_loop(shape, cond, *args, extra=extra, do_cond_slice=do_cond_slice, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def _preamble(self, inputs, normalize=True, preproc=True):
        if normalize and self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=self.concat_dtype)

        if preproc:
            inputs = self._preproc_fn(inputs)
        return inputs

    def dynamics_forward(self, inputs, input_arr=None, training=False, do_preamble=True, preproc=True, postproc=True,
                         normalize=False, unnormalize=None, compute_negatives=0, current_horizon=None, meta=None,
                         **kwargs):
        assert self.dynamics_beta > 0

        if do_preamble and inputs is not None:
            inputs = self._preamble(inputs, normalize=normalize)

        if unnormalize is None:
            unnormalize = normalize

        if input_arr is None:
            input_arr = combine_then_concatenate(inputs, self.inputs, 2)

        # should output states (state_names)
        output_arr = self.dynamics(input_arr)

        outputs = cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(output_arr, self.state_names),
                                  self.state_names, concat_arr=output_arr, after_dim=2)

        if unnormalize:
            outputs = self.normalize_by_statistics(outputs, self.state_names, inverse=True)

        return outputs

    def forward(self, inputs, index=None, single_step=False, inf_steps=0, condition=None, training=False, preproc=True,
                postproc=True, unnormalize_outs=None, compute_negatives=0, current_horizon=None, do_dynamics=False,
                meta=None, do_cond_slice=True, **kwargs):
        """
        Sampling forward:
            fr_steps == 0: Model is generative (no forward steps, inputs not used)
            fr_steps == N, N > 1: Model does N forward and reverse steps, and aggregates all
        Training forward:
            if single_step = True, will do 1 forward and reverse step.

        :param inputs: AttrDict of (B, H, ...) only used if single_step=True or if condition is None ( to parse default condition )

        :param index: Diffusion index of each input element
                If single_step
                    If None, will sample index uniformly
                    Else index.shape == (B, ), will run a single forward and reverse diffusion step.
                Else:
                    index must be None
        :param single_step: see (index) .. if True will sample one forward and reverse step of diffusion for each input (training)
        :param inf_steps: inference steps
                        -1  --> skip inference
                        0   --> do generative sampling
                        > 0 --> do forward + reverse sampling
        :param condition: conditions on sampling process. should be a list of [(time, state_val_dict) ... ]
                          where each state_val_dict item is of shape (B x ...).
                          TODO : mask
        :param unnormalize_outs: If true, will un-normalize the inference outputs. defaults to provided value in params
        :param compute_negatives: If > 0, will train contrastively. only works if single_step=False
        :param current_horizon: If not None and single_step=False, will use this horizon for output sample.
                if None and !single_step, will get the horizon from inputs (MAKE SURE these are provided)
        :param do_cond_slice: Do conditional slice sampling
        :param kwargs:

        :return: NORMALIZED sample and/or NORMALIZED single diffusion step
        """

        if inputs is not None:
            inputs = self._preamble(inputs)

        if condition is None:
            condition = self._get_default_condition_fn(inputs, self.inputs)

        assert isinstance(condition, List)
        # AttrDict -> array for each condition
        cond = [(t, combine_then_concatenate(val_dict, self.inputs, 1)) for t, val_dict in condition]

        outs = d()

        extra = None
        if self.extra_input_size > 0:
            # extra info to condition generator. assume this info comes in (B x H x ...) and we only look at arr[:, 0]
            extra = concatenate((inputs > self.extra_inputs) \
                                .leaf_apply(lambda arr: combine_after_dim(arr[:, 0], 1)), self.extra_inputs, dim=1)

        """ TRAINING """
        if single_step:
            with timeit("diffusion/x_noisy"):
                # B x H x D
                x_start = combine_then_concatenate(inputs, self.inputs, 2)
                if self.relative_to_h is not None:
                    x_start, x_delta = apply_relative(x_start, self.relative_to_h, self.relative_slice_dim)
                    cond = [(t, apply_rebase(v, x_delta, self.relative_slice_dim, inverse=True)) for t, v in cond]

                if index is None:
                    index = torch.randint(0, self.num_diffusion_steps, (x_start.shape[0],),
                                          device=x_start.device).long()
                noise = torch.randn_like(x_start)

                # one forward diffusion step
                x_noisy = self.q_sample(x_start=x_start, t=index, noise=noise)
                x_noisy = apply_conditioning(x_noisy, cond, self.cond_slice_dim if do_cond_slice else None)

            with timeit("diffusion/x_recon"):
                # network does one reverse diffusion step after the forward
                x_recon = self.generator((x_noisy, cond, index, extra))
                x_recon = apply_conditioning(x_recon, cond, self.cond_slice_dim if do_cond_slice else None)

            # CAttrDict for easy alternation between concatenated and unconcatenated views.
            outs = d(
                index=index,
                start=cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(x_start, self.inputs),
                                      self.inputs, concat_arr=x_start, after_dim=2),
                noise=cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(noise, self.inputs),
                                      self.inputs, concat_arr=noise, after_dim=2),
                forward=cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(x_noisy, self.inputs),
                                        self.inputs, concat_arr=x_noisy, after_dim=2),
                reverse=cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(x_recon, self.inputs),
                                        self.inputs, concat_arr=x_recon, after_dim=2),
            )

            # TODO
            # if self.relative_to_h is not None:
            #     outs['delta'] = cd.from_dynamic(self.env_spec.parse_view_from_concatenated_flat(x_delta, self.inputs),
            #                                     self.inputs, concat_arr=x_delta, after_dim=2)

            if compute_negatives > 0:
                with timeit("diffusion/x_negative"):
                    # network does one reverse diffusion step after the forward
                    x_full_noise = torch.randn(x_start.shape[0] * compute_negatives, x_start.shape[1:],
                                               device=self.device)
                    raise NotImplementedError

            if do_dynamics:
                with timeit("diffusion/recon_dynamics"):
                    # reconstructed x will get a dynamics check, used to update diffusion model.
                    with torch_disable_grad(self.dynamics):
                        outs.reverse_next = self.dynamics_forward(None, input_arr=x_recon, do_preamble=False,
                                                                  normalize=False)

                with timeit("diffusion/true_dynamics"):
                    # used for training dynamics model.
                    outs.next = self.dynamics_forward(None, input_arr=x_start, do_preamble=False, normalize=False)

        """ INFERENCE """
        if inf_steps >= 0:
            if unnormalize_outs is None:
                unnormalize_outs = self._unnormalize_outs

            if inf_steps == 0:
                x0 = None
                # PURE GENERATIVE
                assert index is None, "Diffusion index cannot be provided during sampling!"
                batch_size = inputs.get_one().shape[0] if len(cond) == 0 else None
                horizon = current_horizon or inputs.get_one().shape[1]
                # (B x H x ...) and (B x N x H x ...)

                if self.relative_to_h is not None:
                    if self.relative_to_h in [t for t, v in cond]:
                        x_delta = [v[..., self.relative_slice_dim] for t, v in cond if t == self.relative_to_h][
                            0]  # this is what we will rebase to.
                    else:
                        raise NotImplementedError("cannot sample relative inputs with h != conditional timesteps")

                with timeit("diffusion/conditional_sample"):
                    p_sample, all_index_samples = self.conditional_sample(cond=cond, return_diffusion=True,
                                                                          batch_size=batch_size, horizon=horizon,
                                                                          extra=extra, do_cond_slice=do_cond_slice,
                                                                          **kwargs)
            else:
                # FORWARD + REVERSE (inputs required)
                assert inf_steps <= self.num_diffusion_steps
                x0 = combine_then_concatenate(inputs, self.inputs, 2)

                if self.relative_to_h is not None:
                    x0, x_delta = apply_relative(x0, self.relative_to_h, self.relative_slice_dim)

                xt = x0

                diff_states = [xt]

                # forward process
                t = torch.full((xt.shape[0],), 0, device=xt.device).long()
                with timeit("diffusion/q_sample"):
                    for _ in range(inf_steps):
                        xt = self.q_sample(xt, t)
                        xt = apply_conditioning(xt, cond, self.cond_slice_dim if do_cond_slice else None)
                        diff_states.append(xt)
                        t += 1

                # reverse process
                with timeit("diffusion/p_sample"):
                    for _ in range(inf_steps):
                        t -= 1
                        xt = self.p_sample(xt, cond, t, extra=extra)
                        xt = apply_conditioning(xt, cond, self.cond_slice_dim if do_cond_slice else None)
                        diff_states.append(xt)

                p_sample = xt
                # first N are forward steps, last N are reverse steps
                all_index_samples = torch.stack(diff_states, dim=2)

            if self.relative_to_h is not None:
                p_sample = apply_rebase(p_sample, x_delta, self.relative_slice_dim)
                all_index_samples = apply_rebase(all_index_samples, x_delta, self.relative_slice_dim)

            # Aggregate inference results and view
            sample = self.env_spec.parse_view_from_concatenated_flat(p_sample, self.inputs)
            all_sample = self.env_spec.parse_view_from_concatenated_flat(all_index_samples, self.inputs)

            # unnormalize
            if unnormalize_outs:
                to_normalize = list(set(self.normalization_inputs).intersection(self.inputs))
                sample = self.normalize_by_statistics(sample, to_normalize, inverse=True)
                all_sample = self.normalize_by_statistics(all_sample, to_normalize, inverse=True)

            outs.sample = sample
            outs.all_sample = all_sample

            if do_dynamics:
                # sampled x will be used with dynamics.
                outs.sample['next'] = self.dynamics_forward(None, input_arr=p_sample, do_preamble=False,
                                                            normalize=False,
                                                            unnormalize=unnormalize_outs)

                # if you specified inputs, we will also use them for dynamics
                if x0 is None and inputs is not None:
                    x0 = combine_then_concatenate(inputs, self.inputs, 2)

                if x0 is not None:
                    outs['next'] = self.dynamics_forward(None, input_arr=x0, do_preamble=False, normalize=False,
                                                         unnormalize=unnormalize_outs)

        if postproc:
            outs = self._postproc_fn(inputs, outs)

        return outs

    @staticmethod
    def get_default_loss_fn(err_fn, loss_weights=None):
        if loss_weights is not None:
            loss_weights = torch.Tensor(loss_weights)
            logger.info(f"Using loss weights: {loss_weights}")

        state_idxs = None

        def dyn_err(model: DiffusionModel, diff_out, pred_dc, true_arr):
            nonlocal state_idxs
            assert isinstance(pred_dc, cd), type(pred_dc)
            norm_pred = pred_dc.concat()[:, :-1]  # first H-1 outputs
            if state_idxs is None:
                state_idxs = model.env_spec.get_indices_for_flat(model.state_names, model.inputs)
            norm_true = true_arr[:, 1:, ..., state_idxs]  # last H-1 states

            return err_fn(norm_pred, norm_true)

        def loss_fn(model: DiffusionModel, diff_out: d, ins: d, outs: d, i=0, writer=None, writer_prefix="",
                    normalize=True, ret_dict=False, **kwargs):

            # move loss weights to correct device
            nonlocal loss_weights
            if loss_weights is not None and loss_weights.device != model.device:
                loss_weights = loss_weights.to(device=model.device)

            # diff_out will contain all needed to compute the loss
            start, noise, reverse = diff_out.get_keys_required(['start', 'noise', 'reverse'])

            target = start
            if model.predict_epsilon:  # predict the noise (epsilon) rather than the next x
                target = noise

            assert isinstance(reverse, cd), type(reverse)
            assert isinstance(target, cd), type(target)

            # should take (B x H x ...) and return (B x H x ...) shape
            err = err_fn(reverse.concat(), target.concat())
            loss = diff_loss = err

            if loss_weights is not None:
                loss = loss * loss_weights

            mean_scale = 0.
            if model.dynamics_beta > 0:
                time = diff_out["index"]
                assert time.shape[0] == loss.shape[0]  # (B,)
                mask = (time == 0).to(dtype=torch.float32)  # only for index == 0 steps do we compute dynamics loss
                # mask_mean = mask.mean().item()
                # mean_scale = 1. if (mask_mean > 0) else 0.  # OLD version: divide by mask_mean to make mean() correct, given many 0 elements.

                # dynamics loss portion, for all the true x0
                dynamics_train_loss = dyn_err(model, diff_out, diff_out["next"], start.concat())

                # extra dynamics penalty for being far from the true next state (only when index == 0)
                # old used "mean_scale * " here
                dynamics_penalty = unsqueeze_n(mask, 2, dim=-1) * \
                                   dyn_err(model, diff_out, diff_out["reverse_next"], reverse.concat())

                loss = loss.mean() + model.dynamics_beta * dynamics_penalty.mean()
            else:
                loss = loss.mean()

            if writer is not None:
                write_avg_per_last_dim(err, i, writer, writer_prefix + "err/dim_")
                write_avg_per_last_dim(diff_loss, i, writer, writer_prefix + "diff_loss/dim_")
                if model.dynamics_beta > 0:
                    writer.add_scalar(writer_prefix + "diff_loss", diff_loss.mean().item(), i)
                    writer.add_scalar(writer_prefix + "diff_penalty_loss", loss.item(), i)  # just diff + penalty so far
                    writer.add_scalar(writer_prefix + "dyn_penalty", dynamics_penalty.mean().item(), i)
                    writer.add_scalar(writer_prefix + "dynamics_beta", model.dynamics_beta, i)
                    writer.add_scalar(writer_prefix + "mask_sum", mask.sum().item(), i)

                    if mask.sum().item() > 0:
                        write_avg_per_last_dim(dynamics_train_loss, i, writer, writer_prefix + "dyn_loss/dim_")
                        write_avg_per_last_dim(dynamics_penalty, i, writer, writer_prefix + "dyn_penalty/dim_")

            if model.dynamics_beta > 0:
                loss = loss + dynamics_train_loss.mean()

            if ret_dict:
                return d(loss=loss) & \
                    (d(dynamics_loss=dynamics_train_loss, dynamics_penalty=dynamics_penalty,
                       diffusion_loss=diff_loss)
                     if model.dynamics_beta > 0 else d())

            return loss

        return loss_fn

    @staticmethod
    def get_default_default_condition_fn(use_first=False, use_last=False):
        def get_default_cond_fn(inputs, state_names, **kwargs):
            # inputs will be (B x H)
            cond = []
            if use_first or use_last:
                ins = inputs > state_names
                H = ins.get_one().shape[1]
                if use_first:
                    state = ins.leaf_apply(lambda arr: arr[:, 0])
                    cond.append((0, state))
                if use_last:
                    last_state = ins.leaf_apply(lambda arr: arr[:, -1])
                    cond.append((H - 1, last_state))
            return cond

        return get_default_cond_fn
