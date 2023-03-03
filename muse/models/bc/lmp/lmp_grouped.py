import random

import torch
from typing import Callable

from muse.experiments import logger
from muse.grouped_models.grouped_model import GroupedModel
from muse.utils.general_utils import timeit, is_next_cycle

from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.utils.torch_utils import split_dim, broadcast_dims


class LMPGroupedModel(GroupedModel):
    """
    Learning Latent Plans from Play -- algorithm.
    Consists of Plan prior, posterior, and policy.

    Parameters
    * min_horizon: minimum horizon
    * horizon: maximum horizon (horizon used for getting batch)
    * beta: LMP regularization weighting (e.g. KL prior / posterior)
    * plan_size: latent vector dimension

    Names / Name Lists
    TODO which of these do we actually need?
    * plan_name: latent vector plan name
    * action_names: subset of env_spec.action_names, to specify policy actions
    * prior_input_names: subset of env_spec.names, not including goal necessarily
    * prior_goal_state_names: subset of env_spec.names
    * posterior_input_names: subset of env_spec.names, not including goal necessarily
    * posterior_goal_state_names: subset of env_spec.goal_names

    * state_encoder_names: should map to the passed in models
    * plan_sample_fn: samples outputs from the plan distribution.

    GroupedModel model field
    * [1+] state_encoder(s): map to Model, applied first to inputs, each encoder linked by state_encoder_names
    * goal_selector: network to take in H x (encoded states, inputs, goals) -> goal_state
    * posterior_input_selector: network that takes inputs and goals and maps to posterior names
    * prior_input_selector: network that takes inputs and goals and maps to prior names
    * posterior: network to take in H x (encoded states + inputs + goal_state) -> plan distribution
    * prior: network to take H x (subset of encoded states, goal) -> plan distribution
    * policy: network to take inputs + plans -> actions

    """

    required_models = ["goal_selector",
                       "posterior_input_selector",
                       "prior_input_selector",
                       "policy_input_selector",
                       "posterior",
                       "prior",
                       "policy"
                       ]

    goal_parsed_kwargs = {}
    encoder_parsed_kwargs = {}
    plan_prior_parsed_kwargs = {}
    plan_posterior_parsed_kwargs = {}

    def _init_params_to_attrs(self, params: d):
        super(LMPGroupedModel, self)._init_params_to_attrs(params)

        # PARAMS
        self.beta = get_with_default(params, "beta", 1.0)
        self.beta_schedule = 1.  # default is 100% of beta
        self.beta_info = get_with_default(params, "beta_info", 0.)

        # represents the minimum horizon to use for recognition & policy sequences.
        self.horizon = params["horizon"]
        self.min_horizon = get_with_default(params, "min_horizon", self.horizon)
        assert 1 < self.min_horizon <= self.horizon, [self.min_horizon, self.horizon]
        self.plan_size = get_with_default(params, "plan_size", default=64)
        self._optimize_prior = get_with_default(params, "optimize_prior", default=False)
        self._optimize_policy = get_with_default(params, "optimize_policy", default=True)

        if self._optimize_prior:
            logger.warn("Optimizing LfP using the prior!")

        # TODO beta schedule
        # logger.debug("Beta = %f, custom scheduler = %s" % (self.beta, params.has_leaf_key("beta_schedule_fn")))
        # self.beta_schedule_fn = get_with_default(params, "beta_schedule_fn", default=lambda step: 1.)
        # takes in each of the inputs and returns embedding for obs, proprio

        # NAMES
        self.plan_name = get_with_default(params, "plan_name", default="plan")
        self.action_names = get_with_default(params, "action_names", default=["action"])


        # self.all_input_names = get_with_default(params, "all_inputs", default=self._env_spec.all_names)
        # self.include_goal_proprio = get_with_default(params, "include_goal_proprio", default=False)

        # all state encoders, run first, default is all the non-required models
        self.state_encoder_names = list(get_with_default(params, "state_encoder_names",
                                                         set(self._sorted_model_order).difference(
                                                             self.required_models)))
        # (plan) -> (sampled plan)
        self.set_fn("plan_sample_fn", params["plan_sample_fn"], Callable[[d], d])

        # (plan_prior, plan_posterior, inputs, outputs) -> KL loss between plan proposed and recognized
        self.set_fn("plan_dist_fn", params["plan_dist_fn"], Callable[[d, d, d, d], torch.Tensor])

        # (action) -> sampled action
        self.set_fn("action_sample_fn", params["action_sample_fn"], Callable[[d], d])

        # (self, model_outs, ins, outs) ->
        self.set_fn("action_loss_fn", params["action_loss_fn"],
                    Callable[[__class__, d, d, d], torch.Tensor])

        if params.has_leaf_key("batch_mask_fn"):
            # takes inputs, outputs (B,) tensor mask
            self.batch_mask_fn = params["batch_mask_fn"]
        else:
            self.batch_mask_fn = None

        self._block_model_training_steps = get_with_default(params, "block_model_training_steps", 0)
        self._block_kl_training_steps = get_with_default(params, "block_kl_training_steps", 0)
        # in plan_distance, pass in a fixed posterior. makes the prior have no regularizing effect.
        if self._block_model_training_steps > 0:
            logger.info(f"Blocking Model training for {self._block_model_training_steps} steps")
        if self._block_kl_training_steps > 0:
            assert self._block_model_training_steps <= self._block_kl_training_steps
            logger.info(f"Blocking KL training for {self._block_kl_training_steps} steps")

        if self.beta_info > 0:
            logger.info(f"Using Info Gain beta = {self.beta_info}")

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def encoders_forward(self, inputs: d, encoder_names=None, **kwargs):
        """
        :param inputs: (d)  (B x H x ...)
        :param encoder_names
        :return model_outputs: (d)  (B x H x ...)
        """
        results = d()
        if encoder_names is None:
            encoder_names = self.state_encoder_names
        assert set(encoder_names).issubset(self.state_encoder_names), [encoder_names, self.state_encoder_names]

        for n in encoder_names:
            results.safe_combine(self._models[n](inputs, **kwargs), warn_conflicting=True)
        return results

    def _prepare_inputs(self, inputs, preproc=True, current_horizon=None) -> d:
        # accept tuple inputs
        if not isinstance(inputs, d):
            assert len(inputs) == len(self.all_input_names), [self.all_input_names, len(inputs)]
            inputs = d.from_dict({k: v for k, v in zip(self.all_input_names, inputs)})
        else:
            inputs = inputs.leaf_copy()

        # varying time horizon
        if current_horizon is not None:
            with timeit("horizon_truncate"):
                inputs.leaf_assert(lambda arr: arr.shape[1] >= current_horizon)
                # truncate to appropriate length
                inputs.leaf_modify(lambda arr: arr[:, :current_horizon])

        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=torch.float32)

        return self._preproc_fn(inputs) if preproc else inputs

    def compute_plan_prior(self, pl_inputs, model_outs, sample=True, **kwargs):
        # get the proposed plan (start and goal)
        plan_propose_ins = self.prior_input_selector(pl_inputs)
        plan_propose_outs = self.prior(plan_propose_ins)
        model_outs.plan_prior = plan_propose_outs

        if sample:
            # sample from proposal otherwise, in policy for example
            sampled_plan = self.plan_sample_fn(plan_propose_outs)
            model_outs.plan_prior_sample = sampled_plan

        return plan_propose_outs

    def compute_plan_posterior(self, pl_inputs, model_outs, sample=True, **kwargs):
        # get the recognized plan
        plan_recog_ins = self.posterior_input_selector(pl_inputs)
        plan_recog_outs = self.posterior(plan_recog_ins)
        model_outs.plan_posterior = plan_recog_outs

        if sample:
            # sample from recognition for training, for example
            sampled_plan = self.plan_sample_fn(plan_recog_outs)
            model_outs.plan_posterior_sample = sampled_plan

        return plan_recog_outs

    def compute_policy_outs(self, pl_inputs, model_outs, plan_sample_name, sample=False, **kwargs):
        # add the sampled plan + inputs + encodings, optional sampling the output
        policy_ins = self.policy_input_selector(pl_inputs)
        policy_outs = self.policy(policy_ins, **kwargs)
        model_outs[plan_sample_name + "_policy"].combine(policy_outs)

        if sample:
            policy_sample = self.action_sample_fn(model_outs[plan_sample_name + "_policy"])
            assert policy_sample.has_leaf_keys(self.action_names)
            model_outs[plan_sample_name + "_policy"].combine(policy_sample)  # sampled action will be at the root level

        return model_outs[plan_sample_name + "_policy"]

    def compute_goals(self, pl_inputs, model_outs, **kwargs):
        # AttrDict (B x H ...)
        goal_states = self.goal_selector(pl_inputs)
        # default mask is to include all states as attributed to this goal
        model_outs.goal_states = goal_states.leaf_copy()
        return goal_states

    def check_broadcast_plan(self, plan, current_horizon):
        assert current_horizon is not None
        assert len(plan.shape) > 1, f"Plan needs to be batched: {plan.shape}"
        assert plan.shape[-1] == self.plan_size, plan.shape
        if len(plan.shape) == 2:
            plan = plan.unsqueeze(1)  # horizon dim
        return broadcast_dims(plan, dims=[-2], new_shape=[current_horizon])

    # run on a plan and goal
    def forward(self, inputs, preproc=True, postproc=True, run_prepare=True, plan_posterior=False, sample=False, current_horizon=None,
                run_enc=True, run_goal_select=True, run_plan=True, run_random_plan_policy=False, run_policy=True,
                run_all=False, batch_element_mask=None, plan_posterior_policy=None, model_outs=d(), **kwargs):
        """
        LMP forward:
        0. inputs are (B x H x ...)
        1. run all state encoders on inputs, resulting in (B x H x ...)
        2. Run goal selector to get the goals for the sequence, along with the temporal mask (B x H)
            yielding goals: (B x ...) and mask: (B x H).
        3. For posterior (full seq):
            a. inputs from posterior_input_selector(inputs, goals) -> (B x H x ...), run posterior : (B x H x PD)
        4. For prior (seq / name subset):
            a. inputs from prior_input_selector(inputs, goals) -> (B x ...), run prior:  (B x PD)
        5. combine all inps, etc

        :param inputs: (d)  (B x H x ...), H >= 2
        :param preproc:
        :param postproc:
        :param sample: for the OUTPUT action, plan is always sampled for forward
        :param current_horizon: The "window" size to use when running the model forward. 2 <= current_horizon <= H

        :return model_outputs: (d)  (B x ...)
        - embeddings/...
        - plan_posterior(_sample)
        - plan_prior(_sample)
        - action_names

        """
        assert plan_posterior or not run_all, "Cannot run all without running plan_recognition forward"

        if plan_posterior_policy is None:
            plan_posterior_policy = plan_posterior

        if run_prepare:
            inputs = self._prepare_inputs(inputs, preproc=preproc, current_horizon=current_horizon)

        # if plan was not passed in, we need to compute the plan first.
        if not model_outs.has_node_leaf_key('plan_posterior_sample') and not model_outs.has_node_leaf_key('plan_prior_sample'):
            # .leaf_apply(lambda arr: arr.shape).pprint()
            assert run_plan or not run_all, "Cannot run all without running some plan forward"

        # running accumulation
        model_outs = model_outs.leaf_copy()  # outputs

        # for example, computing goals
        pl_inputs = inputs.leaf_copy()
        # pl_inputs.leaf_apply(lambda arr: arr.shape).pprint()

        with timeit("lmp_forward"):

            if batch_element_mask is not None:
                pl_inputs.leaf_modify(lambda arr: arr[batch_element_mask])

            # also accepts a tuple with a specific order

            if run_enc:
                with timeit("lmp_forward/encoders"):
                    # encode states
                    encoder_kwargs = self.parse_kwargs_for_method("encoder", kwargs)
                    embeddings = self.encoders_forward(inputs, **encoder_kwargs)
                    if not embeddings.is_empty():
                        model_outs.embeddings = embeddings
                        pl_inputs.combine(embeddings)

            if run_goal_select:
                with timeit("lmp_forward/goals"):
                    # AttrDict (B x H ...)
                    goal_kwargs = self.parse_kwargs_for_method("goal", kwargs)
                    pl_inputs.goal_states = self.compute_goals(pl_inputs, model_outs, **goal_kwargs)
            else:
                assert pl_inputs.has_node_leaf_key("goal_states"), "Goals required but run_goal_select = False and none present"

            if run_plan:
                with timeit("lmp_forward/prior"):
                    plan_prior_kwargs = self.parse_kwargs_for_method("plan_prior", kwargs)
                    self.compute_plan_prior(pl_inputs, model_outs, **plan_prior_kwargs)

                if plan_posterior:
                    with timeit("lmp_forward/posterior"):
                        plan_post_kwargs = self.parse_kwargs_for_method("plan_posterior", kwargs)
                        self.compute_plan_posterior(pl_inputs, model_outs, **plan_post_kwargs)

            if run_all:
                samples = ("plan_posterior", "plan_prior")
            elif plan_posterior_policy:
                samples = ("plan_posterior",)
            else:
                samples = ("plan_prior",)

            if run_policy:
                # per plan sample, generate the policy output.
                for plan_sample_name in samples:
                    # broadcasting plans (B, zdim...) -> (B, H, ...), and copying it to top level
                    pl_inputs.combine(model_outs[plan_sample_name + "_sample"].leaf_apply(
                        lambda arr: self.check_broadcast_plan(arr, current_horizon)))
                    with timeit("lmp_forward/policy"):
                        # add the sampled plan + inputs + encodings, optional sampling the output
                        self.compute_policy_outs(pl_inputs, model_outs, plan_sample_name, sample=False, **kwargs)

                if run_random_plan_policy:
                    old_plans = model_outs[f"{samples[0]}_sample"]
                    rand_plans = d()
                    for k, plan in old_plans.leaf_items():
                        # either B, H, D or B, D
                        plan = plan.detach()
                        rand_plan_mean = plan.mean(dim=tuple(range(len(plan.shape) - 1)), keepdim=True)
                        rand_plan_std = plan.std(dim=tuple(range(len(plan.shape) - 1)), keepdim=True)
                        rand_plans[k] = torch.randn_like(plan) * rand_plan_std + rand_plan_mean

                    pl_inputs.combine(rand_plans.leaf_apply(
                            lambda arr: self.check_broadcast_plan(arr, current_horizon)))
                    with timeit("lmp_forward/random_plan_policy"):
                        self.compute_policy_outs(pl_inputs, model_outs, "plan_random", sample=False, **kwargs)

                # move this to "top level" to sample appropriate action
                if plan_posterior_policy:
                    model_outs.combine(model_outs.plan_posterior_policy)
                else:
                    model_outs.combine(model_outs.plan_prior_policy)

                if sample:
                    policy_sample = self.action_sample_fn(model_outs)
                    assert policy_sample.has_leaf_keys(self.action_names)
                    model_outs.combine(policy_sample)  # sampled action will be at the root level

            self.compute_additional_forward(pl_inputs, model_outs)

            return self._postproc_fn(pl_inputs, model_outs) if postproc else model_outs

    def compute_additional_forward(self, pl_inputs, model_outs):
        pass

    def additional_losses(self, model_outs, inputs, outputs, i=0, writer=None, writer_prefix="", current_horizon=None):
        # returns losses, extra_scalars
        return d(), d()

    def _get_policy_outputs(self, inputs, outputs, model_outputs, current_horizon=None):
        """
        returns the policy outputs (current_horizon - 1)

        :param inputs:
        :param outputs:
        :param model_outputs:
        :param current_horizon:
        :return:
        """
        outs = outputs.leaf_copy()
        # semantically, an action is an input, even though it is an output in our case
        for key in self.action_names:
            assert inputs[key].shape[1] == self.horizon
            # relevant actions are B, H-1 (predicting the action at state s_0...sH-1)
            outs[key] = inputs[key][:, :current_horizon - 1]
        return outs

    # don't call this and then do backprop!! graphs are not properly retained for some reason.
    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, ret_dict=False,
             randomize_horizon=True, do_prior_policy=False, do_posterior_policy=False, meta=d(), **kwargs):
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

        run_all = (do_posterior_policy if self._optimize_prior else do_prior_policy) or writer is not None
        plan_post_policy = not self._optimize_prior or run_all  # run policy for posterior plan

        if i >= self._block_model_training_steps:

            with timeit("loss/forward"):
                # inputs.leaf_apply(lambda arr: arr.shape).pprint()
                model_outs = self.forward(inputs, preproc=True, postproc=True, plan_posterior=True,
                                          plan_posterior_policy=plan_post_policy,
                                          current_horizon=current_horizon, run_all=run_all, run_random_plan_policy=run_all or self.beta_info > 0, sample=False, meta=meta)

            with timeit("loss/action_and_plan_loss"):

                if self._optimize_policy or run_all:
                    outs = self._get_policy_outputs(inputs, outputs, model_outs, current_horizon=current_horizon)

                    if self._optimize_prior:
                        model_outs.combine(model_outs["plan_prior_policy"])
                        policy_loss = self.action_loss_fn(self, model_outs, inputs, outs, i=i, writer=writer,
                                                                    writer_prefix=writer_prefix + "prior/")
                    else:
                        model_outs.combine(model_outs["plan_posterior_policy"])
                        policy_loss = self.action_loss_fn(self, model_outs, inputs, outs, i=i, writer=writer,
                                                                    writer_prefix=writer_prefix + "posterior/")

                    if run_all or self.beta_info > 0:
                        model_outs.combine(model_outs["plan_random_policy"])
                        random_policy_loss = self.action_loss_fn(self, model_outs, inputs, outs, i=i, writer=writer,
                                                                    writer_prefix=writer_prefix + "random/")
                    else:
                        random_policy_loss = 0
                else:
                    policy_loss = torch.zeros(1, device=self.device)
                    random_policy_loss = 0

                plan_dist_loss = self.plan_dist_fn(model_outs["plan_prior"], model_outs["plan_posterior"],
                                              inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix)
        else:
            model_outs = d()
            # blocks training, effectively.
            policy_loss = torch.zeros(1, device=self.device)
            plan_dist_loss = torch.zeros(1, device=self.device)

        with timeit("loss/additional_losses"):
            additional_losses, extra_scalars = self.additional_losses(model_outs, inputs, outputs, i=i, writer=writer,
                                                                      writer_prefix=writer_prefix,
                                                                      current_horizon=current_horizon)

        if self.beta_info > 0:
            # information gain, pi(s, g, z) - beta * pi(s, g, z_random)
            additional_losses['random/policy_loss'] = (-self.beta_info, random_policy_loss)
        elif run_all:
            extra_scalars['random/policy_loss'] = random_policy_loss.mean().item()

        # print(policy_loss, proposed_plan_and_encoded_start_goal.plan_dist.mean,
        #       recognized_plan_and_encoded_horizon.plan_dist.mean)

        policy_loss = policy_loss.mean()
        plan_dist_loss = plan_dist_loss.mean()

        loss = policy_loss if self._optimize_policy else torch.zeros_like(policy_loss)
        if i >= self._block_kl_training_steps:
            loss = loss + self.lmp_beta * plan_dist_loss
        if not additional_losses.is_empty():
            for key, (weight, added_loss) in additional_losses.leaf_items():
                avg = added_loss.mean()
                additional_losses[key] = (weight, avg)
                loss += weight * avg

        if run_all and i >= self._block_model_training_steps:
            if self._optimize_prior:
                model_outs.combine(model_outs["plan_posterior_policy"])
                policy_prior_loss = policy_loss
                policy_posterior_loss = self.action_loss_fn(self, model_outs, inputs, outs, i=i, writer=writer,
                                                       writer_prefix=writer_prefix + "posterior/")

            else:
                policy_posterior_loss = policy_loss
                model_outs.combine(model_outs["plan_prior_policy"])
                policy_prior_loss = self.action_loss_fn(self, model_outs, inputs, outs, i=i, writer=writer,
                                                       writer_prefix=writer_prefix + "prior/")

        if writer is not None:
            with timeit("writer"):
                writer.add_scalar(writer_prefix + "loss", loss.item(), i)
                if i >= self._block_model_training_steps:
                    writer.add_scalar(writer_prefix + "plan_distance", plan_dist_loss.item(), i)
                    writer.add_scalar(writer_prefix + "policy_loss", policy_loss.item(), i)
                    writer.add_scalar(writer_prefix + "beta", self.lmp_beta, i)
                    writer.add_scalar(writer_prefix + "prior/policy_loss", policy_prior_loss.mean().item(), i)
                    writer.add_scalar(writer_prefix + "posterior/policy_loss", policy_posterior_loss.mean().item(),
                                      i)  # extra but good for consistency
                if batch_mask is not None:
                    writer.add_scalar(writer_prefix + "batch_utilization", torch.count_nonzero(batch_mask) / len(batch_mask), i)
                for key, (weight, add_loss) in additional_losses.leaf_items():
                    writer.add_scalar(writer_prefix + key, add_loss.item(), i)
                for key, scalar in extra_scalars.leaf_items():
                    writer.add_scalar(writer_prefix + key, scalar, i)

        if ret_dict:
            dc = d(
                loss=loss[None],
                plan_dist_loss=plan_dist_loss[None],
                policy_loss=policy_loss[None],
            ) & additional_losses.leaf_apply(lambda vs: vs[1])
            if run_all and i >= self._block_model_training_steps:
                dc.policy_prior_loss = policy_prior_loss.mean()[None]

            dc.model_outs = model_outs
            return dc

        return loss

    def set_beta_schedule(self, bs: float):
        self.beta_schedule = bs

    @property
    def lmp_beta(self):
        return self.beta_schedule * self.beta

    # MODELS
    @property
    def goal_selector(self):
        return self._goal_selector

    @property
    def posterior_input_selector(self):
        return self._posterior_input_selector

    @property
    def prior_input_selector(self):
        return self._prior_input_selector

    @property
    def policy_input_selector(self):
        return self._policy_input_selector

    @property
    def posterior(self):
        return self._posterior

    @property
    def prior(self):
        return self._prior

    @property
    def policy(self):
        return self._policy

    @staticmethod
    def get_default_mem_policy_forward_fn(replan_horizon, action_names, policy_rnn_hidden_name='hidden_policy', recurrent=False, sample_plan=False, flush_horizon=None, **kwargs):
        # online execution using MemoryPolicy or subclass
        if flush_horizon is None:
            flush_horizon = replan_horizon
        def mem_policy_model_forward_fn(model: LMPGroupedModel, obs: d, goal: d, memory: d,
                                        known_sequence=None, **kwargs):
            obs = obs.leaf_copy()
            if memory.is_empty():
                memory.policy_rnn_h0 = None
                memory.count = 0
                # print(timeit)
                # timeit.reset()

            H = max(obs.leaf_apply(lambda arr: arr.shape[1]).leaf_values())
            AH = H
            if not goal.is_empty():
                AH = H + 1  # we will be concatenating.
                obs.goal_states = goal
                if 'sample_first' not in kwargs.keys():
                    kwargs['sample_first'] = False  # disable sampling online

            if H == 1:
                assert not goal.is_empty(), "Goal must be specified if H = 1 for obs"

            action_filler = model.env_spec.get_zeros(action_names, AH, torch_device=model.device) \
                .leaf_apply(lambda arr: split_dim(arr, 0, [1, AH]))  # used only for normalization

            if is_next_cycle(memory.count, flush_horizon) or memory.count == 0:
                memory.policy_rnn_h0 = None

            # happens @ beginning, every H steps
            if is_next_cycle(memory.count, replan_horizon) or memory.count == 0:
                if not goal.is_empty():
                    kwargs['run_goal_select'] = False
                # memory.policy_rnn_h0 = None
                if known_sequence is not None:
                    # get z from plan recog, then run policy on current obs
                    out = model.forward(known_sequence, sample=False, rnn_hidden_init=None, plan_posterior=True, run_policy=False,
                                        current_horizon=known_sequence.get_one().shape[1], **kwargs)
                    dist = out["plan_posterior/plan_dist"]
                else:
                    # plan proposal, filler actions for forward call
                    out = model.forward(obs & action_filler, sample=False, rnn_hidden_init=None, current_horizon=H, run_policy=False, **kwargs)
                    dist = out["plan_prior/plan_dist"]

                if sample_plan:
                    memory.plan = dist.sample()
                else:
                    memory.plan = dist.mean.detach()

            memory.count += 1

            # normal policy w/ fixed plan, we use prior, doesn't really matter here tho since run_plan=False
            model_outs = d(plan_prior_sample=d(plan=memory["plan"]))
            out = model.forward(obs & action_filler, rnn_hidden_init=memory["policy_rnn_h0"], run_enc=True, run_plan=False, run_policy=True,
                                plan_posterior=False,  # use the prior plan
                                model_outs=model_outs,
                                sample=False, current_horizon=H, **kwargs)
            # NEXT OUTPUT
            if recurrent:
                memory.policy_rnn_h0 = out[policy_rnn_hidden_name]
            return out

        return mem_policy_model_forward_fn
