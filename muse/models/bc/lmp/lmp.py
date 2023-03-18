from typing import Callable

import numpy as np
import torch
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from muse.experiments import logger
from muse.models.bc.gcbc import BaseGCBC
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit, value_if_none, is_next_cycle
from muse.utils.torch_utils import broadcast_dims, to_scalar


class LMPBaseGCBC(BaseGCBC):
    """
    Adds a prior and posterior to the basic GCBC architecture.

    These both have their own input names, and output a dictionary with "plan_name"
    """

    required_models = ['prior', 'posterior'] + \
                      BaseGCBC.required_models

    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument('beta', type=float, default=1e-3, help='weight of KL term'),
        Argument('beta_info', type=float, default=0.),
        Argument('plan_size', type=int, default=64),

        Argument('optimize_prior', action='store_true',
                 help='Use prior plan for policy during training'),
    ]

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)

        if self.optimize_prior:
            logger.info("Optimizing LMP using the prior!")

        # NAMES
        self.plan_name = get_with_default(params, "plan_name", default="plan")

        # make sure plan is in spec
        if self.plan_name not in self.env_spec.all_spec_names:
            logger.warn(f"[LMP] Adding {self.plan_name} to env_spec!")
            self.env_spec.add_nsld(self.plan_name, (self.plan_size,), (-np.inf, np.inf), np.float32)

        # (plan) -> (sampled plan)
        self.set_fn("plan_sample_fn", params["plan_sample_fn"], Callable[[d], d])

        # (plan_prior, plan_posterior, inputs, outputs) -> KL loss between plan proposed and recognized
        self.set_fn("plan_dist_fn", params["plan_dist_fn"], Callable[[d, d, d, d], torch.Tensor])

        if self.beta_info > 0:
            logger.debug(f"Using Info Gain beta = {self.beta_info}")

    def _init_setup(self):
        super()._init_setup()

        # forward during loss computation requires the posterior / decoder from posterior plan.
        # run the posterior decoder only if we are not optimizing the prior decoder
        self._loss_forward_kwargs = {'select_goal': True,
                                     'run_posterior': True,
                                     'run_posterior_decoder': not self.optimize_prior}

    def compute_plan_prior(self, pl_inputs, sample=True, **kwargs):
        """ Get the proposed plan (start and goal)

        Parameters
        ----------
        pl_inputs:
            (B x H x ...) inputs to the plan models
        sample:
            whether or not to sample from plan distribution
        kwargs

        Returns
        -------
        plan_prior_outs: outputs of proposal
        """
        # get the first goal and first state only.
        inputs = pl_inputs.leaf_apply(lambda arr: arr[:, :1])

        plan_prior_outs = self.prior(inputs, **kwargs)

        if sample:
            # sample from proposal otherwise, in policy for example
            plan_prior_outs.combine(self.plan_sample_fn(plan_prior_outs))

        return plan_prior_outs

    def compute_plan_posterior(self, pl_inputs, sample=True, **kwargs):
        """ Get the posterior plan (full temporal sequence)

        Parameters
        ----------
        pl_inputs: AttrDict
            (B x H x ...) inputs to the plan models
        sample:
            whether or not to sample from the plan distribution
        kwargs

        Returns
        -------
        plan_posterior_outs: outputs of posterior
        """
        # run the posterior
        plan_posterior_outs = self.posterior(pl_inputs, **kwargs)

        if sample:
            # sample from recognition for training, for example
            plan_posterior_outs.combine(self.plan_sample_fn(plan_posterior_outs))

        return plan_posterior_outs

    def forward(self, inputs,
                select_goal=False,
                run_posterior=False,
                run_posterior_decoder=False,
                prior_plan=None,
                posterior_plan=None,
                prior_kwargs=None,
                posterior_kwargs=None,
                **kwargs):
        """ LMP call to forward()

        NOTE: does not use preproc or postproc actions.
        Does the following
        (1) encoders
        (2) goal selection
        (3) compute the plan prior
        (4) [run_posterior=True] compute the plan posterior
        (5) run the policy (action decoder) conditioned on plan prior
        (6) [run_posterior_policy=True] run the policy (action decoder) conditioned on plan posterior

        Parameters
        ----------
        inputs
        select_goal: bool
            if True and self.use_goal, will select the goal from the inputs if not present
                (otherwise it must be provided in inputs)
        run_posterior: bool
            run the posterior to get the plan
        run_posterior_decoder: bool
            run the policy (action decoder) using the posterior plan
        prior_plan: tensor or none
            if provided, prior will use this instead of calling the prior to compute the plan.
        posterior_plan: tensor or none
            if !run_posterior and run_posterior_policy, this plan is required for the posterior.
        prior_kwargs: dict
            kwargs to prior
        posterior_kwargs: dict
            kwargs to prior
        kwargs:
            decoder arguments.

        Returns
        -------
        outputs: AttrDict
            - prior
                - <all prior output names>
                - <plan name>
            - posterior
                - <all posterior output names>
                - <plan name>
            - prior_decoder
                - <all decoder output names>
                - <all action names>
            - posterior_decoder
                - <all decoder output names>
                - <all action names>
        """

        inputs = inputs.leaf_copy()

        outputs = d()

        prior_kwargs = value_if_none(prior_kwargs, {})
        posterior_kwargs = value_if_none(posterior_kwargs, {})

        if run_posterior_decoder:
            assert run_posterior ^ (posterior_plan is not None), \
                "For run_posterior_decoder=True, either pass in a plan tensor or set run_posterior=True (not both)"

        with timeit("lmp/encoders"):
            # call each encoder forward, and add to inputs
            for enc_name in self.state_encoder_order:
                inputs.combine(self[enc_name](inputs))

        # get the goal and add it
        if self.use_goal:
            if select_goal:
                inputs.combine(self.select_goals(inputs))
            else:
                assert inputs.has_leaf_keys(self.goal_names), f"Missing goal names {self.goal_names} from input!"

        if prior_plan is None:
            with timeit("lmp/prior"):
                outputs.prior = self.compute_plan_prior(inputs, **prior_kwargs)
        else:
            outputs.prior = d.from_dict({self.plan_name: prior_plan})

        if run_posterior:
            # compute the posterior
            with timeit("lmp/posterior"):
                outputs.posterior = self.compute_plan_posterior(inputs, **posterior_kwargs)
        elif posterior_plan is not None:
            # use an externally computed posterior plan.
            outputs.posterior = d.from_dict({self.plan_name: posterior_plan})

        sample_sources = ['prior']
        if run_posterior_decoder:
            sample_sources.append('posterior')

        # per plan sample, generate the policy output.
        for sample_source in sample_sources:
            plan = outputs[sample_source][self.plan_name]
            policy_inputs = inputs.leaf_copy()

            # horizon is computed from any one of the inputs
            horizon = inputs.get_one().shape[1]

            # checking the plan shape
            assert len(plan.shape) > 1, f"Plan from {sample_source} needs to be batched: {plan.shape}"
            assert plan.shape[-1] == self.plan_size, plan.shape
            if len(plan.shape) == 2:
                plan = plan.unsqueeze(1)  # add in the horizon dim

            policy_inputs[self.plan_name] = broadcast_dims(plan, dims=[-2], new_shape=[horizon])

            # actually run the decoder using samples
            with timeit(f"lmp/{sample_source}_decoder"):
                outputs[sample_source + "_decoder"] = self.action_decoder(policy_inputs, **kwargs)
                outputs[f"{sample_source}_decoder"] = self.action_decoder(policy_inputs, **kwargs)

        # move the policy outputs (either posterior / prior) to top level for optimization / policy to use
        if not run_posterior_decoder or self.optimize_prior:
            # move the prior decoder output to the top level
            outputs.combine(outputs.prior_decoder)
        else:
            # move the posterior decoder output to the top level
            outputs.combine(outputs.posterior_decoder)

        return inputs & outputs

    def additional_loss(self, model_outputs, inputs, outputs, i=0, writer=None, writer_prefix="", **kwargs):
        """ Additional losses that operate on posterior/prior, for example.

        Parameters
        ----------
        model_outputs
        inputs
        outputs
        i
        writer
        writer_prefix
        kwargs

        Returns
        -------
        Dict[str, Tuple[Union[Number, torch.Tensor], torch.Tensor]]
            consisting of {
                loss_name: (coeff, loss_tensor)
                ...
            }

        """
        with timeit('loss/plan_dist_loss'):
            # plan divergence loss, e.g. KL divergence
            plan_dist_loss = self.plan_dist_fn(model_outputs.prior, model_outputs.posterior,
                                               inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix)

        return {
            'plan_dist_loss': (self.beta, plan_dist_loss)
        }

    def action_loss(self, model_outputs, inputs, outputs, i=0, writer=None, writer_prefix="", **kwargs):
        loss = self._loss_fn(self, model_outputs, inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix,
                             ret_dict=False, **kwargs)
        return loss

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, ret_dict=False,
             meta=d(), **kwargs):
        """ LMP loss function

        Computes parent loss (BC objective) and also computes plan distance.

        Parameters
        ----------
        inputs
        outputs
        i
        writer
        writer_prefix
        training
        ret_dict
        meta
        kwargs

        Returns
        -------

        """

        # call forward to get all the model outputs
        forward_kwargs = self._loss_forward_kwargs.copy()
        # run posterior decoder if we are only optimizing the prior decoder and we are in a writing step
        if writer is not None and self.optimize_prior:
            forward_kwargs['run_posterior_decoder'] = True
        model_outs = self.forward(inputs, training=training, **forward_kwargs, **kwargs)

        # get decoder losses
        plan_prefix = "prior/" if self.optimize_prior else "posterior/"
        policy_loss = self.action_loss(model_outs, inputs, outputs, i=i,
                                       writer=writer, writer_prefix=writer_prefix + plan_prefix).mean()

        unsupervised_losses = self.additional_loss(model_outs, inputs, outputs, i=i, writer=writer,
                                                   writer_prefix=writer_prefix, ret_dict=ret_dict,
                                                   meta=meta, **kwargs)

        # add all additional unsupervised losses in
        loss = policy_loss
        means = {}
        for k, tup in unsupervised_losses.items():
            coeff, loss_tensor = tup
            means[k] = loss_tensor.mean()
            loss += coeff * means[k]

        if writer is not None:
            with timeit('writer'):
                # write the loss that was used to train end to end.
                writer.add_scalar(writer_prefix + "policy_loss", policy_loss.item(), i)

                # write both the prior and posterior decoder output losses
                #   (one of these will be same as policy_loss above)
                if self.optimize_prior:
                    prior_policy_loss = policy_loss
                    # compute action loss using the posterior decoder outputs
                    posterior_policy_loss = self.action_loss(model_outs & model_outs.posterior_decoder, inputs, outputs,
                                                             i=i, writer=writer,
                                                             writer_prefix=writer_prefix + "posterior/").mean()
                else:
                    # compute action loss using the posterior decoder outputs
                    prior_policy_loss = self.action_loss(model_outs & model_outs.prior_decoder, inputs, outputs, i=i,
                                                         writer=writer, writer_prefix=writer_prefix + "prior/").mean()
                    posterior_policy_loss = policy_loss

                # write each one
                writer.add_scalar(writer_prefix + "prior_policy_loss", prior_policy_loss.item(), i)
                writer.add_scalar(writer_prefix + "posterior_policy_loss", posterior_policy_loss.item(), i)

                # write the unsupervised losses and coefficients
                for k in unsupervised_losses.keys():
                    # add the coefficient
                    writer.add_scalar(writer_prefix + k + "_weight", to_scalar(unsupervised_losses[k][0]), i)
                    writer.add_scalar(writer_prefix + k, means[k].item(), i)

                # also write the total loss.
                if not ret_dict:
                    writer.add_scalar(writer_prefix + "loss", loss.item(), i)

        return loss

    def get_default_mem_policy_forward_fn(self, *args, replan_horizon=0, **kwargs):
        """ Function that policy will use to run model forward (see GCBCPolicy)

        Keeps track of planning horizon.

        Parameters
        ----------
        args
        replan_horizon: int
            how often to replan (default is every step)
        kwargs

        Returns
        -------

        """
        if replan_horizon == 0:
            logger.warn("LMP policy model forward, will replan every time step!")

        fn = super().get_default_mem_policy_forward_fn(*args, **kwargs)

        def inner_fn(model, obs, goal, memory, **inner_kwargs):
            # fill in plan if it is present
            # assumes "count" will be tracked and updated in "fn" above

            if 'count' not in memory.keys() or is_next_cycle(memory.count, replan_horizon):
                inner_kwargs['prior_plan'] = None
            else:
                # this will be filled in from previous step
                inner_kwargs['prior_plan'] = memory.plan

            # call parent fn (GCBC forward)
            out = fn(model, obs, goal, memory, **inner_kwargs)

            # put prior plan into memory for next time.
            memory.plan = out.prior[self.plan_name]

            return out

        return inner_fn
