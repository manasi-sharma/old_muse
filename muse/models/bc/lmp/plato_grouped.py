import torch

from muse.experiments import logger
from muse.models.bc.lmp.lmp_grouped import LMPGroupedModel
from muse.datasets.fast_np_interaction_dataset import NpInteractionDataset
from muse.models.dist.helpers import detach_normal
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from muse.utils.torch_utils import broadcast_dims


class PLATOGroupedModel(LMPGroupedModel):
    """
    A contact interaction is defined as follows:
    [period of non contact]
    [grasp]
    [period of semi-continuous contact]
    [release]
    [optional period of non contact afterwards]

    This model requires inputs to have all of these segments in the horizon.
    LMP is used to learn the affordances, from [a subset] of the contact period

    """

    predefined_arguments = LMPGroupedModel.predefined_arguments + [
        # run policy on init window
        Argument('do_init_policy', action='store_true'),
        # use goals)
        Argument('goal_sampling', action='store_true'),
        # weight of initiation window action loss
        Argument('beta_init', type=float, default=1.),
    ]

    def _init_params_to_attrs(self, params: d):
        super(PLATOGroupedModel, self)._init_params_to_attrs(params)
        # self._get_contact_start_ends = params["get_contact_start_ends"]
        # self._variable_horizon = get_with_default(params, "variable_horizon", True)

        # no gradients from initiation policy to encoder
        self._detach_init_plan = get_with_default(params, "detach_init_plan", False)
        # discounting action reconstruction seq based on how far away from affordance.
        self._init_discount = get_with_default(params, "init_discount", 1.)

        assert not self._detach_init_plan or self.do_init_policy, "Cannot detach init plan without enabling init pol."

        # assert isinstance(self._get_contact_start_ends, Callable)
        if self._dataset_train is not None:
            assert isinstance(self._dataset_train, NpInteractionDataset), \
                f"Dataset must be compatible with reading interactions: {type(self._dataset_train)}"

        if self.do_init_policy:
            logger.info(f"ContactLMP using initiation policy. beta_init = {self.beta_init}")

    def forward(self, inputs, preproc=True, postproc=True, run_prepare=True, current_horizon=None,
                do_init=True, plan_posterior=False, run_all=False, run_policy=None,
                model_outs=d(), **kwargs):
        """ Forward call for PLATO

        Parameters
        ----------
        inputs: AttrDict
            inputs consisting of (B x H x ...) tensors
        preproc: bool
            whether to run self._preproc_fn before
        postproc: bool
            whether to run self._postproc_fn after
        run_prepare: bool
            whether to run preparation actions (input normalization, parsing tuple inputs potentially)
        current_horizon: int
            the horizon to use in place of H (e.g. if using variable length batches) 2 <= current_horizon <= H
        do_init: bool
            do the initiation action decoding
        plan_posterior: bool
            whether to run the posterior network at all (otherwise will just use prior & its plan)

        run_all: bool
            run the policy on *both* prior and posterior plans
            requires that plan_posterior = True
        run_policy: bool
            run the policy using whichever plans were generated (depending on run_all and plan_posterior_policy)

        model_outs: AttrDict
        kwargs: other arguments for LMPGroupedModel.forward()

        Returns
        -------

        """

        if run_policy is None:
            run_policy = run_all or not self.no_contact_policy  # default during training.

        with timeit("contact_lmp/prepare_inputs"):
            model_outs = model_outs.leaf_copy()

            if run_prepare:
                # make sure normalization inputs are in all of these.
                inputs = self._prepare_inputs(inputs, preproc=preproc, current_horizon=None)
                # all keys that aren't initiation keys
                lmp_inputs = inputs.node_leaf_filter_keys([k for k in list(inputs.keys()) if k != "initiation"])
                # max over the shape[1] dims
                contact_current_horizon = lmp_inputs.leaf_reduce(lambda red, val: max(red, val.shape[1]), seed=1)

                if "initiation" in inputs.keys():
                    # do NOT truncate
                    inputs.initiation = self._prepare_inputs(inputs.initiation, preproc=preproc, current_horizon=None)

                if "goal_states" in inputs.keys():
                    # do NOT truncate
                    inputs.goal_states = self._prepare_inputs(inputs.goal_states, preproc=preproc,
                                                              current_horizon=None)
                    inputs.goal_states = inputs.goal_states.leaf_apply(lambda arr:
                                                                       broadcast_dims(arr, [1],
                                                                                      [contact_current_horizon]))

            if self.do_init_policy and do_init:
                init_lmp_inputs = inputs["initiation"]
                # copy over goals
                if "goal_states" not in init_lmp_inputs.keys():
                    init_lmp_inputs.goal_states = lmp_inputs["goal_states"]
                init_current_horizon = lmp_inputs.leaf_reduce(lambda red, val: max(red, val.shape[1]), seed=1)

        with timeit("contact_lmp/lmp_forward"):
            # run full PLATO on the contact window
            if "run_goal_select" not in kwargs.keys():
                kwargs['run_goal_select'] = not self.goal_sampling
            lmp_outputs = super(PLATOGroupedModel, self).forward(lmp_inputs, preproc=preproc, postproc=postproc,
                                                                 run_prepare=False, run_all=run_all,
                                                                 plan_posterior=plan_posterior,
                                                                 model_outs=model_outs, run_policy=run_policy,
                                                                 current_horizon=contact_current_horizon,
                                                                 **kwargs)
            model_outs.combine(lmp_outputs)

        if self.do_init_policy and do_init:
            # run just the PLATO policy on init window, but condition on future contact affordance.
            with timeit("contact_lmp/initiation_lmp_forward"):
                # at least one will be here.
                init_lmp_outs = (model_outs < ['embedding', 'plan_posterior_sample', 'plan_prior_sample']).leaf_copy()
                if self._detach_init_plan:
                    for key, arr in (init_lmp_outs < ['plan_posterior_sample', 'plan_prior_sample']).leaf_items():
                        if isinstance(arr, torch.distributions.Distribution):
                            arr = detach_normal(arr)  # only supports Normal or independent normals
                        else:
                            # tensor
                            arr = arr.detach()
                        init_lmp_outs[key] = arr
                init_lmp_outs = super(PLATOGroupedModel, self).forward(init_lmp_inputs, preproc=preproc,
                                                                       postproc=postproc,
                                                                       run_prepare=False, run_plan=False,
                                                                       run_all=run_all,
                                                                       plan_posterior=plan_posterior,
                                                                       model_outs=init_lmp_outs,
                                                                       current_horizon=init_current_horizon,
                                                                       run_goal_select=not self.goal_sampling)
                model_outs.initiation.combine(init_lmp_outs)

        return self._postproc_fn(inputs, model_outs) if postproc else model_outs

    # def _get_policy_outputs(self, inputs, outputs, model_outputs, current_horizon=None, use_mask=True):
    #     """
    #     returns the policy outputs (current_horizon - 1), aligned with the contact sample
    #
    #     Parameters
    #     ----------
    #     inputs: raw inputs (B x current_horizon)
    #     outputs:
    #     model_outputs:
    #     current_horizon:
    #     :return:
    #     """
    #     outs = outputs.leaf_copy()
    #
    #     new_outs = inputs > list(self.action_names)
    #
    #     new_outs = new_outs.leaf_apply(lambda arr: arr[:, :-1])  # skip last step
    #     if current_horizon is not None:
    #         new_outs.leaf_assert(lambda arr: arr.shape[1] == current_horizon - 1)
    #
    #     return outs & new_outs

    def additional_losses(self, model_outs, inputs, outputs, i=0, writer=None, writer_prefix="", current_horizon=None):
        """
        Computes the initiation window losses

        Parameters
        ----------
        model_outs: AttrDict
            fully computed model outputs
        inputs
        outputs
        i
        writer
        writer_prefix
        current_horizon

        Returns
        -------

        """
        losses, extra_scalars = super(PLATOGroupedModel, self).additional_losses(model_outs, inputs, outputs, i=i,
                                                                                 writer=writer,
                                                                                 writer_prefix=writer_prefix,
                                                                                 current_horizon=current_horizon)

        if self.do_init_policy:
            # if we are computing initiation window, compute the action loss using the posterior affordance
            init_model_outs = (model_outs["initiation"]).leaf_copy()
            init_model_outs.combine(init_model_outs["plan_posterior_policy"])

            # initiation policy outputs, don't require padding mask over loss.
            outs = self._get_policy_outputs(inputs["initiation"], outputs, model_outs, current_horizon=current_horizon)

            # these inputs/outs are the contact window, make sure action loss fn does not rely on these!
            initiation_policy_posterior_loss = self.action_loss_fn(
                self, init_model_outs, inputs, outs,
                i=i, writer=writer, writer_prefix=writer_prefix + "posterior/initiation/"
            )

            # write the prior loss here, since it doesn't get optimized directly.
            if writer is not None:
                init_model_outs.combine(init_model_outs["plan_prior_policy"])

                # write the prior policy
                initiation_policy_prior_loss = self.action_loss_fn(self, init_model_outs, inputs, outs,
                                                                   i=i, writer=writer,
                                                                   writer_prefix=writer_prefix + "prior/initiation/")

                writer.add_scalar(writer_prefix + "prior/initiation/policy_loss",
                                  initiation_policy_prior_loss.mean().item(), i)

            # this loss will get added and logged to writer
            losses['posterior/initiation/policy_loss'] = (self.beta_init, initiation_policy_posterior_loss)

        return losses, extra_scalars
