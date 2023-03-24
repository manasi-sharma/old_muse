from attrdict import AttrDict as d

from muse.experiments import logger
from muse.models.bc.lmp.lmp import LMPBaseGCBC
from muse.utils.abstract import Argument


class PLATO(LMPBaseGCBC):
    predefined_arguments = LMPBaseGCBC.predefined_arguments + [
        # run policy on init window
        Argument('do_pre_policy', action='store_true'),
        # weight of pre window action loss
        Argument('beta_pre', type=float, default=1.),
    ]

    def _init_params_to_attrs(self, params: d):
        super(PLATO, self)._init_params_to_attrs(params)

        if self.do_pre_policy:
            logger.info(f"PLATO using pre policy. beta_pre = {self.beta_pre}")

    def additional_loss(self, model_outputs, inputs, outputs, i=0, writer=None, writer_prefix="", **kwargs):
        loss_dc = super().additional_loss(model_outputs, inputs, outputs, i=i,
                                          writer=writer, writer_prefix=writer_prefix, **kwargs)

        if self.do_pre_policy:
            assert 'pre' in inputs.keys(), "inputs.pre is missing! " \
                                               "Make sure to use an interaction loader (e.g. NpInteractionDataset)"
            # compute the pre-interaction goal (same as in inputs)
            pre_inputs = inputs.pre.leaf_copy()
            if self.use_goal:
                pre_inputs &= model_outputs > self.goal_names

            # compute the pre-interaction loss using the posterior plan
            pre_model_outputs = self.forward(pre_inputs, select_goal=False, prior_plan=model_outputs.posterior.plan,
                                             **kwargs)
            pre_policy_loss = self.action_loss(pre_model_outputs, inputs, outputs, i=i,
                                               writer=writer, writer_prefix=writer_prefix + "pre/posterior/").mean()

            loss_dc['pre/posterior_policy_loss'] = (self.beta_pre, pre_policy_loss)

        return loss_dc
