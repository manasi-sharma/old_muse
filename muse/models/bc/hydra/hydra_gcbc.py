from typing import Callable

import torch
from attrdict import AttrDict as d
from torch.nn import CrossEntropyLoss

from muse.models.bc.gcbc import BaseGCBC
from muse.models.bc.hydra.hydra_decoder import HydraActionDecoder
from muse.utils.abstract import Argument
from muse.utils.general_utils import timeit


class HydraGCBC(BaseGCBC):
    """
    All this does is define the default loss function using two mode losses.
    """
    predefined_arguments = BaseGCBC.predefined_arguments + [
        Argument('gamma', type=float, default=0.5, help="Weighting of action losses by mode per step"),
        Argument('mode_beta', type=float, default=0.01, help="Weighting of mode losses"),
        Argument('label_smoothing', type=float, default=0., help="Apply label smoothing to mode XE"),
        Argument('use_smooth_mode', action='store_true'),
    ]

    def _init_params_to_attrs(self, params: d):
        super()._init_params_to_attrs(params)

        # loss functions for each mode.
        self.mode0_loss_fn: Callable = params["mode0_loss_fn"]
        self.mode1_loss_fn: Callable = params["mode1_loss_fn"]

    def _init_setup(self):
        super()._init_setup()
        assert isinstance(self.action_decoder, HydraActionDecoder), "Not using action decoder for Hydra!"

        self.mode_loss_obj = CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # override this if you want your own loss
    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, ret_dict=False,
             meta=d(), **kwargs):
        """ Overwrite the loss fn for model.

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
        model_outputs = self.forward(inputs, training=training, **self._loss_forward_kwargs, **kwargs)

        true_mode = inputs[self.action_decoder.mode_key].to(dtype=torch.long)  # index

        B, H = true_mode.shape[:2]

        mode_action_losses = []
        coeffs = []

        # sparse mode loss
        with timeit(f'loss/policy_mode0_loss'):
            mode_action_losses.append(
                self.mode0_loss_fn(self, model_outputs, inputs, outputs, i=i, writer=writer,
                                   writer_prefix=writer_prefix + f"policy_mode1_loss/",
                                   **kwargs))

            coeffs.append(torch.where(true_mode == 0, 1 - self.gamma, self.gamma).view(B, H))

        # dense mode loss
        with timeit(f'loss/policy_mode0_loss'):
            mode_action_losses.append(
                self.mode1_loss_fn(self, model_outputs, inputs, outputs, i=i, writer=writer,
                                   writer_prefix=writer_prefix + f"policy_mode1_loss/",
                                   **kwargs))

            coeffs.append(torch.where(true_mode == 1, 1 - self.gamma, self.gamma).view(B, H))

        # mode probability loss
        mode_prob = model_outputs[self.action_decoder.mode_prob_out_name]

        with timeit('loss/mode_loss'):
            # predict the right mode
            if self.use_smooth_mode:
                smooth_mode = (inputs["smooth_mode"]).view(B * H, 1)
                all_prob = torch.cat([1. - smooth_mode, smooth_mode], dim=-1)
                mode_loss = self.mode_loss_obj(mode_prob.view(B * H, -1), all_prob).mean()
            else:
                mode_loss = self.mode_loss_obj(mode_prob.view(B * H, -1), true_mode.view(B * H)).mean()

        if writer is not None:
            with timeit('writer'):
                writer.add_scalar(writer_prefix + f"mode_loss", mode_loss.item(), i)
                for m in range(2):
                    writer.add_scalar(writer_prefix + f"action_mode{m}_loss", mode_action_losses[m].mean().item(),
                                      i)
                    writer.add_scalar(writer_prefix + f"weighted_action_mode{m}_loss",
                                      (coeffs[m] * mode_action_losses[m]).mean().item(), i)

        loss = sum((c * l).mean() for c, l in zip(coeffs, mode_action_losses))
        loss = self.mode_beta * mode_loss + loss

        return loss
