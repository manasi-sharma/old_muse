"""
Similar to a basic model, but requires a full sequence at eval as well.

Sequences should be passed in at training of length seq_len, and
"""
import torch

from muse.models.basic_model import BasicModel
from muse.models.model import Model
from attrdict import AttrDict
from attrdict.utils import get_with_default


class SequenceModel(BasicModel):
    """
    Seq2Seq model 
    """

    def _init_params_to_attrs(self, params):
        self._horizon = params["horizon"]
        # self._online_horizon = get_with_default(params, "online_horizon", self.horizon)
        # which index to get of the (B, H, C) output of the sequence.
        # self.default_output_horizon_idx = get_with_default(params, "default_output_horizon_idx", -1)
        assert self._horizon > 1, "Do not use sequence model with horizon <= 1"
        super()._init_params_to_attrs(params)

    def forward(self, inputs, training=False, preproc=True, postproc=True, mask=None, **kwargs):
        # make sure all inputs match the horizon length (B x H x ...)
        inputs.leaf_assert(lambda arr: arr.shape[1] == self._horizon)
        if mask is not None:
            if hasattr(self.net, "set_mask"):
                self.net.set_mask(mask)
            elif isinstance(self.net, torch.nn.Sequential):
                assert any(hasattr(n, "set_mask") for n in self.net), \
                           "Seq must have at least one layer that can set_mask()"
                for n in self.net:
                    if hasattr(n, "set_mask"):
                        n.set_mask(mask)
            else:
                raise ValueError("Masked sequence models require set_mask()")
        return super().forward(inputs, training=training, preproc=preproc, postproc=postproc, **kwargs)
