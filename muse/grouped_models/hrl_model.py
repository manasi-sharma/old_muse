"""
HRL model: action model for current level, and child model to represent any sub models.
"""
from collections import Callable

import torch

from muse.grouped_models.grouped_model import GroupedModel
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from muse.utils.general_utils import timeit


class HRLModel(GroupedModel):

    required_models = [
        "action_model",
        "child_model",
    ]

    def _init_params_to_attrs(self, params: d):
        super(HRLModel, self)._init_params_to_attrs(params)

        # these let you customize how losses get computed. default is passthru.
        self.set_fn("_action_model_forward_fn", get_with_default(params, "action_model_forward_fn",
                                                                 lambda *args, **kwargs: self.action_model.forward(*args, **kwargs)), Callable)  # mutation okay

        self.set_fn("_action_model_loss_fn", get_with_default(params, "action_model_loss_fn",
                                                                 lambda *args, **kwargs:
                                                                 self.action_model.loss(*args, **kwargs)), Callable)  # mutation okay

        self.set_fn("_child_model_forward_fn", get_with_default(params, "child_model_forward_fn",
                                                                 lambda *args, **kwargs: self.child_model.forward(*args, **kwargs)), Callable)  # mutation okay

        self.set_fn("_child_model_loss_fn", get_with_default(params, "child_model_loss_fn",
                                                                 lambda *args, **kwargs:
                                                                 self.child_model.loss(*args, **kwargs)), Callable)  # mutation okay

        self._do_child_model_loss = get_with_default(params, "do_child_model_loss", True)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def preamble(self, inputs, preproc=True):
        inputs = inputs.leaf_copy()

        if self.normalize_inputs:
            inputs = self.normalize_by_statistics(inputs, self.normalization_inputs, shared_dtype=torch.float32)

        with timeit("hrl_model/preproc"):
            if preproc:
                inputs = self._preproc_fn(inputs)

        return inputs

    def forward(self, inputs, training=False, preproc=True, postproc=True, prior_action_outputs=None, model_outputs=d(), **kwargs):
        inputs = self.preamble(inputs, preproc=preproc)
        # default behavior: compute the action model from parent, and feed that into the child
        action_kwargs = {k[7:]: v for k, v in kwargs.items() if k.startswith("action_")}
        child_kwargs = {k[6:]: v for k, v in kwargs.items() if k.startswith("child_")}
        # run the action model only when prior actions are not specified (this allows for hierarchy at the training level.
        if prior_action_outputs is None:
            action_outs = self._action_model_forward_fn(self.action_model, inputs, training=training, **action_kwargs)
        else:
            action_outs = prior_action_outputs

        child_inputs = inputs & action_outs
        child_outs = self._child_model_forward_fn(self.child_model, child_inputs, training=training, **child_kwargs)

        return self._postproc_fn(inputs, child_outs) if postproc else child_outs

    def loss(self, inputs, outputs, i=0, writer=None, writer_prefix="", training=True, **kwargs):
        action_kwargs = {k[7:]: v for k, v in kwargs.items() if k.startswith("action_")}
        child_kwargs = {k[6:]: v for k, v in kwargs.items() if k.startswith("child_")}

        action_loss = self._action_model_loss_fn(self.action_model, inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix, **action_kwargs)

        if self._do_child_model_loss:
            child_loss = self._child_model_loss_fn(self.child_model, inputs, outputs, i=i, writer=writer, writer_prefix=writer_prefix + "child_loss/", **child_kwargs)
            action_loss += child_loss

        return action_loss

    # MODELS
    @property
    def action_model(self):
        return self._action_model

    @property
    def child_model(self):
        return self._child_model
