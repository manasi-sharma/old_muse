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
        # which index to get of the (B, H, C) output of the sequence.
        self._default_output_horizon_idx = get_with_default(params, "default_output_horizon_idx", -1)
        assert self._horizon > 1, "Do not use sequence model with horizon <= 1"
        super()._init_params_to_attrs(params)
        # history names to get from inputs
        self.online_inputs = get_with_default(params, "online_inputs", self.inputs)
        print("ONLINE", self.online_inputs)

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

    def get_online_inputs(self, inputs):
        return inputs > self.online_inputs

    @staticmethod
    def get_default_mem_policy_forward_fn(*args, add_goals_in_hor=False, output_horizon_idx=None, horizon=None, **kwargs):
        """
        Similar to RnnModel, but keep track of the history of self.inputs online, rather than hidden state

        More memory intensive.
        """

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: SequenceModel, obs: AttrDict, goal: AttrDict, memory: AttrDict,
                                        root_model: Model = None, **inner_kwargs):
            nonlocal output_horizon_idx, horizon

            inputs = obs.leaf_copy()
            if not add_goals_in_hor and not goal.is_empty():
                inputs.goal_states = goal
            else:
                inputs = inputs & goal

            if memory.is_empty():
                if output_horizon_idx is None:
                    output_horizon_idx = model._default_output_horizon_idx
                if horizon is None:
                    horizon = model._horizon
                    # prefer root
                    if hasattr(root_model, "horizon"):
                        horizon = root_model.horizon
                    elif hasattr(root_model, "_horizon"):
                        horizon = root_model._horizon

                memory.count = 0  # total steps

                # list of inputs, shape (B x 1 x ..), will be concatenated latervi
                memory.input_history = [model.get_online_inputs(inputs) for _ in range(horizon)]

                # avoid allocating memory again
                memory.alloc_inputs = AttrDict.leaf_combine_and_apply(memory.input_history,
                                                                      lambda vs: torch.cat(vs, dim=1))

            memory.count += 1

            # add new inputs, maintaining sequence length
            memory.input_history = memory.input_history[1:] + [model.get_online_inputs(inputs)]

            # normal policy w/ fixed plan, we use prior, doesn't really matter here tho since run_plan=False
            base_model = (model if root_model is None else root_model)

            def set_vs(k, vs):
                # set allocated array, return None
                torch.cat(vs, dim=1, out=memory.alloc_inputs[k])

            AttrDict.leaf_combine_and_apply(memory.input_history, set_vs, pass_in_key_to_func=True)

            # run the base model forward on the inputs
            out = base_model.forward(memory.alloc_inputs,
                                     **inner_kwargs)

            # grab a horizon index from the base model's output arrays (default is last idx)
            # then reshape to (B x 1 x ...)
            out.combine(
                out.leaf_arrays().leaf_apply(lambda arr: arr[:, output_horizon_idx, None])
            )

            # default online postproc defined in BasicModel (parent class)
            return base_model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn
