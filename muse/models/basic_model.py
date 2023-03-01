"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
import os

import torch
from typing import List

from muse.envs.param_spec import ParamEnvSpec
from muse.experiments import logger
from muse.models.model import Model
from muse.utils.param_utils import SequentialParams, LayerParams
from muse.utils.python_utils import timeit
from attrdict import AttrDict
from attrdict.utils import get_with_default
from muse.utils.torch_utils import concatenate


class BasicModel(Model):

    # @abstract.overrides
    def _init_params_to_attrs(self, params):
        self.inputs = params["model_inputs"]
        self.output = str(params["model_output"])

        # run net on each input, rather than all inputs at once
        self.call_separate = get_with_default(params, "call_separate", False)
        self.use_shared_params = get_with_default(params, "use_shared_params", True)
        assert self.call_separate or self.use_shared_params, "Must use shared params if not using call_separate"

        self.net = params["network"]
        if self.call_separate and not self.use_shared_params:
            if isinstance(self.net, List):
                assert len(self.net) == len(self.inputs), \
                    f"Specify 1 net, or 1 per input, for len(inputs)={len(self.inputs)}, but len(nets)={len(self.net)}"
                # multiple nets specified, since we are calling it separately
                self.net = torch.nn.ModuleList([
                    n.to_module_list(as_sequential=True) for n in self.net
                ]).to(self.device)
            else:
                # clone model for each input, since we are calling it separately
                self.net = torch.nn.ModuleList([
                    self.net.to_module_list(as_sequential=True) for _ in range(len(self.inputs))
                ]).to(self.device)
        else:
            # one (potentially reused) input
            assert isinstance(self.net, LayerParams), "Network must be passed in as LayerParams instance!"
            self.net = self.net.to_module_list(as_sequential=True).to(self.device)

        self.concat_dim = int(params.get("concat_dim", -1))
        self.concat_dtype = params.get("concat_dtype", torch.float32)

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    @staticmethod
    def concat_forward(model: Model, net: torch.nn.Module, inputs, input_names: List[str], output_name: str,
                       concat_dim: int, concat_dtype,
                       training=False, preproc_fn=None, postproc_fn=None,
                       call_separate=False, use_shared_params=True, timeit_prefix="basic_model/", **kwargs):
        # runs the model forward
        if not isinstance(inputs, AttrDict):
            if isinstance(inputs, torch.Tensor):
                inputs = [inputs]
            assert len(inputs) == len(model.inputs), [model.inputs, len(inputs)]
            inputs = AttrDict.from_dict({k: v for k, v in zip(model.inputs, inputs)})
        else:
            inputs = inputs.leaf_copy()

        if model.normalize_inputs:
            inputs = model.normalize_by_statistics(inputs, model.normalization_inputs, shared_dtype=concat_dtype)

        with timeit(f"{timeit_prefix}preproc"):
            if preproc_fn:
                inputs = preproc_fn(inputs)

        # either call net on each separate, or concatenate first.
        if call_separate:
            each_obs = [inputs[key] for key in input_names]
            each_obs = [arr.to(dtype=concat_dtype) for arr in each_obs]
        else:
            with timeit(f'{timeit_prefix}cat'):
                obs = concatenate((inputs > input_names)
                                  .leaf_apply(lambda arr: arr.to(dtype=concat_dtype)),
                                  input_names, dim=concat_dim)
            each_obs = [obs]

        # run model and concatenate after (if calling separate)
        with timeit(f'{timeit_prefix}forward'):
            out_ls = []
            for i, eo in enumerate(each_obs):
                # index into module dict if separate
                encoder_net_i = net if use_shared_params else net[i]
                out_ls.append(encoder_net_i(eo))

        out = AttrDict({output_name: torch.cat(out_ls, dim=concat_dim)})

        return postproc_fn(inputs, out) if postproc_fn else out

    # @abstract.overrides
    def forward(self, inputs, training=False, preproc=True, postproc=True, **kwargs):
        """
        :param inputs: (AttrDict)  (B x ...)
        :param training: (bool)
        :param preproc: (bool) run preprocess fn
        :param postproc: (bool) run postprocess fn

        :return model_outputs: (AttrDict)  (B x ...)
        """
        return BasicModel.concat_forward(self, self.net, inputs, self.inputs, self.output, self.concat_dim,
                                         self.concat_dtype, training=training,
                                         preproc_fn=self._preproc_fn if preproc else None,
                                         postproc_fn=self._postproc_fn if postproc else None,
                                         call_separate=self.call_separate,
                                         use_shared_params=self.use_shared_params,
                                         **kwargs)

    @staticmethod
    def get_default_mem_policy_forward_fn(*args, add_goals_in_hor=False, **kwargs):

        # online execution using MemoryPolicy or subclass
        def mem_policy_model_forward_fn(model: BasicModel, obs: AttrDict, goal: AttrDict, memory: AttrDict,
                                        root_model: Model = None, **inner_kwargs):
            obs = obs.leaf_copy()
            if memory.is_empty():
                memory.count = 0

            if not add_goals_in_hor and not goal.is_empty():
                obs.goal_states = goal

            memory.count += 1

            # normal policy w/ fixed plan, we use prior, doesn't really matter here tho since run_plan=False
            base_model = (model if root_model is None else root_model)
            out = base_model.forward(obs, **inner_kwargs)
            return base_model.online_postproc_fn(model, out, obs, goal, memory, **inner_kwargs)

        return mem_policy_model_forward_fn

    def print_parameters(self, prefix="", print_fn=logger.debug):
        print_fn(prefix + "[BasicModel]")
        for n, p in self.named_parameters():
            print_fn(prefix + "[BasicModel] %s <%s> (requires_grad = %s)" % (n, list(p.shape), p.requires_grad))


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    md = BasicModel(AttrDict(model_inputs=["ins"], model_output="out", device="cpu", normalization_inputs=[],
                             network=SequentialParams([
                                 LayerParams("linear", in_features=5, out_features=10),
                                 LayerParams("relu"),
                                 LayerParams("linear", in_features=10, out_features=10),
                                 LayerParams("gaussian_dist_cap", params=AttrDict(use_log_sig=False))
                             ]), postproc_fn=lambda inps, outs: outs.out),
                    ParamEnvSpec(AttrDict()), None)

    writer = SummaryWriter(os.path.expanduser("~/.test"))

    print("Adding graph...")
    writer.add_graph(md, input_to_model=torch.zeros((10, 5)), verbose=True)

    print("DONE.")
