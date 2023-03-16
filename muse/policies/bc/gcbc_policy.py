import numpy as np
import torch

from muse.policies.memory_policy import MemoryPolicy
from attrdict import AttrDict as d
from attrdict.utils import get_with_default


class GCBCPolicy(MemoryPolicy):
    def _init_params_to_attrs(self, params):
        params.policy_model_forward_fn = get_with_default(params, "policy_model_forward_fn",
                                                          self.default_mem_policy_model_forward_fn)
        super()._init_params_to_attrs(params)

        self.online_action_postproc_fn = params["online_action_postproc_fn"]

        self.velact = get_with_default(params, "velact", True)
        self.recurrent = get_with_default(params, "recurrent", True)
        self.sample_plan = get_with_default(params, "sample_plan", False)
        self.replan_horizon = get_with_default(params, "replan_horizon", 0)
        self.flush_horizon = get_with_default(params, "flush_horizon", self.replan_horizon)
        self.mode_key = params << "mode_key"

        self._policy_out_names = get_with_default(params, "policy_out_names", self._out_names)
        self._policy_out_norm_names = get_with_default(params, "policy_out_norm_names", self._policy_out_names)

        self.max_gripper_vel = get_with_default(params, "max_gripper_vel", 150.)
        self.max_orn_vel = get_with_default(params, "max_orn_vel", 10.)
        self.free_orientation = get_with_default(params, "free_orientation", False)

        self.fill_extra_policy_names = get_with_default(params, "fill_extra_policy_names", True)

    def _init_setup(self):
        super()._init_setup()
        self.model_forward_fn = None

    def postproc_action(self, model, obs, out, memory, vel_act, **kwargs):

        # postprocessing of action (e.g. target action)
        self.online_action_postproc_fn(model, obs, out, self._policy_out_names,
                                       relative=False, vel_act=vel_act, memory=memory,
                                       max_gripper_vel=self.max_gripper_vel,  # 1000 if real else 150.
                                       policy_out_norm_names=self._policy_out_norm_names,
                                       max_orn_vel=self.max_orn_vel,  # 5. if "drawer" in obs.keys() else 10.
                                       free_orientation=self.free_orientation,
                                       **kwargs)

        # adding in extra info
        if self.fill_extra_policy_names:
            shp = list((out >> self._policy_out_names[0]).shape[:1])
            if not out.has_leaf_key("policy_type"):
                out['policy_type'] = np.broadcast_to([253], shp + [1])
            if not out.has_leaf_key("policy_name"):
                out['policy_name'] = np.broadcast_to(["lmp_policy"], shp + [1])
            if not out.has_leaf_key("policy_switch"):
                out['policy_switch'] = np.broadcast_to([False], shp + [1])

        return out

    # policy first rolls out model, then postprocess action(s) as defined by utils
    def default_mem_policy_model_forward_fn(self, model, obs: d, goal: d, memory: d, known_sequence=None, **kwargs):

        if self.model_forward_fn is None:
            self.model_forward_fn = model.get_default_mem_policy_forward_fn(self.replan_horizon,
                                                                            self._policy_out_names,
                                                                            recurrent=self.recurrent,
                                                                            sample_plan=self.sample_plan,
                                                                            flush_horizon=self.flush_horizon)

        obs = obs.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))
        goal = goal.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))

        # model/rnn forward, root_model used for model.forward, decoder used for recurrent state tracking.
        out = self.model_forward_fn(model, obs, goal, memory, known_sequence=known_sequence, **kwargs)

        # optional mode switching between velocity & target/position action
        if self.mode_key is None:
            vel_act = self.velact
        else:
            vel_act = (out >> self.mode_key).item() > 0.5  # mode is 1 for velact, 0 for posact.

        return self.postproc_action(model, obs, out, memory, vel_act, **kwargs)
