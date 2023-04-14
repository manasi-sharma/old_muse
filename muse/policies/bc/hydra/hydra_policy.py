from typing import Tuple

import numpy as np
import torch

from muse.envs.mode_param_spec import BiModalParamEnvSpec
from muse.experiments import logger
from muse.policies.bc.gcbc_policy import GCBCPolicy
from muse.utils.torch_utils import combine_then_concatenate

from attrdict import AttrDict as d
from attrdict.utils import get_with_default


class HYDRAPolicy(GCBCPolicy):
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        # control sparse via waypoints
        self.use_online_sparse_control = get_with_default(params, "use_online_sparse_control", True)
        # hidden state is not forwarded through mode=0 states.
        self.skip_online_hidden_state = get_with_default(params, "skip_online_hidden_state", False)
        # hidden state gets flushed at mode = 0-><anything> transition.
        self.flush_online_hidden_state = get_with_default(params, "flush_online_hidden_state", False)

        self.online_sparse_timeout = get_with_default(params, "online_sparse_timeout", 100)

        # dynamics
        if isinstance(self._env_spec, BiModalParamEnvSpec):
            self.state_keys = self._env_spec.dynamics_state_names
            self.wp_dynamics_fn = self._env_spec.wp_dynamics_fn
            self._sparse_policy_out_names = get_with_default(params, "sparse_policy_out_names",
                                                             self._env_spec.mode0_action_names)

            assert self.wp_dynamics_fn is not None, "BiModal Env Spec does not have a dynamics fn!"
        else:
            logger.warn("Env spec is not a BiModal Env Spec! "
                        "Will try to get state_keys / dynamics / sparse_names from params manually.")
            self.state_keys = params["state_keys"]
            self.wp_dynamics_fn = params["wp_dynamics_fn"]
            self._sparse_policy_out_names = params["sparse_policy_out_names"]

    def default_mem_policy_model_forward_fn(self, model, obs: d, goal: d, memory: d, **kwargs):
        # set the forward fn from the model

        obs = obs.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))
        goal = goal.leaf_arrays().leaf_apply(lambda arr: arr.to(dtype=torch.float32))

        # model/rnn forward
        out = model.online_forward(obs & goal, memory, **kwargs)

        # controlling sparse / dense modes differently.
        wp_done = False

        if self.use_online_sparse_control:
            if not memory.has_leaf_key("curr_waypoint"):
                memory.curr_waypoint = None
                memory.curr_norm_waypoint_dict = None
                memory.waypoint_count = 0

            unnorm_out = model.normalize_by_statistics(out,
                                                       self._policy_out_norm_names,
                                                       inverse=True) > (
                                     self._policy_out_names + self._sparse_policy_out_names)

            if not self.velact:
                is_dense_mode = False
            else:
                is_dense_mode = (out >> self.mode_key).item() > 0.5

            # print("dense:", is_dense_mode)
            # if is_sparse and there's no waypoint set, start one.
            if not is_dense_mode and memory.curr_waypoint is None:
                # NEW WAYPOINT
                memory.curr_norm_waypoint_dict = (out > self._sparse_policy_out_names).leaf_apply(
                    lambda arr: arr.clone())
                memory.curr_waypoint = combine_then_concatenate(unnorm_out, self._sparse_policy_out_names,
                                                                dim=1).reshape(-1)
                # print(memory.curr_waypoint)
                if self.skip_online_hidden_state:
                    # record the hidden state here. policy will use this.
                    if isinstance(memory.policy_rnn_h0, Tuple):
                        memory.curr_policy_rnn_h0 = tuple(a.clone() for a in (memory >> "policy_rnn_h0"))
                    else:
                        memory.curr_policy_rnn_h0 = (memory >> "policy_rnn_h0").clone()

            # reset the waypoint to None if reached.
            if memory.curr_waypoint is not None:
                memory.waypoint_count += 1
                # use the same waypoint until reached or timeout if one was set.
                state = combine_then_concatenate(obs, self.state_keys, dim=2).reshape(-1)
                _, _, reached = self.wp_dynamics_fn(state, memory.curr_waypoint)

                # print(state, reached)

                # DONE WITH WAYPOINT
                if reached.item() or memory.waypoint_count > self.online_sparse_timeout:
                    wp_done = True
                    if memory.waypoint_count > self.online_sparse_timeout:
                        logger.warn(f"Waypoint {memory.curr_waypoint} timed out!!")
                    # else:
                    #     logger.warn(f"Waypoint {memory.curr_waypoint} reached!!")
                    memory.curr_waypoint = None

            # now if a waypoint has been set, override the wp keys in the policy_out
            if memory.curr_waypoint is not None:
                out.combine(memory.curr_norm_waypoint_dict)

                if self.velact:
                    out[self.mode_key][:] = 0  # set to sparse mode manually.

                if self.skip_online_hidden_state:
                    # rnn state is set to h0 throughout sparse mode
                    if isinstance(memory.policy_rnn_h0, Tuple):
                        memory.policy_rnn_h0 = tuple(a.clone() for a in memory.curr_policy_rnn_h0)
                    else:
                        # reset hidden state to its first value (affects policy)
                        memory.policy_rnn_h0 = memory.curr_policy_rnn_h0.clone()

                elif self.flush_online_hidden_state:
                    # rnn state is set to None throughout sparse mode
                    memory.policy_rnn_h0 = None
                    memory.flush_count = 0  # reset the flushing count as well for rnn policy (redundancy)

            else:
                # print(f'unchanged: dense = {is_dense_mode}')
                memory.waypoint_count = 0  # reset

        if self.velact:
            is_dense_mode = (out >> self.mode_key).item() > 0.5

        # specific postprocessing on the action (recompute dense mode in case it changed)
        out = self.postproc_action(model, obs, out, memory, is_dense_mode, **kwargs)

        shp = [1]
        out['is_dense'] = np.broadcast_to([is_dense_mode], shp + [1])
        out['wp_done'] = np.broadcast_to([wp_done], shp + [1])

        return out
