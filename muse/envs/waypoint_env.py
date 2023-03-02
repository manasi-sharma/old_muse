import numpy as np
import torch

from muse.envs.env import Env
from muse.envs.env_spec import EnvSpec
from muse.experiments import logger
from muse.models.model import Model
from muse.policies.memory_policy import MemoryPolicy
from attrdict import AttrDict as d
from attrdict.utils import get_with_default, get_or_instantiate_cls
from muse.utils.torch_utils import combine_then_concatenate, dc_add_horizon_dim, to_numpy


class WaypointEnv(Env):
    """
    Holds the following
    - model (for the low level policy)
    - policy (low level policy)
    - base env.
    - base env_spec

    Step (variable underlying step length)
    - steps the base environment with a waypoint reaching controller
        (with the action as a delta on base_policy in waypoint space)
    - steps the base policy/environment until the policy outputs mode=SPARSE.

    Issue
     -  DELTA_WP -> NEW_WAYPOINT -> ACTION POSTPROC -> STEP

    """

    def __init__(self, params, env_spec):
        super(WaypointEnv, self).__init__(params, env_spec)
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        # dynamics of waypoint -> action
        self.wp_dynamics_fn = params["wp_dynamics_fn"]
        self.state_keys = params["state_keys"]
        self.sparse_timeout = params["sparse_timeout"]
        self.horizon = params["horizon"]
        # if true, will encompass waypoints that have been reached
        self._terminate_on_reward = get_with_default(params, "terminate_on_reward", False)
        self._penalize_act_norm = get_with_default(params, "penalize_act_norm", 0)
        self._penalize_wp_timeout = get_with_default(params, "penalize_wp_timeout", 0)

        if self._penalize_act_norm > 0:
            logger.debug(f"Penalizing action norm with alpha={self._penalize_act_norm}")
        if self._penalize_wp_timeout > 0:
            logger.debug(f"Penalizing wp timeouts with alpha={self._penalize_wp_timeout}")

        # will run through all waypoints that are reachable from start state (only meaningful waypoints will be used).
        self.collapse_wp_timeout = get_with_default(params, "collapse_wp_timeout", 0)
        if self.collapse_wp_timeout:
            logger.info(f"Waypoint env will collapse, with timeout = {self.collapse_wp_timeout}")

        # policy / mode stuff
        self.mode_key = get_with_default(params, "mode_key", "mode")
        self.policy_out_names = params["policy_out_names"]
        self.wp_policy_out_names = params["wp_policy_out_names"]
        self.policy_out_norm_names = get_with_default(params, "policy_out_norm_names", list(self.wp_policy_out_names))
        self.policy_timeout = get_with_default(params, "policy_timeout", 0)

        if "base_env_spec" in params.keys() and params.base_env_spec is not None:
            self._base_env_spec = get_or_instantiate_cls(params, "base_env_spec", EnvSpec)
        else:
            logger.debug("WP spec defaulting to self._env_spec")
            self._base_env_spec = self._env_spec

        self._base_env = get_or_instantiate_cls(params, "base_env", Env,
                                                constructor=lambda cls, prms: cls(prms, self._base_env_spec))
        #
        # self._base_policy = get_or_instantiate_cls(params, "base_policy", Policy,
        #                                            constructor=lambda cls, prms: cls(prms, self._base_env_spec,
        #                                                                              env=self._base_env))

        # override policy postproc function (this is how we update the waypoint inside the policy)
        self._model = get_or_instantiate_cls(params, "model", Model,
                                             constructor=lambda cls, prms: cls(prms, self._base_env_spec, None))

        # initialize a memory policy (rnn compatible)
        base_mem_forward_fn = self._model.get_default_mem_policy_forward_fn(self.policy_out_names,
                                                                            flush_horizon=self.horizon)
        self.online_action_postproc_fn = params["online_action_postproc_fn"]
        self.online_action_postproc_fn_kwargs = get_with_default(params, "online_action_postproc_fn_kwargs", {})

        def mem_policy_model_forward_fn(model, obs: d, goal: d, memory: d, known_sequence=None,
                                        **kwargs):
            # call the model
            policy_out = base_mem_forward_fn(model, obs, goal, memory, known_sequence=known_sequence, **kwargs)
            policy_out = self.update_wp_action(policy_out)
            is_dense_mode = (policy_out[self.mode_key]).item() > 0.5
            self.online_action_postproc_fn(model, obs, policy_out, self.policy_out_names,
                                           relative=False, vel_act=is_dense_mode, memory=memory,
                                           max_gripper_vel=150.,
                                           policy_out_norm_names=self.policy_out_norm_names,
                                           max_orn_vel=5. if "drawer" in obs.keys() else 10.,
                                           **self.online_action_postproc_fn_kwargs,
                                           **kwargs)

            return policy_out

        is_terminated_fn = None
        if self.policy_timeout > 0:
            # requires mem policy to keep track of count
            is_terminated_fn = lambda model, obs, goal, mem, **kwargs: \
                False if mem.is_empty() else mem["count"] >= self.policy_timeout

        self._base_policy = MemoryPolicy(d(
            policy_model_forward_fn=mem_policy_model_forward_fn,
            is_terminated_fn=is_terminated_fn), self._base_env_spec, env=self._base_env)

        # model file to load.
        self._model_file = params["model_file"]

        # self._skip_hidden = get_with_default(params, "skip_hidden", False)
        # TODO implement this for 3D envs..
        self.combine_waypoint_fn = get_with_default(params, "combine_waypoint_fn", lambda wp, wp_delta: wp + wp_delta)

    def _init_setup(self):
        logger.debug(f"Restoring base_model using: {self._model_file}")
        self._model.restore_from_file(self._model_file, strict=True)

        self.curr_obs, self.curr_goal = None, None
        self.curr_waypoint = None
        self.curr_wp_dc_norm = None
        self.waypoint_count = 0

    def reset(self, presets: d = d()):
        # reset base environment
        self.curr_obs, self.curr_goal = self._base_env.reset(presets)
        # reset base policy with first state.
        self._base_policy.reset_policy(next_obs=self.curr_obs, next_goal=self.curr_goal)
        self.curr_waypoint = None
        self.curr_wp_dc_norm = None
        self.waypoint_count = 0

        with torch.no_grad():
            # the first output of the policy.
            self.curr_policy_out = self._base_policy.get_action(self._model, dc_add_horizon_dim(self.curr_obs),
                                                                dc_add_horizon_dim(self.curr_goal))

            # step base env until policy outputs mode = 0.
            i = 0
            done = np.array([False])
            while not done[0] and (self.curr_policy_out[self.mode_key]).item() > 0.5:
                self.curr_obs, self.curr_goal, done = self._base_env.step(self.curr_policy_out)
                self.curr_policy_out = self._base_policy.get_action(self._model, dc_add_horizon_dim(self.curr_obs),
                                                                    dc_add_horizon_dim(self.curr_goal))
                done[0] = done[0] or self._base_policy.is_terminated(self._model, self.curr_obs, self.curr_goal)
                i += 1

        if done[0]:
            logger.warn('env finished during reset.. there is a bug')

        logger.debug(f"Reset complete for WP environment: pre steps = {i}")
        return self.get_wp_obs(), self.curr_goal

    def step(self, action: d):
        obs, goal, done = self._step(action)
        # if self.collapse_waypoints_timeout > 0:
        # keep doing inner step until we
        i = 0
        new_waypoint = to_numpy(combine_then_concatenate(self.curr_policy_out, self.wp_policy_out_names,
                                                         dim=1).reshape(-1))
        rew = (obs['reward'])
        while i < self.collapse_wp_timeout and self.compute_reached(new_waypoint, check_timeout=False):
            obs, goal, done = self._step(d(delta_waypoint=torch.zeros_like(action['delta_waypoint'])))
            new_waypoint = to_numpy(combine_then_concatenate(self.curr_policy_out, self.wp_policy_out_names,
                                                             dim=1).reshape(-1))
            rew = np.maximum(rew, obs['reward'])
            i += 1

        obs.reward = rew

        return obs, goal, done

    def _step(self, action: d):
        """
        Will process one waypoint (or until timeout) and as many extra dense segments as necessary

        :param action: (DELTA) on top of true waypoint.
        :return: (obs & policy_output, goal) where the policy has output a waypoint.
        """
        assert self.curr_policy_out is not None, "Must reset environment properly!"
        # action is a waypoint (normalized)
        wp_delta = action["delta_waypoint"][0]  # index batch dim
        # self.curr_norm_waypoint_dict = (policy_out > self.wp_policy_out_names).leaf_apply(lambda arr: arr.clone())

        step_reward = 0.  # will compute the max reward

        with torch.no_grad():

            self.curr_waypoint = combine_then_concatenate(self.curr_policy_out, self.wp_policy_out_names,
                                                          dim=1).reshape(-1)

            # adds as a delta in unnormalized space, then converts back to normalized
            assert wp_delta.shape == self.curr_waypoint.shape
            self.curr_waypoint = self.combine_waypoint_fn(self.curr_waypoint, wp_delta)  # torch
            curr_waypoint_dc = self._base_env_spec.parse_view_from_concatenated_flat(self.curr_waypoint,
                                                                                     self.wp_policy_out_names)
            # renormalize waypoint dict (for later adding), add back in batch dimension
            self.curr_wp_dc_norm = self._model.normalize_by_statistics(
                curr_waypoint_dc.leaf_apply(lambda arr: arr[None]), list(set(self.policy_out_norm_names)
                                                                         .intersection(self.wp_policy_out_names)))

            reached_next_waypoint = False
            done = np.array([False])
            self.waypoint_count = 0
            self.curr_waypoint = to_numpy(self.curr_waypoint)
            self.curr_policy_out = self.curr_policy_out.leaf_copy()

            # ---- SPARSE ---- #
            # loop until either done or we reach the next waypoint
            while not done[0]:
                # step the environment.
                self.curr_obs, self.curr_goal, done = self._base_env.step(self.curr_policy_out)
                step_reward = max(step_reward, (self.curr_obs['reward']).item())

                # import ipdb; ipdb.set_trace()
                # update count to reflect number of added steps
                self.waypoint_count += 1
                # run model, override curr_policy_out
                reached_next_waypoint = self.compute_reached(self.curr_waypoint)
                if reached_next_waypoint:
                    break

                self.curr_policy_out = self._base_policy.get_action(self._model, dc_add_horizon_dim(self.curr_obs),
                                                                    dc_add_horizon_dim(self.curr_goal))
                done[0] = done[0] or self._base_policy.is_terminated(self._model, self.curr_obs, self.curr_goal)

            # if self.waypoint_count == 0:
            #     import ipdb; ipdb.set_trace()  # TODO

            # ---- DENSE ---- #
            self.curr_waypoint = None
            self.curr_wp_dc_norm = None
            # compute first action of next phase.
            if reached_next_waypoint:
                self.curr_policy_out = self._base_policy.get_action(self._model, dc_add_horizon_dim(self.curr_obs),
                                                                    dc_add_horizon_dim(self.curr_goal))
                done[0] = done[0] or self._base_policy.is_terminated(self._model, self.curr_obs, self.curr_goal)

            # set the "first" mode0 to be the current policy out (might not be mode 0)
            is_dense = (self.curr_policy_out[self.mode_key]).item() > 0.5

            # loop until we get through all the remaining dense segments
            # step base env until policy outputs mode = 0 or done
            dense_steps = 0
            while not done[0] and is_dense:
                self.curr_obs, self.curr_goal, done = self._base_env.step(self.curr_policy_out)
                dense_steps += 1
                step_reward = max(step_reward, (self.curr_obs['reward']).item())
                self.curr_policy_out = self._base_policy.get_action(self._model, dc_add_horizon_dim(self.curr_obs),
                                                                    dc_add_horizon_dim(self.curr_goal))
                is_dense = self.curr_policy_out[self.mode_key].item() > 0.5
                done[0] = done[0] or self._base_policy.is_terminated(self._model, self.curr_obs, self.curr_goal)

        # returns state and the first next policy_out where mode = 0.

        if self._terminate_on_reward and step_reward > 0.5:
            done[:] = True

        # reward goes down for a failed waypoint
        if self.waypoint_count > self.sparse_timeout and self._penalize_wp_timeout > 0:
            step_reward -= self._penalize_wp_timeout

        if self._penalize_act_norm > 0:
            wp_delta = to_numpy(wp_delta, check=True)
            step_reward -= self._penalize_act_norm * np.linalg.norm(wp_delta) / np.sqrt(len(wp_delta))

        return self.get_wp_obs(step_reward), self.curr_goal, done

    def compute_reached(self, curr_waypoint, check_timeout=True):

        # use the same waypoint until reached or timeout if one was set.
        state = combine_then_concatenate(self.curr_obs, self.state_keys, dim=1).reshape(-1)
        _, _, reached = self.wp_dynamics_fn(state, curr_waypoint)

        # DONE WITH WAYPOINT
        if reached.item() or (check_timeout and self.waypoint_count > self.sparse_timeout):
            if check_timeout and self.waypoint_count > self.sparse_timeout:
                logger.warn(f"Waypoint {curr_waypoint} timed out!!")

            return True

        return False

    def update_wp_action(self, policy_out):
        if self.curr_wp_dc_norm is not None:
            # override with the current waypoint (normalized)
            policy_out.combine(dc_add_horizon_dim(self.curr_wp_dc_norm))
            policy_out[self.mode_key][:] = 0  # set to sparse mode manually.

        return policy_out

    def get_wp_obs(self, rew=0):
        wp_obs = self.curr_obs & \
                 (self.curr_policy_out > self.wp_policy_out_names) \
                     .leaf_apply(lambda arr: to_numpy(arr, check=True))

        wp_obs.reward = np.array([[rew]])  # (1, 1)

        return wp_obs
