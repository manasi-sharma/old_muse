import numpy as np
import torch

from muse.envs.mode_param_spec import BiModalParamEnvSpec
from muse.utils.general_utils import is_array
from muse.utils.np_utils import clip_norm, clip_scale
from muse.utils.torch_utils import to_numpy, to_torch, cat_any
import muse.utils.transform_utils as T


GRIPPER_WIDTH_TOL = 0.05
GRIPPER_WIDTH_DELTA_TOL = 1e-3


def parse_orientations(unnorm_out, targ_prefix):
    if f'{targ_prefix}orientation' in unnorm_out.list_leaf_keys():
        targ_quat = to_numpy(unnorm_out[f"{targ_prefix}orientation"], check=True)[:, 0]
        targ_quat = targ_quat / (
                    np.linalg.norm(targ_quat, axis=-1, keepdims=True) + 1e-8)  # make sure quaternion is normalized
        targ_eul = T.fast_quat2euler(targ_quat)
    else:
        targ_eul = to_numpy(unnorm_out[f"{targ_prefix}orientation_eul"], check=True)[:, 0]
        targ_quat = T.fast_euler2quat(targ_eul)
    return targ_quat, targ_eul


def get_wp_dynamics_fn(fast_dynamics=True, no_ori=False, max_pos_vel=0.4, max_ori_vel=5.0, DT=0.05, TOL=0.0075, ORI_TOL=0.075,
                       true_max_pos_vel=1., true_max_ori_vel=10., scale_action=True):
    """ (robot pose, mode0_action) -> (next robot pose, mode1_action, reached) """
    max_ac_dpos = true_max_pos_vel * DT
    max_ac_dori = true_max_ori_vel * DT

    # faster max velocities
    if fast_dynamics:
        max_pos_vel = true_max_pos_vel
        max_ori_vel = true_max_ori_vel

    def wp_dyn(robot_pose, targ_pose):
        torch_device = None if isinstance(robot_pose, np.ndarray) else robot_pose.device
        if torch_device is not None:
            robot_pose = to_numpy(robot_pose).copy()
            targ_pose = to_numpy(targ_pose).copy()

        front_shape = list(robot_pose.shape[:-1])
        robot_pose = robot_pose.reshape(-1, robot_pose.shape[-1])
        targ_pose = targ_pose.reshape(-1, targ_pose.shape[-1])

        # targ_pose can be either
        # - 6 dim (pos, ori_eul)
        # - 10 dim (pos, ori, ori_eul)
        pos = robot_pose[..., :3]
        dpos = (targ_pose[..., :3] - pos)
        goal_gr = np.zeros_like(dpos[..., :1])

        # TODO torch implementation of these

        if isinstance(dpos, np.ndarray):

            dpos = clip_norm(dpos, max_pos_vel * DT, axis=-1)  # see robosuite_policies.py
            next_pos = pos + dpos

            if no_ori:
                assert robot_pose.shape[-1] == 3, robot_pose.shape
                next_state = next_pos
            else:
                target_q = T.get_normalized_quat(targ_pose)
                curr_q = T.get_normalized_quat(robot_pose)

                q_angle = T.quat_angle(target_q, curr_q).astype(targ_pose.dtype)
                abs_q_angle_clipped = np.minimum(np.abs(q_angle), max_ori_vel * DT)
                scale = abs_q_angle_clipped.copy()
                scale[np.abs(q_angle) > 0] /= np.abs(q_angle[np.abs(q_angle) > 0])
                goal_q = T.batch_quat_slerp(curr_q, target_q, scale)
                # goal_eul = T.quat2euler_ext(goal_q)

                dori_q = T.quat_difference(goal_q, curr_q)
                dori = T.fast_quat2euler(dori_q)

                dpos = clip_scale(dpos, max_ac_dpos)
                dori = clip_scale(dori, max_ac_dori)
                dori_q = T.fast_euler2quat(dori)
                next_ori_q = T.quat_multiply(curr_q, dori_q)
                next_ori = T.fast_quat2euler(next_ori_q)

                next_state = np.concatenate([next_pos, next_ori_q, next_ori], axis=-1)

            if scale_action:
                # actions are scaled to (-1 -> 1)
                unscaled_dpos = dpos / max_ac_dpos
                if not no_ori:
                    unscaled_dori = dori / max_ac_dori
            else:
                # actions are scaled to (-1 -> 1)
                unscaled_dpos = dpos
                if not no_ori:
                    unscaled_dori = dori

            # hopefully gripper will be replaced later.. hard to compute now.

            reached = (np.linalg.norm(dpos, axis=-1) < TOL)
            if no_ori:
                mode1_action = np.concatenate([unscaled_dpos, unscaled_dori, goal_gr], axis=-1)
            else:
                mode1_action = np.concatenate([unscaled_dpos, unscaled_dori, goal_gr], axis=-1)
                reached = reached & (np.abs(q_angle) < ORI_TOL)
        else:
            raise NotImplementedError

        if torch_device is not None:
            mode1_action = to_torch(mode1_action, device=torch_device)
            next_state = to_torch(next_state, device=torch_device)
            reached = to_torch(reached, device=torch_device)

        # reshape back
        mode1_action = mode1_action.reshape(front_shape + [mode1_action.shape[-1]])
        next_state = next_state.reshape(front_shape + [next_state.shape[-1]])
        return mode1_action, next_state, reached

    return wp_dyn


def rs_generic_online_action_postproc_fn(model, obs, out, policy_out_names, no_ori=False,
                                         policy_out_norm_names=None, vel_act=True,
                                         pos_key="robot0_eef_pos", quat_key="robot0_eef_quat",
                                         targ_prefix="target/", wp_dyn=None, fast_dynamics=True, **kwargs):
    if policy_out_norm_names is None:
        policy_out_norm_names = policy_out_names

    if wp_dyn is None:
        wp_dyn = get_wp_dynamics_fn(no_ori=no_ori, fast_dynamics=fast_dynamics)

    unnorm_out = model.normalize_by_statistics(out, policy_out_norm_names, inverse=True) > policy_out_names
    # import ipdb; ipdb.set_trace()
    targ_eul, targ_quat = None, None

    if vel_act:
        ac = to_numpy(unnorm_out["action"], check=True)
        assert ac.shape[-1] == (4 if no_ori else 7), ac.shape
    else:
        if unnorm_out.has_leaf_key('target/gripper'):
            base_gripper = to_numpy(unnorm_out['target/gripper'], check=True)
        else:
            base_action = to_numpy(unnorm_out["action"], check=True)  # needs this for gripper
            base_gripper = base_action[..., -1:]
        # front_shape = base_action.shape[:-1]
        pos = to_numpy(obs[pos_key], check=True)[:, 0]
        if quat_key in obs.leaf_keys():
            quat = to_numpy(obs[quat_key], check=True)[:, 0]
            quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)  # make sure quaternion is normalized
            eul = T.fast_quat2euler(quat)
        else:
            eul = to_numpy(obs[pos_key], check=True)[:, 0]
            quat = T.fast_euler2quat(eul)

        targ_pos = to_numpy(unnorm_out[f"{targ_prefix}position"], check=True)[:, 0]
        if no_ori:
            curr_pose = pos
            targ_pose = targ_pos
        else:
            targ_quat, targ_eul = parse_orientations(unnorm_out, targ_prefix)
            curr_pose = np.concatenate([pos, quat, eul], axis=-1)
            targ_pose = np.concatenate([targ_pos, targ_quat, targ_eul], axis=-1)

        # dynamics of waypoint -> action
        ac, _, _ = wp_dyn(curr_pose, targ_pose)

        ac = ac[:, None]  # re-expand to match horizon
        ac[..., -1:] = base_gripper  # fill in gripper action from base action.
        # print(np.linalg.norm(base_action[..., 3:6]), np.linalg.norm(ac[..., 3:6]))

    out.combine(unnorm_out)
    out.action = to_torch(ac, device="cpu").to(device=model.device, dtype=torch.float32)

    # add in orientation keys if missing
    if not no_ori and unnorm_out.has_leaf_key(f'{targ_prefix}orientation') or unnorm_out.has_leaf_keys(
            f'{targ_prefix}orientation_eul'):
        if targ_quat is None or targ_eul is None:
            # extract orientations
            targ_quat, targ_eul = parse_orientations(unnorm_out, targ_prefix)

        out[f'{targ_prefix}orientation'] = to_torch(targ_quat[:, None], device=model.device)
        out[f'{targ_prefix}orientation_eul'] = to_torch(targ_eul[:, None], device=model.device)

    out.leaf_modify(lambda arr: (arr[:, 0] if is_array(arr) else arr))
    return out


def get_rs_online_action_postproc_fn(**kwargs):
    """
    Gets an online postprocessing fn for the actions, using some arbitrary overriding kwargs.

    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    def online_fn(*args, **inner_kwargs):
        inner_kwargs.update(kwargs)
        return rs_generic_online_action_postproc_fn(*args, **inner_kwargs)

    return online_fn


def modify_spec_prms(prms, no_names=False, raw=False, minimal=False, no_reward=False, no_object=False,
                     include_click_state=False, include_mode=False,
                     include_real=False, include_target_names=False, include_target_gripper=False,
                     include_policy_actions=True):
    if no_names:
        prms.action_names.remove('policy_name')

    # the raw data doesn't contain euler angle keys
    if raw:
        prms.observation_names.remove('robot0_eef_eul')

    if minimal:
        allowed = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        prms.observation_names = allowed
        prms.action_names = ['action']
        if include_policy_actions:
            prms.action_names.extend(["policy_type", "policy_name", "policy_switch"])
    elif not include_policy_actions:
        prms.action_names.remove("policy_type")
        prms.action_names.remove("policy_name")
        prms.action_names.remove("policy_switch")

    if no_reward:
        prms.output_observation_names.remove('reward')

    if no_object and 'object' in prms.observation_names:
        prms.observation_names.remove("object")

    if include_click_state:
        prms.action_names.append('click_state')

    if include_mode:
        prms.observation_names.append('mode')

        # change to support mode0 and mode1 actions (HYDRA)
        prms.cls = BiModalParamEnvSpec
        prms.mode0_action_names = ['target/position', 'target/orientation', 'target/orientation_eul']
        prms.mode1_action_names = ['action']
        prms.dynamics_state_names = ["robot0_eef_pos", "robot0_eef_quat", "robot0_eef_eul"]

    if include_real:
        prms.observation_names.append('real')

    if include_target_names:
        prms.action_names.extend(['target/position', 'target/orientation', 'target/orientation_eul'])

    if include_target_gripper:
        prms.action_names.extend(['target/gripper'])

    return prms


def rs_gripper_width_as_contact_fn(inputs):
    """
    Takes an episode

    Parameters
    ----------
    self
    inputs

    Returns
    -------

    """
    c0 = inputs.robot0_gripper_qpos[..., 0]
    c1 = inputs.robot0_gripper_qpos[..., 1]

    gw = c1 - c0

    contact = np.abs(gw) < GRIPPER_WIDTH_TOL
    return contact



