import muse.envs.robosuite.robosuite_utils as rs
from muse.envs.mode_param_spec import BiModalParamEnvSpec


def get_wp_dynamics_fn(max_pos_vel=0.1, max_ori_vel=1.0, DT=0.1, TOL=0.0075, ORI_TOL=0.075,
                       true_max_pos_vel=0.25, true_max_ori_vel=2.5, **kwargs):
    """ (robot pose, mode0_action) -> (next robot pose, mode1_action, reached) """
    return rs.get_wp_dynamics_fn(max_pos_vel=max_pos_vel,
                                 max_ori_vel=max_ori_vel, DT=DT, TOL=TOL,
                                 ORI_TOL=ORI_TOL, scale_action=False,
                                 true_max_pos_vel=true_max_pos_vel,
                                 true_max_ori_vel=true_max_ori_vel, **kwargs)


def get_polymetis_online_action_postproc_fn(pos_key="ee_position", quat_key="ee_orientation",
                                            eul_key="ee_orientation_eul", targ_prefix="target/ee_", wp_dyn=None,
                                            **kwargs):
    if wp_dyn is None:
        wp_dyn = get_wp_dynamics_fn()
    return rs.get_rs_online_action_postproc_fn(pos_key=pos_key,
                                               quat_key=quat_key,
                                               eul_key=eul_key,
                                               targ_prefix=targ_prefix,
                                               wp_dyn=wp_dyn, **kwargs)


def modify_spec_prms(prms, no_names=False, raw=False, minimal=False, no_reward=False,
                     include_click_state=False, include_mode=False,
                     include_real=False, include_target_names=False, include_target_gripper=False,
                     include_policy_actions=True):
    if no_names:
        prms.action_names.remove('policy_name')

    # the raw data doesn't contain euler angle keys
    if raw:
        prms.observation_names.remove('ee_orientation_eul')

    if minimal:
        allowed = ["ee_position", "ee_orientation", "gripper_pos"]
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

    if include_click_state:
        prms.action_names.append('click_state')

    if include_mode:
        prms.observation_names.append('mode')

        # change to support mode0 and mode1 actions (HYDRA)
        prms.cls = BiModalParamEnvSpec
        prms.mode0_action_names = ['target/ee_position', 'target/ee_orientation', 'target/ee_orientation_eul']
        prms.mode1_action_names = ['action']
        prms.dynamics_state_names = ["ee_position", "ee_orientation", "ee_orientation_eul"]

    if include_real:
        prms.observation_names.append('real')

    if include_target_names:
        prms.action_names.extend(['target/ee_position', 'target/ee_orientation', 'target/ee_orientation_eul'])

    if include_target_gripper:
        prms.action_names.extend(['target/gripper_pos'])

    return prms
