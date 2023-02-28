from attrdict import AttrDict as d

from muse.envs.robosuite.robosuite_env import RobosuiteEnv

export = d(
    cls=RobosuiteEnv,
    env_name='KitchenEnv',
    img_width=128,
    img_height=128,
    imgs=True,
    ego_imgs=True,
    no_ori=True,
    onscreen_camera_name='agentview',
    offscreen_camera_name='agentview',
    render=False,
    controller=None,
    done_on_success=True,
    use_reward_stages=False,
    enable_preset_sweep=False,
    preset_sweep_pos=8,
    preset_sweep_ori=8,
    pos_noise_std=0,
    ori_noise_std=0,
)
