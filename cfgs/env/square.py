from muse.envs.robosuite.robosuite_env import RobosuiteEnv
from attrdict import AttrDict as d

export = d(
    cls=RobosuiteEnv,
    env_name='NutAssemblySquare',
    img_width=84,
    img_height=84,
    imgs=False,
    ego_imgs=False,
    no_ori=False,
    onscreen_camera_name='agentview',
    offscreen_camera_name='agentview',
    render=False,
    controller=None,
    done_on_success=True,
    use_reward_stages=False,
    enable_preset_sweep=True,
    preset_sweep_pos=5,
    preset_sweep_ori=10,
    pos_noise_std=0.,
    ori_noise_std=0.,
)
