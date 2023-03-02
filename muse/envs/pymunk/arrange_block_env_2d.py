import numpy as np

from muse.envs.env import make
from muse.envs.pymunk.block_env_2d import BlockEnv2D

from attrdict import AttrDict

from muse.envs.pymunk.teleop_functions import pygame_key_teleop_step


class ArrangeBlockEnv2D(BlockEnv2D):
    default_params = BlockEnv2D.default_params & AttrDict(
        keep_in_bounds=True,
        grab_action_binary=True,
        num_maze_cells=8,
        grid_size=np.array([400., 400.]),
        gravity=(0., -100.),
        damping=0.75,
        ego_block_size=40,
        block_size=(40, 40),
        block_size_lower=(30, 30),
        block_size_upper=(80, 80),
        block_mass=40,
        block_corner_radius=3.,
        block_grabbing_frac=1.3,
        block_friction=0.45,
        block_bbox=True,
        static_line_friction=0.5,
        grab_one_only=True,
        num_blocks=6,
        default_teleop_speed=120.,
        initialization_steps=30,
        grab_add_rotary_limit_joint=True,
        break_constraints_on_large_impulse=True,
        grab_slider_min_frac=0.5,
        max_velocity=100,
        fixed_np_maze=np.zeros((8, 8)),
    )

    def reset(self, presets: AttrDict = AttrDict()):
        obs, goal = super(ArrangeBlockEnv2D, self).reset(presets)
        while (np.abs([self.bodies[i].angle for i in range(self.num_blocks)]) > np.pi / 10).any():
            obs, goal = super(ArrangeBlockEnv2D, self).reset(presets)
        return obs, goal


if __name__ == "__main__":
    env_params = AttrDict(
        render=True,
        realtime=True,
    )

    env = make(ArrangeBlockEnv2D, env_params)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    env.reset()  # trolling with a fake UI

    running = True
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    while running:
        obs, goal, act, last_act, done, running = pygame_key_teleop_step(env, act, last_act, gamma)

        # cv2.imshow("image_test", obs.image)
        # cv2.waitKey(1)

