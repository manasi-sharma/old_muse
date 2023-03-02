import numpy as np

from muse.envs.env import make
from muse.envs.pymunk.block_env_2d import BlockEnv2D
from attrdict import AttrDict as d

from muse.envs.pymunk.teleop_functions import pygame_key_teleop_step


class SimpleBlockEnv2D(BlockEnv2D):
    default_params = BlockEnv2D.default_params & d(
        fixed_np_maze=np.array([
            # bottom left ---> bottom right
            [0, 2, 0, 2, 0],
            [1, 13, 5, 13, 4],
            [1, 5, 7, 5, 4],
            [1, 6, 10, 3, 4],
            [0, 8, 8, 8, 0],
        ]),
        valid_start_idxs=np.array([
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0],                         [1, 4],
            [2, 0], [2, 1], [2, 2],         [2, 4],
            [3, 0],                         [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
        ]),
    )


if __name__ == "__main__":

    params = d(
        render=True,
        realtime=True,
        keep_in_bounds=True,
        grab_action_binary=True,
        initialization_steps=5,
        do_wall_collisions=False,
    )

    env = make(SimpleBlockEnv2D, params)

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
