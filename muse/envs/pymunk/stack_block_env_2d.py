import numpy as np

from muse.envs.env import make
from muse.envs.pymunk.block_env_2d import BlockEnv2D

from attrdict import AttrDict
from attrdict.utils import get_with_default

from muse.envs.pymunk.teleop_functions import pygame_key_teleop_step


class StackBlockEnv2D(BlockEnv2D):
    """
    Stacking environment with gravity, and a single other block.
    """

    default_params = BlockEnv2D.default_params & AttrDict(
        num_maze_cells=8,
        num_blocks=1,
        grid_size=np.array([400., 400.]),
        gravity=(0., -100.),
        damping=0.75,
        ego_block_size=40,
        block_size=(40, 80),
        block_size_lower=(30, 30),
        block_size_upper=(60, 80),
        block_mass=10,
        block_corner_radius=3.,
        block_grabbing_frac=1.5,
        block_friction=0.2,
        block_bbox=True,
        static_line_friction=0.4,
        default_teleop_speed=120.,
        initialization_steps=30,
        grab_add_rotary_limit_joint=False,
        break_constraints_on_large_impulse=True,
        grab_slider_min_frac=0.5,
        max_velocity=100,
        grab_action_max_force=10000,
        keep_in_bounds=True,
    )

    def _init_params_to_attrs(self, params):
        # maze
        maze = np.zeros((params.num_maze_cells, params.num_maze_cells))
        maze[params.num_maze_cells - 3, 0] = 8  # left most ledge
        maze[params.num_maze_cells - 3, 1] = 8  # left most ledge
        maze[params.num_maze_cells - 5, params.num_maze_cells - 1] = 8  # right most ledge (lowest)
        maze[params.num_maze_cells - 5, params.num_maze_cells - 2] = 8  # right most ledge (lowest)
        maze[params.num_maze_cells - 2, params.num_maze_cells // 2] = 8  # middle ledge (highest)
        maze[params.num_maze_cells - 2, params.num_maze_cells // 2 + 1] = 8  # middle ledge (highest)
        params.fixed_np_maze = get_with_default(params, "fixed_np_maze", maze)
        # bottom half only
        # vs_default = np.meshgrid(range(params.num_maze_cells), range(params.num_maze_cells // 2))
        # vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
        # params.valid_start_idxs = get_with_default(params, "valid_start_idxs", vs_default)

        super(StackBlockEnv2D, self)._init_params_to_attrs(params)

    def reset(self, presets: AttrDict = AttrDict()):
        obs, goal = super(StackBlockEnv2D, self).reset(presets)
        invalid_angle = (np.abs([self.bodies[i].angle for i in range(self.num_blocks)]) > np.pi / 10).any()
        invalid_block_pos = any([body.position.x < 0 or body.position.x > self.grid_size[0] or body.position.y < 0 or
                                 body.position.y > self.grid_size[1] for body in (self.bodies + [self.player_body])])
        invalid_reset = invalid_angle or invalid_block_pos
        while invalid_reset:
            obs, goal = super(StackBlockEnv2D, self).reset(presets)
            invalid_angle = (np.abs([self.bodies[i].angle for i in range(self.num_blocks)]) > np.pi / 10).any()
            invalid_block_pos = any(
                [body.position.x < 0 or body.position.x > self.grid_size[0] or body.position.y < 0 or
                 body.position.y > self.grid_size[1] for body in (self.bodies + [self.player_body])])
            invalid_reset = invalid_angle or invalid_block_pos
        return obs, goal

    def get_block_positions(self, presets):
        locations = super(StackBlockEnv2D, self).get_block_positions(presets)
        # ego block should always be the highest to start. (permute the order accordingly)
        highest_idx = np.argmax(locations[:, 1])
        return np.concatenate([
            locations[:highest_idx],
            locations[highest_idx + 1:],
            locations[highest_idx:highest_idx + 1]  # ego last
        ], axis=0)


if __name__ == "__main__":
    start_near_bottom = True

    grid_size = StackBlockEnv2D.default_params.grid_size
    block_max_size = StackBlockEnv2D.default_params.block_size_upper

    num_blocks = (np.asarray(grid_size) / np.asarray(block_max_size)).astype(int)

    env_params = AttrDict(
        render=True,
        realtime=True,
    )

    if start_near_bottom:
        vs_default = np.meshgrid(range(num_blocks[0]), range(num_blocks[1] // 2))
        vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
        env_params.valid_start_idxs = vs_default

    env = make(StackBlockEnv2D, env_params)
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

