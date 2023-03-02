import matplotlib.pyplot as plt
import numpy as np

from muse.envs.env import Env, make
from muse.envs.env_spec import EnvSpec
from muse.utils.general_utils import value_if_none
from muse.utils.np_utils import line_circle_intersection
from attrdict import AttrDict
from attrdict.utils import get_with_default
from muse.utils.torch_utils import to_numpy


class PMObstacle:
    """
    TODO:
    - reset: set the state of obstacle
    - get_obs: return object position
    - step: alter path of target based on obstacle
    - rendering
    """
    def __init__(self, center=(0., 0.), radius=0.1):
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)

    def is_colliding(self, point, ret_distance=False):
        norm = np.linalg.norm(np.asarray(point, dtype=np.float32) - self.center)
        return (norm <= self.radius, norm) if ret_distance else norm <= self.radius

    def collision_safe_next_point(self, point, next_point):
        point, next_point = np.asarray(point), np.asarray(next_point)
        initial_norm = np.linalg.norm(next_point - point)

        # same point check
        if initial_norm < 1e-11:
            return next_point

        intersections = line_circle_intersection(point, next_point, self.radius, self.center)

        all_dists = [np.linalg.norm(inter - point) for inter in intersections]
        all_dot = [(inter - point).dot(next_point - point) for inter in intersections]
        # in direction of motion and close by
        intersections = [io for i, io in enumerate(intersections) if all_dists[i] < initial_norm and all_dot[i] >= 0]

        if len(intersections) == 0:
            return next_point
        elif len(intersections) == 1:
            pt = intersections[0]
            dist = np.linalg.norm(pt - point)
        else:
            # multiple intersections, pick the one closest
            closest_idx = np.argmin(all_dists)
            pt = intersections[closest_idx]
            dist = all_dists[closest_idx]

        assert dist <= initial_norm, "Norm increased!! this is a bug"
        return point + 0.9 * (pt - point)  # some tolerance needed


class PointMassEnv(Env):
    """
    Simple 2D point mass environment.
    """
    def __init__(self, params, env_spec: EnvSpec):
        super().__init__(params, env_spec)

        # PARAMS
        self._render = params["render"]
        self._num_steps = params["num_steps"]
        self._noise_std = params["noise_std"]
        self._theta_noise_std = params["theta_noise_std"]
        self._sparse_reward = get_with_default(params, "sparse_reward", True)
        self._target_speed = params["target_speed"]
        self._ego_speed = params["ego_speed"]

        self._init_obs = params << "initial_obs"
        self._init_targ = params << "initial_target"

        # optional obstacles (circles) will be spawned on the field, which the target and ego cannot go through.
        self.num_obstacles = get_with_default(params, "num_obstacles", 0)
        self.obstacle_bounds = get_with_default(params, "obstacle_bounds", [0.25, 0.75])  # random bounds for init
        self.obstacle_radii = get_with_default(params, "obstacle_radii", [0.05 for _ in range(self.num_obstacles)])
        self.obstacle_face_colors = get_with_default(params, "obstacle_face_colors",
                                                     ['g', 'c', 'm', 'y', 'k', 'w', 'r', 'b'])
        assert len(self.obstacle_radii) == self.num_obstacles, len(self.obstacle_radii)

        self.init_obstacle_positions = params << "initial_obstacle_positions"
        # self.init_obstacle_radii = params << "initial_obstacle_radii" TODO

        self.obstacles = [PMObstacle(radius=self.obstacle_radii[i]) for i in range(self.num_obstacles)]
        self.obstacles_changed = [True for _ in range(self.num_obstacles)]

        if self._render:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.line, = self.ax.plot(0, 0, "ro-")
            self.line2, = self.ax.plot(0, 0, "bo-")
            self.obstacle_patches = []
            self.obstacle_face_colors = self.obstacle_face_colors[:self.num_obstacles]
            if len(self.obstacle_face_colors) < self.num_obstacles:
                raise NotImplementedError("specify the right number of colors!!")

            for obstacle, color in zip(self.obstacles, self.obstacle_face_colors):
                circle = plt.Circle(tuple(obstacle.center.tolist()), obstacle.radius, fc=color)
                self.ax.add_patch(circle)
                self.obstacle_patches.append(circle)

            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            plt.ion()
            plt.show()

            # plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self._obs = np.zeros(2)
        self._target = np.zeros(2)
        self._target_vel = np.zeros(2)
        self._reward = 0
        self._done = False
        self._next_done = False
        self._curr_step = 0

    def _init_parameters(self, params):
        pass

    # this will be overriden
    def step(self, action):
        # batch input
        base_action = to_numpy(action.action[0], check=True)

        # OBSTACLE SAFE noisy velocity control
        next_obs = self._obs + self._ego_speed * base_action + np.random.randn(2) * self._noise_std
        for o in self.obstacles:
            # each can only reduce the norm
            next_obs = o.collision_safe_next_point(self._obs, next_obs)
        self._obs = next_obs

        vel = base_action / np.linalg.norm(base_action)

        direction = self._target - self._obs
        norm_direction = direction / np.linalg.norm(direction)
        cos_phi = vel.dot(norm_direction)  # cos(phi), phi is the angle between
        cos_phi = max(cos_phi, 0)  # ignore the other half plane motion
        # cos phi is 1 when the directions match

        prob_run_away = 0.1 + 0.7 * cos_phi  # range: 0.1 -> 0.6

        if np.random.random() < prob_run_away:  # running away behavior
            theta = np.random.normal(loc=np.arctan2(vel[1], vel[0]), scale=self._theta_noise_std) % (2 * np.pi)
        else:
            theta = np.random.uniform(0, 2*np.pi)
        self._target_vel = self._target_speed * np.array([np.cos(theta), np.sin(theta)])

        # OBSTACLE SAFE target motion
        next_targ = self._target + 0.01 * self._target_vel
        for o in self.obstacles:
            # each can only reduce the norm
            next_targ = o.collision_safe_next_point(self._target, next_targ)
        self._target = next_targ

        self._obs = np.clip(self._obs, 0, 1)
        self._target = np.clip(self._target, 0, 1)

        self._curr_step += 1
        self._done = self._next_done or self._curr_step >= self._num_steps

        if self._sparse_reward:
            self._reward = float(np.linalg.norm(self._obs - self._target) <= 0.01)
            self._next_done = self._reward > 0
            if any(self.obstacles[i].is_colliding(self._obs) for i in range(self.num_obstacles)):
                self._reward += -1  # penalty for collision
        else:
            self._reward = - np.linalg.norm(self._obs - self._target)
            self._next_done = self._reward > -0.01

        if self._render:
            self.line.set_data([self._obs[0]], [self._obs[0]])
            self.line2.set_data([self._target[0]], [self._target[0]])
            for i, o in enumerate(self.obstacles):
                # update center if changed.
                if self.obstacles_changed[i]:
                    self.obstacle_patches[i].center = tuple(o.center.tolist())
                    self.obstacles_changed[i] = False
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.show()
            plt.pause(0.04)  # 25Hz (4 seconds per episode)

        return self.get_obs(), self.get_goal(), np.array([self._done])

    def reset(self, presets: AttrDict = None):
        presets = value_if_none(presets, AttrDict())
        # EGO / TARGET
        # random if no default / preset is specified.
        self._obs = np.random.uniform(0, 1, size=(2,)) if self._init_obs is None else self._init_obs
        self._target = np.random.uniform(0, 1, size=(2,)) if self._init_targ is None else self._init_targ

        if "obs" in presets.leaf_keys():
            self._obs[:] = presets.obs[0, :2]
            self._target[:] = presets.obs[0, 2:]

        # OBSTACLES
        # random if no default / preset is specified.
        for i in range(self.num_obstacles):
            if self.init_obstacle_positions is None:
                # rejection sampling, TODO fail after N iterations?
                while True:
                    self.obstacles[i].center = np.random.uniform(*self.obstacle_bounds, size=(2,))
                    # list of (center, radius) to compare against (includes prev obstacles, ego, and target points)
                    margin = 0.005  # separation between obstacle and new obstacle to enforce
                    to_compare = [(self._obs, 0.), (self._target, 0.)] + \
                                 [(self.obstacles[j].center, self.obstacles[j].radius) for j in range(i)]
                    has_clearance = [np.linalg.norm(self.obstacles[i].center - c) >= self.obstacles[i].radius + r + margin
                                     for c, r in to_compare]  # compare to previous obstacles
                    if all(has_clearance):
                        break  # random initialization is non overlapping.
            else:
                self.obstacles[i].center = np.array(self.init_obstacle_positions[i])

        if "objects/position" in presets.leaf_keys():
            opos = presets["objects/position"]
            assert list(opos.shape) == [1, self.num_obstacles, 2]
            for i in range(self.num_obstacles):
                self.obstacles[i].center = opos[0, i]  # i'th object pos

        self._target_vel = np.zeros((2,))
        self._reward = 0
        self._curr_step = 0

        if self._render:  # render on reset
            self.line.set_data([self._obs[0]], [self._obs[0]])
            self.line2.set_data([self._target[0]], [self._target[0]])
            for i in range(self.num_obstacles):
                self.obstacle_patches[i].center = tuple(self.obstacles[i].center.tolist())
                self.obstacles_changed[i] = False
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.show()
            plt.pause(0.01)  # faster than DT

        self._next_done = False
        return self.get_obs(), self.get_goal()

    def get_obs(self):
        observation = AttrDict(
            ego=self._obs[None],
            target=self._target[None],
            reward=np.array([self._reward])[None],
        )
        if self.num_obstacles > 0:
            object_pos = np.stack([o.center for o in self.obstacles], axis=0)
            observation['objects/position'] = object_pos[None]
            observation['objects/size'] = np.asarray(self.obstacle_radii)[None]
            observation['objects/contact'] = np.asarray([self.obstacles[i].is_colliding(self._obs) for i in range(self.num_obstacles)])[None]
        return self._env_spec.map_to_types(observation)

    def get_goal(self):
        return self._env_spec.map_to_types(
            AttrDict(
            )
        )

    default_params = AttrDict(
        render=False,
        num_steps=100,
        noise_std=0,
        theta_noise_std=0,
        target_speed=0.0125,
        ego_speed=0.025,
        num_obstacles=0,
    )

    @staticmethod
    def get_default_env_spec_params(params: AttrDict = None) -> AttrDict:
        params = value_if_none(params, AttrDict())
        from muse.envs.param_spec import ParamEnvSpec
        n = params['num_obstacles']

        obstacle_names = []
        obstacle_param_names = []
        if n > 0:
            obstacle_names.append('objects/position')
            obstacle_param_names.append('objects/size')
            # obstacle_param_names.append('objects/contact') # TODO

        return AttrDict(
            cls=ParamEnvSpec,
            names_shapes_limits_dtypes=[
                ('ego', (2,), (0, 1), np.float32),
                ('target', (2,), (0, 1), np.float32),
                ('objects/position', (n, 2), (0, 1), np.float32),
                ('objects/size', (n,), (0, 1), np.float32),
                ('objects/contact', (n,), (False, True), bool),
                ('reward', (1,), (-np.inf, np.inf), np.float32),
                ('action', (2,), (-1, 1), np.float32),
            ],
            observation_names=['ego', 'target'] + obstacle_names,
            output_observation_names=['reward'],
            action_names=['action'],
            goal_names=[],
            param_names=obstacle_param_names,
            final_names=[],
        )


if __name__ == '__main__':
    from muse.experiments import logger
    env = make(PointMassEnv, AttrDict(render=True))

    env.reset()

    done = [False]
    while not done[0]:
        action = env.env_spec.get_uniform(env.env_spec.action_names, 1)
        obs, goal, done = env.step(action)

    logger.debug('Done.')
