import os
import time

import numpy as np
import pybullet as p

import muse.envs.bullet_envs.teleop_functions
from muse.envs.env import Env
from muse.envs.env_spec import EnvSpec
from muse.envs.env_interfaces import VRInterface
from muse.experiments import logger
from muse.utils.input_utils import UserInput
from muse.utils.general_utils import timeit
from muse.utils import transform_utils as T
from muse.utils.torch_utils import to_numpy

from attrdict import AttrDict
from attrdict.utils import get_with_default


class RobotBulletEnv(Env, VRInterface):
    """
    Implements a single robot + scene abstraction as a gym-like Env, in line with the root Env class.

    Scenes consist of a robot and a set of objects.
    initialization is split into the following:

    _init_bullet_world
    _init_figure

    loading is split into:

    _load_robot
    _load_assets
    _load_dynamics

    reset is split into

    pre_reset
    reset_robot
    reset_assets
    reset_dynamics
    _reset_images

    """

    # map mode -> action dim (add more when implemented)
    action_modes = {
        'ee_euler': 7,
        'ee_euler_delta': 7,
        'ee_quat': 8,
        'ee_quat_delta': 8,
        'ee_axisangle': 7,
        'ee_axisangle_delta': 7,
        'ee_rot6d': 10,
    }

    action_high = {
        'ee_euler': np.array([10., 10., 10., np.pi, np.pi, np.pi, 1.]),  # TODO gripper
        'ee_euler_delta': np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.]),
        'ee_quat': np.array([10., 10., 10., 1., 1., 1., 1., 1.]),
        'ee_quat_delta': np.array([1.0, 1.0, 1.0, 1., 1., 1., 1., 1.]),
        'ee_axisangle': np.array([10., 10., 10., np.pi, np.pi, np.pi, 1.]),
        'ee_axisangle_delta': np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.]),
        'ee_rot6d': np.array([10., 10., 10.] + [1.] * 7),
    }

    action_low = {
        m: -h for m, h in action_high.items()
    }

    def __init__(self, params: AttrDict, env_spec: EnvSpec):
        super(RobotBulletEnv, self).__init__(params, env_spec)
        self.asset_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../assets')
        logger.debug("Assets in: %s" % self.asset_directory)
        assert os.path.exists(self.asset_directory)

        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params: AttrDict):
        self.camdist = 0.05

        self.debug_cam_dist = get_with_default(params, "debug_cam_dist", 1.3)
        self.debug_cam_p = get_with_default(params, "debug_cam_p", -45)
        self.debug_cam_y = get_with_default(params, "debug_cam_y", 40)
        self.debug_cam_target_pos = get_with_default(params, "debug_cam_target_pos", [-0.2, 0, 0.75])
        self.gui_width, self.gui_height = get_with_default(params, "gui_width", 1920), \
            get_with_default(params, "gui_height", 1080)
        self.img_width, self.img_height = params << "img_width", params << "img_height"

        """ control specific """
        # stepSimulation dt
        self.time_step = get_with_default(params, "time_step", 0.02)
        # base environment controls in absolute ee_euler, so the default is the same
        self.action_mode = get_with_default(params, "action_mode", 'ee_euler')
        # what to use for delta actions as the base action (ground truth robot pose, or expected pose after last action)
        self.delta_pivot = get_with_default(params, "delta_pivot", 'ground_truth')
        # 10Hz default
        self.skip_n_frames_every_step = get_with_default(params, "skip_n_frames_every_step", 5)
        self.dt = self.time_step * self.skip_n_frames_every_step
        assert self.action_mode in self.action_modes, f"Action mode {self.action_mode} not implemented!"
        assert self.delta_pivot in ['ground_truth', 'expected']
        # gripper action will be -1 to 1, this is the internal range for _control()
        self.gripper_range = get_with_default(params, "gripper_range", np.array([0., 255.]))

        # no GUI default
        self._render = get_with_default(params, "render", False)
        # no gravity default
        self._use_gravity = get_with_default(params, "use_gravity", True)
        # where to call _control
        self._control_inner_step = get_with_default(params, "control_inner_step", True)
        # how many steps to run before quitting
        self._max_steps = get_with_default(params, "max_steps", np.inf, map_fn=int)

        # returning images or not, False should speed things up
        self.compute_images = get_with_default(params, "compute_images", True)
        # returning ego images or not, False should speed things up
        self.compute_ego_images = get_with_default(params, "compute_ego_images", False)
        # tilt off vertical
        self.ego_tilt_angle = get_with_default(params, "ego_tilt_angle", np.deg2rad(25))

        # allows GL rendering,
        self.non_gui_mode = get_with_default(params, "non_gui_mode",
                                             p.GUI if (self.compute_images or self.compute_ego_images)
                                                      and "DISPLAY" in os.environ.keys() else p.DIRECT)

        # computes reward, optional: 
        #   env_reward_fn(curr_obs, goal, action, next_obs=next_obs, done=done, env=self) -> reward
        self.env_reward_fn = get_with_default(params, "env_reward_fn", None)
        # pass in env
        self._env_reward_requires_env = get_with_default(params, "env_reward_requires_env", True)

        self.debug = get_with_default(params, "debug", False)

        self._teleop_fn = get_with_default(params, "teleop_fn",
                                           muse.envs.bullet_envs.teleop_functions.bullet_keys_teleop_fn)

    def _init_setup(self):
        self.id = None  # THIS SHOULD BE SET
        self._curr_obs = AttrDict()
        self.last_action = None

        self._init_bullet_world()

        self.load()

        self.setup_timing()  # todo add to resetr

    def _init_bullet_world(self):
        if self._render:
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 '
                                               '--background_color_blue=1.0 --width=%d --height=%d' % (self.gui_width,
                                                                                                       self.gui_height))
            logger.warn('Render physics server ID: %d' % self.id)

            # if self.compute_images:
            #     print("Showing plot")
            #     plt.show(block=False)
            #     self.fig.canvas.draw()
        else:
            self.id = p.connect(self.non_gui_mode)
            logger.warn('Physics server ID: %d' % self.id)

        p.resetSimulation(physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

        # Disable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        p.setTimeStep(self.time_step, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=200, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)

    def _init_figure(self):
        raise NotImplementedError

    """ LOADERS """
    def load(self):
        """ 0. plane """
        p.loadURDF(os.path.join(self.asset_directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # load in robots
        self._load_robot()

        # load in assets
        self._load_assets()

        # load in dynamics (constraints, etc)
        self._load_dynamics()

    def _load_robot(self, presets: AttrDict = AttrDict()):
        raise NotImplementedError

    # Load all models off screen and then move them into place
    def _load_assets(self, presets: AttrDict = AttrDict()):
        pass

    # do stuff here like collision handling
    def _load_dynamics(self, presets: AttrDict = AttrDict()):
        pass

    """ GETTERS """
    def get_id(self):
        assert self.id is not None, "Id must be set!"
        return self.id

    def _get_obs(self, **kwargs):
        raise NotImplementedError

    def _get_goal(self, **kwargs):
        return AttrDict()

    def _get_reward(self, curr_obs, next_obs, goal, action, done):
        if self.env_reward_fn is None:
            rew = self.is_success()
        elif self._env_reward_requires_env:
            rew = self.env_reward_fn(curr_obs, goal, action, next_obs=next_obs, done=done, env=self)
        else:
            rew = self.env_reward_fn(curr_obs, goal, action, next_obs=next_obs, done=done)
        return np.array([[float(rew)]])

    def _get_images(self, **kwargs):
        raise NotImplementedError

    def get_joint_indices(self):
        raise NotImplementedError

    def get_gripper_joint_indices(self):
        raise NotImplementedError

    def get_initial_joint_positions(self):
        raise NotImplementedError

    def get_joint_limits(self):
        raise NotImplementedError

    """ SETTERS """
    def reset_joint_positions(self, q, **kwargs):
        pass

    def reset_gripper_positions(self, g, **kwargs):
        pass

    def _control(self, action, **kwargs):
        pass

    def set_external_forces(self, action):
        pass

    def update_targets(self):
        pass

    """ TIMING """
    def slow_time(self, record=False):
        if record and self.last_sim_time is None:
            self.last_sim_time = time.time()
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)

        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    """ ENV """

    def _read_state(self):
        pass

    def _register_obs(self, obs: AttrDict, done: np.ndarray):
        self._curr_obs = obs.copy()

    def _get_done(self, obs: AttrDict = None):
        return np.array([self.iteration >= self._max_steps])

    def _process_action(self, action):
        """ Conversion from whatever input action space to ee pos / ori_eul

        Gripper is converted from [-1 (open), 1 (closed)] to self.gripper_range

        Parameters
        ----------
        action: (1 x ac_dim)

        Returns
        -------
        new_action: (1 x 7) pos / ori_eul / gripper

        """
        # parse pos / orientation (dimension should match the conversion function input)
        ee_pos = action[..., :3]
        ori = action[..., 3:-1]
        gripper = np.clip(action[..., -1:], -1, 1)  # -1 to 1
        if 'euler' in self.action_mode:
            assert ori.shape[-1] == 3
            ee_eul = ori  # 3D
        elif 'quat' in self.action_mode:
            ee_eul = T.fast_quat2euler(ori)
        elif 'rot6d' in self.action_mode:
            ee_eul = T.mat2euler(T.rot6d2mat(ori))
        elif 'axisangle' in self.action_mode:
            ee_eul = T.mat2euler(T.axisangle2mat(ori))
        else:
            raise NotImplementedError(self.action_mode)

        # turns action input to absolute ee euler (output space)
        if self.action_mode.endswith('delta'):
            if self.delta_pivot == 'ground_truth':
                pivot = np.concatenate([self.get_ee_pos(), self.get_ee_eul()])[None]
            elif self.delta_pivot == 'expected':
                pivot = self.last_action[..., :6]
            else:
                raise NotImplementedError(self.delta_pivot)

            # ee pos / eul interpreted as delta
            ee_pos = pivot[..., :3] + ee_pos
            ee_eul = T.add_euler(ee_eul, pivot[..., :3])

        # un-scale gripper from (-1, 1) to gripper_range
        gripper = self.gripper_range[0] + (self.gripper_range[1] - self.gripper_range[0]) * (gripper + 1) * 0.5

        new_action = np.concatenate([ee_pos, ee_eul, gripper], axis=-1)

        self.last_action = new_action.copy()
        return new_action

    def step(self, action, ret_images=True, skip_render=False, **kwargs):
        with timeit("env_step/total"):
            self.iteration += 1
            if self.last_sim_time is None:
                self.last_sim_time = time.time()

            with timeit("env_step/read_state"):
                self._read_state()

            # action = np.clip(action, a_min=self.robot_action_low, a_max=self.robot_action_high)  # TODO
            # action_robot = action * self.robot_action_multiplier
            if isinstance(action, AttrDict):
                act_np = to_numpy(action.action[0], check=True).copy()  # assumes batched (1,..)
            elif isinstance(action, np.ndarray):
                act_np = action.copy()  # assumes not batched
            else:
                pass

            # process the action to the type expected for control(), usually absolute ee pose (eul)
            act_np = self._process_action(act_np)

            # sets motor control for each joint
            if not self._control_inner_step:
                with timeit("env_step/control"):
                    self._control(act_np)

            # Update robot position
            for _ in range(self.skip_n_frames_every_step):
                if self._control_inner_step:
                    with timeit("env_step/control"):
                        self._control(act_np)

                with timeit("env_step/step"):
                    self.set_external_forces(act_np)  # each stepSim clears external forces
                    self._step_simulation()
                    self.update_targets()

                if self._render and not skip_render:
                    # Slow down time so that the simulation matches real time
                    with timeit("env_step/slow_time"):
                        self.slow_time()

            self._after_step_simulation()
            # self.record_video_frame()

            with timeit("env_step/post_step"):
                next_obs, next_goal, done = self._step_suffix(action, ret_images=ret_images, **kwargs)
        return next_obs, next_goal, done

    def _step_suffix(self, action, ret_images=True, **kwargs):
        next_obs = self._get_obs(ret_images=ret_images and self.compute_images,
                                 ret_ego_images=ret_images and self.compute_ego_images)
        done = self._get_done(obs=next_obs)
        if not next_obs.has_leaf_key("reward"):
            next_obs.reward = self._get_reward(self._curr_obs, next_obs, AttrDict(), action, done)
        self._register_obs(next_obs, done)
        return next_obs, AttrDict(), done

    def _step_simulation(self):
        p.stepSimulation(physicsClientId=self.id)

    def _after_step_simulation(self):
        pass

    def clear_gui_elements(self):
        pass

    def cleanup(self):
        pass  # this is where you delete objects between resets

    # override safe
    def pre_reset(self, presets: AttrDict = AttrDict()):
        pass

    def reset_robot(self, presets: AttrDict = AttrDict()):
        raise NotImplementedError

    def reset_assets(self, presets: AttrDict = AttrDict()):
        pass

    def reset_dynamics(self, presets: AttrDict = AttrDict()):
        pass

    def _reset_images(self, presets: AttrDict = AttrDict()):
        pass

    # DO NOT OVERRIDE
    # @profile
    def reset(self,
              presets: AttrDict = AttrDict()):  # ret_images=False, food_type=None, food_size=None, food_orient_eul=None, mouth_orient_eul=None):

        self.setup_timing()

        # if is_next_cycle(self.num_resets, self.reset_full_every_n):  # TODO
        #     logger.warn("Resetting sim fully")
        #     p.resetSimulation(physicsClientId=self.id)
        #     self._load_assets()
        # process = psutil.Process(os.getpid())
        # before = process.memory_info().rss
        # logger.debug("Clearing old things: %s" % process.memory_info().rss)
        # cleaning up old objects (normal)
        self.cleanup()

        if self._render:
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # any initial setup actions
        self.pre_reset(presets)

        self.clear_gui_elements()

        self.reset_robot(presets)

        self.reset_assets(presets)

        self.reset_dynamics(presets)

        if self._render:
            p.resetDebugVisualizerCamera(cameraDistance=self.debug_cam_dist, cameraYaw=self.debug_cam_y, cameraPitch=self.debug_cam_p,
                                         cameraTargetPosition=self.debug_cam_target_pos, physicsClientId=self.id)

        if self._use_gravity:
            p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        else:
            p.setGravity(0, 0, 0, physicsClientId=self.id)

        # initialize everything
        for i in range(10):
            p.stepSimulation()

        # here is where we compute the camera images
        if self.compute_images:
            self._reset_images(presets)

        # process = psutil.Process(os.getpid())
        # after = process.memory_info().rss
        # logger.debug("-> After resetting things: %s | delta = %s" % (after, after - before))
        obs = self._get_obs(ret_images=presets.get("ret_images", True) and self.compute_images,
                            ret_ego_images=presets.get("ret_images", True) and self.compute_ego_images)
        if not obs.has_leaf_key("reward"):
            obs.reward = np.zeros((1, 1))
        done = self._get_done(obs=obs)
        # first obs
        self._register_obs(obs, done)

        self.last_action = np.concatenate([self.get_ee_pos(), self.get_ee_eul(), [self.get_gripper(normalized=False)]])
        return obs, self._get_goal()

    def get_link_info(self, object_id):
        numJoint = p.getNumJoints(object_id)
        LinkList = ['base']
        for jointIndex in range(numJoint):
            jointInfo = p.getJointInfo(object_id, jointIndex)
            # print("jointINFO", jointInfo)
            link_name = jointInfo[12]
            if link_name not in LinkList:
                LinkList.append(link_name)
        return LinkList

    def get_ee_pos(self):
        raise NotImplementedError('EE pos should be implemented by subclasses')

    def get_ee_eul(self):
        raise NotImplementedError('EE eul should be implemented by subclasses')

    def get_gripper(self, normalized=True):
        raise NotImplementedError('Gripper should be implemented by subclasses')

    def get_num_links(self, object_id):
        return len(self.get_link_info(object_id))

    @property
    def zero_action(self):
        if self.action_mode.endswith('delta'):
            ac = np.zeros((self.action_modes[self.action_mode]))
            if 'quat' in self.action_mode:
                ac[3:7] = np.array([0, 0, 0, 1.])
            ac[-1] = self.get_gripper()
        else:
            ori_eul = self.get_ee_eul()
            if 'euler' in self.action_mode:
                ori = ori_eul
            elif 'quat' in self.action_mode:
                ori = T.euler2quat(ori_eul)
            elif 'axisangle' in self.action_mode:
                ori = T.mat2axisangle(T.euler2mat(ori_eul))
            elif 'rot6d' in self.action_mode:
                ori = T.mat2rot6d(T.euler2mat(ori_eul))
            else:
                raise NotImplementedError(self.action_mode)

            ac = np.concatenate([self.get_ee_pos(), ori, [self.get_gripper()]])

        return ac

    def get_aabb(self, object_id):
        num_links = self.get_num_links(object_id)
        aabb_list = []
        # get all link bounding boxes, pick the max on each dim
        for link_id in range(-1, num_links-1):
            aabb_list.append(p.getAABB(object_id, link_id, physicsClientId=self.id))
        aabb_array = np.array(aabb_list)
        aabb_obj_min = np.min(aabb_array[:, 0, :], axis=0)
        aabb_obj_max = np.max(aabb_array[:, 1, :], axis=0)
        aabb_obj = np.array([aabb_obj_min, aabb_obj_max])
        return aabb_obj

    def get_default_teleop_model_forward_fn(self, user_input: UserInput):  # TODO add this to parents
        return self._teleop_fn(self, user_input)

    """ VR interface """
    def get_safenet(self):
        return np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])

    def change_view(self, delta_dist=0, delta_yaw=0, delta_pitch=0, delta_target=np.zeros(3)):
        self.debug_cam_dist += delta_dist
        self.debug_cam_y += delta_yaw
        self.debug_cam_p += delta_pitch
        self.debug_cam_target_pos = (np.asarray(self.debug_cam_target_pos) + delta_target).tolist()

        p.resetDebugVisualizerCamera(cameraDistance=self.debug_cam_dist, cameraYaw=self.debug_cam_y,
                                     cameraPitch=self.debug_cam_p,
                                     cameraTargetPosition=self.debug_cam_target_pos, physicsClientId=self.id)
