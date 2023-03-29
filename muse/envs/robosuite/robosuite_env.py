import numpy as np
from attrdict import AttrDict as d
from attrdict.utils import get_with_default
from scipy.spatial.transform import Rotation

from muse.envs.env import Env, make
from muse.envs.param_spec import ParamEnvSpec
from muse.experiments import logger
from muse.utils import transform_utils
from muse.utils.torch_utils import to_numpy
from muse.utils.transform_utils import quat_multiply


def controller_defaults(env_name):
    if env_name == "KitchenEnv":
        return "OSC_POSITION"
    return "OSC_POSE"


class RobosuiteEnv(Env):
    def __init__(self, params, env_spec):
        super(RobosuiteEnv, self).__init__(params, env_spec)
        self._init_params_to_attrs(params)
        self._init_setup()

        self._reward = 0
        self._done = False
        self._draw_actions = []

    def _init_params_to_attrs(self, params):
        self._env_name = params.env_name
        self._control_freq = get_with_default(params, "control_frequency", 20)
        self._robots = get_with_default(params, "robots", "Panda")
        self._controller = get_with_default(params, "controller", controller_defaults(self._env_name))
        self._render = get_with_default(params, "render", False)
        self._imgs = get_with_default(params, "imgs", False)
        self._ego_imgs = get_with_default(params, "ego_imgs", False)
        self._img_postproc = get_with_default(params, "img_postproc", False)
        self._done_on_success = get_with_default(params, "done_on_success", False)
        self._parse_objects = get_with_default(params, "parse_objects", False)

        # disables orientation control
        self._no_ori = get_with_default(params, "no_ori", False)
        assert self._controller != "OSC_POSITION" or self._no_ori, "Cannot enable OSC POSITION control if using orientations"

        logger.debug(f"Env {self._env_name} using controller: {self._controller}")

        # rendering (onscreen and offscreen)
        self._onscreen_camera_name = get_with_default(params, "onscreen_camera_name", "agentview")
        self._offscreen_camera_name = get_with_default(params, "offscreen_camera_name", "agentview")
        self._offscreen_ego_camera_name = get_with_default(params, "offscreen_ego_camera_name", "robot0_eye_in_hand")
        self._W = get_with_default(params, "img_width", 256)
        self._H = get_with_default(params, "img_height", 256)

        self._enable_preset_sweep = get_with_default(params, "enable_preset_sweep", False)
        self._preset_sweep_pos = get_with_default(params, "preset_sweep_pos", 8)
        self._preset_sweep_ori = get_with_default(params, "preset_sweep_ori", 8)

        # noise on actions
        self._pos_noise_std = get_with_default(params, "pos_noise_std", 0)
        self._ori_noise_std = get_with_default(params, "ori_noise_std", 0)

        # will split reward into #stages axes. this is implemented per env.
        self._use_reward_stages = get_with_default(params, "use_reward_stages", False)

        if self._pos_noise_std > 0 or self._ori_noise_std > 0:
            logger.debug(f"Using noise in action: pos = {self._pos_noise_std}, ori = {self._ori_noise_std}")

    def _init_setup(self, **kwargs):
        # base robosuite environment (robosuite).
        from robomimic.envs.env_robosuite import EnvRobosuite
        from robomimic.envs.env_robosuite import ObsUtils
        from robosuite.controllers.controller_factory import load_controller_config
        image_modalities = ['agentview_image'] if self._render or self._imgs else []
        if self._ego_imgs:
            image_modalities.append('robot0_eye_in_hand')
        obs_modality_specs = {
            "obs": {
                "low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat",
                            "robot0_eef_vel_ang",
                            "robot0_eef_vel_lin", "robot0_gripper_qpos", "robot0_gripper_qvel",
                            "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                            "robot0_joint_vel"],
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)  # populates some dynamic keys
        # default meta data from ph/low_dim.hdf5
        env_kwargs = {'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': False,
                      'control_freq': self._control_freq, 'robots': self._robots, 'reward_shaping': False,
                      'controller_configs': load_controller_config(default_controller=self._controller)}
        env_kwargs['controller_configs']['damping'] = 1
        # logger.debug(f"Env config:\n{(json.dumps(env_kwargs, indent=4, sort_keys=True))}")
        if self._env_name == 'KitchenEnv':
            # loads the environments for zoo.
            import robosuite_task_zoo
        self._base_env = EnvRobosuite(self._env_name,
                                      render=self._render,
                                      render_offscreen=self._imgs,
                                      use_image_obs=self._imgs,
                                      postprocess_visual_obs=self._img_postproc,
                                      **env_kwargs)

        if self._enable_preset_sweep:
            self._preset_counter = 0  # for keeping track of things
            self._preset_list = get_object_presets_sweep(self, sweep_pos=self._preset_sweep_pos,
                                                         sweep_orn=self._preset_sweep_ori)

    def generate_preset(self):
        presets = self._preset_list[self._preset_counter]
        self._preset_counter = (self._preset_counter + 1) % len(self._preset_list)  # cycle
        return presets

    def step(self, action):
        # will be scaled -1 -> 1
        base_action = to_numpy((action.action)[0], check=True)
        if self._pos_noise_std > 0:
            base_action = base_action.copy()
            base_action[:3] += self._pos_noise_std * np.random.randn(3)
        if self._ori_noise_std > 0:
            assert not self._no_ori, "Cannot add ori noise without ori action"
            base_action = base_action.copy()
            base_action[3:6] += self._ori_noise_std * np.random.randn(3)
        if self._no_ori:
            assert base_action.shape[-1] == 4, base_action.shape[-1]
            # fill in with zeros, to project up to dim = 7
            if self.rs_env.action_dim == 7:
                base_action = np.concatenate(
                    [base_action[..., :3], np.zeros_like(base_action[..., :3]), base_action[..., -1:]], axis=-1)
        self._action = base_action
        self._obs, self._reward, self._done, info = self._base_env.step(base_action)
        if self._env_name in override_reward_fns.keys():
            self._reward = override_reward_fns[self._env_name](self, self._obs, base_action, self._reward, self._done)
        # waits 2 steps after success to stop
        if self._done_on_success and (self.is_success() or self._stop_counter > 0):
            self._stop_counter += 1
            # stop after success and + N-1 additional steps
            if self._stop_counter >= 2:  # needs one step to register the reward I think
                self._done = True
        if self._render:
            self._base_env.render(mode="human", camera_name=self._onscreen_camera_name)
        return self.get_obs(), self.get_goal(), np.array([self._done])

    def unscale_action(self, action, idx=0):
        # converts raw scaled action to unscaled action (input)
        c = self.rs_env.robots[idx].controller
        omx, omn = c.output_max, c.output_min  # unscaled
        imx, imn = c.input_max, c.input_min

        zero_to_1 = (action[:len(omx)] - omn) / (omx - omn)
        return np.concatenate([np.clip(zero_to_1 * (imx - imn) + imn, imn, imx), action[len(omx):]])

    def get_control_range(self, idx=0):
        return self.rs_env.robots[idx].controller.output_min, self.rs_env.robots[idx].controller.output_max

    def reset(self, presets: d = None):
        if presets is None:
            presets = d()
        if self._enable_preset_sweep:
            presets = self.generate_preset() & presets

        logger.debug(f"Resetting {self._env_name}... [sweeped={self._enable_preset_sweep}]")
        self._obs = self._base_env.reset()  # TODO preset version

        # reset the mujoco objects if specified in presets.
        objects = presets << 'objects'
        if presets.has_leaf_key('object'):
            objects = get_ordered_objects_from_arr(self, presets.object)[0].objects

        if objects is not None:
            objs_pos, objs_quat = objects.get_keys_required(['position', 'orientation'])
            accepted_names = get_ordered_object_names(self)
            i = 0
            for obj in self.rs_env.model.mujoco_objects:
                if obj._name in accepted_names:
                    # mujoco uses (w,x,y,z) but we use (x,y,z,w)
                    self.rs_env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([objs_pos[..., i, :].reshape(3),
                                                                                       transform_utils.convert_quat(
                                                                                           objs_quat[..., i, :].reshape(
                                                                                               4), to="wxyz")]))
                    i += 1

        # FROM ROBOMIMIC: hack that is necessary for robosuite tasks for deterministic action playback
        if self._env_name == 'KitchenEnv':
            # some initial
            initial_mjstate = self.rs_env.sim.get_state().flatten()
            self.rs_env.sim.set_state_from_flattened(initial_mjstate)

            # # pre-steps
            for _ in range(5):
                self._obs, self._reward, self._done, info = self._base_env.step(np.array([0.] * 6 + [-1.]))

        else:
            self._obs = self._base_env.reset_to(self._base_env.get_state())
        self._reward = 0
        self._stop_counter = 0
        self._draw_actions = []
        obs = self.get_obs()

        # logger.debug(f"Object state: {obs.object}")
        return obs, self.get_goal()

    def _get_stage_rewards(self):
        if self._env_name == "ToolHang":
            return [float(self.rs_env._check_frame_assembled()), float(self.rs_env._check_tool_on_frame())]
        else:
            raise NotImplementedError(f"Staged rewards for {self._env_name}")

    def get_obs(self):
        # TODO imgs
        if self._imgs:
            self._obs['image'] = self._base_env.render(mode="rgb_array", height=self._H, width=self._W,
                                                       camera_name=self._offscreen_camera_name)
            # any drawing on the obs
            for fn in self._draw_actions:
                self._obs['image'] = fn(self, self._obs['image'])
        if self._ego_imgs:
            self._obs['ego_image'] = self._base_env.render(mode="rgb_array", height=self._H, width=self._W,
                                                           camera_name=self._offscreen_ego_camera_name)
        self._obs['reward'] = np.array([self._reward])
        # more reward keys, based on env.
        if self._use_reward_stages:
            rews = np.asarray(self._get_stage_rewards())
            for i in range(len(rews)):
                self._obs[f'reward_{i}'] = rews[i:i + 1]
        # print(self._obs['robot0_eef_vel_lin'])
        for i, r in enumerate(self.base_env.env.robots):
            self._obs[f'robot{i}_eef_vel_lin'] = r._hand_vel
            self._obs[f'robot{i}_eef_vel_ang'] = r._hand_ang_vel

        obs = d.from_dict(self._obs).leaf_apply(lambda arr: arr.copy())
        for k in obs.list_leaf_keys():  # do not iterate generator if mutating
            if "_eef_quat" in k:
                # copy ensures it is contiguous
                obs[k.replace('_eef_quat', '_eef_eul')] = transform_utils.quat2euler(obs[k]).copy()

        # parse the object fields for this environment ( bc robosuite flattens everything :/ )
        if self._parse_objects and 'object' in obs:
            obs.combine(get_ordered_objects_from_arr(self, obs['object'])[0])

        return obs.leaf_apply(lambda arr: arr[None])

    def is_success(self):
        if self._env_name in override_success_fns.keys():
            return override_success_fns[self._env_name](self, self._obs, self._action, self._reward, self._done)
        if hasattr(self._base_env, 'is_success'):
            return self._base_env.is_success()['task']
        if hasattr(self._base_env, '_check_success'):
            return self._base_env._check_success()
        else:
            raise NotImplementedError

    def get_goal(self):
        return d()

    @property
    def base_env(self):
        return self._base_env

    @property
    def rs_env(self):
        return self._base_env.env

    @property
    def name(self):
        return self._env_name

    @property
    def dt(self):
        return 1. / self._control_freq

    default_params = d(
        onscreen_camera_name='agentview',
        offscreen_camera_name='agentview',
        done_on_success=True,
    )

    @staticmethod
    def get_default_env_spec_params(params: d = None) -> d:
        name = params['env_name']
        img_height = params << "img_height"
        img_width = params << "img_height"
        imgs = get_with_default(params, "imgs", False)
        ego_imgs = get_with_default(params, "ego_imgs", False)

        no_ori = get_with_default(params, "no_ori", False)
        parse_objects = get_with_default(params, "parse_objects", False)

        if name == "NutAssemblySquare":
            n_obj = 1
            obdim = 14
            img_height = img_height or 84
            img_width = img_width or 84
        elif name == "ToolHang":
            n_obj = 3
            obdim = 44
            img_height = img_height or 240
            img_width = img_width or 240
        elif name == "PickPlaceCan":
            n_obj = 1
            obdim = 14
            img_height = img_height or 84
            img_width = img_width or 84
        elif name == "KitchenEnv":
            n_obj = 4
            obdim = 70  # not used
            img_height = img_height or 128
            img_width = img_width or 128
        else:
            raise NotImplementedError

        prms = d(
            cls=ParamEnvSpec,
            names_shapes_limits_dtypes=[
                ("image", (img_height, img_width, 3), (0, 255), np.uint8),
                ("ego_image", (img_height, img_width, 3), (0, 255), np.uint8),

                ("object", (obdim,), (-np.inf, np.inf), np.float32),
                ("objects/position", (n_obj, 3), (-np.inf, np.inf), np.float32),
                ("objects/orientation", (n_obj, 4), (-np.inf, np.inf), np.float32),
                ("objects/orientation_eul", (n_obj, 3), (-np.inf, np.inf), np.float32),
                ("robot0_eef_pos", (3,), (-np.inf, np.inf), np.float32),
                ("robot0_eef_eul", (3,), (-np.pi, np.pi), np.float32),
                ("robot0_eef_quat", (4,), (-1., 1.), np.float32),
                ("robot0_eef_vel_ang", (3,), (-np.inf, np.inf), np.float32),
                ("robot0_eef_vel_lin", (3,), (-np.inf, np.inf), np.float32),
                ("robot0_gripper_qpos", (2,), (-np.inf, np.inf), np.float32),
                ("robot0_gripper_qvel", (2,), (-np.inf, np.inf), np.float32),
                ("robot0_joint_pos", (7,), (-np.inf, np.inf), np.float32),
                ("robot0_joint_pos_cos", (7,), (-np.inf, np.inf), np.float32),
                ("robot0_joint_pos_sin", (7,), (-np.inf, np.inf), np.float32),
                ("robot0_joint_vel", (7,), (-np.inf, np.inf), np.float32),

                ('action', (4 if no_ori else 7,), (-1, 1.), np.float32),
                ('reward', (1,), (-np.inf, np.inf), np.float32),

                ("click_state", (1,), (0, 255), np.uint8),
                ("mode", (1,), (0, 255), np.uint8),
                ("real", (1,), (False, True), bool),

                ("policy_type", (1,), (0, 255), np.uint8),
                ("policy_name", (1,), (0, 1), object),
                ("policy_switch", (1,), (False, True), bool),  # marks the beginning of a policy

                # target
                ('target/position', (3,), (-np.inf, np.inf), np.float32),
                ('target/orientation', (4,), (-1, 1.), np.float32),
                ('target/orientation_eul', (3,), (-np.pi, np.pi), np.float32),
                ('target/gripper', (1,), (-np.pi, np.pi), np.float32),

                # delta wp
                ('delta_waypoint', (6,), (-1., 1.), np.float32),

                # raw actions
                ('raw/action', (4 if no_ori else 7,), (-1, 1.), np.float32),
                ('raw/target/position', (3,), (-np.inf, np.inf), np.float32),
                ('raw/target/orientation', (4,), (-1, 1.), np.float32),
                ('raw/target/orientation_eul', (3,), (-np.pi, np.pi), np.float32),
                ('raw/target/gripper', (1,), (-np.pi, np.pi), np.float32),

            ], observation_names=["object", "robot0_eef_pos", "robot0_eef_eul", "robot0_eef_quat", "robot0_eef_vel_ang",
                                  "robot0_eef_vel_lin", "robot0_gripper_qpos", "robot0_gripper_qvel",
                                  "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                                  "robot0_joint_vel",
                                  ],
            param_names=[],
            final_names=[],
            action_names=["action", "policy_type", "policy_name", "policy_switch"],
            output_observation_names=["reward"]
        )

        if name == 'KitchenEnv':
            prms.observation_names = ["object", "robot0_eef_pos", "robot0_eef_eul", "robot0_eef_quat",
                                      "robot0_gripper_qpos", "robot0_joint_pos"]
            prms.output_observation_names = []

        if parse_objects:
            # parse flat object into useful keys (environment dependent)
            prms.observation_names.remove('object')
            prms.observation_names.extend(['objects/position', 'objects/orientation', 'objects/orientation_eul'])

        if no_ori:
            if 'robot0_eef_quat' in prms.observation_names:
                prms.observation_names.remove('robot0_eef_quat')
            if 'robot0_eef_eul' in prms.observation_names:
                prms.observation_names.remove('robot0_eef_eul')

        if imgs:
            prms.observation_names.append('image')

        if ego_imgs:
            prms.observation_names.append('ego_image')

        return prms


override_reward_fns = {
    # 'KitchenEnv': buds_kitchen_reward
}

override_success_fns = {
    # 'KitchenEnv': buds_kitchen_success
}


def get_ordered_objects_from_arr(env, obj):
    # obj is an array
    if env.name == "NutAssemblySquare":
        assert obj.shape[-1] == 14
        nut_pos = obj[..., :3]
        nut_quat = obj[..., 3:7]
        nut_eul = Rotation.from_quat(nut_quat).as_euler("xyz")
        num_objects = 1
        od = d(objects=d(position=nut_pos[..., None, :],
                         orientation=nut_quat[..., None, :],
                         orientation_eul=nut_eul[..., None, :]))
    elif env.name == "ToolHang":
        assert obj.shape[-1] == 44
        poss, quats, euls = [], [], []
        for i, on in enumerate(['base', 'frame', 'tool']):
            obj_i = obj[..., i * 14:(i + 1) * 14]
            pos, quat = obj_i[..., :3], obj_i[..., 3:7]
            eul = Rotation.from_quat(quat).as_euler("xyz")
            poss.append(pos)
            quats.append(quat)
            euls.append(eul)
        num_objects = 3
        od = d(objects=d(position=np.stack(poss, axis=-2),
                         orientation=np.stack(quats, axis=-2),
                         orientation_eul=np.stack(euls, axis=-2), ))
    else:
        raise NotImplementedError

    return od, num_objects


def get_ordered_object_names(env):
    # obj is an array
    if env.name == "NutAssemblySquare":
        return ['SquareNut']
    elif env.name == "ToolHang":
        return ['stand', 'frame', 'tool']
    elif env.name == "KitchenEnv":
        return ['cube_bread', 'PotObject']
    else:
        raise NotImplementedError


def get_object_presets_sweep(env, sweep_pos=6, sweep_orn=8):
    # obj is an array
    if env.name == "NutAssemblySquare":
        x = np.average([-0.115, -0.11])
        yvec = np.linspace(0.11, 0.225, num=sweep_pos)
        rot_z_vec = np.linspace(0., 2 * np.pi, num=sweep_orn, endpoint=False)

        base_pos = env.rs_env.table_offset - env.rs_env.nuts[0].bottom_offset + np.array([x, 0., 0.02])
        ally, allrz = np.meshgrid(yvec, rot_z_vec)
        presets = []
        for (y, rz) in zip(ally.reshape(-1), allrz.reshape(-1)):
            pos = base_pos + np.array([0., y, 0.])
            # TODO this orientation order seems wrong but works... look into this.
            orientation = np.array([0, 0, np.sin(rz / 2), np.cos(rz / 2)])  # quat (x,y,z,w)
            presets.append(d(objects=d(position=pos[None], orientation=orientation[None])))

    elif env.name == "ToolHang":
        stand_base = np.array([-env.rs_env.table_full_size[0] * 0.1, 0., 0.001])
        frame_base = np.array([-env.rs_env.table_full_size[0] * 0.05, -env.rs_env.table_full_size[1] * 0.3,
                               (env.rs_env.frame_args["frame_thickness"] - env.rs_env.frame_args[
                                   "frame_height"]) / 2. + 0.001 +
                               (env.rs_env.stand_args["base_thickness"] / 2.) + (
                                   env.rs_env.frame_args["grip_size"][1])])
        tool_base = np.array([env.rs_env.table_full_size[0] * 0.05, -env.rs_env.table_full_size[1] * 0.25, 0.001])

        if ("tip_size" in env.rs_env.frame_args) and (env.rs_env.frame_args["tip_size"] is not None):
            frame_base[2] -= (env.rs_env.frame_args["tip_size"][0] + 2. * env.rs_env.frame_args["tip_size"][3])

        frame_x = 0.
        frame_yvec = np.linspace(-0.02, 0.02, num=sweep_pos)
        tool_x = 0.
        tool_yvec = np.linspace(-0.02, 0.02, num=sweep_pos)
        frame_ori = np.linspace(-np.pi / 18, np.pi / 18, num=sweep_orn)
        tool_ori = np.linspace(-np.pi / 18, np.pi / 18, num=sweep_orn)
        all_fy, all_ty, all_for, all_tor = np.meshgrid(frame_yvec, tool_yvec, frame_ori, tool_ori)

        presets = []
        for (fy, ty, fry, trz) in zip(all_fy.reshape(-1), all_ty.reshape(-1), all_for.reshape(-1), all_tor.reshape(-1)):
            spos = stand_base + np.array([0., 0., -env.rs_env.stand.bottom_offset[-1]]) + env.rs_env.table_offset
            fpos = frame_base + np.array([frame_x, fy, -env.rs_env.frame.bottom_offset[-1]]) + env.rs_env.table_offset
            tpos = tool_base + np.array([tool_x, ty, -env.rs_env.tool.bottom_offset[-1]]) + env.rs_env.table_offset

            fry = (-np.pi / 2) + (np.pi / 6) + fry
            trz = (-np.pi / 2) - (np.pi / 9.) + trz

            sori = env.rs_env.stand.init_quat  # fixed.
            fori = np.array([0, np.sin(fry / 2), 0, np.cos(fry / 2)])  # y axis
            tori = np.array([0, 0, np.sin(trz / 2), np.cos(trz / 2)])  # z axis

            # base orientation adjustment
            fori = quat_multiply(env.rs_env.frame.init_quat, fori)

            presets.append(d(objects=d(position=np.stack([spos, fpos, tpos]),
                                       orientation=np.stack([sori, fori, tori]))))
    else:
        raise NotImplementedError

    return presets


if __name__ == '__main__':

    params = d(
        env_name='NutAssemblySquare',
        # env_name='ToolHang',
        # env_name='KitchenEnv',
        render=True,
        enable_preset_sweep=False,
    )

    # # square example (2 * 8)
    # params.preset_sweep_pos = 2
    # params.preset_sweep_ori = 8

    # tool hang example (2^2 * 2^2)
    params.preset_sweep_pos = 2
    params.preset_sweep_ori = 2

    env = make(RobosuiteEnv, params)

    for step in range(20):
        logger.debug("Resetting...")
        env.reset()

        logger.debug("Stepping...")

        ac = np.zeros((1, 7))

        for i in range(100):
            action = d(action=ac + np.array([0., 0, np.sin(i * 2 * np.pi / 50), 0., 0., 0., 0.]))
            obs, goal, done = env.step(action)
            # print(obs.robot0_eef_pos)

    logger.debug("Done.")
