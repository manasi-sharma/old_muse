import os
import hid
import numpy as np

from configs.helpers import get_script_parser, load_base_config
from muse.datasets.np_dataset import NpDataset
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c, is_next_cycle
from muse.utils.torch_utils import reduce_map_fn

from attrdict import AttrDict as d

from scripts.collect import rollout, parse_history, save


from muse.envs.polymetis.polymetis_panda_env import PolymetisPandaEnv



class KinestheticTeachingInterface:
    def __init__(
        self,
        env: PolymetisPandaEnv,
        max_time_per_demo: int = 30
    ) -> None:
        """
        Initialize a Kinesthetic Teaching Interface, with the requisite parameters.

        :param env: Base "PolymetisPandaEnv robot environment wrapper (for controlling the robot/setting gains).
       .
        :param max_time_per_demo: Max time (in seconds) to record demo -- default = 21 seconds.
      
        """
        self.max_time_per_demo = max_time_per_demo
        self.env = env
        self.demo_index = 0

        # for device in hid.enumerate():
        #     print(f"0x{device['vendor_id']:04x}:0x{device['product_id']:04x} {device['product_string']}")
        # foot pedal 0x1a86:0xe026 
        self.gamepad = hid.device()
        self.gamepad.open(0x1a86, 0xe026)
        self.gamepad.set_nonblocking(True)
        self.open_gripper = True

    def reset(self):
        self.open_gripper = True

    def record_single_demo_traj(self):

        logger.debug(f"[*] Starting to Record Demonstration `{self.demo_index}`...")
        
        # Set "kinesthetic recording", reset environment, and wait on user input...
        self.env.set_kinesthetic_and_reset(do_kinesthetic=True)
        user_input = input(
            f"Ready to record!\n\tYou have `{self.max_time_per_demo}` secs to complete the demo, and can use"
            " `Ctrl-C` to stop anytime.\n\tPress (r) to reset, and any other key to start..."
        )

        # Reset...
        if user_input == "r":
            return None

        # Go, go, go --> for `kinesthetic` (raw) recording mode, we only care about ee positions & gripper widths
        logger.debug("\t=>> Started recording...")
        observations = []
        try:
            for _ in range(int(self.max_time_per_demo * self.env.hz) - 1):
                # Use foot pedal to toggle gripper
                step_action = np.zeros((1,7))  # Last digit is for gripper, 0: open, 1:closed
                report = self.gamepad.read(64)
                if report and report[3] > 0:
                    self.open_gripper = not self.open_gripper
                step_action[0][-1] = 0 if self.open_gripper else 1

                obs, _, _ = self.env.step(d(action=step_action))
                
                obs["policy_type"] = [300]
                obs["policy_name"] = ["kinesthetic_teaching"]
                obs["policy_switch"] =  [False]
                obs["click_state"] =  [False]
                obs["gripper_open"] = [0 if self.open_gripper else 1]
                observations.append(obs)
        except (KeyboardInterrupt, SystemExit):
            logger.debug("\t=>> Caught Keyboard Interrupt, stopping recording...")

        # Close environment (kill errant controllers)...
        self.env.close()
        self.env.set_kinesthetic_and_reset(do_kinesthetic=False)

        # Write full dictionary to file...
        playback_dict = {k: [] for k in ["ee_pose", "q", "gripper_width", "gripper_open", "policy_type", "policy_name", "policy_switch", "click_state"]}

        for obs in observations:
            for k in playback_dict:
                playback_dict[k].append(obs[k]) 
        
        playback_dict = {k: np.array(v) for k, v in playback_dict.items()}

        gripper_widths = playback_dict['gripper_width']
        
        gripper_open_pedal = 1 - playback_dict['gripper_open']
        
        playback_dict["action"] = gripper_open_pedal

        playback_dict['ee_position'] = np.squeeze(playback_dict['ee_pose'][:,:,:3], axis = 1)
        playback_dict['ee_orientation'] = np.squeeze(playback_dict['ee_pose'][:,:,3:], axis = 1)
       
        self.demo_index += 1
        return playback_dict




if __name__ == '__main__':
    # things we can use from command line
    parser = get_script_parser()
    parser.add_argument('config', type=str)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_eps', type=int, default=None)
    parser.add_argument('--capacity', type=int, default=1e6,
                        help='if max_steps is not None, will use whatever is smaller (2*max_steps) or this')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--save_every_n_episodes', type=int, default=0, help='Set to nonzero to save')
    parser.add_argument('--save_chunks', action='store_true',
                        help='Save each group of {save_every_n_episodes} in their own file')
    parser.add_argument('--save_start_ep', type=int, default=0,
                        help='If saving chunks, this is the start offset for saving '
                             '(e.g., if you resumed after stopping)')
    parser.add_argument('--dataset_save_group', type=str, default=None)
    parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--random_policy', action="store_true")
    parser.add_argument('--print_last_obs', action="store_true")
    parser.add_argument('--print_policy_name', action="store_true")
    parser.add_argument('--track_returns', action="store_true")
    parser.add_argument('--reduce_returns', type=str, default='sum', choices=list(reduce_map_fn.keys()),
                        help='If tracking returns, will apply this func to the returns before tracking..')
    args, unknown = parser.parse_known_args()
         

    # determine when to stop collecting (either episode or step based)
    assert (args.max_steps is None) ^ (args.max_eps is None), "Specify either max_steps or max_eps!"
    if args.max_steps:
        terminate_fn = lambda steps, eps: steps >= args.max_steps
        capacity = min(args.max_steps * 2, args.capacity)
    else:
        terminate_fn = lambda steps, eps: eps >= args.max_eps
        capacity = args.capacity

    # load the config
    params, root = load_base_config(args.config, unknown)
    exp_name = root.get_exp_name()

    exit_on_ctrl_c()
    file_manager = ExperimentFileManager(exp_name, is_continue=True)

    if args.model_file is not None:
        model_fname = args.model_file
        model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
        assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
        logger.debug("Using model: {}".format(model_fname))
    else:
        model_fname = os.path.join(file_manager.models_dir, "model.pt")
        logger.debug("Using default model for current eval: {}".format(model_fname))

    # generate env
    env_spec = params.env_spec.cls(params.env_spec)
    env = params.env_train.cls(params.env_train, env_spec)

    # generate model and policy
    model = params.model.cls(params.model, env_spec, None)

    # kinesthetic policy (NOT Implemented)
    # policy_kinesthetic = params.policy_kinesthetic.cls(params.policy_kinesthetic, env_spec, env=env)
    
    kt_interface = KinestheticTeachingInterface(env)

    # generate dataset
    if args.save_every_n_episodes > 0:
        logger.debug(f"Saving to file -> {args.save_file} every {args.save_every_n_episodes} eps.")
    if args.dataset_save_group is None:
        ds = d(cls=NpDataset, batch_size=10, horizon=10, capacity=capacity, output_file=args.save_file)
    else:
        ds = params[args.dataset_save_group]
        ds.file = None
        ds.output_file = args.save_file

    # instantiate the dataset
    dataset_save = ds.cls(ds, env_spec, file_manager)

    # restore model from file (if provided)
    if not args.no_model_file:
        model.restore_from_file(model_fname)

    model.eval()

    logger.debug("Beginning Evaluation.")

    step = 0
    ep = 0
    last_save = None
    first_to_save_ep = 0
    while not terminate_fn(step, ep):
        logger.info(f"[{step}] Rolling out episode {ep}...")
        
        kt_interface.reset()
        # record demo with kt_interface
        playback_dict = None
        while not playback_dict:
            playback_dict = kt_interface.record_single_demo_traj()
        # replay policy and pass data to replay policy (TODO)
        params.policy.demo_file = playback_dict
        policy = params.policy.cls(params.policy, env_spec, env=env)

        # obs, goal = env.reset()
        # policy.reset_policy(next_obs=obs, next_goal=goal)

        # kin_obs_history, kin_goal_history, kin_ac_history = rollout(args, policy, env, model, obs, goal)
        # kin_inputs, kin_outputs = parse_history(env_spec, kin_obs_history, kin_goal_history, kin_ac_history)

        obs, goal = env.reset()
        # TODO add demo file as as input to reset policy here.
        policy.reset_policy(next_obs=obs, next_goal=goal)

        obs_history, goal_history, ac_history = rollout(args, policy, env, model, obs, goal)

        step += len(obs_history) - 1
        ep += 1

        inputs, outputs = parse_history(env_spec, obs_history, goal_history, ac_history)

        # actually add to dataset
        dataset_save.add_episode(inputs, outputs)

        if is_next_cycle(ep, args.save_every_n_episodes):
            logger.warn(f"[{step}] Saving data after {ep} episodes, data len = {len(dataset_save)}")
            # save to dataset (optionally in chunks)
            first_to_save_ep = save(dataset_save, start_offset=args.save_start_ep,
                                    first_to_save_ep=first_to_save_ep, save_chunks=args.save_chunks)
            last_save = step

    logger.info(f"[{step}] Terminating after {ep} episodes. Final data len = {len(dataset_save)}")

    # save one last time if there's new stuff and we are supposed to save
    if not last_save != step and step > 0 and args.save_every_n_episodes > 0:
        logger.warn("Saving final data...")
        save(dataset_save, start_offset=args.save_start_ep,
             first_to_save_ep=first_to_save_ep, save_chunks=args.save_chunks)

    logger.info("Done.")
