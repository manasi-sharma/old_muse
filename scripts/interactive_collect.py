"""
Very similar to collect.py, but uses the interactive resetting functionality.
"""
import os
import sys

from attrdict import AttrDict as d

from collect import rollout, parse_history, save
from configs.helpers import get_script_parser, load_base_config
from muse.datasets.np_dataset import NpDataset
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c, timeit
from muse.utils.input_utils import KeyInput as KI, wait_for_keydown_from_set
from muse.utils.pygame_utils import TextFillPygameDisplay, PygameOnlyKeysInput
from muse.utils.torch_utils import reduce_map_fn

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
    parser.add_argument('--save_episodic', action='store_true',
                        help='Save each episode in its own file')
    parser.add_argument('--save_start_ep', type=int, default=0,
                        help='If saving episodic, this is the start offset for saving '
                             '(e.g., if you resumed after stopping)')
    parser.add_argument('--dataset_save_group', type=str, default=None)
    parser.add_argument('--no_model_file', action="store_true")
    parser.add_argument('--random_policy', action="store_true")
    parser.add_argument('--print_last_obs', action="store_true")
    parser.add_argument('--print_policy_name', action="store_true")
    parser.add_argument('--track_returns', action="store_true")
    parser.add_argument('--reduce_returns', type=str, default='sum', choices=list(reduce_map_fn.keys()),
                        help='If tracking returns, will apply this func to the returns before tracking..')
    parser.add_argument('--use_env_display', action="store_true",
                        help="If true, will use display defined in env (may not be process compatible)")

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

    # generate model
    model = params.model.cls(params.model, env_spec, None)

    if args.dataset_save_group is None:
        ds = d(cls=NpDataset, batch_size=10, horizon=10, capacity=capacity, output_file=args.save_file)
    else:
        ds = params[args.dataset_save_group]
        ds.file = None
        ds.output_file = args.save_file

    # instantiate the
    dataset_save = ds.cls(ds, env_spec, file_manager)

    # restore model from file (if provided)
    if not args.no_model_file:
        model.restore_from_file(model_fname)

    model.eval()

    # Generate input handle

    # pygame display details
    if args.use_env_display:
        # environment owns display
        display = env.display
    else:
        # script owns display
        display = TextFillPygameDisplay(d())

    assert display is not None, "Env (%s) doesn't have a display set up" % type(env)

    # handling user input (uses pygame)
    empty_handler = lambda ui, ki: None
    input_handle = PygameOnlyKeysInput(d(display=display), {})

    # global running variable, to respond to "quit" message
    running = True
    def quit(*local_args):
        global running
        running = False
        sys.exit(0)

    # reset action
    def reset(*local_args):
        global running
        running = False

    def early_terminate_fn(*args):
        input_handle.run(rate_limited=False, once=True)
        return not running

    input_handle.register_callback(KI("r", KI.ON.down), reset)
    input_handle.register_callback(KI("y", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("n", KI.ON.down), empty_handler)
    input_handle.register_callback(KI("q", KI.ON.down), quit)

    # generate policy
    policy = params.policy.cls(params.policy, env_spec, env=env, input_handle=input_handle)

    # generate dataset
    logger.debug(f"Will save to file -> {args.save_file}")


    def populate_display(*args, **kwargs):
        if hasattr(env, "populate_display_fn"):
            env.populate_display_fn(*args, **kwargs)
        else:
            display.populate_display(*args, **kwargs)


    logger.debug("Beginning Interactive Evaluation.")
    populate_display("Beginning...")

    do_save = False

    def extra_reset_fn(**kwargs):
        # queries the user to save or trash during the reset (after the first one).
        global do_save

        logger.info("UI: Save [y] or Trash [n]")
        populate_display('UI: Save [y] or Trash [n]')
        res = wait_for_keydown_from_set(input_handle, [KI('y', KI.ON.down), KI('n', KI.ON.down)],
                                        do_async=False)
        do_save = res.key == 'y'

    step = 0
    ep = 0

    # reset once to start (UI)
    obs, goal = env.user_input_reset(input_handle)
    policy.reset_policy(next_obs=obs, next_goal=goal)

    while not terminate_fn(step, ep):
        running = True

        # clear the inputs
        input_handle.run(once=True)

        logger.info(f"[{step}] Running episode {ep}...")
        populate_display("'r' to reset and 'q' to quit")
        obs_history, goal_history, ac_history, _ = rollout(args, policy, env, model, obs, goal,
                                                        early_terminate_fn=early_terminate_fn)

        # UI reset fn with the extra reset function that will ask if we should save or not
        obs, goal = env.user_input_reset(input_handle, reset_action_fn=extra_reset_fn)
        policy.reset_policy(next_obs=obs, next_goal=goal)

        if do_save:
            # parse history if we are saving
            inputs, outputs = parse_history(env_spec, obs_history, goal_history, ac_history)

            # actually increment and add to dataset only if we are saving
            step += len(obs_history) - 1
            ep += 1
            dataset_save.add_episode(inputs, outputs)

            logger.warn(f"[{step}] Saving data after {ep} episodes, data len = {len(dataset_save)}")
            # save either whole dataset or each individually
            save(dataset_save, start_offset=args.save_start_ep,
                 first_to_save_ep=ep - 1, save_chunks=args.save_episodic)

    logger.info(f"[{step}] Done after {ep} episodes. Final data len = {len(dataset_save)}")
