"""
Evaluate a policy and model in an environment.
Optionally save to a dataset (if --save_every_n_episodes > 0)
"""

import os

import numpy as np
import torch
from attrdict import AttrDict as d

from configs.helpers import get_script_parser, load_base_config
from muse.datasets.np_dataset import NpDataset
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c, is_next_cycle
from muse.utils.torch_utils import reduce_map_fn, to_numpy


def rollout(local_args, local_policy, local_env, local_model, local_obs, local_goal, early_terminate_fn=None):
    """ Rollout environment for one episode (or earlier if max_steps

    Parameters
    ----------
    local_args:
    local_policy: Policy
    local_env: Env
    local_model: Model
    early_terminate_fn: Optional[Callable]
        returns a bool


    Returns
    -------
    obs_history: List[AttrDict], length H+1
    goal_history: List[AttrDict], length H+1
    ac_history: List[AttrDict], length H

    """
    done = [False]
    reward_reduce = reduce_map_fn[local_args.reduce_returns]
    returns = 0.
    i = 0

    local_obs_history = []
    local_goal_history = []
    local_ac_history = []

    while not done[0] and (not early_terminate_fn or not early_terminate_fn(local_env, local_obs, local_goal)):
        # empty axes for (batch_size, horizon)
        expanded_obs = local_obs.leaf_apply(lambda arr: arr[:, None])
        expanded_goal = local_goal.leaf_apply(lambda arr: arr[:, None])
        with torch.no_grad():
            # query the model for the action
            if local_args.random_policy:
                action = local_policy.get_random_action(local_model, expanded_obs, expanded_goal)
            else:
                action = local_policy.get_action(local_model, expanded_obs, expanded_goal)

            if i == 0 and local_args.print_policy_name and action.has_leaf_key("policy_name"):
                logger.info(f"Policy: {action.policy_name.item()}")

        # only keep arrays -> to numpy
        local_obs_history.append(local_obs.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))
        local_goal_history.append(local_goal.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))
        local_ac_history.append(action.leaf_arrays().leaf_apply(lambda arr: to_numpy(arr, check=True)))

        # step the environment with the policy action
        local_obs, local_goal, done = local_env.step(action)
        i += 1

        if local_policy.is_terminated(local_model, local_obs, local_goal):
            done[0] = True

        if local_args.track_returns:
            rew = reward_reduce(local_obs.reward).item()
            returns += rew

    # append the last observation/goal
    local_obs_history.append(local_obs)
    local_goal_history.append(local_goal)

    if local_args.track_returns:
        logger.info(f'Returns: {returns}')

    return local_obs_history, local_goal_history, local_ac_history


def parse_history(spec, local_obs_history, local_goal_history, local_ac_history):
    """

    Parameters
    ----------
    spec
    local_obs_history: List[AttrDict], length H + 1
    local_goal_history: List[AttrDict], length H + 1
    local_ac_history: List[AttrDict], length H

    Returns
    -------
    local_inputs: AttrDict
    local_outputs: AttrDict

    """

    # de-prefix the names that we will get from output
    raw_out_obs_names = [(oo[5:] if oo.startswith('next/') else oo) for oo in spec.output_observation_names]
    raw_out_goal_names = [(og[5:] if og.startswith('next/') else og) for og in spec.output_goal_names]

    # concatenate into one big episode
    obs = d.leaf_combine_and_apply([(o > spec.observation_names) for o in local_obs_history[:-1]], np.concatenate)
    goals = d.leaf_combine_and_apply([(g > spec.goal_names) for g in local_goal_history[:-1]], np.concatenate)
    acs = d.leaf_combine_and_apply([(a > spec.action_names) for a in local_ac_history], np.concatenate)
    out_obs = d.leaf_combine_and_apply([(o > raw_out_obs_names) for o in local_obs_history[1:]], np.concatenate)
    out_goals = d.leaf_combine_and_apply([(g > raw_out_goal_names) for g in local_goal_history[1:]], np.concatenate)

    # inputs are just the obs / goals / actions
    local_inputs = obs & goals & acs
    # parse outputs from next obs / next goal
    local_outputs = d()
    raw_outputs = out_obs & out_goals
    for k, mapped_k in zip(raw_out_obs_names + raw_out_obs_names,
                           spec.output_observation_names + spec.output_goal_names):
        local_outputs[mapped_k] = raw_outputs[k]

    # extra keys for outputs (done and rollout_timestep)
    local_outputs.done = np.zeros(len(local_ac_history), dtype=bool)
    local_outputs.done[-1] = True
    local_outputs.rollout_timestep = np.arange(len(local_ac_history))

    return local_inputs, local_outputs


def save(dataset, start_offset=0, first_to_save_ep=None, save_chunks=False):
    """
    Saves to a dataset, optionally in chunks, to the dataset's output file

    If saving in chunks, will add a suffix to the output file name with the current chunk range.
    For example if start_offset=4, first_to_save_ep=2, and the dataset has 4 episodes, and output_file=dataset.npz,
     this will save to:
        dataset_ep6-7.npz

    Parameters
    ----------
    dataset
    start_offset: int

    latest: June 2022: FILE_ID = 1TR0ph1uUmtWFYJr8B0yo1hASjj0cP8fc

Create several additional directories in the root:

    data/: This will store all the data
    experiments/: This is where runs will log and write models to.
    plots/: (optional)
    videos/: (optional)

sode to start from (e.g. num episodes saved before starting this script)
    first_to_save_ep: int
        if saving chunks, which episode to start saving in the dataset enumeration
    save_chunks: bool
        if True, will save from [ first_to_save_ep, dataset.get_num_episodes() )
            using a suffix to the dataset file for this range (e.g. for chunk 0 -> 2, suffix="_ep0-2")
        else will save all episodes.

    Returns
    -------
    episodes: int
        The next "first_to_save_ep" value, where to start saving from next time.

    """
    episodes = dataset.get_num_episodes()

    if save_chunks:
        assert first_to_save_ep is not None, "Must pass in value for first_to_save_ep to save()!"
        if first_to_save_ep == episodes - 1:
            # single episode
            suffix = f'_ep{start_offset + first_to_save_ep}'
        elif first_to_save_ep < episodes - 1:
            suffix = f'_ep{start_offset + first_to_save_ep}-{start_offset + episodes - 1}'
        else:
            raise NotImplementedError('first_save should always be less than episodes.')
        # [inclusive, exclusive)
        dataset.save(suffix=suffix, ep_range=(first_to_save_ep, episodes))
        return episodes
    else:
        dataset.save()
        return episodes


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
    if params.model is not None:
        model = params.model.cls(params.model, env_spec, None)
        model.eval()
    else:
        #create dummy model
        model = torch.empty([])
    policy = params.policy.cls(params.policy, env_spec, env=env)

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

        obs, goal = env.reset()
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
