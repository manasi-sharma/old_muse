"""
Evals multi-task play environment with a scripted policy and a learned policy.
"""
import os
from pydoc import locate

import numpy as np

import torch
from attrdict import AttrDict

from configs.helpers import load_base_config, get_script_parser
from muse.experiments import logger

from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import timeit, exit_on_ctrl_c
from muse.utils.torch_utils import dc_add_horizon_dim, to_numpy

exit_on_ctrl_c()

parser = get_script_parser()
parser.add_argument('config', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--success_metrics', type=str, required=True,
                    help="a dictionary to import (with inspect) that maps policy_name -> success metric, "
                         "for example, muse.envs.pymunk.success_functions.pname_to_success")
parser.add_argument('--num_eps', type=int, required=True)
parser.add_argument('--num_proc', type=int, default=1)
parser.add_argument('--no_model_file', action="store_true")
parser.add_argument('--random_policy', action="store_true")
parser.add_argument('--print_policy_name', action="store_true")
args, unknown = parser.parse_known_args()


# load the config
params, root = load_base_config(args.config, unknown)
exp_name = root.get_exp_name()

file_manager = ExperimentFileManager(exp_name, is_continue=True)

if args.model_file is not None:
    model_fname = args.model_file
    model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current eval: {}".format(model_fname))

# create all the global modules
env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)

# lookup from policy_name --> success_metric_fn (batch-supported)
success_metric_fns: dict = dict(locate(args.success_metrics))

# restore model
if not args.no_model_file:
    model.restore_from_file(model_fname)


def eval_rollout(local_args, prefix, env, obs, goal, policy, policy_hardcoded, stack_goal=True):
    """
    Rolls out hardcoded policy to get the "goal", then runs policy to reach that.
    Parameters
    ----------
    local_args
    prefix
    env
    obs
    goal
    policy
    policy_hardcoded
    stack_goal

    Returns
    -------

    """
    obs_names = obs.list_leaf_keys()
    goal_obs, goal_goal = obs, goal
    ptype, raw_name, name = -1, None, None
    # stop counter for policy rollout
    counter = 0

    # helpers for detecting & stopping on success
    def prep_for_success_fn(obs, goal):
        return (obs, np.array([0]), goal)

    def stop_condition(obs, goal, name):
        nonlocal counter
        tobs, didx, gl = prep_for_success_fn(obs, goal)
        if raw_name is None:
            this_success = False
        else:
            succ_out = success_metric_fns[raw_name](tobs, didx, gl)
            # success is range [0.5, inf]
            this_success = succ_out['best'] >= 0.5
        if this_success or counter > 0:
            counter += 1

        if counter > 5:
            counter = 0
            return True

        return False

    # logger.debug(f'pos before: {obs.objects.position} {obs.cabinet.joint_position} {obs.drawer.joint_position}')

    # Run HARDCODED POLICY to GET GOAL
    for j in range(1000):
        with timeit("hardcoded/policy"):
            act = policy_hardcoded.get_action(model, dc_add_horizon_dim(goal_obs), dc_add_horizon_dim(goal_goal))

        if j == 0:
            if policy_hardcoded.curr_policy_idx == -1:
                logger.warn(f"{prefix}Policy type is -1! skipping this rollout")
                break

            ptype = policy_hardcoded.curr_policy_idx

            raw_name = policy_hardcoded.curr_policy.curr_name
            name = raw_name.replace("_", " ").title()

        with timeit("hardcoded/env_step"):
            goal_obs, goal_goal, done = env.step(act)

        # finish early
        if done[0] or policy_hardcoded.is_terminated(model, goal_obs, goal_goal):
            break

    # if terminated early, skip rollout
    if j == 0:
        return None, None, None, None

    # Run POLICY
    sequence = []

    # start from root start, make sure this reset() is properly implemented!
    presets = obs.leaf_apply(lambda arr: arr[0])
    obs, goal = env.reset(presets)

    # logger.debug(f'new pos: {obs.objects.position} {obs.cabinet.joint_position} {obs.drawer.joint_position}')

    # run eval for 50 extra steps than hardcoded.
    run_time = j + 50

    for j in range(run_time):
        if stack_goal:
            # stack B x 2 x ...
            policy_obs = obs & AttrDict.leaf_combine_and_apply([obs, goal_obs], lambda vs: np.stack(vs, axis=1))
            policy_goal = AttrDict()
        else:
            # pass in as separate obs of B x 1 x ...
            policy_obs = dc_add_horizon_dim(obs)
            policy_goal = dc_add_horizon_dim(goal_obs)

        with timeit("prior/policy"):
            with torch.no_grad():
                # ignore the action, sample the plan
                action = policy.get_action(model, policy_obs, policy_goal, sample_first=False)
                action.leaf_modify(lambda arr: to_numpy(arr, check=True))

        # starts from first obs & action
        sequence.append(obs & (action < env_spec.action_names))

        with timeit("prior/env_step"):
            obs, goal, done = env.step(action)

        # optional stopping condition.
        if j > 1 and stop_condition(obs, goal_obs, raw_name):
            logger.debug(f"{prefix}Early stopping at i = {j}")
            break

    # stack arrays
    full_seq = AttrDict.leaf_combine_and_apply(sequence, lambda vs: np.stack(vs, axis=1))

    # compute success across all steps
    tobs, didx, gl = prep_for_success_fn((full_seq > obs_names).leaf_apply(lambda arr: arr[0]), goal_obs)
    success_dc = success_metric_fns[raw_name](tobs, didx, gl)

    # if local_args.timeit:
    #     print(timeit)
    #     timeit.reset()

    return full_seq, name, ptype, success_dc


def eval_process(inps):
    """
    Runs a single eval thread, recording only the success.

    Parameters
    ----------
    inps

    Returns
    -------

    """
    proc_id, num_eps = inps
    prefix = f"[{proc_id}]: "
    env = params.env_train.cls(params.env_train.params, env_spec)
    policy = params.policy.cls(params.policy.params, env_spec, env=env)
    policy_hardcoded = params.policy_hardcoded.cls(params.policy_hardcoded.params, env_spec, env=env)

    # begin eval loop
    successes_by_name = {}
    ep = 0
    while ep < num_eps:
        # reset environment
        obs, goal = env.reset()
        policy.reset_policy(next_obs=obs, next_goal=goal)
        policy_hardcoded.reset_policy(next_obs=obs, next_goal=goal)
        policy.warm_start(model, obs, goal)

        full_seq, pname, ptype, success_dc = eval_rollout(args, prefix, env, obs, goal,
                                                          policy, policy_hardcoded, stack_goal=True)
        if success_dc is not None:
            succ = success_dc['best'][0]
            logger.debug(f"{prefix}Episode {ep}: policy={pname}, success#={succ}")
            if pname not in successes_by_name.keys():
                successes_by_name[pname] = []
            successes_by_name[pname].append(int(succ >= 0.5))
            ep += 1

    # return as tuple for pool.
    return successes_by_name,


num_proc = args.num_proc
assert num_proc > 0
logger.debug(f"Num proc: {num_proc}")

# splitting the work
assert args.num_eps % num_proc == 0, "Work must be evenly divisable"

if num_proc > 1:
    import torch.multiprocessing as mp

    mp.set_start_method('forkserver')

    all_args = []
    for i in range(num_proc):
        all_args.append((i, args.num_eps // num_proc))

    logger.debug(f"launching {num_proc} processes...")
    with mp.Pool(num_proc) as p:
        rets = p.map(eval_process, all_args)

    # combine success data
    successes_by_name = {}
    for succ, in rets:
        for key in succ.keys():
            if key not in successes_by_name:
                successes_by_name[key] = []
            successes_by_name[key] = successes_by_name[key] + succ[key]

else:
    successes_by_name, = eval_process((0, args.num_eps))

logger.debug("Successes:")
AttrDict.from_dict(successes_by_name).leaf_apply(lambda vs: np.mean(vs)).pprint()

logger.debug("Done.")
