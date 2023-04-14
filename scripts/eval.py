"""
Evaluate a policy and model in an environment. No saving of data (see scripts/collect.py)
"""

import os
import sys

import numpy as np
import torch

from configs.helpers import get_script_parser, load_base_config
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c, is_next_cycle
from muse.utils.torch_utils import reduce_map_fn
from muse.utils.general_utils import timeit

exit_on_ctrl_c()

# things we can use from command line
parser = get_script_parser()
parser.add_argument('config', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--max_eps', type=int, default=0)
parser.add_argument('--no_model_file', action="store_true")
parser.add_argument('--random_policy', action="store_true")
parser.add_argument('--print_last_obs', action="store_true")
parser.add_argument('--print_policy_name', action="store_true")
parser.add_argument('--timeit_freq', type=int, default=0)
parser.add_argument('--track_returns', action="store_true")
parser.add_argument('--reduce_returns', type=str, default='sum', choices=list(reduce_map_fn.keys()),
                    help='If tracking returns, will apply this func to the returns before tracking..')
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

# generate env
env_spec = params.env_spec.cls(params.env_spec)
env = params.env_train.cls(params.env_train, env_spec)

# generate model and policy
model = params.model.cls(params.model, env_spec, None)
policy = params.policy.cls(params.policy, env_spec, env=env)

# reset the environment and policy
obs, goal = env.reset()
policy.reset_policy(next_obs=obs, next_goal=goal)

# warm start the policy
policy.warm_start(model, obs, goal)

# restore model from file (if provided)
if not args.no_model_file:
    model.restore_from_file(model_fname)

model.eval()

# actual eval loop
done = [False]
all_returns = []
rew_list = []
i = 0
ep = 0
steps = 0
reward_reduce = reduce_map_fn[args.reduce_returns]

while True:
    if done[0] or policy.is_terminated(model, obs, goal):
        logger.info(f"[{ep}] Resetting env after {i} steps")
        if args.track_returns:
            returns = reward_reduce(torch.tensor(rew_list)).item()
            logger.info(f'Returns: {returns}')
            all_returns.append(returns)
        rew_list = []
        steps += i
        i = 0
        ep += 1
        # terminate condition
        if ep >= args.max_eps > 0:
            break
        obs, goal = env.reset()
        policy.reset_policy(next_obs=obs, next_goal=goal)
        timeit.reset()

    # empty axes for (batch_size, horizon)
    expanded_obs = obs.leaf_apply(lambda arr: arr[:, None])
    expanded_goal = goal.leaf_apply(lambda arr: arr[:, None])
    with torch.no_grad():
        # query the model for the action
        if args.random_policy:
            action = policy.get_random_action(model, expanded_obs, expanded_goal)
        else:
            action = policy.get_action(model, expanded_obs, expanded_goal)

        if i == 0 and args.print_policy_name and action.has_leaf_key("policy_name"):
            logger.info(f"Policy: {action.policy_name.item()}")

    # step the environment with the policy action
    obs, goal, done = env.step(action)
    i += 1

    if is_next_cycle(i, args.timeit_freq):
        print(timeit)
        timeit.reset()

    if args.track_returns:
        rew_list.append(obs.reward.item())

    if done and args.print_last_obs:
        logger.debug("Last obs:")
        obs.pprint()

if args.track_returns:
    logger.info(f"----------- Done w/ {ep} episode(s), {steps} steps  -----------")
    logger.info(f"exp_name = {exp_name}\n")
    logger.debug(f"Raw command: \n{' '.join(sys.argv)} \n")

    all_returns = np.asarray(all_returns)
    logger.info(f"Returns: mean={np.mean(all_returns)}, std={np.std(all_returns)}, "
                f"%>0={np.mean(all_returns > 0) * 100}")
    logger.info(f"---------------------------------------------------------------")
