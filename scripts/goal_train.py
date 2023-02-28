import argparse
import sys

from configs.helpers import load_base_config
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('config', type=str, help="common params for all modules.")
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--print_all', action='store_true')
    parser.add_argument('--do_holdout_env', action='store_true')
    parser.add_argument('--different_env_holdout', action='store_true')
    parser.add_argument('--num_datasets', type=int, default=1)
    parser.add_argument('--model_dataset_idx', type=int, default=-1)
    parser.add_argument('--run_async', action='store_true')
    local_args, unknown = parser.parse_known_args()

    logger.debug(f"Raw command: \n{' '.join(sys.argv)}")

    # load the config
    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    logger.debug(f"Using: {exp_name}")
    file_manager = ExperimentFileManager(exp_name,
                                         is_continue=getattr(local_args, 'continue'),
                                         log_fname='log_train.txt',
                                         config_fname=local_args.config,
                                         extra_args=unknown)

    # instantiate classes from the params
    env_spec = params.env_spec.cls(params.env_spec)
    env_train = params.env_train.cls(params.env_train, env_spec)

    if not local_args.do_holdout_env:
        env_holdout = None
    else:
        if local_args.different_env_holdout:
            env_holdout = params.env_holdout.cls(params.env_holdout, env_spec)
        else:
            env_holdout = params.env_train.cls(params.env_train, env_spec)

    datasets_train, datasets_holdout = [], []
    for i in range(local_args.num_datasets):
        suffix = f"_{i}" if local_args.num_datasets > 1 else ""
        datasets_train.append(params[f"dataset_train{suffix}"].cls(params[f"dataset_train{suffix}"],
                                                                   env_spec, file_manager))
        datasets_holdout.append(params[f"dataset_holdout{suffix}"].cls(params[f"dataset_holdout{suffix}"],
                                                                       env_spec, file_manager,
                                                                       base_dataset=datasets_train[-1]))

    # making model, default use the last dataset.
    model = params.model.cls(params.model, env_spec, datasets_train[local_args.model_dataset_idx])

    # for train env
    policy = params.policy.cls(params.policy, env_spec, env=env_train)
    goal_policy = params.goal_policy.cls(params.goal_policy, env_spec, env=env_train, is_goal=True)

    # for holdout env
    policy_holdout, goal_policy_holdout = None, None
    if local_args.do_holdout_env:
        policy_holdout = params.policy_holdout.cls(params.policy_holdout, env_spec, env=env_holdout)
        goal_policy_holdout = params.goal_policy_holdout.cls(params.goal_policy_holdout, env_spec,
                                                             env=env_holdout, is_goal=True)

    reward = params.reward.cls(params.reward, env_spec) if params.has_leaf_key("reward/cls") else None

    # trainer
    trainer = params.trainer.cls(params.trainer,
                                 file_manager=file_manager,
                                 model=model,
                                 policy=policy,
                                 goal_policy=goal_policy,
                                 datasets_train=datasets_train,
                                 datasets_holdout=datasets_holdout,
                                 env_train=env_train,
                                 env_holdout=env_holdout,
                                 policy_holdout=policy_holdout,
                                 goal_policy_holdout=goal_policy_holdout,
                                 reward=reward)

    if local_args.run_async:
        trainer.run_async()
    else:
        # run training
        trainer.run()
