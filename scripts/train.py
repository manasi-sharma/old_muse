import argparse
import sys

from attrdict import AttrDict

from configs.helpers import load_base_config, get_script_parser
from muse.envs.env import Env
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager

if __name__ == '__main__':
    parser = get_script_parser()
    parser.add_argument('config', type=str, help="common params for all modules.")
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--print_all', action='store_true')
    parser.add_argument('--no_env', action='store_true')
    parser.add_argument('--do_holdout_env', action='store_true')
    local_args, unknown = parser.parse_known_args()

    logger.debug(f"Raw command: \n{' '.join(sys.argv)}")

    if local_args.local_rank is not None:
        import torch
        torch.cuda.set_device(local_args.local_rank)

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
    env_spec = params.env_spec.cls(params["env_spec"])
    if local_args.no_env:
        env_train = Env(AttrDict(), env_spec)
    else:
        env_train = params.env_train.cls(params["env_train"], env_spec)
    env_holdout = None if "env_holdout" not in params \
        else params.env_holdout.cls(params.env_holdout, env_spec)

    dataset_train = params.dataset_train.cls(params.dataset_train, env_spec, file_manager)
    dataset_holdout = params.dataset_holdout.cls(params.dataset_holdout, env_spec, file_manager,
                                                 base_dataset=dataset_train)

    # making model
    model = params.model.cls(params.model, env_spec, dataset_train)

    # policy
    policy = params.policy.cls(params.policy, env_spec, env=env_train)

    # trainer
    trainer = params.trainer.cls(params.trainer,
                                 file_manager=file_manager,
                                 model=model,
                                 policy=policy,
                                 dataset_train=dataset_train,
                                 dataset_holdout=dataset_holdout,
                                 env_train=env_train,
                                 env_holdout=env_holdout)

    # run training
    trainer.run()
