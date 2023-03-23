import argparse
import os

from configs.config_node import ConfigNode
from configs.utils import import_config_file
from muse.experiments import logger


def make_base_config(base_params):
    class BaseConfigNode(ConfigNode):
        default_params = base_params

    return BaseConfigNode('')


def load_base_config(file, cmd_args):
    if file.startswith('exp='):
        exp = file[4:]
        logger.debug(f"Loading exp {exp}...")
        file = os.path.join(exp, "config.py")
        conf_arg_file = os.path.join(exp, "config_args.txt")
        if os.path.exists(conf_arg_file):
            logger.debug(f"Loading config args for {exp}...")
            with open(conf_arg_file, 'r') as f:
                arg_str = f.read().replace('\n', ' ')
            arg_ls = arg_str.split()
            if len(arg_ls) > 0:
                logger.debug(f'Prepending command args: {arg_str}')
            cmd_args = arg_ls + cmd_args

    params = import_config_file(file).export
    root = make_base_config(params)
    return root.load(cmd_args), root


def get_script_parser(**kwargs):
    # parser defines its own help
    if 'add_help' not in kwargs:
        kwargs['add_help'] = False

    # ensures no namespace collisions
    if 'allow_abbrev' not in kwargs:
        kwargs['allow_abbrev'] = False

    return argparse.ArgumentParser(**kwargs)
