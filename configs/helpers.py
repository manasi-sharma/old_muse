import argparse

from configs.config_node import ConfigNode
from configs.utils import import_config_file


def make_base_config(base_params):
    class BaseConfigNode(ConfigNode):
        default_params = base_params

    return BaseConfigNode('')


def load_base_config(file, cmd_args):
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
