import argparse
import inspect
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Tuple, Callable

from attrdict import AttrDict as d
from configs import utils as cu
from configs.fields import Field
from configs.macro_parser import MacroEnabledArgumentParser
from muse.experiments import logger

PREFIX_CHAR = '%'


class ConfigNode:
    """
    Config for a specific group, where the groups are modular

    Methods
    -------

    """

    # default class for subclasses to use to anchor to
    default_cls = None
    # default local chunk of experiment name
    default_local_exp_name = ""
    # default arguments to skip from cls.predefined_arguments
    default_cls_skip_args = []
    # default parameters for the class to be instantiated with
    default_params = d(

    )

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

        # set the nested name of this node, based on the parent.
        if self.parent is None or len(self.parent.full_name) == 0:
            self.full_name = self.name
        else:
            self.full_name = self.parent.full_name + '/' + self.name

        # will be set during load
        self.is_loaded = False
        self.parser = None

        # things to update in other functions
        self.cls = None
        self.params = None
        self.cls_skip_args = None

        # parts of the params
        self.local_exp_name = None
        self.local_params = None
        self.subgroups = None
        self.subgroup_names = []

    # declares the arguments for this level
    def declare_arguments(self, parser, defaults, cls_skip_args) -> ArgumentParser:
        """ Add arguments that should be read by command line.

        By default this loads arguments from...
        (1) the specified class

        Parameters
        ----------
        parser: argparse.ArgumentParser
        defaults: d
            The default params
        cls_skip_args: List[str] args from cls to skip

        Returns
        -------

        """
        defaults = defaults.as_dict()

        if self.cls is not None and hasattr(self.cls, 'predefined_arguments'):
            # class defines some arguments
            logger.debug(f'[{self.name}] Loading predefined args for {self.cls}...')

            # load all cls args and change the defaults to match the config defaults
            added = self.cls.declare_arguments(parser, skip_args=cls_skip_args, defaults=defaults)

            # try to add what remains (some will already have been added above likely)
            defaults_to_add = {k: v for k, v in defaults.items() if k not in added}
        else:
            # try to add all defaults to command lines
            defaults_to_add = defaults

        if len(defaults_to_add) > 0:
            # use default loading for extra params in the defaults.
            cu.parser_from_params(parser, defaults_to_add)

        return parser

    def process_params(self, namespace, node_params, global_params=None) -> d:
        """ Process arguments loaded from command line back into params, optionally using parent params.

        Using non-global parent_params when specified, although allowed, is not good.
        This makes a config too dependent on what else has been loaded...
        By default, process just combines params with the namespace, processing fields if present

        Parameters
        ----------
        namespace: argparse.Namespace
        node_params: d
            The existing params for this node so far (does not include any subgroup params)
        global_params: d
            The global tree of params that has been generated so far, with node params inserted in place.

        Returns
        -------
        d

        """
        out_d = node_params & cu.ns_to_attrs(namespace)
        # for looking up current keys in a global way

        if global_params is None:
            global_params = node_params

        # go through and process all the fields with the command line updated values at this level (and global)
        local_params, _ = cu.filter_local_and_group_params(out_d)
        for key, item in local_params.leaf_items():
            if isinstance(item, Field):
                out_d[key] = item.process(key, local_params, global_params, path=self.full_name)

        return out_d

    def load_subgroups(self, node_params, subgroups, cmd_group_args) -> Tuple[d, d]:
        """ Process all subgroup specs with optional command line overrides, calls load on the subgroups after creating them.

        Parameters
        ----------
        node_params: d
            Existing parameters for node.
        subgroups: d
            Groups with defaults, where each has the format:
            group_name = d(
                cls = ...
                ...
            )
        cmd_group_args: Dict[str, List[str]]
            Command line overrides for groups for this node.

        Returns
        -------
        params: d
            updated parameters with subgroup (loaded)
        subgroups: d
            maps group name to subgroup ConfigNode

        """
        nested_params = node_params.leaf_copy()

        # this is what we will fill
        self.subgroups = d()
        self.subgroup_names = list(subgroups.keys()) + [k for k in cmd_group_args.keys() if k not in subgroups.keys()]

        # iterate through all groups (default and cmd line overrides)
        for key in self.subgroup_names:
            # default inner params for groups that were specified in self.params
            grp_params = subgroups[key].leaf_filter(lambda k, v: k != 'cls') if key in subgroups else d()
            grp_arg_strings = []

            if key in cmd_group_args:
                # check that override groups have at least 1 argument
                assert len(cmd_group_args[key]) > 0, f"[{self.full_name}] Missing arguments for cmd group {key}"
                # try importing it
                imp_source = cu.try_import_source(cmd_group_args[key][0])
                if key in subgroups and imp_source is None:
                    # if source not present in override, default to subgroup cls, but use override args
                    grp_source = subgroups[key].cls
                    grp_arg_strings = cmd_group_args[key]
                else:
                    # use both override cls and override args (partitioned)
                    grp_source = cmd_group_args[key][0]
                    grp_arg_strings = cmd_group_args[key][1:]
            else:
                grp_source = subgroups[key].cls

            grp_source = cu.try_import_source(grp_source)
            # instantiate the cmd line modified groups
            self.subgroups[key] = GroupStruct(key, grp_source, grp_params, arg_strings=grp_arg_strings)

        # recursively instantiate and parse all subgroups
        n = self.full_name
        for key in self.subgroups:
            logger.debug(f"[{n if len(n) > 0 else 'root'}] Creating and loading group {key}.")
            gstruct = self.subgroups[key]
            self.subgroups[key] = self.subgroups[key].create(parent=self)
            # now load the subgroup with the command line strings
            nested_params[key] = self.subgroups[key].load(gstruct.arg_strings,
                                                          defaults=gstruct.params,
                                                          cls=gstruct.cls)

        return nested_params, self.subgroups

    def load(self, cmd_args, defaults=d(), cls=None) -> d:
        """ Combines everything, including resolving subgroups and loading them, into final set of params

        Parameters
        ----------
        cmd_args: List[str]
            Specifies the command line arguments for this group.
        defaults: d
            Any defaults to use to override or add to default_params
        cls: class
            Load in place of default_cls

        Returns
        -------

        """

        assert not self.is_loaded, f"[{self.full_name}] Already loaded but load() was called!"

        # update the defaults and cls
        self.cls = self.default_cls if cls is None else cls
        self.params = self.default_params & defaults
        self.cls_skip_args = self.default_cls_skip_args

        # things that aren't groups
        self.local_params, _ = cu.filter_local_and_group_params(self.params)

        # add an option for the extra name (added to end)
        if 'exp_name' in self.local_params and 'extra_exp_name' not in self.local_params:
            self.local_params['extra_exp_name'] = ""

        # declare the arguments for the command line parser (macro-enabled, prog=group name)
        kwargs = {'formatter_class': ArgumentDefaultsHelpFormatter, 'add_help': False}
        if self.parent is None:
            self.parser = MacroEnabledArgumentParser(**kwargs)
            self.parser.add_argument('-h', '--help', action='store_true')
        else:
            self.parser = MacroEnabledArgumentParser(prog=self.name, **kwargs)
        self.declare_arguments(self.parser, self.local_params, self.cls_skip_args)

        # split the command line arguments
        local_args, cmd_group_args = cu.split_local_and_group_args(cmd_args, PREFIX_CHAR)

        # now process parameters at the local level, passing in parent parameters
        namespace = self.parser.parse_args(local_args)
        do_help = False
        if self.parent is None:
            do_help = namespace.help
            delattr(namespace, 'help')

        global_params = None
        if self.parent and self.parent.is_loaded:
            global_params = self.global_params

        self.params = self.process_params(namespace, self.params, global_params=global_params)

        # set the local exp name if it was provided, and remove it from local_params
        self.local_exp_name = self.default_local_exp_name
        if 'exp_name' in self.local_params:
            self.local_exp_name = self.local_params.exp_name
            del self.local_params['exp_name']

        # this notes that all the local params have been loaded (and maybe some subgroups)
        self.is_loaded = True

        # reload subgroups (again) since they might have been changed in process_params
        _, subgroups = cu.filter_local_and_group_params(self.params)

        # load the subgroup structures into params.
        self.params, self.subgroups = self.load_subgroups(self.params, subgroups, cmd_group_args)

        if do_help:
            # after loading things, we can print a more useful "help" message
            self.help()
            sys.exit(0)

        return self.full_params

    def get_local_exp_name(self) -> str:
        """ Local experiment name (just this node)

        Returns
        -------
        str

        """
        # allow support for a callable exp_name fn
        local_exp_name = self.local_exp_name
        if isinstance(local_exp_name, Callable):
            local_exp_name = local_exp_name(self.params, self.full_params)
        assert isinstance(local_exp_name, str), \
            f"Local experiment name ({local_exp_name}) should be str, but was {type(local_exp_name)}!"
        # add extra name in
        if 'extra_exp_name' in self.params:
            local_exp_name += self.params['extra_exp_name']
        return cu.find_replace_brackets(local_exp_name, self.full_params)

    def get_exp_name(self, nested=True) -> str:
        """ Get the experiment name string for this node (nested: and its subclasses in order)

        Parameters
        ----------
        nested: bool
            include the subclass experiment name in the return

        Returns
        -------
        str

        """
        exp_name = self.get_local_exp_name()
        if nested:
            # depth first lookup of the experiment names of each subgroup.
            for key in self.subgroup_names:
                exp_name += self.subgroups[key].get_exp_name(nested=True)
        return exp_name

    # calling help on parsers to print messages using each help string
    def usage(self, prefix=''):
        print(cu.prefix_lines(self.parser.format_usage(), prefix))
        for key in self.subgroup_names:
            self.subgroups[key].usage(prefix=f'{prefix}\t')

    def help(self, prefix=''):
        lines = self.parser.format_help()
        length = max(int(0.8 * os.get_terminal_size().columns) - len(prefix.expandtabs()), 1)
        print(prefix + ('-' * length))
        print(cu.prefix_lines(lines[:-1], prefix))
        print(prefix + ('-' * length))
        for key in self.subgroup_names:
            self.subgroups[key].help(prefix=f'{prefix}\t')

    @property
    def full_params(self):
        assert self.is_loaded
        return d(cls=self.cls) & self.params

    @property
    def global_params(self):
        if self.parent is None:
            return self.params.leaf_copy()

        gparams = self.parent.global_params.leaf_copy()
        if self.full_name not in gparams:
            gparams[self.full_name] = d()
        gparams[self.full_name] &= self.params

        return gparams


class GroupStruct:
    """
    Sub Group format to simplify loading process. Don't create these manually.
    """

    def __init__(self, name, source, params, arg_strings):
        self.name = name
        self.source = self.original_source = source
        self.params = params.leaf_copy()
        self.arg_strings = list(arg_strings)

        # class of the group
        self.cls = self.params << 'cls'

        # get the <export> from modules
        if inspect.ismodule(self.source):
            try:
                self.source = self.source.export
            except AttributeError as e:
                raise AttributeError(f'[{self.name}], source={self.source}, {str(e)}')

        # figure out what class we need to instantiate
        if isinstance(self.source, d):
            # if source is a dict, wrap as ConfigNode, and add dict params to existing ones.
            self.config_cls = ConfigNode
            self.params = self.params & self.source

            self.cls = self.params << 'cls'

        elif self.source and issubclass(self.source, ConfigNode):
            # source specifies some sub class of ConfigNode
            self.config_cls = self.source
        else:
            # source is specifying the inner class, which means wrap it.
            self.config_cls = ConfigNode
            self.cls = self.source

        assert issubclass(self.config_cls, ConfigNode), f"Invalid config class: {self.config_cls}"

    def create(self, parent=None) -> ConfigNode:
        return self.config_cls(self.name, parent=parent)


if __name__ == '__main__':
    from muse.models.model import Model
    from muse.policies.policy import Policy

    class TestConfigNode(ConfigNode):
        default_params = d(
            batch_size=10,
            horizon=20,
            device="cuda",
            lr=1e-4,
            more_stuff=['lol', 'this', 'is', 'a', 'list'],

            model=d(
                cls=Model,
                ignore_inputs=True,
                nested_group=d(
                    cls=Model,
                    inside_param=10,
                    inner_nested_group=d(
                        cls=Model,
                        guess=100,
                    )
                )
            ),
            policy=d(
                cls=Policy,
            ),
        )


    root = TestConfigNode('')
    params = root.load(sys.argv[1:])

    params.pprint()
