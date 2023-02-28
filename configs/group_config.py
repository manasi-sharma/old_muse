"""
Example dynamic config module, loaded by LoadableGroupedArgumentParser.

 *Dynamic* Config files should define two functions:

- declare_arguments(): returns a parser that will get the specific arguments required for this config.
<the calling script will then call parser.parse_args()>
- process_params(): takes in the parameterization (included by group in common_params), and makes any changes
                    to the global config based on all the loaded / default params.

Finally, they should export these as an AttrDict named "params"
"""
from argparse import ArgumentParser
from pydoc import locate
from typing import Tuple

from muse.experiments import logger
from muse.utils.config_utils import Utils
from muse.utils.file_utils import prepend_to_base_name, import_config_module
# declares this group's parser, and defines any sub groups we need
from muse.utils.python_utils import AttrDict as d, get_with_default


class GroupConfig:
    """
    Config for a specific group, where the groups are modular
    Use name = 'root' for the root node.

    Methods
    -------
    declare_defaults: sets up the cls and params which define the structure.
    declare_arguments: declares the arguments to check for this object

    """

    def __init__(self, name: str, parent=None, cls=None, params: d = d(), subclasses=None, skip_args=()):
        """

        Parameters
        ----------
        name: name of the group
        parent: the parent GroupConfig node
        params: local params for this level to use as defaults.
        subclasses: dict mapping submodule -> class (partial list)
        skip_args:
        """
        self.name = name
        self.parent = parent

        # the class for this group
        self.cls = cls

        # the argument defaults (will be loaded and then optionally mutated by cmd line)
        self.params = params.leaf_copy()

        # arguments in the class to skip
        self.skip_args = list(skip_args)

        self.submodule_names = None

        # map submodule -> str module or class
        self.subclasses = subclasses or {}

        # override this to declare defaults in a config specific fashion
        self.declare_defaults()

        # this will search for submodules, and then similarly make group configs for them, returning a big dictionary.
        self.tree = self._resolve_structure()

    def declare_defaults(self):
        """ Defines the static "structure", including any default arguments, for this level.

        Override this for each config that has a static config.

        Returns
        -------
        """
        pass

    def _resolve_structure(self) -> dict:
        """
        Resolves the submodule structure, passing in defaults where relevant

        You can pass in cls=[instantiable class, str, or GroupConfigSubclass]
        If str, it will load the class from file.

        Returns
        -------
        mapping from submodule name to its GroupConfig

        """
        assert self.submodule_names is None
        self.submodule_names = []
        tree = {}
        for group_name, val in self.params.items():
            if isinstance(val, d) and val.has_key('cls'):
                logger.debug(f"[{self.name}]: Loading submodule {group_name}...")
                self.submodule_names.append(self.name)

                # params for the submodule, if present
                params = get_with_default(val, 'params', d())

                # check if its in the submodule names, and replace if so.
                if group_name in self.subclasses.keys():
                    val.cls = self.subclasses[group_name]
                    logger.debug(
                        f"[{self.name}/{group_name}] Overwriting given {val.cls} with {self.subclasses[group_name]}")

                if isinstance(val.cls, str):
                    # find the class as a string
                    cls = locate(val.cls)
                    assert issubclass(cls,
                                      GroupConfig), f"[{self.name}/{group_name}]: Loaded a class that does not subclass GroupConfig!"
                    tree[group_name] = cls(group_name, parent=self, params=params)

                elif issubclass(val.cls, GroupConfig):
                    # instantiate this from a loaded config.
                    tree[group_name] = val.cls(group_name, parent=self, params=params)

                else:
                    # instantiate a generic group config from the cls params defined.
                    assert issubclass(val.cls, object), f"[{self.name}/{group_name}]: Must pass in a valid class!"
                    tree[group_name] = GroupConfig(group_name, parent=self, cls=val.cls, params=params)

        assert set(self.subclasses.keys()).issubset(self.submodule_names), \
            f"[{self.name}]: Extra subclass names provided: " \
            f"subclass_names={list(self.subclasses.keys())}, " \
            f"submodule_names={self.submodule_names}"
        return tree if len(tree) > 0 else None

    def declare_arguments(self, parser=ArgumentParser()):
        """ Declares command line parsable arguments for the group

        Parameters
        ----------
        parser: an argument parser to add new arguments to.

        Returns: argument parser
        -------

        1. cls.predefined_arguments are loaded if params.cls is not None
        2. add arguments unique & local to this level

        """

        if self.cls is not None and hasattr(self.cls, 'predefined_args'):
            # class defines some arguments
            logger.debug(f'[{self.name}] Loading predefined args for {self.cls}...')

            # load all cls args and change the defaults to match the config defaults
            self.cls.declare_arguments(parser, skip_args=self.skip_args, defaults=self.local_defaults)
        else:
            # use default loading for params in the defaults.
            self.parser_from_params(parser, self.params)

        return parser

    def process_params(self, global_params, utils: Utils) -> Tuple[d, d]:
        """ compile arguments into the params that this file defines

        Parameters
        ----------
        global_params: params that are globally defined.
        utils: the utils module for the current experiment.

        Returns
        -------
        new global params, optionally changing it...
        new instantiable params, which can be loaded for this class

        """
        return global_params, d(
            cls=self.cls,
            params=self.params.leaf_copy(),
        )

    @staticmethod
    def parser_from_params(parser, params):
        """ Looks into params keys, adds arguments for known dtypes

        Parameters
        ----------
        parser
        params

        Returns
        -------

        """
        parsable_types = [
            float,
            int,
            str,
        ]

        # parse params from local arguments.
        for k, v in params.items():
            if type(v) in parsable_types:
                parser.add_argument(f'--{k}', type=type(v), default=v)
            elif type(v) is bool:
                if not v:
                    parser.add_argument(f'--{k}', action='store_true')
                else:
                    parser.add_argument(f"--{prepend_to_base_name(k, 'no-')}", dest=k, action='store_false')
                parser.set_defaults(**{k: v})

        return parser

    @property
    def local_defaults(self) -> dict:
        # get the params at the local level
        return (self.params > self.params.keys()).as_dict()

    @property
    def full_name(self) -> str:
        full = self.name
        if self.parent is not None:
            pname = self.parent.full_name
            if len(pname) > 0:
                full = pname + '/' + full
        return full


def count_start_char(s, char):
    count = 0
    while s.startswith(char):
        s = s[1:]
        count += 1

    return count


def split_flat_then_nested(list, nexts, char):
    if len(list) == 0:
        return []

    depref = [l[1:] for l in list]
    assert not depref[0].startswith(char)
    curr_group = depref[0]
    i = 1
    groups_this_level = {curr_group: [nexts[0]]}
    while i < len(list):
        count = count_start_char(depref[i], char)
        if count > 0:
            groups_this_level[curr_group].append((depref[i], nexts[i]))
        elif count == 0:
            curr_group = depref[i]
            groups_this_level[curr_group] = [nexts[i]]

    group_nested = {}
    for key, group_ls in groups_this_level.items():
        if len(group_ls) > 1:
            # if there are children keep going
            group_nested[key] = (group_ls[0], split_flat_then_nested([g[0] for g in group_ls[1:]],
                                                                     [g[1] for g in group_ls[1:]],
                                 char))
        else:
            # if base, then include
            groups_this_level[key] = group_ls[0]

    # return a dictionary of group mapping to (next, children_dict())
    return group_nested


def load_group_config(config_file, raw_args, group_char='%'):
    # base_config = import_config_module(config_file)

    # get subclasses from raw_args
    group_args = [a for a in raw_args if a.startswith(group_char)]
    group_idxs = [i for i, a in enumerate(raw_args) if a.startswith(group_char)]
    group_next = [raw_args[i + 1] if len(raw_args) > i + 1 else '' for i in group_idxs]
    group_next_is_py = [gn.endswith('.py') for gn in group_next]

    out = split_flat_then_nested(group_args, group_next_is_py, group_char)


    # group_pref_count = {g: count_start_char(g, group_char) for g in group_args}
    #
    # structure = d()
    # curr_prefix = None
    # last_prefix = None
    # curr_level = 0
    #
    #
    #
    # for g, count in group_pref_count.items():
    #     if curr_prefix is None:
    #         # assert count == 1, "First group must be first level!"
    #         curr_prefix = g[count:]
    #         last_prefix = curr_prefix
    #         curr_level = count
    #         structure[curr_prefix] = {}
    #     else:
    #         if count == curr_level + 1:
    #             # add nested group
    #             structure[curr_prefix][g[count:]] = {}
    #             last_prefix = curr_prefix + '/' + g[count:]
    #         elif count == curr_level:
    #             # terminate group
    #         else:
    #
    #
    #
    # # nested structure
    # if len(group_args) > 0:
    #     curr_group = group_args[1:]
    #     assert curr_group[0] != group_args, f"Sequence cannot begin with a nested group ({curr_group})!"

    # root = GroupConfig('', params=base_config.params)


if __name__ == '__main__':

    pass