import importlib
import os
import re
from collections import OrderedDict
from pydoc import locate
from typing import List

from attrdict import AttrDict as d

from muse.utils.file_utils import prepend_to_base_name
from muse.utils.general_utils import round_to_n


def hr_name(float_arg, fp=None):
    if fp is not None:
        float_arg = round_to_n(float_arg, n=fp)
    return str(float_arg).replace('.', '_')


def find_replace_brackets(string, params, fp=None):
    pattern = '{([^{]+)}'
    while re.search(pattern, string) is not None:
        s = re.search(pattern, string).group(1)
        name = str(s)

        # bool argument replace with name if True else nothing.
        use_conditional = s.startswith('?')
        if use_conditional:
            # conditional of form {?name:true_replace[:false_replace]}
            name = name[1:]
            cond = name.split(':')
            assert 1 <= len(cond) <= 3, f"conditional: {s} is not a valid format!"
            # first element is the name to search
            name = cond[0]
            # if not specified, true string is the argument name
            if len(cond) == 1:
                cond = cond * 2
            # if not specified, false string is empty
            if len(cond) == 2:
                cond = cond + ['']

        assert name in params.leaf_keys(), f"Value for {name} (found in {string}) not found in params! "
        string_value = params[name]

        # parsing the string value.
        if use_conditional:
            # conditional {?name:true_replace:false_replace}
            if string_value:
                string_value = cond[1]
            else:
                string_value = cond[2]
        elif isinstance(string_value, float):
            string_value = hr_name(string_value, fp=fp)
        else:
            string_value = str(string_value)
        string = string.replace(f'{{{s}}}', string_value)

    return string


def prefix_lines(long_str, prefix):
    lines = long_str.split('\n')

    new_lines = []
    for line in lines:
        new_lines.append(prefix + line)

    return '\n'.join(new_lines)


def ns_to_attrs(namespace, out=None) -> d:
    if out is None:
        out = d()
    # locals
    for key in namespace.__dict__:
        # print(out, key)
        attr = getattr(namespace, key)
        out[key] = attr

    return out


def filter_local_and_group_params(params: d):
    dc = d()
    groups = d()
    for key in params.keys():
        if isinstance(params[key], d) and 'cls' in params[key]:
            groups[key] = params[key]
        else:
            dc[key] = params[key]
    return dc, groups


def count_start_char(s, char):
    count = 0
    while s.startswith(char):
        s = s[1:]
        count += 1

    return count


def split_local_and_group_args(arg_strings, char, only_local=False):
    """ Split a list of grouped arguments into local and group_args

    Nested arguments will only parse groups at the local level, not sub nested groups.

    Parameters
    ----------
    arg_strings: List[str]
        all arguments on command line as strings
    char: str
        the character for prefixing
    only_local: bool
        return just the local arguments without parsing the nested ones

    Returns
    -------
    local_arg_strings: List[str]
    nested_child_arg_strings: Dict[str, List[str]]

    """

    local_arg_strings = []
    nested_child_arg_strings = OrderedDict()
    curr_group = None
    # split the arguments
    for i, arg in enumerate(arg_strings):
        char_cnt = count_start_char(arg, char)

        """ LOCAL ARGS """
        if curr_group is None:
            assert char_cnt <= 1, f"{arg} is a nested group but no group was specified!"
            if char_cnt == 0:
                # add argument and don't bother looking for nested groups
                local_arg_strings.append(arg)
                continue
            elif only_local:
                # done if only parsing local and a group was found.
                return local_arg_strings, nested_child_arg_strings

        """ SUBGROUP (nested) ARGS """
        if char_cnt == 0 or char_cnt > 1:
            # check for specified group (can't be unspecified, but just in case)
            assert curr_group is not None, f"Cannot add {arg} since no group was specified..."
            # strip nested sub-group
            if char_cnt > 1:
                arg = arg[1:]
            # add the argument to the current group for regular arguments and further nested arguments for this group.
            nested_child_arg_strings[curr_group].append(arg)
            # print(f"{arg} going to {curr_group}")
        elif char_cnt == 1:
            # if this is a valid group for this level, change the group
            assert len(arg) > 1, f"Floating {char}!"

            # set the new group for future args
            curr_group = arg[1:]

            # create argument list for new group if not already there (append to existing)
            if curr_group not in nested_child_arg_strings.keys():
                nested_child_arg_strings[curr_group] = []

    return local_arg_strings, nested_child_arg_strings


def import_config_file(config_fname):
    if config_fname.endswith('.py'):
        spec = importlib.util.spec_from_file_location('config', config_fname)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    elif config_fname.endswith('.yaml'):
        raise NotImplementedError("YAML files not implemented yet")
    else:
        raise NotImplementedError(f"Cannot load {config_fname}")

    return config


def try_import_source(source):
    # import source as a module with export, or a class, from string
    if isinstance(source, str):
        if os.path.exists(source):
            # load a file as a module
            source = import_config_file(source)
        elif importlib.util.find_spec(source):
            # load a module directly (with dot notation)
            source = importlib.import_module(source)
        else:
            # load a class
            source = locate(source)

    return source


def parser_from_params(parser, params):
    """ Looks into params keys, adds arguments for known dtypes

    Default help string will consist of "type = %(type)s"

    Parameters
    ----------
    parser: argparse.ArgumentParser
    params: d

    Returns
    -------
    argparse.ArgumentParser

    """
    parsable_types = [
        float,
        int,
        str,
    ]

    # parse params from local arguments.
    for k, v in params.items():
        if type(v) in parsable_types:
            parser.add_argument(f'--{k}', type=type(v), default=v, help='type = %(type)s')
        elif type(v) is bool:
            if not v:
                parser.add_argument(f'--{k}', action='store_true', help='type = bool')
            else:
                parser.add_argument(f"--{prepend_to_base_name(k, 'no-')}", dest=k, action='store_false',
                                    help='type = bool')
            parser.set_defaults(**{k: v})

    return parser


if __name__ == '__main__':
    import json

    """ testing split_local_and_group_args... """

    test_args = "--a --b --c 10 --d 1.0 %group1 --arg1 --arg2 %%subgroup11 --a 10 %group2 --arg1 --b %group3 --a string --yolo " \
                "%%subgroup31 test --a %%subgroup32 test2 --another --arg".split()

    local_args, nested_args = split_local_and_group_args(test_args, '%')

    print('local arguments:', local_args)
    print('nested_args ->', json.dumps(nested_args, indent=4))

    """ testing filter... """

    fdc = d(
        real_arg1=1,
        real_arg2=10.,
        group1=d(
            cls=1,
            arg1=2,
        ),
        group2=d(
            cls=3,
            arg1=4,
        ),
    )
    loc, grp = filter_local_and_group_params(fdc)
    print(f'locals -> {loc.pprint(ret_string=True)}')
    print(f'groups -> {grp.pprint(ret_string=True)}')

    print('exp_name:', find_replace_brackets('test_{real_arg1}-{real_arg2}', fdc))
