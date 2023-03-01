"""
Extra utilities for the config module.
"""
import importlib.util
from typing import Tuple

import numpy as np

from muse.envs.env import Env
from muse.utils.file_utils import prepend_to_base_name
from muse.utils.python_utils import AttrDict as d

# TODO move this elsewhere
class Utils:
    """ Utils base class, each module should experiment some options here """
    pass


def bool_cond_add_to_exp_name(name, params, args_or_arg2abbrev, abbrevs=None, sep="_"):
    # args_or_arg2abbrev can be either [arg1....] (in which case abbrevs must be specified)
    #                               or [(arg1, abb1)...]
    N = len(args_or_arg2abbrev)
    assert N > 0, "empty args passed to bool_cond_add_to_exp_name()"
    if isinstance(args_or_arg2abbrev[0], str):
        assert abbrevs is not None and len(abbrevs) == N, "Abbreviations must be specified externally!"
        arg2abbrev = list(zip(args_or_arg2abbrev, abbrevs))
    else:
        arg2abbrev = args_or_arg2abbrev
        assert abbrevs is None, "Cannot specify abbreviations!"

    for key, abbrev in arg2abbrev:
        # checks if key is True
        if params >> key:
            name += f"{sep}{abbrev}"

    return name


##############

def default_process_env_step_output_fn(env: Env, obs: d, goal: d, next_obs: d, next_goal: d, policy_outputs: d,
                                       env_action: d, done: d) -> Tuple[d, d]:
    return next_obs, next_goal


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_dims_for_keys(nsld, names):
    dims = 0

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            dims += np.prod(shape)

    return dims


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_lims_for_keys(nsld, names):
    lims = {}

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            lims[name] = limits

    return [lims[name] for name in names]


# used to get model input dims flexibly from env spec precursor nlsd
def nsld_get_dtypes_for_keys(nsld, names):
    dtypes = []

    all_names = [tup[0] for tup in nsld]
    assert all([n in all_names for n in names]), "%s is not a subset of %s" % (names, all_names)

    for name, shape, limits, dtype in nsld:
        if name in names:
            dtypes.append(dtype)

    return dtypes


def nsld_get_names_to_shapes(nsld):
    nts = d()
    for n, s, _, _ in nsld:
        nts[n] = s
    return nts


def nsld_get_shape(nsld, name):
    return nsld_get_row(nsld, name)[1]


def nsld_get_row(nsld, name):
    for row in nsld:
        if row[0] == name:
            return row
    raise ValueError(name)
