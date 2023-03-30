"""
Parsing each object key from robosuite object format (flat array... ew)
"""
import argparse

import numpy as np
from attrdict import AttrDict as d

from muse.envs.robosuite.robosuite_env import get_ordered_objects_from_arr
from muse.experiments import logger
from muse.utils.input_utils import query_string_from_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True, help='name of robosuite environment')
    parser.add_argument('--file', type=str, required=True, help='NPZ file to load from')
    parser.add_argument('--output_file', type=str, required=True, help='NPZ file to save to')
    args = parser.parse_args()

    logger.debug(f"Loading {args.file}...")
    dc = d.from_dict(dict(np.load(args.file, allow_pickle=True)))

    logger.debug(f"Parsing key object from dict.")

    # spoof the env as an AttrDict (this function only looks for env.name
    obj_arr = dc['object']
    dc.combine(get_ordered_objects_from_arr(d(name=args.env_name), obj_arr)[0])

    logger.debug(f"New shapes:")
    dc.leaf_shapes().pprint()

    do_save = query_string_from_set(f'Save to {args.output_file}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        logger.warn('Saving...')
        np.savez_compressed(args.output_file, **dc.as_dict())

    logger.debug("Done.")
