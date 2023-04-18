"""
Parsing each object key from robosuite object format (flat array... ew)
"""
import argparse

import numpy as np
from attrdict import AttrDict as d

from muse.experiments import logger
from muse.utils.input_utils import query_string_from_set
from muse.utils import transform_utils as T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='NPZ file to load everything from')
    parser.add_argument('--action_key', type=str, required=True, help='Action key to load from file')
    parser.add_argument('--start_dim', type=int, default=3, help='Where does the rotation start in action')
    parser.add_argument('--end_dim', type=int, default=6, help='Where does the rotation end in action')
    parser.add_argument('--rot_format', type=str, default='axisangle',
                        help='format of the rotation -> T.{format}2mat should exist')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to add to key. Default will overwrite existing key')
    parser.add_argument('--output_file', type=str, required=True, help='NPZ file to save to')
    args = parser.parse_args()

    logger.debug(f"Loading {args.file}...")
    dc = d.from_dict(dict(np.load(args.file, allow_pickle=True)))

    key = args.action_key

    assert key in dc.leaf_keys(), f"{key} is not in file!"
    action = dc[key]
    mat = getattr(T, f'{args.rot_format}2mat')(action[..., args.start_dim:args.end_dim])
    rot6d = T.mat2rot6d(mat)
    new_action = np.concatenate([action[..., :args.start_dim], rot6d, action[..., args.end_dim:]], axis=-1)
    if not args.suffix:
        logger.debug(f'Overwriting {key} with shape={action.shape} to shape={new_action.shape}')
    dc[f'{key}{args.suffix}'] = new_action

    logger.debug(f"New shapes:")
    dc.leaf_shapes().pprint()

    do_save = query_string_from_set(f'Save to {args.output_file}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        logger.warn('Saving...')
        np.savez_compressed(args.output_file, **dc.as_dict())

    logger.debug("Done.")
