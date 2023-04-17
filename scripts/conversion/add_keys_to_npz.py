"""
Parsing each object key from robosuite object format (flat array... ew)
"""
import argparse

import numpy as np
from attrdict import AttrDict as d

from muse.experiments import logger
from muse.utils.input_utils import query_string_from_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='NPZ file to load everything from')
    parser.add_argument('--keys_file', type=str, required=True, help='NPZ file to load keys from')
    parser.add_argument('--keys', type=str, nargs='+', required=True, help='Keys to load from keys_file')
    parser.add_argument('--output_file', type=str, required=True, help='NPZ file to save to')
    parser.add_argument('--no_overwrite', action='store_true', help='No overlap between keys and whats in file')
    args = parser.parse_args()

    logger.debug(f"Loading {args.file}...")
    dc = d.from_dict(dict(np.load(args.file, allow_pickle=True)))

    logger.debug(f"Loading {args.keys_file}")
    dc_with_keys = np.load(args.keys_file, allow_pickle=True)

    for key in args.keys:
        assert not args.no_overwrite or key not in dc.leaf_keys(), f"{key} is in file, but no_overwrite=True!"
        if key in dc.leaf_keys():
            logger.warn(f"Overwriting {key} in dc!")
        dc[key] = dc_with_keys[key]

    logger.debug(f"New shapes:")
    dc.leaf_shapes().pprint()

    do_save = query_string_from_set(f'Save to {args.output_file}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        logger.warn('Saving...')
        np.savez_compressed(args.output_file, **dc.as_dict())

    logger.debug("Done.")
