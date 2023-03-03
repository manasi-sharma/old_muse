"""
This script takes robosuite configs
"""
import os
from argparse import ArgumentParser
from typing import List

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from muse.envs.robosuite.robosuite_env import RobosuiteEnv, modify_spec_prms
from muse.experiments import logger
from attrdict import AttrDict as d
from attrdict.utils import get_with_default

from convert_robosuite_hdf5_to_npz import zoo_remap, regular_remap, get_hdf5_leaf_names, parse_hdf5


def parse_episode(ep_name, episodic_names, nonspecified_required_keys, prefix, get_obs_extra=False, get_ac_extra=False):
    this_ep = d()

    for key in set(episodic_names).difference(nonspecified_required_keys):
        # load the array for this key and save it
        hd_name = os.path.join(prefix + ep_name, key)
        this_ep[key] = parse_hdf5(hd_name, node[hd_name])

    # fill in the others as needed
    ep_len = len(this_ep[episodic_names[0]])
    if get_obs_extra:
        this_ep['done'] = np.array([False] * ep_len, dtype=bool)
        this_ep['done'][-1] = True
        this_ep['rollout_timestep'] = np.arange(ep_len)
    if get_ac_extra:
        this_ep['policy_type'] = get_with_default(this_ep, 'policy_type', np.array([-1] * ep_len)[:, None])
        this_ep['policy_name'] = get_with_default(this_ep, 'policy_name',
                                                  np.array(["unknown" for _ in range(ep_len)],
                                                           dtype=object)[:,
                                                  None])
        this_ep['policy_switch'] = get_with_default(this_ep, 'policy_switch',
                                                    np.array([False] * ep_len, dtype=bool)[:, None])

    return this_ep


def check_episodic_hdf5(node, episodic_names, flat_names=None, prefix="/data/", ep_prefix="demo_", get_obs_extra=True,
                        get_ac_extra=True, integer_sort_episodes=True, mask=None):
    """
    Will get episodic_names, stacked, and flat_names, not stacked, from the given hdf5 node.

    :param node: the hdf5 node
    :param episodic_names: names that will be looked for under prefix+ep_prefix in node
    :param flat_names: names that are outside and do not need to be modified (shape is already correct in hdf5)
    :param prefix: the hdf5 prefix. must start with '/'. we only get keys under this prefix.
    :param ep_prefix: the episode prefix. the integer episode number will follow this prefix.
    :param get_obs_extra: if True, will also return 'done' and 'rollout_timestep' keys.
    :param get_ac_extra: if True, will also return 'policy_type', 'policy_name', and 'policy_switch' keys.
    :param integer_sort_episodes: if True, will sort episodes as integers, else by string
    """

    assert prefix.startswith("/"), "Hdf5 format does not start with /, pls add"
    assert len(episodic_names) > 0, "Provide some names!"

    # all the names present in hdf5
    all_hdf5_names = get_hdf5_leaf_names(node)
    depref_hdf5_names = [n[len(prefix):] for n in all_hdf5_names if n.startswith(prefix)]

    if flat_names is None:
        # if not specified, flat_names will be any name starting with prefix, but outside of the prefix+ep_prefix
        flat_names = [n for n in depref_hdf5_names if not n.startswith(ep_prefix)]

    assert len(set(flat_names).intersection(episodic_names)) == 0, \
        f"Overlapping flat/episodic names: {[flat_names, episodic_names]}"

    # placeholder
    zero_d = d.from_kvs(depref_hdf5_names, [0] * len(depref_hdf5_names))

    # find the episode names
    if integer_sort_episodes:
        sorted_episode_names = sorted([n for n in zero_d.keys() if n.startswith(ep_prefix)],
                                      key=lambda k: int(k[len(ep_prefix):]))
    else:
        sorted_episode_names = sorted([n for n in zero_d.keys() if n.startswith(ep_prefix)])

    if mask is not None:
        # mask determines which episode names to keep
        assert isinstance(mask, List)
        allowed_keys = []
        for m in list(mask):
            allowed_keys += [elem.decode("utf-8") for elem in np.array(node["mask/{}".format(m)][:])]
        # print(len(sorted_names) - len(allowed_keys))
        sorted_episode_names = [s for s in sorted_episode_names if s in allowed_keys]

    # print("SORTED NAMES:", sorted_names)
    assert len(sorted_episode_names) > 0, f"No names found to match ep prefix!: {ep_prefix}" \
                                          + (f", with masks: {mask}" if mask is not None else "")

    all_episode_keys = zero_d[sorted_episode_names[0]].list_leaf_keys()
    print(all_episode_keys)
    optional_keys = []
    if get_obs_extra:
        optional_keys.extend(['done', 'rollout_timestep'])
    if get_ac_extra:
        optional_keys.extend(['policy_type', 'policy_name', 'policy_switch'])

    nonspecified_required_keys = list(set(optional_keys).difference(all_episode_keys))  # nonspecified keys

    assert set(episodic_names).issubset(all_episode_keys + optional_keys), ["missing keys:", list(
        set(episodic_names).difference(all_episode_keys + optional_keys))]

    structure = d()

    for n in sorted_episode_names:
        this_ep = d()
        # for all the keys that we need to get AND can actually get...
        for key in set(episodic_names).difference(nonspecified_required_keys):
            this_ep[key] = None

        if get_obs_extra:
            this_ep['done'] = None
            this_ep['rollout_timestep'] = None
        if get_ac_extra:
            this_ep['policy_type'] = None
            this_ep['policy_name'] = None
            this_ep['policy_switch'] = None

        structure[n] = this_ep

    return structure, d(
        flat_names=flat_names,
        all_hdf5_names=all_hdf5_names,
        depref_hdf5_names=depref_hdf5_names,
        sorted_episode_names=sorted_episode_names,
        episodic_names=episodic_names,
        nonspecified_required_keys=nonspecified_required_keys,
        optional_keys=optional_keys,
    )


def robosuite_hdf5_to_native(node, save_file, obs_names, ac_names, out_obs_names, flat_names=None, prefix="/data/",
                             ep_prefix="demo_",
                             get_obs_extra=True, get_ac_extra=True, integer_sort_episodes=True, mask=None,
                             add_ee_ori=False,
                             lookup_names=regular_remap):
    """
    loads episodic obs / ac, but maps some weird naming conventions back.
    """

    mapped_obs_names = [lookup_names[n] if n in lookup_names.keys() else ('obs/' + n) for n in obs_names]
    mapped_ac_names = ac_names.copy()
    if "action" in mapped_ac_names:
        idx = mapped_ac_names.index("action")
        mapped_ac_names[idx] = "actions"  # stupid
    mapped_out_obs_names = ["rewards" if n == "reward" else 'next_obs/' + n for n in out_obs_names]

    mapped_episodic_names = mapped_obs_names + mapped_ac_names + mapped_out_obs_names

    # structure will have each episode with the keys that we can read from hdf5.
    structure, info = check_episodic_hdf5(node, mapped_episodic_names, flat_names=flat_names, prefix=prefix,
                                          ep_prefix=ep_prefix,
                                          get_obs_extra=get_obs_extra, get_ac_extra=get_ac_extra,
                                          integer_sort_episodes=integer_sort_episodes, mask=mask)

    if save_file is not None:
        data_shapes = d()
        with h5py.File(save_file, 'w') as out_node:
            for ep_name in info.sorted_episode_names:
                ep_group = out_node.create_group(prefix + ep_name)
                this_ep = parse_episode(ep_name, info.episodic_names, info.nonspecified_required_keys,
                                        prefix=prefix, get_obs_extra=get_obs_extra, get_ac_extra=get_ac_extra)

                # mapping back
                for mo, o in zip(mapped_episodic_names, obs_names + ac_names + out_obs_names):
                    ep_group.create_dataset(o, data=this_ep[mo])

                    if add_ee_ori and "_eef_quat" in o:
                        # add euler angles to episode (replace new key, not mapped one...)
                        ep_group.create_dataset(o.replace("_eef_quat", "_eef_eul"),
                                                data=Rotation.from_quat(this_ep[mo]).as_euler("xyz"))

                # add any non present optional keys
                for mo in info.optional_keys:
                    if mo in this_ep.leaf_keys() and mo not in mapped_episodic_names:
                        ep_group.create_dataset(mo, data=this_ep[mo])

                for key in ep_group.keys():
                    data_shapes[f"{ep_name}/{key}"] = ep_group[key].shape

        return data_shapes
    else:
        return structure


def robosuite_zoo_hdf5_to_native(node, save_file, obs_names, ac_names, out_obs_names, flat_names=None, prefix="/data/",
                                 ep_prefix="demo_", get_obs_extra=True, get_ac_extra=True, integer_sort_episodes=True,
                                 mask=None, add_ee_ori=False,
                                 lookup_names=zoo_remap):
    """
    very specific mapping, only some keys supported. no orientation...
    """

    # not including images
    supported_state_names = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_joint_pos', 'robot0_gripper_qpos']
    required_mapped_names = ['ee_states', 'ee_states', 'joint_states', 'gripper_states']
    map_n = dict(zip(supported_state_names, required_mapped_names))

    mapped_obs_names = []
    for name in obs_names:
        if name in map_n.keys():
            mapped_name = f"obs/{map_n[name]}"
            if mapped_name not in mapped_obs_names:
                mapped_obs_names.append(mapped_name)
        elif name in lookup_names.keys():
            mapped_obs_names.append(lookup_names[name])
        else:
            raise ValueError(f"Key: {name} is has no equivalent name in robosuite zoo!")

    assert len(ac_names) == 1 and ac_names[0] == 'action'
    mapped_ac_names = ['actions']

    mapped_out_obs_names = ["rewards" if n == "reward" else 'next_obs/' + n for n in out_obs_names]

    mapped_episodic_names = mapped_obs_names + mapped_ac_names + mapped_out_obs_names

    # mapped_episodic_names += ['obs/centernet_bbox_20']

    logger.debug(f"Looking for mapped names: {mapped_episodic_names}")

    structure, info = check_episodic_hdf5(node, mapped_episodic_names, flat_names=flat_names, prefix=prefix,
                                          ep_prefix=ep_prefix,
                                          get_obs_extra=get_obs_extra, get_ac_extra=get_ac_extra,
                                          integer_sort_episodes=integer_sort_episodes, mask=mask)

    if save_file is not None:
        data_shapes = d()
        with h5py.File(save_file, 'w') as out_node:
            for ep_name in info.sorted_episode_names:
                ep_group = out_node.create_group(prefix + ep_name)

                this_ep = parse_episode(ep_name, info.episodic_names, info.nonspecified_required_keys,
                                        prefix=prefix, get_obs_extra=get_obs_extra, get_ac_extra=get_ac_extra)

                # manually assigning keys since this format is weird
                if 'robot0_eef_pos' in obs_names:
                    ep_group.create_dataset('robot0_eef_pos', data=this_ep['obs/ee_states'][..., :3])
                if 'robot0_eef_quat' in obs_names:
                    ee_states = this_ep['obs/ee_states']
                    assert ee_states.shape[-1] == 7, "Cannot parse quat!"
                    ep_group.create_dataset('robot0_eef_pos', data=ee_states[..., :3])
                    if add_ee_ori:
                        # add euler angles to episode (replace new key, not mapped one...)
                        ep_group.create_dataset('robot0_eef_eul',
                                                data=Rotation.from_quat(ee_states[..., :3]).as_euler("xyz"))

                if 'robot0_gripper_qpos' in obs_names:
                    ep_group.create_dataset('robot0_gripper_qpos', data=this_ep['obs/gripper_states'])
                if 'robot0_joint_pos' in obs_names:
                    ep_group.create_dataset('robot0_joint_pos', data=this_ep['obs/joint_states'])

                # action
                ep_group.create_dataset('action', data=this_ep['actions'])

                # extra lookup names like images
                for n, mn in lookup_names.items():
                    if n in obs_names:
                        ep_group.create_dataset(n, data=this_ep[mn])

                # add any non present optional keys
                for mo in info.optional_keys:
                    if mo in this_ep.leaf_keys() and mo not in mapped_episodic_names:
                        ep_group.create_dataset(mo, data=this_ep[mo])

                for key in ep_group.keys():
                    data_shapes[f"{ep_name}/{key}"] = ep_group[key].shape

        return data_shapes
    else:
        return structure


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help='hdf5 file to load from')
    parser.add_argument('env_name', type=str,
                        help='what type of environment is this? [NutAssemblySquare, ToolHang] supported so far')
    parser.add_argument('--save_file', type=str, default=None, help='Optional hdf5 file to output to')
    parser.add_argument('--ep_prefix', type=str, default="demo_", help='Prefix for episodes')
    parser.add_argument('--mask', type=str, nargs="*", default=None,
                        help='Optional episode mask(s) (will look under mask/{}) by default')
    parser.add_argument('--add_ee_ori', action='store_true')
    parser.add_argument('--imgs', action='store_true')
    parser.add_argument('--ego_imgs', action='store_true')
    parser.add_argument('--zoo', action='store_true',
                        help='use the alternate spec that people behind robosuite zoo use.')
    args = parser.parse_args()

    # loading robosuite data
    file = args.file  # "data/robosuite/human_square_low_dim.hdf5", for example
    env_name = args.env_name  # "NutAssemblySquare", for example
    # file = "human_tool_hang_low_dim.hdf5"

    es_prms = RobosuiteEnv.get_default_env_spec_params(params=d(env_name=env_name, no_ori=args.zoo))
    es_prms = modify_spec_prms(es_prms, raw=True, minimal=args.zoo, no_object=args.zoo, no_reward=args.zoo)

    if args.zoo:
        args.imgs = True
        es_prms.observation_names.append('robot0_joint_pos')  # add in joints too
        assert not args.add_ee_ori, "Cannot add ee ori, zoo is a position only action space"
    if args.imgs:
        es_prms.observation_names.append('image')
    if args.ego_imgs:
        es_prms.observation_names.append('ego_image')
    env_spec = es_prms.cls(es_prms)

    img_key = 'obs/sideview_image' if env_name == 'ToolHang' else 'obs/agentview_image'
    regular_remap['image'] = img_key

    if args.mask is not None:
        logger.debug(f"Mask keys: {[f'mask/{m}' for m in args.mask]}")

    with h5py.File(file, 'r') as node:
        # will load
        if args.zoo:
            # add action names in after, these will be loaded too
            env_spec.action_names.extend(["policy_type", "policy_name", "policy_switch"])
            data = robosuite_zoo_hdf5_to_native(node, args.save_file, env_spec.observation_names, env_spec.action_names,
                                                env_spec.output_observation_names, mask=args.mask,
                                                lookup_names=zoo_remap, ep_prefix=args.ep_prefix,
                                                add_ee_ori=args.add_ee_ori)
        else:
            data_shapes = robosuite_hdf5_to_native(node, args.save_file, env_spec.observation_names,
                                                   env_spec.action_names,
                                                   env_spec.output_observation_names, mask=args.mask,
                                                   lookup_names=regular_remap, ep_prefix=args.ep_prefix,
                                                   add_ee_ori=args.add_ee_ori)

    logger.debug(f"Data shapes:\n{data_shapes.pprint(ret_string=True)}")

    logger.debug(f"Done loading")
