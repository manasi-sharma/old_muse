import math

import h5py
import numpy as np
import torch
from attrdict.utils import get_with_default

from muse.datasets.dataset import Dataset
from muse.experiments import logger
from muse.utils.python_utils import AttrDict
from muse.utils.torch_utils import broadcast_dims_np


class Hdf5Dataset(Dataset):
    """
    Loads from a sequence of .hdf5 files

    Each file can follow one of these structure:

    (1) Multiple episodes

    - {prefix}{ep_prefix}1
        - key1
        - key2
        ...
    - {prefix}{ep_prefix}2
        ...
    ...

    (2) Single episode
    - key1
    - key2
    ...

    """

    def _init_params_to_attrs(self, params):
        super(Hdf5Dataset, self)._init_params_to_attrs(params)
        # None if we are starting from scratch
        self._input_files = params << "file"
        # multi episode structure parameters
        self._prefix = get_with_default(params, 'prefix', "/data/")
        self._ep_prefix = get_with_default(params, 'ep_prefix', "demo_")
        self._pad_end_sequence = get_with_default(params, "pad_end_sequence", True)

        self._cache = get_with_default(params, "cache", False)

        # will use base_dataset to get the whole datadict.
        self._load_from_base = get_with_default(params, "load_from_base", False)
        self._load_episode_range = params << "load_episode_range"
        self._min_frac = 0  # round up, inclusive
        self._max_frac = 1  # round up, exclusive
        if self._load_episode_range is not None:
            assert len(self._load_episode_range) == 2, self._load_episode_range
            assert all(1. >= r >= 0. for r in self._load_episode_range), self._load_episode_range
            self._min_frac, self._max_frac = self._load_episode_range
            assert self._min_frac < self._max_frac, f"{self._min_frac} must be less than {self._max_frac}"

        # TODO: effective horizon for skipping elements
        self.effective_horizon = self.horizon

    def _get_load_range(self):
        # keeping only some range of the dataset
        if self._load_episode_range is not None:
            ep_lower = int(math.ceil(self._min_frac * self.get_num_episodes()))
            ep_upper = int(math.ceil(self._max_frac * self.get_num_episodes()))
            logger.warn(
                f"Dataset keeping range [{ep_lower}, {ep_upper}), or {ep_upper - ep_lower} / {self.get_num_episodes()} episodes.")
            return ep_lower, ep_upper
        else:
            return 0, self.get_num_episodes()

    def _init_setup(self):
        if not self._load_from_base:
            self._pointers, self._dones, self._rollout_timesteps, self._ep_lens, self._data_len = self._load_hdf5s()
        else:
            logger.debug("Loading from base dataset...")
            # if loading from base,
            assert isinstance(self._base_dataset, self.__class__), "Base dataset is not the right class!"
            low, high = self._get_load_range()
            self._pointers = self._base_dataset._pointers[low:high]
            self._dones = self._base_dataset._dones[low:high]
            self._rollout_timesteps = self._rollout_timesteps[low:high]
            self._ep_lens = self._ep_lens[low:high]
            self._data_len = sum(self._ep_lens)

        # per episode, where does it start, how long is it, and where is the last position we can start.
        self.ep_starts = np.concatenate([[0], self._ep_lens[:-1]])
        self.ep_lengths = np.array(self._ep_lens)
        logger.debug(
            'Dataset Ep Lengths: min {}, max {}, med {}'.format(np.min(self.ep_lengths), np.max(self.ep_lengths),
                                                                np.median(self.ep_lengths)))

        # first startable is always 0
        # last startable is either L-1 (if padding the end) or L-1-H
        self.last_startable = np.maximum(
            self.ep_lengths - 1 - int(not self._pad_end_sequence) * self.effective_horizon, 0)

        self._cached_dicts = {}

        self._data_setup()

    def _data_setup(self):
        # static sampling likelihoods
        self._period_lengths = np.array([self.period_length(i) for i in range(self.get_num_periods())])

        # if padding end, use period_lengths, else period_lengths - H : for sampling padding.
        self._sampling_period_lengths = np.maximum(
            self._period_lengths - int(not self._pad_end_sequence) * self.effective_horizon, 1).astype(np.int64)
        self._num_valid_samples = sum(self._sampling_period_lengths)

        # number of valid samples, sample proportional to period length
        self._period_probs = 1 / self._sampling_period_lengths
        self._period_probs /= self._period_probs.sum()

        # create map from sequence idx -> episode
        self._seq_idx_to_ep_idx = np.repeat(np.arange(len(self._sampling_period_lengths)),
                                            self._sampling_period_lengths)
        self._seq_idx_to_start_in_ep_idx = np.concatenate([np.arange(l) for l in self._sampling_period_lengths])
        period_starts = np.concatenate([[0] + np.cumsum(self._period_lengths)[:-1]]) if len(
            self._period_lengths) > 1 else [0]
        self._seq_idx_to_true_idx = np.concatenate(
            [ps + np.arange(l) for l, ps in zip(self._sampling_period_lengths, period_starts)])

    # getting the indices for a chunk from episode i (human ordering)
    def batch_sample_indices_within_episode(self, indices):
        relevant_ep_lengths = self._ep_lens[indices]
        relevant_ep_last_starts = self.last_startable[indices]
        random_start_offset = np.random.randint(0, relevant_ep_last_starts + 1)

        # start = relevant_ep_starts + random_start_offset
        # horizon forward (B x H) -> list of BxH, make sure they don't go outside ep boundary
        within_ep_idxs = random_start_offset[:, None] + np.arange(self.horizon)[None] * 1

        # if self._allow_padding and np.any(relevant_ep_lengths < self.horizon):
        #     raise NotImplementedError("need to implement padding the sequences")

        bounded_ep_idxs = np.minimum(within_ep_idxs, (relevant_ep_lengths - 1)[:, None])
        indices = bounded_ep_idxs

        # how long each sample really is
        chunk_lengths = np.minimum(relevant_ep_lengths, self.effective_horizon)

        return indices.reshape(-1), chunk_lengths

    def _get_hdf5_leaf_names(self, node, name=''):
        if isinstance(node, h5py.Dataset):
            return [name]
        else:
            names = []
            for child_name, child in node.items():
                names += self._get_hdf5_leaf_names(child, name=name+'/'+child_name)
            return names

    @staticmethod
    def _load_multi_episode_hdf5(node, prefix, ep_prefix):
        node = node[prefix]
        episodes = np.array([k for k in node.keys() if k.startswith(ep_prefix)])
        episode_num = np.array([int(k[len(ep_prefix):]) for k in episodes])
        order = np.argsort(episode_num)

        episodes = episodes[order]
        # episode_num = episode_num[order]

        assert len(episodes) > 0, f"Hdf5 node has no episodes of prefix={prefix}, ep_prefix={ep_prefix}"

        out_nodes = []
        for ep_name in episodes:
            out_nodes.append(node[ep_name])

        return out_nodes

    def _load_hdf5s(self):
        self._input_files = [self._input_files] if isinstance(self._input_files, str) else self._input_files

        # initialize to empty lists
        pointers = []
        dones = []
        rollout_timesteps = []
        data_len = 0
        ep_lens = []

        # concatenate each h5py
        for hdf5_fname in self._input_files:
            # load it now (read only)
            logger.debug('Loading ' + hdf5_fname)
            f = h5py.File(hdf5_fname, 'r')

            # get all names in the hdf5 dataset
            all_hdf5_names = self._get_hdf5_leaf_names(f)

            # option to load multiple episodes, prefixed by {ep_prefix}
            if self._ep_prefix and any(n.startswith(self._ep_prefix) for n in all_hdf5_names):
                nodes = self._load_multi_episode_hdf5(f, self._prefix, self._ep_prefix)
            else:
                nodes = [f]

            # now check that all episodes are included, and filter the relevant keys, and compute sizes
            for i, ep in enumerate(nodes):
                hdf5_names = self._get_hdf5_leaf_names(ep)
                # make sure all names are present
                for name in self._env_spec.names:
                    assert name in hdf5_names, f"Episode {i} from file={hdf5_fname} missing name {name}!"

                # make sure lengths match (or lengths are 1) between all keys for this ep
                hdf5_lens = np.array([len(ep[name]) for name in self._env_spec.names])
                ep_len = max(hdf5_lens[0])

                for key in self._onetime_names:
                    assert len(ep[key]) in [1, ep_len], \
                        f"Key {key} must have length 1 or {ep_len}, but was {len(ep[key])}"

                for key in self._step_names:
                    assert len(ep[key]) == ep_len, \
                        f"Key {key} must have length {ep_len}, but was {len(ep[key])}"

                if hdf5_lens[0] == 0:
                    logger.warn('hdf5 lengths are 0, skipping!')
                    continue

                # add optional keys to node for later loading
                dones.append(ep['done'] if 'done' in ep else ([False] * (ep_len - 1) + [True]))
                rollout_timesteps.append(ep['rollout_timestep'] if 'rollout_timestep' in ep else np.arange(ep_len))
                data_len += ep_len
                ep_lens.append(ep_len)

            pointers.extend(nodes)

        logger.debug('Dataset length: {}'.format(data_len))

        return pointers, dones, rollout_timesteps, ep_lens, data_len

    def get_episode(self, i, names, split=True, torch_device=None, **kwargs):

        if i in self._cached_dicts:
            return self._cached_dicts[i]

        # load ep if not already loaded
        new_dc = AttrDict()
        for n in names:
            arr = self._pointers[i][n]
            # truncate onetime names and broadcast
            if n in self._onetime_names:
                if n in self._env_spec.final_names:
                    arr = arr[-1:]
                else:
                    arr = arr[:1]
                # match ep len shape
                arr = broadcast_dims_np(arr, [0], [self.ep_lengths[i]])
            new_dc[n].append(arr)

        if self._cache:
            self._cached_dicts[i] = new_dc

        if split:
            all_ds = self.split_into_inout(new_dc)
        else:
            all_ds = (new_dc,)

        if torch_device is not None:
            for dc in all_ds:
                if self._is_torch:
                    dc.leaf_modify(lambda x: x.to(torch_device))
                else:
                    dc.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return all_ds[0] if len(all_ds) == 1 else tuple(all_ds)

    def load_episodes(self, episodes, names):

        unique, idxs = np.unique(episodes, return_index=True)

        # load the arrays from scratch if we are not scratching

        unique_dcs = [self.get_episode(ep, names, split=False) for ep in unique]
        return [unique_dcs[i] for i in idxs]

    def get_batch(self, indices=None, names=None, torch_device=None, min_idx=0, max_idx=0, local_batch_size=None, **kwargs):
        if names is None:
            names = self._env_spec.all_names

        local_batch_size = self.batch_size if local_batch_size is None else local_batch_size

        if max_idx <= 0:
            max_idx += self._num_valid_samples
        assert 0 <= min_idx < max_idx <= self._num_valid_samples

        # now indices will refer to the indexable **sample sequences** not the episodes.
        if indices is None:
            # equally ranked sequences
            indices = np.random.choice(max_idx - min_idx, local_batch_size, replace=max_idx - min_idx < local_batch_size)
        else:
            assert len(indices) == local_batch_size, [local_batch_size, len(indices), indices]

        indices = self.idx_map_to_real_idx(indices)

        # get the episodes from file (or cached)
        episode_indices = self._seq_idx_to_ep_idx[indices]
        relevant_eps = self.load_episodes(episode_indices, names)

        # get the (B*H) indices we will load
        true_indices, _ = self.batch_sample_indices_within_episode(indices)
        true_indices = true_indices.reshape(len(indices), self.horizon)

        dcs = []
        for ep_dc, seq in zip(relevant_eps, true_indices):
            # get the indices within each episode
            dcs.append(ep_dc.leaf_apply(lambda arr: arr[seq]))

        inputs, outputs = self.split_into_inout(AttrDict.leaf_combine_and_apply(dcs, np.concatenate))

        if torch_device is not None:
            for d in (inputs, outputs):
                d.leaf_modify(lambda x: torch.from_numpy(x).to(device=torch_device))

        return inputs, outputs

    def __len__(self):
        return self._data_len

    def idx_map_to_real_idx(self, idxs):
        return idxs

    def get(self, name, idxs):
        pass

    def set(self, name, idxs, values):
        pass

    def add(self, inputs, outputs, **kwargs):
        pass

    def add_episode(self, inputs, outputs, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_statistics(self, names):
        """
        This function will load all of the stat names into memory,
        so make sure to not compute this for big things...

        Parameters
        ----------
        names

        Returns
        -------

        """
        all_eps = self.load_episodes(list(range(self.get_num_episodes())), names)
        datadict = AttrDict.leaf_combine_and_apply(all_eps, np.concatenate)

        means = datadict.leaf_apply(lambda arr: np.mean(arr, axis=0))
        means.leaf_modify(lambda arr: np.where(np.isnan(arr), 0, arr))

        stds = datadict.leaf_apply(lambda arr: np.std(arr, axis=0))
        # stdev should not be zero or inf
        stds.leaf_modify(
            lambda arr: np.where(np.logical_or(np.isnan(arr), arr == 0), 1, arr))

        mins = datadict.leaf_apply(lambda arr: np.min(arr, axis=0))
        maxs = datadict.leaf_apply(lambda arr: np.max(arr, axis=0))

        out = AttrDict(mean=means, std=stds, min=mins, max=maxs)
        return out

    def get_num_episodes(self):
        return len(self.ep_lengths)

    def episode_length(self, i):
        return self.ep_lengths[i]

