import math

import h5py
import numpy as np
import torch
from attrdict import AttrDict
from attrdict.utils import get_with_default
from tqdm import tqdm

from muse.datasets.dataset import Dataset
from muse.experiments import logger
from muse.utils.general_utils import timeit
from muse.utils.torch_utils import broadcast_dims_np, pad_dims, split_dim_np


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

        self._cache = get_with_default(params, "cache", True)

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

        self._batch_names_to_get = get_with_default(params, "batch_names_to_get", self._env_spec.all_names)

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

            self._pointers = self._base_dataset._pointers
            self._dones = self._base_dataset._dones
            self._rollout_timesteps = self._rollout_timesteps
            self._ep_lens = self._ep_lens

        # parse a range of the dataset
        low, high = self._get_load_range()
        self._pointers = self._pointers[low: high]
        self._dones = self._dones[low: high]
        self._rollout_timesteps = self._rollout_timesteps[low: high]
        self._ep_lens = self._ep_lens[low: high]
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

        # self._cached_dicts = {}

        self._data_setup()

        self.temporary_names = []

        if self._cache:
            logger.debug("Caching all episodes...")

            self._cache_ins = []
            self._cache_outs = []
            for i in tqdm(range(int(math.ceil(self._num_valid_samples / self.batch_size)))):
                start, end = i*self.batch_size, min((i + 1)*self.batch_size, self._num_valid_samples)
                ins, outs = self.get_batch(np.arange(start, end), local_batch_size=end - start, cache=False)
                self._cache_ins.append(ins)
                self._cache_outs.append(outs)

            self._cache_ins = AttrDict.leaf_combine_and_apply(self._cache_ins, np.concatenate)
            self._cache_outs = AttrDict.leaf_combine_and_apply(self._cache_outs, np.concatenate)

            logger.debug("Done caching.")

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

    @staticmethod
    def parse_hdf5(key, value):
        new_value = np.array(value)
        #     if type(new_value[0]) == np.bytes_:
        #         new_value = np_utils.bytes2im(new_value)
        if new_value.dtype == np.float64:
            new_value = new_value.astype(np.float32)
        if len(new_value.shape) == 1:
            new_value = new_value[:, np.newaxis]
        return new_value

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
            if self._ep_prefix and any(n.startswith(self._prefix + self._ep_prefix) for n in all_hdf5_names):
                nodes = self._load_multi_episode_hdf5(f, self._prefix, self._ep_prefix)
            else:
                nodes = [f]

            hdf5_step_names = [s for s in self._step_names if s not in ['done', 'rollout_timestep']]

            # now check that all episodes are included, and filter the relevant keys, and compute sizes
            for i, ep in enumerate(nodes):
                hdf5_names = self._get_hdf5_leaf_names(ep)
                hdf5_names_no_root = [h[1:] for h in hdf5_names]
                # make sure all names are present
                for name in self._env_spec.names:
                    assert name in hdf5_names_no_root, f"Episode {i} from file={hdf5_fname} missing name {name}!"

                # make sure lengths match (or lengths are 1) between all keys for this ep
                hdf5_lens = np.array([len(ep[name]) for name in self._env_spec.names])
                ep_len = max(hdf5_lens)

                for key in self._onetime_names:
                    assert len(ep[key]) in [1, ep_len], \
                        f"Key {key} must have length 1 or {ep_len}, but was {len(ep[key])}"

                for key in hdf5_step_names:
                    assert len(ep[key]) == ep_len, \
                        f"Key {key} must have length {ep_len}, but was {len(ep[key])}"

                if hdf5_lens[0] == 0:
                    logger.warn('hdf5 lengths are 0, skipping!')
                    continue

                # add optional keys to node for later loading
                dones.append(self.parse_hdf5(self._done_key, ep[self._done_key])
                             if self._done_key in ep else ([False] * (ep_len - 1) + [True]))
                rollout_timesteps.append(self.parse_hdf5('rollout_timestep', ep['rollout_timestep'])
                                         if 'rollout_timestep' in ep else np.arange(ep_len))
                data_len += ep_len
                ep_lens.append(ep_len)

            pointers.extend(nodes)

        logger.debug('Dataset length: {}'.format(data_len))

        return pointers, dones, rollout_timesteps, ep_lens, data_len

    def get_episode(self, i, names, split=True, torch_device=None, indices=None, pad_horizon=False, **kwargs):
        # load ep if not already loaded
        if indices is None:
            indices = slice(None)

        new_dc = AttrDict()
        for n in names:

            with timeit('get_episode/access'):
                # TODO case for temp names
                if n == self._done_key:
                    arr = self._dones[i][indices]
                elif self._use_rollout_steps and n == 'rollout_timestep':
                    arr = self._rollout_timesteps[i][indices]
                else:
                    arr = self.parse_hdf5(n, self._pointers[i][n][indices])

            with timeit('get_episode/after'):
                if pad_horizon and arr.shape[0] < self.horizon:
                    arr = np.pad(arr, ((0, self.horizon - arr.shape[0]), (0, 0)), mode='edge')

                # truncate onetime names and broadcast
                if n in self._onetime_names:
                    if n in self._env_spec.final_names:
                        arr = arr[-1:]
                    else:
                        arr = arr[:1]
                    # match ep len shape
                    arr = broadcast_dims_np(arr, [0], [self.ep_lengths[i]])
                new_dc[n] = arr

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

    def load_batches(self, episodes, slices, names, use_unique=True):
        if use_unique:
            assert slices is None, "Not Implemented slicing and unique eps"
            unique = np.unique(episodes)
            # N x U
            idxs = np.argmax(episodes[:, None] == unique[None], axis=-1)
            unique_dc = [self.get_episode(ep, names, split=False) for ep in unique]
            return [unique_dc[i] for i in idxs]
        else:
            return [self.get_episode(ep, names, indices=slices[i], pad_horizon=True, split=False) for i, ep in enumerate(episodes)]

    def get_batch(self, indices=None, names=None, torch_device=None, min_idx=0, max_idx=0, local_batch_size=None, cache=True, **kwargs):
        if names is None:
            names = self._batch_names_to_get + [self._done_key] + \
                       (["rollout_timestep"] if self._use_rollout_steps else []) + \
                       self.temporary_names

        local_batch_size = self.batch_size if local_batch_size is None else local_batch_size

        if max_idx <= 0:
            max_idx += self._num_valid_samples
        assert 0 <= min_idx < max_idx <= self._num_valid_samples

        # now indices will refer to the indexable **sample sequences** not the episodes.
        if indices is None:
            # equally ranked sequences
            # NOTE: this is slow, you should use dataset.sampler instead to generate indices
            indices = np.random.choice(max_idx - min_idx, local_batch_size, replace=max_idx - min_idx < local_batch_size)
        else:
            assert len(indices) == local_batch_size, [local_batch_size, len(indices), indices]

        indices = self.idx_map_to_real_idx(indices)

        with timeit('get_batch/index'):
            if self._cache and cache:
                inputs = self._cache_ins.leaf_apply(lambda arr: arr[indices])
                outputs = self._cache_outs.leaf_apply(lambda arr: arr[indices])
            else:
                episode_indices = self._seq_idx_to_ep_idx[indices]
                # produce B x H indices, within each episode, and clip to not overflow past ep end.
                ep_lens = self.ep_lengths[episode_indices]
                # get the start index within the batch of episodes (since we load the whole ep)
                ep_starts = np.cumsum(np.concatenate([[0], ep_lens[:-1]]))
                range_start = self._seq_idx_to_start_in_ep_idx[indices] + ep_starts
                # range_end = np.minimum(range_start + self.horizon, ep_lens)
                max_ep_indices = ep_starts + ep_lens - 1
                indices = np.minimum(range_start[:, None] + np.arange(self._horizon)[None], max_ep_indices[:, None])
                # slices = [slice(rs, re) for rs, re in zip(range_start, range_end)]

                dcs = self.load_batches(episode_indices, None, names)
                cat_dcs = AttrDict.leaf_combine_and_apply(dcs, np.concatenate)

                # index
                flat_indices = indices.reshape(-1)
                # each is (B x H x ...)
                batch = cat_dcs.leaf_apply(lambda arr: split_dim_np(arr[flat_indices], 0, list(indices.shape)))

                inputs, outputs = self.split_into_inout(batch)

        with timeit('get_batch/to_torch'):
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
        return len(self._ep_lens)

    def episode_length(self, i):
        return self.ep_lengths[i]

    def period_length(self, i):
        return self.episode_length(i)
