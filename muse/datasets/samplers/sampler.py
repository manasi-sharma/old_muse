from typing import Union, List

import numpy as np
from attrdict.utils import get_with_default

from muse.datasets.dataset import Dataset
from muse.experiments import logger


class Sampler:
    """
    Sampler returns indices for dataset(s).
    """
    def __init__(self, dataset: Union[Dataset, List[Dataset]], params):
        if isinstance(dataset, Dataset):
            self._ds_list = [dataset]
            self._ds = dataset
        else:
            self._ds_list = dataset
            self._ds = dataset[0]
        self._num_datasets = len(self._ds_list)

    def get_indices(self, **kwargs):
        return None


class SequentialSampler(Sampler):
    def __init__(self, dataset, params):
        super(SequentialSampler, self).__init__(dataset, params)
        assert self._num_datasets == 1, f"[{__class__}] not implemented to load from multiple datasets!"
        self._curr_idx = None
        self._bs = self._ds.batch_size
        self._shuffle = get_with_default(params, "shuffle", True)
        self._order = None
        self._num_batches = None
        self._sample_weights = params << "sample_weights"
        assert self._sample_weights is None or len(self._sample_weights) == len(self._ds)
        self._reset()

    def _reset(self):
        self._curr_idx = 0

        if len(self._ds) > self._bs:
            num_pad_to_order = self._bs - (len(self._ds) % self._bs)  # pad to be a multiple of batch_size
        else:
            num_pad_to_order = self._bs - len(self._ds)  # pad to have 1 batch

        if self._sample_weights is not None:
            if not self._shuffle:
                logger.debug("SequentialSampler: sampling always by weights, "
                             "but shuffle=False (which will be ignored)...")
            self._order = np.random.choice(len(self._ds), len(self._ds) + num_pad_to_order, replace=True, p=self._sample_weights)
        else:
            self._order = np.arange(len(self._ds))
            if num_pad_to_order > 0:
                # pad up to batch size with some random idxs.
                self._order = np.concatenate([self._order,
                                              np.random.choice(len(self._ds), num_pad_to_order, replace=False)])
            if self._shuffle:
                self._order = np.random.permutation(self._order)

        self._num_batches = len(self._order) // self._bs  # will be at least one

    def get_indices(self, **kwargs):
        # predetermined order
        if self._curr_idx >= self._num_batches:
            self._reset()
        indices = self._order[self._curr_idx * self._bs: (self._curr_idx + 1) * self._bs]
        assert len(indices) == self._bs, "Bug, this shouldn't happen"
        self._curr_idx += 1
        return indices


class WeightedSequentialSampler(SequentialSampler):
    """
    Sample weighted by some "mode", each mode will have its own class.
    """

    def __init__(self, dataset, params):
        super(WeightedSequentialSampler, self).__init__(dataset, params)
        self._mode_key = get_with_default(params, "mode_key", "mode")
        # for each value.
        self._num_modes = get_with_default(params, "num_modes", 2)
        # default uniform
        self._default_weights = get_with_default(params, "default_weights", np.ones(self._num_modes), map_fn=np.asarray)
        assert len(self._default_weights) == self._num_modes

        self._mtw = {
            'max': self.max_mode_to_weight,
            'first': self.nth_mode_to_weight,
            'last': lambda mode: self.nth_mode_to_weight(mode, n=-1),
        }
        self._default_mtw = get_with_default(params, "default_mtw", 'first')

        self._mode_to_weight_fn = get_with_default(params, "mode_to_weight_fn", self._mtw[self._default_mtw])

        assert self._mode_key in self._ds._env_spec.all_names

        # initial pass through the data to determine all weights (goes through get batch)
        all_weights = []
        for i in range(0, len(self._ds), self._bs):
            idxs = np.minimum(np.arange(i, i + self._bs), len(self._ds) - 1)
            res = self._ds.get_batch(indices=idxs, local_batch_size=len(idxs))
            mode = (res[0] & res[1])[self._mode_key]
            all_weights.append(self._mode_to_weight_fn(mode))

        self._sample_weights = np.concatenate(all_weights)[:len(self._ds)]  # remove the last extra things

        unique, counts = np.unique(self._sample_weights, return_counts=True)
        logger.debug(f"Before normalizing: sample weight -> min = {min(self._sample_weights)}, max = {max(self._sample_weights)}")
        logger.debug(f"  unique = {unique}, counts = {counts}")
        self._sample_weights = self._sample_weights / self._sample_weights.sum()
        self._reset()  # do this again now that we've updated weights

    def nth_mode_to_weight(self, mode, n=0):
        mode = mode.astype(np.long) % self._num_modes  # mode -> index, then wrap around in case its long
        if mode.size == self._bs:
            mode_idx = mode.reshape(self._bs)
        else:
            mode_idx = mode.reshape(self._bs, self._ds.horizon)[:, n]  # n'th element

        return self._default_weights[mode_idx]  # weight is 0 -> num_modes, shape (B,)

    def max_mode_to_weight(self, mode):
        mode = mode.astype(np.long) % self._num_modes  # mode -> index, then wrap around in case its long
        if mode.size == self._bs:
            mode_idx = mode.reshape(self._bs)
        else:
            mode_idx = mode.reshape(self._bs, self._ds.horizon).max(axis=-1)

        return self._default_weights[mode_idx]  # weight is 0 -> num_modes, shape (B,)
