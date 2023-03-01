import io
import queue
import threading
import warnings

import PIL
import PIL.Image
import numba
import numpy as np
from muse.experiments.file_manager import FileManager
from muse.utils.file_utils import file_path_with_default_dir
import torch
from numba import NumbaPendingDeprecationWarning

from muse.experiments import logger
from muse.utils.python_utils import timeit
from attrdict import AttrDict

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.dot(v1, v1) / np.dot(v2, v2))

def clip_norm(arr, norm, axis=None):
    return arr * np.minimum(norm / (np.linalg.norm(arr, axis=axis, keepdims=axis is not None) + 1e-11), 1)

def clip_scale(arr, scale_max):
    # scale down so all terms are less than max
    scale = np.maximum(np.max(np.abs(arr) / scale_max, axis=-1), 1)
    if len(arr.shape) > len(scale.shape):
        scale = scale[..., None]
    return arr / (scale + 1e-11)


# @numba.jit(cache=True, nopython=True, parallel=True)
def np_pad_sequence(list_of_arr):
    first = list_of_arr[0]
    pad_sizes = np.array([arr.shape[0] for arr in list_of_arr])
    max_size = int(np.max(pad_sizes))
    # extra_sizes = max_size - pad_sizes

    shape = (len(list_of_arr), max_size, *first.shape[1:])
    # for i in range(len(shape)):
    #     shape[i] = int(shape[i])
    # print(type(shape))
    cat = np.zeros(shape, dtype=first.dtype)
    # pads = [(0, 0) for _ in range(len(list_of_arr[0].shape))]
    for i, arr in enumerate(list_of_arr):
        # pads[0] = (0, extra)
        # assign to output, sequentially
        # slices.append(slice(arr.shape[0]))

        cat[i, :arr.shape[0]] = arr

    return cat


def np_add_to_buffer(buffer, new, max_len=0, end=True):
    assert list(buffer.shape[1:]) == list(new.shape), "Shapes do not match: %s should be %s" % (new.shape, buffer.shape[:1])
    max_len = buffer.shape[0] + 1 if max_len <= 0 else max_len
    if end:
        buffer = np.concatenate([buffer, new[None]], axis=0)[-max_len:]
    else:
        buffer = np.concatenate([new[None], buffer], axis=0)[:max_len]
    return buffer


# TODO try jit
def np_idx_range_between(start_idxs, end_idxs, length, spill_end_idxs=None, ret_idxs=True):
    # start, end, and spill_end are inclusive.
    # end is the last sampling start, spill end is the last idx that we can possible sample from (just in case |range| < L)
    if spill_end_idxs is None:
        spill_end_idxs = end_idxs

    assert length > 0, length
    #
    endp1_idxs = end_idxs + 1
    # should not be less than start_idxs (e.g, if 0, 49, L=30, last_start = 20)
    last_sample_start = np.maximum(endp1_idxs - length, start_idxs)
    sampled_start = np.random.randint(start_idxs, last_sample_start + 1)

    if ret_idxs:
        # N x L corresponding to sampled ranges, truncated if necessary.
        return np.minimum(sampled_start[:, None] + np.arange(length)[None], spill_end_idxs[:, None])
    else:
        sampled_end = np.minimum(sampled_start + length, spill_end_idxs)  # can't go past spill_end
        return sampled_start, sampled_end

# TODO implement fully
class DynamicNpArrayWrapper:
    def __init__(self, shape, capacity=0, dtype=np.float32):
        self._capacity = capacity
        self._size = 0
        self._data = np.empty((capacity, *list(shape)), dtype=dtype)

    def dynamic_add(self, x):
        if self._size == self._capacity:
            self._capacity *= 4
            newdata = np.zeros((self._capacity,))
            newdata[:self._size] = self._data
            self._data = newdata

        self._data[self._size] = x
        self._size += 1


def np_split_dataset_by_key(data: AttrDict, onetime_data: AttrDict, done_arr: np.ndarray, complete=True):
    assert len(done_arr.shape) == 1
    done_arr = done_arr.astype(bool)
    data.leaf_assert(lambda arr: arr.shape[0] == done_arr.shape[0])
    if not complete:
        last_true = np.argwhere(done_arr[::-1])[0]
        logger.debug(f"Cutting length {len(done_arr)} down to {last_true + 1} elements")
        done_arr = done_arr[:last_true + 1]
        data = data.leaf_modify(lambda arr: arr[:last_true + 1])

    assert done_arr[-1], "Last value must be true"
    last_idxs_per_ep = np.flatnonzero(done_arr)
    # boundaries for splitting
    splits = last_idxs_per_ep + 1
    onetime_data.leaf_assert(lambda arr: arr.shape[0] == len(splits))
    data_ep_tup = data.leaf_apply(lambda arr: np.split(arr, splits[:-1], axis=0))
    onetime_data_ep_tup = onetime_data.leaf_apply(lambda arr: np.split(arr, len(splits), axis=0))
    data_ep = []
    onetime_data_ep = []
    for ep in range(len(splits)):
        data_ep.append(data_ep_tup.leaf_apply(lambda vs: vs[ep]))
        onetime_data_ep.append(onetime_data_ep_tup.leaf_apply(lambda vs: vs[ep]))
    return splits, data_ep, onetime_data_ep


def np_split_arr_by_done(arr, done):
    assert len(arr) == len(done), [arr.shape, done.shape]
    ep_ends = done.nonzero()[0] + 1
    return np.split(arr, ep_ends[:-1])


if __name__ == '__main__':
    lengths = np.random.randint(20, 100, (1024,))
    sumH = lengths.sum()
    big_arr = np.zeros((sumH, 15))
    sequences = np.split(big_arr, np.cumsum(lengths)[:-1].astype(int), axis=0)

    timeit.reset()
    with timeit("loop"):
        for i in range(500):
            padded = np_pad_sequence(sequences)

    from torch.nn.utils.rnn import pad_sequence
    torch_arr = torch.from_numpy(big_arr)
    torch_seqs = torch.split_with_sizes(torch_arr, lengths.tolist())

    print(padded.shape)
    print(timeit)

    timeit.reset()
    with timeit("torch_loop"):
        for i in range(500):
            padded = pad_sequence(torch_seqs)

    print(padded.shape)
    print(timeit)


def np_load_from_files(files, load_keys, debug=True):
    
    data = []
    if load_keys is None:
        # keep track of the shared keys
        common_keys = None

    for f in files:
        path = file_path_with_default_dir(f, FileManager.base_dir, expand_user=True)
        if debug:
            logger.debug("File Path: %s" % path)
        load_obj = np.load(path, allow_pickle=True)
        # lazy loading
        if load_keys is None:
            new_data = AttrDict.from_dict(dict(load_obj))
            
            if common_keys is None:
                common_keys = set(new_data.list_leaf_keys())
            common_keys = common_keys.intersection(new_data.list_leaf_keys())
        else:
            new_data = AttrDict()
            for k in load_keys:
                new_data[k] = load_obj[k]
        # add the new data
        data.append(new_data)

    if load_keys is None:
        data = AttrDict.leaf_combine_and_apply([da > common_keys for da in data], np.concatenate)
    else:
        data = AttrDict.leaf_combine_and_apply(data, np.concatenate)
        
    return data


def limit_norm(vector: np.ndarray, limit, axis=-1):
    norms = np.linalg.norm(vector, axis=axis, keepdims=True)
    scaling = limit / (norms + 1e-11)

    return vector * np.minimum(scaling, 1.)


def line_circle_intersection(p1, p2, r, circle_center=np.zeros(2)):
    # line from p1 -> p2 intersections w/ circle at either 0, 1, or 2 points

    # recenter around circle center
    p1 = p1 - circle_center
    p2 = p2 - circle_center

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dr2 = dx**2 + dy**2
    D = p1[0]*p2[1] - p1[1]*p2[0]

    incidence = r ** 2 * dr2 - D ** 2
    if incidence < 0:
        return []
    elif incidence == 0:
        x = D * dy / dr2
        y = -D * dx / dr2
        return [np.array([x,y]) + circle_center]
    else:
        sgny = -1 if dy < 0 else 1
        x1 = (D * dy + sgny * dx * np.sqrt(incidence)) / dr2
        x2 = (D * dy - sgny * dx * np.sqrt(incidence)) / dr2
        y1 = (-D * dx + abs(dy) * np.sqrt(incidence)) / dr2
        y2 = (-D * dx - abs(dy) * np.sqrt(incidence)) / dr2
        return [np.array([x1, y1]) + circle_center,
                np.array([x2, y2]) + circle_center]
