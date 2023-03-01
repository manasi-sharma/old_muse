from typing import List, Callable

import numpy as np
import torch
from attrdict import AttrDict
from torch import nn as nn

from muse.utils.torch_utils import view_unconcatenate, split_dim, combine_dims


class ExtractKeys(nn.Module):
    """ Extracts keys of given shapes from input flat array, or from a specific key in input dict."""

    def __init__(self, keys, shapes, from_key=None, postproc_fn=None):
        super(ExtractKeys, self).__init__()
        self.keys = keys
        self.shapes = shapes
        assert len(self.keys) == len(shapes)
        self.names_to_shapes = {k: s for k, s in zip(keys, shapes)}
        self.from_key = from_key
        self.postproc_fn = (lambda x: x) if postproc_fn is None else postproc_fn

    def forward(self, x):
        out = AttrDict()
        if self.from_key is not None:
            assert isinstance(x, AttrDict)
            out = x.leaf_copy()
            x = x[self.from_key]

        assert isinstance(x, torch.Tensor)
        return self.postproc_fn(out & view_unconcatenate(x, self.keys, self.names_to_shapes))


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def forward(self, x):
        return x.reshape([-1] + self.shape)


class SplitDim(nn.Module):
    def __init__(self, dim, new_shape):
        super(SplitDim, self).__init__()
        self.dim = dim
        self.new_shape = list(int(d) for d in new_shape)
        assert sum(int(d == -1) for d in self.new_shape) <= 1, "Only one dimension can be negative: %s" % self.new_shape
        self.idx, = np.nonzero([d == -1 for d in self.new_shape])

    def forward(self, x):
        new_sh = list(self.new_shape)
        if len(self.idx) > 0:
            new_sh[self.idx[0]] = 1
            new_sh[self.idx[0]] = int(x.shape[self.dim] / np.prod(new_sh))
        return split_dim(x, self.dim, new_sh)


class CombineDim(nn.Module):
    def __init__(self, dim, num_dims=2):
        super(CombineDim, self).__init__()
        self.dim = dim
        self.num_dims = num_dims

    def forward(self, x):
        return combine_dims(x, self.dim, self.num_dims)


class ListSelect(nn.Module):
    def __init__(self, list_index, dim=0):
        super(ListSelect, self).__init__()
        self.list_index = list_index
        self.dim = dim

    def forward(self, x):
        if self.dim == 0:
            return x[self.list_index]
        else:
            slice_obj = [slice(None) for _ in range(len(x.shape))]
            slice_obj[self.dim] = self.list_index
            return x[tuple(slice_obj)]


class ListConcat(nn.Module):
    def __init__(self, dim=0):
        super(ListConcat, self).__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, dim=self.dim)


class ListFromDim(nn.Module):
    def __init__(self, dim, split_size_or_sections=1, check_length=None):
        super(ListFromDim, self).__init__()
        self.dim = dim
        self.split_size_or_sections = split_size_or_sections
        self.check_length = check_length

    def forward(self, x):
        x_ls = torch.split(x, self.split_size_or_sections, dim=self.dim)
        if self.check_length and len(x_ls) != self.check_length:
            raise ValueError(
                f"For shape: {x.shape}, splits: {self.split_size_or_sections}. Expected len={self.check_length} but len={len(x_ls)}!")
        if self.split_size_or_sections == 1:  # special case, all elements will flatten since each chunk is size=1
            x_ls = [arr.squeeze(self.dim) for arr in x_ls]
        return x_ls


class Permute(nn.Module):
    def __init__(self, order, order_includes_batch=False, contiguous=True):
        super(Permute, self).__init__()
        self.order = list(order)
        self.order_includes_batch = order_includes_batch

        self.contiguous = contiguous

        self.full_ord = self.order if self.order_includes_batch else [0] + self.order
        assert len(np.unique(self.full_ord)) == len(self.full_ord) and np.amax(self.full_ord) < len(
            self.full_ord), self.full_ord  # all idxs are unique and within range

    def forward(self, x):
        px = x.permute(self.full_ord)
        return px.contiguous() if self.contiguous else px


class Functional(nn.Module):
    def __init__(self, func):
        super(Functional, self).__init__()
        self.func = func
        assert isinstance(func, Callable), "Requires a callable function as input"

    def forward(self, x, **kwargs):
        return self.func(x, **kwargs)


class Assert(Functional):
    def forward(self, x, **kwargs):
        cond, err_msg = self.func(x, **kwargs)
        assert cond, err_msg
        return x
