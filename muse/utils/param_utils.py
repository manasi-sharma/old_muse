from collections import OrderedDict

import torch.nn as nn
from typing import List, Mapping

from muse.models.pretrained.pretrained_models import custom_model_map, pretrained_model_map
from muse.models.dist.layers import GaussianDistributionCap, SoftmaxMixtureSameFamily
from attrdict import AttrDict as d
from muse.models.layers.supported import activation_map, layer_map, reshape_map, dist_cap_map
from muse.utils.torch_utils import BranchedModules, zero_weight_init

all_layer_map = {**layer_map, **activation_map, **reshape_map, **dist_cap_map,
                 **custom_model_map, **pretrained_model_map}

# dictionary for unique layer lookup
_initialized_layer_registry = {

}


# default layer
class LayerParams:
    def __init__(self, ltype, weight_init_fn=None, shared=False, **kwargs):
        # type is either...
        #  1. cls
        #  2. string
        #  3. callable w/ kwargs that returns model
        # shared = True --> to_module_list will return shared underlying module.
        self.type = ltype if (type(ltype) == type or callable(ltype)) else all_layer_map[ltype.lower()]
        self.weight_init_fn = weight_init_fn
        self.shared = shared
        # unique layer id is the object id
        self._id = id(self)
        self.kwargs = kwargs

    def to_module_list(self, **opt):
        if self.shared and self._id in _initialized_layer_registry.keys():
            return _initialized_layer_registry[self._id]
        try:
            # initialize
            mod = self.type(**self.kwargs)
        except Exception as e:
            raise type(e)(f"[{str(self.type)}]: {str(e)}")

        if self.weight_init_fn is not None:
            self.weight_init_fn(mod)
        if self.shared:
            _initialized_layer_registry[self._id] = mod
        return mod

    # has no effect if params already created
    def set_argument(self, name, val, allow_new=False):
        assert allow_new or name in self.kwargs.keys(), "Argument must be present in name: %s" % name
        self.kwargs[name] = val


class SequentialParams(LayerParams):
    def __init__(self, layer_params: List[LayerParams], **kwargs):
        self.params = layer_params
        self.length = len(layer_params)

    def to_module_list(self, as_sequential=True, **opt):
        block = []
        for i in range(self.length):
            block.append(self.params[i].to_module_list())

        if as_sequential:
            return nn.Sequential(*block)
        else:
            return block


class BranchedModuleParams(LayerParams):
    def __init__(self, layer_params: Mapping[str, LayerParams], cat_dim=None, split_dim=None,
                 split_sizes: List[int] = None):
        self.params = layer_params
        assert isinstance(layer_params, OrderedDict), "Branched Module Params requires an ordered dict of modules"
        self.order = list(layer_params.keys())
        self.length = len(self.order)
        self.cat_dim = cat_dim
        self.split_dim = split_dim
        self.split_sizes = split_sizes

    def to_module_list(self, as_sequential=True, **opt):
        modules = {}
        for k in self.params.keys():
            modules[k] = (self.params[k].to_module_list())

        if as_sequential:
            return BranchedModules(self.order, modules, cat_dim=self.cat_dim,
                                   split_dim=self.split_dim, split_sizes=self.split_sizes)
        else:
            return modules


def build_mlp_param_list(in_features: int, layer_out_sizes: List[int], activation='relu', out_bias=True, mask_chunks=1,
                         mask_layer_in_sizes=None, residual=None, dropout_p=0.):
    seq = []
    last_out = in_features

    if mask_chunks > 1 and mask_layer_in_sizes is None:
        mask_layer_in_sizes = [None for _ in range(len(layer_out_sizes))]

    if residual is None:
        residual = [False] * len(layer_out_sizes)

    for i, h in enumerate(layer_out_sizes):
        resid = residual[i]
        if mask_chunks > 1:
            in_mf = mask_layer_in_sizes[i]
            seq.append(LayerParams('mask_linear', num_chunks=mask_chunks, in_features=last_out, out_features=h,
                                   in_masked_features=in_mf, residual=resid))
        else:
            name = 'residual_linear' if resid else 'linear'
            seq.append(LayerParams(name, in_features=last_out, out_features=h))
        seq.append(LayerParams(activation))
        if dropout_p > 0:
            seq.append(LayerParams('dropout', p=dropout_p))
        last_out = h

    assert len(seq) > 0
    # remove last dropout & activation
    seq = seq[:-1 - int(dropout_p > 0)]
    if not out_bias:
        # will be linear, set to no bias for output layer
        seq[-1].set_argument('bias', False)

    return seq  # removed last activation


def get_dist_out_size(dim, prob=False, num_mix=1):
    if prob:
        if num_mix > 1:
            return (2 * dim + 1) * num_mix
        else:
            return 2 * dim
    else:
        return dim


def get_dist_cap(prob, use_tanh_out, num_mix=1, sig_min=1e-5, sig_max=1e5):
    if not prob:
        # deterministic
        return LayerParams('tanh') if use_tanh_out else LayerParams('empty')
    elif num_mix == 1:
        # Gaussian
        return LayerParams("gaussian_dist_cap",
                           params=d(use_log_sig=False, use_tanh_mean=use_tanh_out, event_dim=0,
                                    sig_min=sig_min, sig_max=sig_max))

    else:
        # GMM
        return LayerParams("mixed_dist_cap", params=d(
            num_mix=num_mix,
            combine_dists_fn=lambda ls_in, out, cat: SoftmaxMixtureSameFamily(mixture_distribution=cat,
                                                                              component_distribution=out,
                                                                              temperature=0),
            chunk_dim=-1,
            is_categorical=True,  # learn the mixture weights
            base_dist=d(
                cls=GaussianDistributionCap,
                params=d(use_log_sig=False, use_tanh_mean=use_tanh_out, event_dim=1,
                         sig_min=sig_min, sig_max=sig_max)
            ),
        ))


def add_policy_dist_cap(network: SequentialParams, num_mix, use_tanh_out, hidden_size,
                        policy_out_size, policy_sig_min, policy_sig_max):
    """
    Adds a linear map + gaussian dist cap to existing SequentialParams.
    """
    assert num_mix >=1, f"{num_mix} must be >= 1"

    # layer param list, except the last linear layer and optional tanh layer
    base_layers = network.params[:-1 - int(use_tanh_out)]
    cap = get_dist_cap(True, use_tanh_out, num_mix=num_mix, sig_min=policy_sig_min, sig_max=policy_sig_max)
    raw_out_size = get_dist_out_size(policy_out_size, prob=True, num_mix=num_mix)
    layers = base_layers + [
        LayerParams("linear", in_features=hidden_size, out_features=raw_out_size,
                    bias=True),
        cap,
    ]

    return SequentialParams(layer_params=layers)


def build_conv2d_param_list(in_channels: int, out_channels: List[int], kernel_sizes: List, strides: List,
                            paddings: List, activation='relu', dropout_p=0., truncate_last=True):
    seq = []
    last_out = in_channels
    assert len(kernel_sizes) == len(out_channels) == len(paddings) == len(strides)
    for h, k, s, p in zip(out_channels, kernel_sizes, strides, paddings):
        seq.append(LayerParams('conv2d', in_channels=last_out, out_channels=h, kernel_size=k, padding=p, stride=s))
        seq.append(LayerParams("batchnorm2d", num_features=h))
        seq.append(LayerParams(activation))
        if dropout_p > 0:
            seq.append(LayerParams("dropout2d", p=dropout_p))
        last_out = h

    assert len(seq) > 0
    # don't batchnorm, relu, or dropout on the last layer
    if truncate_last:
        seq = seq[:-2 - int(dropout_p > 0)]

    return seq


if __name__ == '__main__':
    all = [
        LayerParams("linear", in_features=30, out_features=50),
        LayerParams("relu"),
        LayerParams("linear", in_features=50, out_features=50),
        LayerParams("relu"),
        LayerParams("linear", in_features=50, out_features=10, weight_init_fn=zero_weight_init),
    ]

    final = SequentialParams(all).to_module_list(as_sequential=True)

    print(final)
    print(final[2].weight)  # nonzero
    print(final[4].weight)  # all zero
