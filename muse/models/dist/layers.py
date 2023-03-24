import math
from typing import List

import numpy as np
import torch
from torch import nn as nn, distributions as D
from torch.nn import functional as F

from muse.experiments import logger
from muse.utils.torch_utils import unsqueeze_then_gather

from attrdict import AttrDict
from attrdict.utils import get_with_default, get_or_instantiate_cls


class DistributionCap(nn.Module):
    """
    Creates a distribution at the "output" of a network. This can be used as a sequential layer through layer params
    """

    def __init__(self, params: AttrDict):
        super().__init__()
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.out_name = get_with_default(params, "out_name", None)
        self.sample_out_name = get_with_default(params, "sample_out_name", None)
        self.use_argmax = get_with_default(params, "use_argmax", False)  # for sampling
        self.event_dim = get_with_default(params, "event_dim", 0)  # default distribution over last element

    def forward(self, x, **kwargs):
        return x

    def get_argmax(self, out_dist):
        raise NotImplementedError(str(__class__))

    def get_return(self, out_dist):
        # helper for dictionary based return
        if self.out_name is not None:
            out_dc = AttrDict.from_dict({self.out_name: out_dist})
            if self.sample_out_name is not None:
                if self.use_argmax:
                    sample = self.get_argmax(out_dist)
                else:
                    sample = out_dist.rsample() if out_dist.has_rsample else out_dist.sample()
                out_dc[self.sample_out_name] = sample
            return out_dc
        else:
            return out_dist


class MultiDistributionCap(nn.Module):
    """
    packages DistributionCaps at each specified name. (AttrDict packaging) -> TODO where does this apply?
    """

    def __init__(self, params: AttrDict):
        super().__init__()
        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.names, self.distributions = params.get_keys_required(['names', 'distributions'])
        self.output_names = get_with_default(params, "output_names", [n + "_dist" for n in self.names])

        assert len(self.names) == len(self.distributions), "Must pass in same number of distributions as inputs"

        self._map = {}
        # create or get handed each DistributionCap
        for name, dist in zip(self.names, self.distributions):
            if isinstance(dist, AttrDict):
                dist = dist.cls(dist.params)
            assert isinstance(dist, DistributionCap)
            self._map[name] = dist

    def forward(self, inputs: AttrDict, **kwargs):
        """
        :return: each input name, with mapped keys, as a distribution
        """
        inputs = inputs.copy()
        # make sure all inputs present
        assert inputs.has_leaf_keys(self.names), [inputs.list_leaf_keys(), self.names]
        # make sure not overriding
        assert not any(inputs.has_leaf_key(o) for o in self.output_names), [inputs.list_leaf_keys(), self.output_names]

        for oname, name in zip(self.output_names, self.names):
            inputs[oname] = self._map[name].forward(inputs[name])

        return inputs


class GaussianDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(GaussianDistributionCap, self)._init_params_to_attrs(params)
        self.use_log_sig = get_with_default(params, "use_log_sig", True)
        self.use_tanh_mean = get_with_default(params, "use_tanh_mean", False)
        self.clamp_with_tanh = get_with_default(params, "clamp_with_tanh", False)
        self.sig_min = get_with_default(params, "sig_min", 1e-6)
        self.sig_max = get_with_default(params, "sig_max", 1e6)

        # inferred
        self.log_sig_min = np.log(self.sig_min)
        self.log_sig_max = np.log(self.sig_max)

    def get_distribution(self, raw_mu, raw_sig):
        # subclasses might override, example gaussian
        if self.use_log_sig:
            if self.clamp_with_tanh:
                raw_sig = self.log_sig_min + 0.5 * (self.log_sig_max - self.log_sig_min) * (torch.tanh(raw_sig) + 1)
                raw_sig = raw_sig.exp()
            else:
                # continuous log_sigma, exponentiated (less stable than softplus)
                raw_sig = raw_sig.clamp(min=self.log_sig_min, max=self.log_sig_max).exp()
        else:
            # continuous sigma, converted to R+
            raw_sig = F.softplus(raw_sig).clamp(min=self.sig_min, max=self.sig_max)
        if self.use_tanh_mean:
            raw_mu = torch.tanh(raw_mu)
        dist = D.Normal(loc=raw_mu, scale=raw_sig)
        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)

        return dist

    def forward(self, x, **kwargs):
        m, s = torch.chunk(x, 2, -1)
        return self.get_return(self.get_distribution(m, s))

    def get_argmax(self, out_dist):
        return out_dist.mean


class SquashedGaussianDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(SquashedGaussianDistributionCap, self)._init_params_to_attrs(params)
        self.use_log_sig = get_with_default(params, "use_log_sig", True)
        self.clamp_with_tanh = get_with_default(params, "clamp_with_tanh", True)
        self.sig_min = get_with_default(params, "sig_min", 1e-6)
        self.sig_max = get_with_default(params, "sig_max", 1e6)

        # inferred
        self.log_sig_min = np.log(self.sig_min)
        self.log_sig_max = np.log(self.sig_max)

    def get_distribution(self, raw_mu, raw_sig):
        # subclasses might override, example gaussian
        if self.use_log_sig:
            if self.clamp_with_tanh:
                raw_sig = self.log_sig_min + 0.5 * (self.log_sig_max - self.log_sig_min) * (torch.tanh(raw_sig) + 1)
                raw_sig = raw_sig.exp()
            else:
                # continuous log_sigma, exponentiated (less stable than softplus)
                raw_sig = raw_sig.clamp(min=self.log_sig_min, max=self.log_sig_max).exp()
        else:
            # continuous sigma, converted to R+
            raw_sig = F.softplus(raw_sig).clamp(min=self.sig_min, max=self.sig_max)
        dist = SquashedNormal(loc=raw_mu, scale=raw_sig)
        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)
        return dist

    def forward(self, x, **kwargs):
        m, s = torch.chunk(x, 2, -1)
        return self.get_return(self.get_distribution(m, s))

    def get_argmax(self, out_dist):
        return out_dist.mean


class CategoricalDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(CategoricalDistributionCap, self)._init_params_to_attrs(params)
        self.num_bins = int(params["num_bins"])

    def forward(self, x, **kwargs):
        # x should be (..., num_bins)
        assert x.shape[-1] == self.num_bins

        dist = D.Categorical(logits=x)

        if self.event_dim:
            dist = D.Independent(dist, self.event_dim)

        return self.get_return(dist)

    def get_argmax(self, out_dist):
        if self.event_dim:
            out_dist = out_dist.base_dist
        return torch.argmax(out_dist.probs, dim=-1)


class MixedDistributionCap(DistributionCap):
    def _init_params_to_attrs(self, params):
        super(MixedDistributionCap, self)._init_params_to_attrs(params)
        self.num_mix = int(params["num_mix"])
        assert self.num_mix > 0
        self.split_dim = get_with_default(params, "split_dim", -1, map_fn=int)
        self.chunk_dim = get_with_default(params, "chunk_dim", -1, map_fn=int)
        self.is_categorical = get_with_default(params, "is_categorical", True)
        self.all_mix = []
        self.base_dist: DistributionCap = get_or_instantiate_cls(params, "base_dist", DistributionCap)

        # goes from (input tensor, output dist, Cat tensor) -> Distribution or
        self.combine_dists_fn = get_with_default(params, "combine_dists_fn", lambda ls_in, ls_out,
                                                                                    cat: MixedDistributionCap.default_combine_to_mixture_fn)

    def forward_i(self, x, i):
        return self.all_mix[i].forward(x)

    def forward(self, x, **kwargs):
        # each gets equal inputs
        if isinstance(x, List):
            assert len(x) == 2 and self.is_categorical, len(x)
            x, cat = x
        elif self.is_categorical:
            # TODO handle chunk dim here...
            assert self.chunk_dim % len(x.shape) == len(x.shape) - 1, "Not implemented cat w/ chunk_dim neq -1"
            cat = x[..., :self.num_mix]
            x = x[..., self.num_mix:]
        else:
            cat = torch.zeros(list(x.shape[:-1]) + [self.num_mix], device=x.device)

        cat = torch.distributions.Categorical(logits=cat)

        # split dim
        sh = list(x.shape)
        split_dim = self.split_dim % len(sh)
        new_shape = sh[:split_dim] + [self.num_mix, x.shape[-1] // self.num_mix] + sh[split_dim + 1:]
        x = x.view(new_shape)

        # forward
        out = self.base_dist.forward(x)
        out_dist = self.combine_dists_fn(x, out, cat)
        return self.get_return(out_dist)

    def get_argmax(self, out_dist):
        # maximum likelihood sample.
        mean = out_dist.component_distribution.mean  # .. x k x D
        _, max_idxs = torch.max(out_dist.mixture_distribution.probs, dim=-1)  # ..

        return unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D

    @staticmethod
    def default_combine_to_mixture_fn(ins, out_dist, cat):
        return D.MixtureSameFamily(mixture_distribution=cat, component_distribution=out_dist)


class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class AffineTransformNoLogDet(D.AffineTransform):
    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        if self.event_dim:
            shape = shape[:-self.event_dim]
        return torch.zeros(shape, dtype=x.dtype, device=x.device)


class SquashedNormal(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, out_low=np.array(-1.), out_high=np.array(1.), event_dim=0):
        self.loc = loc
        self.scale = scale

        self.base_dist = D.Normal(loc, scale)
        self.event_dim = event_dim
        # reinterpret last N as part of dist
        if self.event_dim > 0:
            self.base_dist = D.Independent(self.base_dist, event_dim)

        self.bound_low = torch.tensor(out_low, dtype=self.base_dist.mean.dtype, device=self.base_dist.mean.device)
        self.bound_high = torch.tensor(out_high, dtype=self.base_dist.mean.dtype, device=self.base_dist.mean.device)

        mid = (self.bound_low + self.bound_high) / 2.
        range = (self.bound_low - self.bound_high) / 2.

        transforms = [TanhTransform()]  # , D.AffineTransform(mid, range, event_dim)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    # def rsample(self, sample_shape=torch.Size()):
    #     out = super(SquashedNormal, self).rsample(sample_shape)
    #     return SquashedNormal.smooth_clamp(out)
    #
    # def sample(self, sample_shape=torch.Size()):
    #     out = super(SquashedNormal, self).sample(sample_shape)
    #     return SquashedNormal.smooth_clamp(out)
    #
    # @staticmethod
    # def smooth_clamp(out, beta=30):
    #
    #     # don't allow samples to be exactly 1. or -1. (for gradient stability through logprob, for example)
    #     clamp_upper = 1. - F.softplus(1 - out, beta=beta)
    #     clamp_lower = F.softplus(out + 1, beta=beta) - 1.
    #
    #     gez = (out > 0).float()
    #
    #     # double ended softplus to "clamp" smoothly
    #     return gez * clamp_upper + (1 - gez) * clamp_lower


class BestCategorical(D.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super(BestCategorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)


class SoftmaxMixtureSameFamily(D.MixtureSameFamily):
    """
    Same as mixture of same family but component dist is
    """
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None, temperature=1.):
        super(SoftmaxMixtureSameFamily, self).__init__(mixture_distribution, component_distribution,
                                                       validate_args=validate_args)
        # high temperature means more of a hard-max, only for log prob TODO extend this
        self._temp = temperature
        # logger.debug(f"Softmax init w/ log_prob temperature: {temperature}")

    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        # k best per batch item, TODO
        if self._temp > 0:
            log_best_prob = torch.log_softmax((1. / self._temp) * log_prob_x, dim=-1).detach()
        else:
            log_best_prob = 0
        # categorical weights TODO normalize properly?
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_best_prob + log_mix_prob, dim=-1)  # [S, B]

    def rsample(self, sample_shape=torch.Size(), sample_all=False):
        # rsampling a mixture doesn't work that well...
        if self.component_distribution.has_rsample:
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.rsample(sample_shape)
            if sample_all:
                return comp_samples

            # mixture samples [n, B, k]
            if isinstance(self.mixture_distribution, D.Categorical):
                mix_sample = F.gumbel_softmax(self.mixture_distribution.logits)
            else:
                logger.warn("Not backpropping sampling mixture!")
                mix_sample = self.mixture_distribution.sample(sample_shape)  # if not categorical..
                mix_sample = F.one_hot(mix_sample, num_classes=comp_samples.shape[-2])
            mix_shape = mix_sample.shape

            for i in range(len(mix_shape), len(comp_samples.shape)):
                mix_sample = mix_sample.unsqueeze(-1)

            # Gather along the k dimension
            # mix_sample_r = mix_sample.reshape(
            #     mix_shape + torch.Size([1] * (len(es) + 1)))
            # mix_sample_r = mix_sample_r.repeat(
            #     torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            # samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            # return samples.squeeze(gather_dim)

            # mixed sample via gumbel
            samples = (comp_samples * mix_sample).sum(gather_dim)
            return samples
        else:
            raise NotImplementedError

    def rsample_each(self, sample_shape=torch.Size()):
        return self.component_distribution.rsample(sample_shape)

    @property
    def mean(self):
        # maximum likelihood sample.
        mean = self.component_distribution.mean  # .. x k x D
        _, max_idxs = torch.max(self.mixture_distribution.probs, dim=-1)  # ..

        return unsqueeze_then_gather(mean, max_idxs, dim=len(max_idxs.shape))  # .. x D


class Logistic(D.TransformedDistribution):
    def __init__(self, loc, scale):
        self.base_dist = D.Uniform(torch.zeros_like(loc), torch.ones_like(scale))
        self.transforms = [D.SigmoidTransform().inv, D.AffineTransform(loc=loc, scale=scale)]
        super(Logistic, self).__init__(base_distribution=self.base_dist, transforms=self.transforms)


class DiscreteLogistic(D.Distribution):
    # Pixel CNN
    def __init__(self, loc, scale, quantized_dim, min=-1, max=1):
        """
        :param loc: element wise
        :param scale: element wise
        :param quantized_dim: number of bins per element
        :param min:
        :param max:
        """
        self.loc = loc
        self.scale = scale
        self.min = min
        self.max = max
        self.quantized_dim = quantized_dim
        self.half_range = 0.5 * (self.max - self.min)
        self.mid = 0.5 * (self.min + self.max)
        self.pm = 1. / (self.quantized_dim - 1)
        super(DiscreteLogistic, self).__init__(batch_shape=loc.shape)

    has_rsample = False

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale

    def log_prob(self, value):
        """

        :param value: tensor of shape (d0, ... di,)
        :return: log probs of each element in this tensor, according the scale and range, (d0, ... di,)
        """
        x = (value - self.mid) / self.half_range  # scale to -1 -> 1
        centered_x = x - self.loc

        inv_stdv = 1. / self.scale
        plus_in = inv_stdv * (centered_x + self.pm)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - self.pm)
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - self.scale.log() - 2. * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - np.log((self.quantized_dim - 1) / 2))
        inner_cond = (x > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (x < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        return log_probs

    def cdf(self, value):
        x = (value - self.mid) / self.half_range  # scale to -1 -> 1
        centered_x = value - self.loc

        inv_stdv = 1. / self.scale
        plus_in = inv_stdv * (centered_x + self.pm)
        min_in = inv_stdv * (centered_x - self.pm)
        outer_cond = (x > 0.999).float()
        inner_cond = (x <= 0.999).float()

        outer_cond * (1 - torch.sigmoid(min_in)) + inner_cond * (torch.sigmoid(plus_in))

        return torch.sigmoid(plus_in)

    # we sample from the underlying logistic
    def sample(self, sample_shape=torch.Size()):
        u = self.loc.data.new(sample_shape).uniform_(1e-5, 1.0 - 1e-5)
        x = self.loc + self.scale * (torch.log(u) - torch.log(1. - u))

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        # NOTE samples are continuous here
        return x * self.half_range + self.mid
