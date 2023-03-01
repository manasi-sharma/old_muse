import numpy as np
import torch
from torch import distributions as D
from torch.nn import functional as F

from muse.models.dist.layers import SquashedNormal, SoftmaxMixtureSameFamily
from muse.utils.general_utils import is_array
from muse.utils.torch_utils import split_dim, torch_clip

from attrdict import AttrDict


def detach_normal(dist):
    assert isinstance(dist, D.Normal) or (isinstance(dist, D.Independent) and isinstance(dist.base_dist, D.Normal))
    event_dim = 0 if isinstance(dist, D.Normal) else dist.reinterpreted_batch_ndims
    bd = dist if isinstance(dist, D.Normal) else dist.base_dist
    new_loc = bd.loc.detach()
    new_scale = bd.scale.detach()
    new_normal = D.Normal(new_loc, new_scale)
    if event_dim > 0:
        new_normal = D.Independent(new_normal, event_dim)
    return new_normal


def upper_kl_normal_softmaxmixnormal(p, q, temp=None, min_weight=0):
    # p will have (... x N), q will be (..., M x N)
    qc = q.component_distribution
    qm = q.mixture_distribution
    if isinstance(qc, D.Independent):
        qc = qc.base_dist
    assert isinstance(qc, D.Normal)
    all_var_ratio = (p.scale[..., None, :] / qc.scale).pow(2)
    all_t1 = ((p.loc[..., None, :] - qc.loc) / qc.scale).pow(2)
    all_kl = 0.5 * (all_var_ratio + all_t1 - 1 - all_var_ratio.log())
    # all_kl = []
    # for j in range(qc.batch_shape[-2]):
    #     loc = qc.loc[..., j, :]
    #     scale = qc.scale[..., j, :]
    #     assert list(loc.shape) == list(p.loc.shape)
    #     var_ratio = (p.scale / scale).pow(2)
    #     t1 = ((p.loc - loc) / scale).pow(2)
    #     kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    #     all_kl.append(kl)
    # mix of the kl (... x M x N)
    # all_kl = torch.stack(all_kl, dim=-2)
    each_kl_avg = all_kl.mean(-1)  # KL mean over rightmost dimensions used to estimate the "best" distribution (of M)
    if temp is not None or q._temp > 0:
        temp = q._temp if temp is None else temp

    if temp is not None and temp < np.inf:
        # weight each KL as a probability.
        each_kl_avg_normalized = each_kl_avg / each_kl_avg.sum(-1, keepdim=True)
        log_softmin_each_kl = torch.log_softmax(- 1 / temp * each_kl_avg_normalized.detach().log(), dim=-1)  # soft min on KL
    else:
        log_softmin_each_kl = 0

    # (..., M)
    log_alphas = qm.logits + log_softmin_each_kl

    if temp == np.inf:
        # pick the best kl, match component dist to that. then match mixture to the idx of the best kl.
        best_kl, best_kl_idxs = each_kl_avg.min(-1)
        if min_weight > 0:
            mask = F.one_hot(best_kl_idxs, num_classes=each_kl_avg.shape[-1]).to(dtype=best_kl.dtype)
            mask = torch.where(mask < min_weight, min_weight, mask)
            best_kl = (each_kl_avg * mask).sum(-1)
        # (B, ) logit loss
        xe_logits = F.cross_entropy(qm.logits, best_kl_idxs, reduction='none')
        # best_all_kl = unsqueeze_then_gather(all_kl, best_kl_idxs, dim=len(best_kl_idxs))
        # (B, ) min dist loss
        return best_kl + xe_logits

    # normalize to sum to 1
    log_alphas = log_alphas - torch.logsumexp(log_alphas, -1, keepdim=True)
    return (log_alphas.exp().unsqueeze(-1) * all_kl).sum(-2)


def upper_kl_normal_softmaxmix_indnormal(p, q, temp=None):
    # p will have (... x N), q will be (..., M x N)
    assert isinstance(p.base_dist, D.Normal)
    if temp is None or temp < np.inf:
        return upper_kl_normal_softmaxmixnormal(p.base_dist, q, temp=temp).sum(-1)
    else:
        return upper_kl_normal_softmaxmixnormal(p.base_dist, q, temp=temp)  # return is (B,)


def get_squashed_normal_dist_from_mu_sigma_tensor_fn(out_low, out_high, event_dim=0):
    def squashed_normal_dist_from_mu_sigma_tensor_fn(key, raw, mu, log_std):
        return SquashedNormal(loc=mu, scale=F.softplus(log_std), out_low=out_low, out_high=out_high,
                              event_dim=event_dim)

    return squashed_normal_dist_from_mu_sigma_tensor_fn


def get_sgmm_postproc_fn(num_mixtures, names_in, names_out, act_lows, act_highs,
                         log_std_min=-5., log_std_max=2., logit_min=-4., logit_max=4.):
    assert not isinstance(names_in, str) and not isinstance(names_out, str)
    assert isinstance(num_mixtures, int) and num_mixtures > 0, num_mixtures
    zipped = list(zip(names_in, names_out, act_lows, act_highs))

    log_std_max = torch.tensor(log_std_max, dtype=torch.float32)
    log_std_min = torch.tensor(log_std_min, dtype=torch.float32)

    logit_max = torch.tensor(logit_max, dtype=torch.float32)
    logit_min = torch.tensor(logit_min, dtype=torch.float32)

    def sgmm_postproc_fn(inputs, model_output):
        result = AttrDict()
        for nin, nout, act_low, act_high in zipped:
            assert model_output.has_leaf_key(nin), nin
            ldim = (model_output[nin].shape[-1] - num_mixtures) // 2  # first num_mixtures are the weights (categorical)
            logits, mu, log_std = torch.split(model_output[nin], [num_mixtures, ldim, ldim], -1)
            mu = split_dim(mu, -1, (num_mixtures, ldim // num_mixtures))
            log_std = split_dim(log_std, -1, (num_mixtures, ldim // num_mixtures))

            mix = D.Categorical(logits=torch.clamp(logits, logit_min, logit_max))  # (..., num_mix)
            comp = SquashedNormal(mu, F.softplus(torch_clip(log_std, log_std_min.to(log_std.device),
                                                            log_std_max.to(log_std.device))),
                                  act_low, act_high, event_dim=1)  # (..., num_mix, dim)
            gmm = D.MixtureSameFamily(mix, comp)
            result[nout] = gmm
        return result

    return sgmm_postproc_fn


def get_dist_first_horizon(arr):
    if is_array(arr):
        return arr[:, 0]
    elif isinstance(arr, torch.distributions.Distribution):
        if isinstance(arr, D.Normal):
            return D.Normal(arr.loc[:, 0], arr.scale[:, 0])
        elif isinstance(arr, D.Categorical):
            return D.Categorical(arr.probs[:, 0])
        elif isinstance(arr, D.Independent):
            return D.Independent(get_dist_first_horizon(arr.base_dist), arr.reinterpreted_batch_ndims)
        elif isinstance(arr, SoftmaxMixtureSameFamily):
            return SoftmaxMixtureSameFamily(get_dist_first_horizon(arr.mixture_distribution), get_dist_first_horizon(arr.component_distribution), temperature=arr._temp)
        else:
             raise NotImplementedError
