from torch import nn as nn

from muse.models.diffusion.temporal_diffusion_layers import TemporalUnet
from muse.models.gpt.bet_layers import GPT
from muse.models.layers.linear_vq_vae import VectorQuantize
from muse.models.vision.vision_core import VisionCore
from muse.models.dist.layers import GaussianDistributionCap, SquashedGaussianDistributionCap, \
    CategoricalDistributionCap, MixedDistributionCap
from muse.models.layers.common import SamplingLinearLayer, ResidualLinear, MaskLinear, Empty, MaskConv2d, CausalConv1d, \
    SpatialSoftmax, ResMaskBlock, LayerNorm, LinConv, ResidualBlock
from muse.models.layers.functional import ExtractKeys, Reshape, SplitDim, CombineDim, ListSelect, ListConcat, \
    ListFromDim, Permute, Functional, Assert

activation_map = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax,
    'softmax2d': nn.Softmax2d,
    'leakyrelu': nn.LeakyReLU,
    'none': None
}

layer_map = {
    "conv3d": nn.Conv3d,
    "conv2d": nn.Conv2d,
    "conv1d": nn.Conv1d,
    "causalconv1d": CausalConv1d,
    "convtranspose2d": nn.ConvTranspose2d,
    "convtranspose1d": nn.ConvTranspose1d,
    "linear": nn.Linear,
    "residual_linear": ResidualLinear,
    "mask_linear": MaskLinear,
    "sampling_linear": SamplingLinearLayer,
    "residualblock": ResidualBlock,
    "linearconv": LinConv,
    "batchnorm2d": nn.BatchNorm2d,
    "layernorm": LayerNorm,
    "maskconv2d": MaskConv2d,
    "residualmaskblock": ResMaskBlock,
    "temporal_unet": TemporalUnet,
    'vision_core': VisionCore,
    'gpt': GPT,
    "vq": VectorQuantize,
    "lstm": nn.LSTM,
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "avg_pool_1d": nn.AvgPool1d,
    "avg_pool_2d": nn.AvgPool2d,
    "avg_pool_3d": nn.AvgPool3d,
    "max_pool_1d": nn.MaxPool1d,
    "max_pool_2d": nn.MaxPool2d,
    "max_pool_3d": nn.MaxPool3d,
    "lp_pool_1d": nn.LPPool1d,
    "lp_pool_2d": nn.LPPool2d,
    "spatial_softmax": SpatialSoftmax,
    "dropout": nn.Dropout,
    "dropout2d": nn.Dropout2d,
    "dropout3d": nn.Dropout3d,
    "empty": Empty,
}

try:
    from muse.models.layers.graph_layers import GCNConv, EdgeConv
    layer_map.update({
        "gcnconv": GCNConv,
        "edgeconv": EdgeConv,
    })
except Exception as e:
    pass

reshape_map = {
    "reshape": Reshape,
    "split_dim": SplitDim,  # splits a dim
    "combine_dim": CombineDim,
    "list_select": ListSelect,
    "list_from_dim": ListFromDim,
    "list_concat": ListConcat,
    "permute": Permute,
    "functional": Functional,  # catch all
    "assert": Assert,
    "extract_keys": ExtractKeys,
}
dist_cap_map = {
    "gaussian_dist_cap": GaussianDistributionCap,
    "squashed_gaussian_dist_cap": SquashedGaussianDistributionCap,
    "categorical_dist_cap": CategoricalDistributionCap,
    "mixed_dist_cap": MixedDistributionCap,
}
