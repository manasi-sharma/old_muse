import math

import numpy as np
import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F

from muse.utils.torch_utils import combine_after_dim


class SamplingLinearLayer(nn.Linear):
    def __init__(self, sample_prob, in_features, bias=True):
        super(SamplingLinearLayer, self).__init__(in_features=in_features, out_features=2 * in_features, bias=bias)
        self.sample_prob = sample_prob

    def forward(self, x):
        output2x = super().forward(x).view(list(x.shape[:-1]) + [self.in_features, 2])
        idxs = torch.rand(output2x.shape[:-1]).unsqueeze(-1)  # (..., in_features, 1)
        idxs = (idxs > self.sample_prob).type(torch.int64)
        return torch.gather(output2x, -1, idxs), idxs

    # bernoulli
    def log_prob(self, idxs):
        return torch.where(idxs == 0, torch.ones_like(idxs) * self.sample_prob,
                           torch.ones_like(idxs) * (1 - self.sample_prob)).log()


class ResidualLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ResidualLinear, self).__init__(in_features, out_features, bias=bias)
        assert in_features == out_features, "Residual layers require equal num inputs and outputs"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super(ResidualLinear, self).forward(input)


class MaskLinear(nn.Linear):
    def __init__(self, num_chunks, in_features, out_features, in_masked_features=None, bias=True, residual=False):
        super(MaskLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        effective_in_features = in_features
        if in_masked_features is not None:
            # NOTE: will compute mask only for the first N features.
            assert 0 < in_masked_features <= in_features, "Must mask nonzero # features, less than/eq # total in features"
            effective_in_features = in_masked_features

        assert effective_in_features % num_chunks == 0, f"ef_in={effective_in_features} must be divisible by masking chunks {num_chunks}"
        assert out_features % num_chunks == 0, f"f_out={out_features} must be divisible by masking chunks {num_chunks}"

        # outf x inf
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self._residual = residual
        if residual:
            assert in_features == out_features, "Residual layers require equal num inputs and outputs"

        col_spacing = effective_in_features // num_chunks
        last_col = effective_in_features
        row_spacing = out_features // num_chunks
        for i in range(num_chunks):
            # outputs i*rs:(i+1)*rs don't see anything after (i+1)*cs
            # e.g., for in=6, out=4, num_ch=2, in_masked_f=3
            # mask = [[1. 1. 1. 0. 0., 0.,  1., 1., 1.]
            #         [1. 1. 1. 0. 0., 0.,  1., 1., 1.]
            #         [1. 1. 1. 1. 1., 1.,  1., 1., 1.]
            #         [1. 1. 1. 1. 1., 1.,  1., 1., 1.]]
            self.mask[i * row_spacing:(i + 1) * row_spacing, (i + 1) * col_spacing:last_col] = 0.

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # element-wise mask to ensure visibility
        out = F.linear(input, self.mask * self.weight, self.bias)
        if self._residual:
            return input + out
        else:
            return out


class Empty(nn.Module):
    def forward(self, x):
        return x


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']

        _, _, H, W = self.weight.size()

        self.register_buffer('mask', self.weight.data.clone())

        self.mask.fill_(1)

        is_mask_b = 0
        if mask_type == 'B':
            is_mask_b = 1

        # right segment nulled
        self.mask[:, :, H // 2, W // 2 + is_mask_b:] = 0
        # downward segment nulled
        self.mask[:, :, H // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(-1), (self.left_padding, 0, 0, 0)).squeeze(-1)

        return super(CausalConv1d, self).forward(x)


class SpatialSoftmax(torch.nn.Module):
    """
    Spatial Softmax Layer + Pooling.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
            self,
            input_shape,
            num_kp=None,
            temperature=1.,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert (len(input_shape) == 3)
        assert (input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert (feature.shape[1] == self._in_c), [feature.shape, "expected:", (self._in_c, self._in_h, self._in_w)]
        assert (feature.shape[2] == self._in_h), [feature.shape, "expected:", (self._in_c, self._in_h, self._in_w)]
        assert (feature.shape[3] == self._in_w), [feature.shape, "expected:", (self._in_c, self._in_h, self._in_w)]
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class SpatialProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim,
                 num_kp=None):
        super().__init__()

        num_kp = num_kp or out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(input_shape, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(combine_after_dim(out, -2))
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResMaskBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels, 1, **kwargs)
        )

    def forward(self, x):
        return self.block(x) + x


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class LinConv(nn.Linear):
    def __init__(self, permute_in, permute_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pin = permute_in
        self.pout = permute_out

    def forward(self, x):
        if self.pin:
            x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        if self.pout:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x) + x


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, batch_first=False, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] (flip first to if batch first)
            output: [sequence length, batch size, embed dim]
        Examples:
            # >>> output = pos_encoder(x)
        """

        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    """
    Generates a temporal pos encoding, meant to add to the last dimension

    adapted from VIOLA
    https://github.com/UT-Austin-RPL/VIOLA/blob/637aeed3554089a45b2a62df8c8a395ec213b23e/viola_bc/modules.py
    """
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        # output will be (H x channels)
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape


class BBoxTrueSinusoidalPositionEncodingFactor(nn.Module):
    """
    adapted from VIOLA:
    https://github.com/UT-Austin-RPL/VIOLA/blob/637aeed3554089a45b2a62df8c8a395ec213b23e/viola_bc/modules.py
    """

    def __init__(
            self,
            channels=3,
            scaling_ratio=128.,
            use_noise=False,
            pixel_var=2,
            dim=4,
            factor_ratio=1.,
    ):
        super().__init__()
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels

        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.scaling_ratio = scaling_ratio
        self.use_noise = use_noise
        self.pixel_var = pixel_var
        factor = nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)

    def forward(self, x):
        if self.use_noise and self.training:
            # Only add noise if use_noise is True and it's in training mode
            noise = (torch.rand_like(x) * 2 - 1) * self.pixel_var
            x += noise
        x = torch.divide(x, self.scaling_ratio)

        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        self.spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        return self.spatial_pos_embedding * self.factor


class RoIAlignWrapper(nn.Module):
    """
    ROI Alignment
    https://github.com/UT-Austin-RPL/VIOLA/blob/637aeed3554089a45b2a62df8c8a395ec213b23e/viola_bc/modules.py
    """
    def __init__(self,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True):
        super().__init__()
        assert (aligned == True)
        self.output_size = output_size
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size,
                                                  spatial_scale=spatial_scale,
                                                  sampling_ratio=sampling_ratio,
                                                  aligned=aligned)

    def forward(self, x, bbox_list):
        batch_size, channel_size, h, w = x.shape
        bbox_size = bbox_list[0].shape[0]
        out = self.roi_align(x, bbox_list)
        out = out.reshape(batch_size, bbox_size, channel_size, *self.output_size)
        return out

    def output_shape(self, input_shape):
        """Return a batch of input sequences"""
        return (input_shape[0], self.output_size[0], self.output_size[1])
