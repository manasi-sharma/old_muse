import abc

import torch
import torchvision

from muse.utils.torch_utils import sample_random_image_crops, combine_dims, split_dim


"""
================================================
Image Randomizer Networks
================================================

Taken from https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/obs_nets.py

"""


class ImageRandomizer(torch.nn.Module):
    """
    Base class for randomizer networks. Each randomizer should implement the @output_shape_in,
    @output_shape_out, @forward_in, and @forward_out methods. The randomizer's @forward_in
    method is invoked on raw inputs, and @forward_out is invoked on processed inputs
    (usually processed by a @VisualCore instance). Note that the self.training property
    can be used to change the randomizer's behavior at train vs. test time.
    """

    def __init__(self):
        super(ImageRandomizer, self).__init__()

    def output_shape(self, input_shape=None):
        """
        This function is unused. See @output_shape_in and @output_shape_out.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_in(self, inputs):
        """
        Randomize raw inputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        return inputs


class CropRandomizer(ImageRandomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
            self,
            input_shape,
            crop_height,
            crop_width,
            num_crops=1,
            pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CropRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        out, _ = sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return combine_dims(out, 0, 2)

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = split_dim(inputs, 0, [batch_size, self.num_crops])
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg


class PaddedCropRandomizer(CropRandomizer):
    def __init__(self, input_shape, pad_len, num_crops=1, pos_enc=False):
        # pads to same size
        crop_height, crop_width = input_shape[-2:]
        new_in_shape = list(input_shape)
        new_in_shape[1] = new_in_shape[1] + 2 * pad_len
        new_in_shape[2] = new_in_shape[2] + 2 * pad_len
        self.pad_len = pad_len
        assert self.pad_len > 0
        super().__init__(new_in_shape, crop_height, crop_width, num_crops, pos_enc)
        self.pad_layer = torch.nn.ReplicationPad2d(pad_len)

    def forward_in(self, inputs):
        inputs = self.pad_layer(inputs)
        return super().forward_in(inputs)


class TransformRandomizer(ImageRandomizer):
    """
    Arbitrary shape preserving transform
    """

    def __init__(
            self,
            input_shape,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
        """
        super(TransformRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)

        self.input_shape = input_shape

    def output_shape_in(self, input_shape=None):
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        return self.transform(inputs)

    def forward_out(self, inputs):
        return inputs

    def __repr__(self):
        """Pretty print network."""
        print(self.transform)


class ColorRandomizer(TransformRandomizer):
    """
    Color augmentation
    """

    def __init__(
            self,
            input_shape,
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
    ):
        super(ColorRandomizer, self).__init__(input_shape)
        self.transform = torchvision.transforms.ColorJitter(brightness=brightness,
                                                            contrast=contrast,
                                                            saturation=saturation,
                                                            hue=hue)


class ErasingRandomizer(TransformRandomizer):
    """
    Random erasing augmentation
    """

    def __init__(
            self,
            input_shape,
            p=0.5,
            scale=(0.02, 0.33),
            value=0,
    ):
        super(ErasingRandomizer, self).__init__(input_shape)
        self.transform = torchvision.transforms.RandomErasing(p=p, scale=scale, value=value)


class Downsampler(TransformRandomizer):

    def __init__(self, input_shape, downsample_frac):
        super(Downsampler, self).__init__(input_shape)
        h, w = self.input_shape[-2:]
        self.downsample_frac = downsample_frac
        self.transform = torchvision.transforms.Resize((int(round(h * downsample_frac)), int(round(w * downsample_frac))))

    def output_shape_in(self, input_shape=None):
        h, w = input_shape[-2:]
        return list(input_shape[:-2]) + [int(round(h * self.downsample_frac)), int(round(w * self.downsample_frac))]
