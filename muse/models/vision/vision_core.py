import numpy as np
import torch
from torchvision import models as vms

from muse.utils.torch_utils import combine_dims, split_dim


class VisionCore(torch.nn.Module):
    """ 
    Takes a pretrained torchvision model by name, cuts it and slaps on a flatten layer 
    
    (Optionally) performs some transform on the 
    (Optionally) changes the input channels
    (Optionally) projects to a new size
    """

    def __init__(self, name, in_channels=3, flip=True, flatten=True, cut_last=0, out_shape=None,
                 projection_size=None, randomizers=None, extra_conv_layers=[], **kwargs):
        super(VisionCore, self).__init__()
        # cut off last two layers
        self.name = name
        net = getattr(vms, name)(**kwargs)

        # replace first layer if in channels are different.
        if in_channels != 3:
            net.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last few layers (..., pooling, projection)
        layers = list(net.children())
        if cut_last > 0:
            layers = layers[:-cut_last]

        # add extra layers before optionally flattening things
        for cl in extra_conv_layers:
            if not isinstance(cl, torch.nn.Module):
                # must be LayerParams instance
                cl = cl.to_module_list(as_sequential=True)
            layers.append(cl)

        if len(extra_conv_layers) > 0:
            assert out_shape is not None, "Must pass in out_shape if using extra conv layers!"

        self.flip = flip
        self.flatten = flatten
        self.out_shape = out_shape
        self.cut_last = cut_last
        self.projection_size = projection_size

        if self.flatten:
            # flatten after the batch*horizon dim
            layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # projection
        if projection_size is not None:
            layers.append(torch.nn.Linear(int(np.prod(out_shape)), projection_size))
            self.out_shape = [projection_size]

        from muse.models.vision.randomizers import ImageRandomizer
        self.randomizers = randomizers or []
        for r in self.randomizers:
            assert isinstance(r, ImageRandomizer), f"Expected randomizer but class was: {type(r)}"

        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs, randomize_in=True, randomize_out=True):
        # preprocess shape
        front_shape = obs.shape[:-3]
        obs = combine_dims(obs, 0, len(front_shape))
        assert len(obs.shape) == 4, obs.shape
        if self.flip:
            obs = obs.permute((0, 3, 1, 2)).contiguous()  # move channel to front

        if randomize_in:
            # randomize
            for r in self.randomizers:
                obs = r.forward_in(obs)

        # pass through backbone
        out = self.net(obs)

        if randomize_out:
            # out-randomize in reverse order
            for r in self.randomizers[::-1]:
                out = r.forward_out(out)

        if self.flip and not self.flatten:
            out = out.permute((0, 2, 3, 1)).contiguous()  # move channel back to end
        # postprocess shape
        return split_dim(out, 0, front_shape)

    def get_output_shape(self, input_shape):
        # input_shape should be (C x H x W)
        if self.out_shape is not None:
            return self.out_shape

        input_shape = list(input_shape)
        for r in self.randomizers:
            input_shape = r.output_shape_in(input_shape)

        output_shape = vision_output_shape_fns[self.name](input_shape, cut_last=self.cut_last)

        for r in self.randomizers[::-1]:
            output_shape = r.output_shape_out(output_shape)

        return output_shape

    def get_randomizer_in_output_shape(self, input_shape):
        input_shape = list(input_shape)
        for r in self.randomizers:
            input_shape = r.output_shape_in(input_shape)
        return input_shape


""" Specific output shape functions """


def resnet18_compute_output_shape(image_shape, cut_last=0, crop_frac=None):
    image_shape = np.asarray((image_shape[2], image_shape[0], image_shape[1]))
    if crop_frac is not None and crop_frac > 0:
        image_shape = (crop_frac * image_shape).astype(int)
    shape_l1 = np.concatenate([[64], ((image_shape[-2:] - 1) / 2).astype(int) + 1])
    shape_l2 = np.concatenate([[64], ((shape_l1[-2:] - 1) / 2).astype(int) + 1])
    shape_l3 = np.concatenate([[128], ((shape_l2[-2:] - 1) / 2).astype(int) + 1])
    shape_l4 = np.concatenate([[256], ((shape_l3[-2:] - 1) / 2).astype(int) + 1])
    shape_l5 = np.concatenate([[512], ((shape_l4[-2:] - 1) / 2).astype(int) + 1])

    shape_pool = np.prod(shape_l5, keepdims=True)
    shape_project = np.array([1000])

    # in reverse order (flatten, pool, 2block, 2block, 2block, 2block, pool, relu, norm, conv, input)
    shapes = [shape_project, shape_pool, shape_l5, shape_l4, shape_l3, shape_l2, shape_l1, shape_l1, shape_l1, image_shape]
    return shapes[cut_last]


vision_output_shape_fns = {
    'resnet18': resnet18_compute_output_shape,
}
