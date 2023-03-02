import numpy as np

from muse.experiments import logger
from muse.models.vision.vision_core import CropRandomizer, Downsampler, PaddedCropRandomizer, ColorRandomizer, ErasingRandomizer
from muse.utils.param_utils import LayerParams


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


def get_resnet18_encoder_layer(image_shape, crop_random_frac=0.9, use_spatial_softmax=True, use_color_randomizer=False,
                               use_erasing_randomizer=False, img_embed_size=64, downsample_frac=1.0):

    # randomizers
    randomizers = []
    curr_sh = image_shape
    if downsample_frac < 1.:
        randomizers.append(get_downsampler(curr_sh, downsample_frac=downsample_frac))
        curr_sh = np.array(curr_sh)
        curr_sh[:2] = np.round(curr_sh[:2] * downsample_frac).astype(int)
        curr_sh = list(curr_sh)

    if crop_random_frac is not None and crop_random_frac > 0:
        randomizers.append(get_crop_randomizer(curr_sh, crop_random_frac=crop_random_frac))
        curr_sh=get_crop_image_shape(curr_sh, crop_random_frac)

    if use_color_randomizer:
        randomizers.append(get_color_randomizer(curr_sh))

    if use_erasing_randomizer:
        randomizers.append(get_erasing_randomizer(curr_sh))
    
    cut_last = 1 + int(use_spatial_softmax)
    out_shape = resnet18_compute_output_shape(curr_sh, cut_last)
    final_out_shape = resnet18_compute_output_shape(curr_sh, 1)
    
    # spatial softmax layer
    extra_conv = []
    if use_spatial_softmax:
        # conv to 32, keypoint softmax, output embeddings will be (64, 2), which will then get flattened
        extra_conv = [
            LayerParams('spatial_softmax', input_shape=out_shape, num_kp=img_embed_size // 2, temperature=1.)
        ]
    # resnet18 outputs 512 channels, optional projection to img_embed_size if not using spatial_softmax
    return LayerParams('vision_core', name='resnet18', flip=True, flatten=True, cut_last=cut_last,
                       raw_out_size=final_out_shape[0], out_size=None if use_spatial_softmax else img_embed_size,
                       extra_conv_layers=extra_conv, randomizers=randomizers)


def get_reset18_partial_embedding_layer(remove_last=4, flip=False):
    # resnet18 outputs 512 channels, optional projection to img_embed_size if not using spatial_softmax
    return LayerParams('vision_core', name='resnet18', flip=flip, flatten=False, cut_last=remove_last)


def get_crop_randomizer(image_shape, crop_random_frac=0.9):
    assert len(image_shape) == 3, image_shape
    # permute the shape to match randomizer expects
    shape = [image_shape[2], image_shape[0], image_shape[1]]  # (C, H, W)
    crop_size = [int(round(crop_random_frac * shape[1])), int(round(crop_random_frac * shape[2]))]
    logger.debug(f"Using crop randomizer of size={crop_size} for image.shape={shape}")
    return CropRandomizer(shape, *crop_size)

def get_crop_image_shape(image_shape, crop_frac):
    return [int(round(crop_frac * image_shape[0])), int(round(crop_frac * image_shape[1])), image_shape[2]]

def get_padded_crop_randomizer(image_shape, pad_size=2):
    # will add pad size to each edge of image.
    assert len(image_shape) == 3, image_shape
    # permute the shape to match randomizer expects
    shape = [image_shape[2], image_shape[0], image_shape[1]]  # (C, H, W)
    logger.debug(f"Using pad crop randomizer of pad={pad_size} for image.shape={shape}")
    return PaddedCropRandomizer(shape, pad_size)


def get_color_randomizer(image_shape, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
    assert len(image_shape) == 3, image_shape
    # permute the shape to match randomizer expects
    shape = [image_shape[2], image_shape[0], image_shape[1]]  # (C, H, W)
    logger.debug(f"Using color randomizer (b={brightness}, c={contrast}, s={saturation}, h={hue}) for image.shape={shape}")
    return ColorRandomizer(shape, brightness, contrast, saturation, hue)


def get_erasing_randomizer(image_shape, p=0.5):
    assert len(image_shape) == 3, image_shape
    # permute the shape to match randomizer expects
    shape = [image_shape[2], image_shape[0], image_shape[1]]  # (C, H, W)
    logger.debug(f"Using erasing randomizer with p={p} for image.shape={shape}")
    return ErasingRandomizer(shape, p=p)


def get_downsampler(image_shape, downsample_frac):
    assert len(image_shape) == 3, image_shape
    # permute the shape to match randomizer expects
    shape = [image_shape[2], image_shape[0], image_shape[1]]  # (C, H, W)
    logger.debug(f"Using downsampling with frac={downsample_frac} for image.shape={shape}")
    return Downsampler(shape, downsample_frac)