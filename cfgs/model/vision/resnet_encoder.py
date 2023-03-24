from attrdict import AttrDict as d

from configs.fields import Field as F
from muse.models.vision.helpers import get_resnet18_encoder_layer
from muse.models.vision.vision_encoder import VisionEncoder

export = d(
    cls=VisionEncoder,

    device="cuda",
    model_inputs=['image'],
    model_output='img_embedding',

    # macro we use to create
    image_shape=None,
    crop_random_frac=0.9,
    use_spatial_softmax=True,
    use_color_randomizer=False,
    use_erasing_randomizer=False,
    img_embed_size=64,
    downsample_frac=1.0,

    network=F(["image_shape", "crop_random_frac", "use_spatial_softmax", "use_color_randomizer",
               "use_erasing_randomizer", "img_embed_size", "downsample_frac"],
              lambda ims, crop, smax, color, erase, emb, down:
              get_resnet18_encoder_layer(ims, crop_random_frac=crop, use_spatial_softmax=smax,
                                         use_color_randomizer=color,
                                         use_erasing_randomizer=erase, img_embed_size=emb, downsample_frac=down))
)
