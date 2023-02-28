from attrdict import AttrDict as d

from cfgs.model import hydra

export = hydra.export & d(
    use_vision_encoder=True,
    default_img_embed_size=64,
    default_use_spatial_softmax=True,
    default_use_crop_randomizer=True,
    goal_names=[],
    image_keys=['image', 'ego_image'],
    hidden_size=1000,
)