from cfgs.model import bc_rnn
from attrdict import AttrDict as d

export = bc_rnn.export & d(
    exp_name='_vis' + bc_rnn.export.exp_name,
    use_vision_encoder=True,
    default_img_embed_size=64,
    default_use_spatial_softmax=True,
    default_use_crop_randomizer=True,
    goal_names=[],
    image_keys=['image', 'ego_image'],
    hidden_size=1000,
)
