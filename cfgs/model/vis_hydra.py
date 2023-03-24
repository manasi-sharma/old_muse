from attrdict import AttrDict as d

from cfgs.model import hydra
from configs.fields import Field as F

from cfgs.model.vision import dual_resnet_encoder

export = hydra.export & d(
    exp_name='_vis' + hydra.export.exp_name,

    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    extra_names=['img_embedding'],

    # encoders
    state_encoder_order=['proprio_encoder', 'vision_encoder'],
    model_order=['proprio_encoder', 'vision_encoder', 'action_decoder'],
    vision_encoder=dual_resnet_encoder.export & d(
        # either specify this or override!
        image_shape=F('image_shape'),
    ),

    head_size=400,
    action_decoder=d(
        hidden_size=1000,
    )
)
