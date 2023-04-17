from cfgs.model import dp_conv_1d
from attrdict import AttrDict as d

from cfgs.model.vision import dual_resnet_encoder
from configs.fields import Field as F

export = dp_conv_1d.export & d(
    exp_name='_vis' + dp_conv_1d.export.exp_name,

    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    extra_names=['img_embedding'],

    # encoders
    state_encoder_order=['proprio_encoder', 'vision_encoder'],
    model_order=['proprio_encoder', 'vision_encoder', 'action_decoder'],
    vision_encoder=dual_resnet_encoder.export & d(
        # either specify this or override!
        image_shape=F('image_shape'),
    ),
    action_decoder=d(
        hidden_size=1000,
    )
)
