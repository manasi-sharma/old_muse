from attrdict import AttrDict as d

from cfgs.model.vision import resnet_encoder

export = resnet_encoder.export & d(
    model_inputs=['image', 'ego_image'],
    call_separate=True,
    use_shared_params=True,
)
