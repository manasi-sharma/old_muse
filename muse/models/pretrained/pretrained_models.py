"""
Usable by LayerParams
"""
from muse.models.pretrained.conv_3D_classifier import MultiColumn, Conv3DFeatureModel
from muse.models.pretrained.detic import DeticPredictor
from torchvision import models as vms


from attrdict import AttrDict as d


custom_model_map = {
    'resnet18': vms.resnet18,
    'resnet50': vms.resnet50,
    'multi_column_model': MultiColumn,
    'conv3d_feature_model': Conv3DFeatureModel,
    'detic': DeticPredictor,
}

pretrained_model_map = {
    'smth_smth_classifier': lambda **kwargs: MultiColumn(174, d(cls=Conv3DFeatureModel, params=d(feature_size=512)), 512),
}
