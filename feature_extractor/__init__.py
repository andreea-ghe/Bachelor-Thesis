from .attention_mechanisms import *
from .pointnet_architecture import *
from .utils_encoder_decoder import *
from .utils import *


def build_feature_extractor(arch, features_dimension, **kwargs):
    if 'in_feat_dim' in kwargs:
        input_features_dim = kwargs['in_feat_dim']
    else:
        input_features_dim = 3  # default

    use_gabriel = kwargs.get('use_gabriel', False)
    gabriel_min_keep_ratio = kwargs.get('gabriel_min_keep_ratio', 0.75)

    if isinstance(features_dimension, list):
        model = PointNetPTMSG(features_dimension[0], features_dimension[1], use_gabriel=use_gabriel, gabriel_min_keep_ratio=gabriel_min_keep_ratio)
    else:
        model = PointNetPTMSG(input_features_dim, features_dimension, use_gabriel=use_gabriel, gabriel_min_keep_ratio=gabriel_min_keep_ratio)

    return model