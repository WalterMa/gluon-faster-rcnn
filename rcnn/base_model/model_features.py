from .vgg import *
from .resnet_v1b import *

__all__ = ['get_model_features']

_model_features = {
    'vgg16': vgg16_features,
    'resnet50_v1b': resnet50_v1b_features,
}


def get_model_features(name, **kwargs):
    """Returns a pre-defined model feature and top feature extractor layers by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model feature layers.
    HybridBlock
        The model top feature layers.
    """
    name = name.lower()
    if name not in _model_features:
        raise ValueError('%s' % ('\n\t'.join(sorted(_model_features.keys()))))
    features, top_features = _model_features[name](**kwargs)
    return features, top_features
