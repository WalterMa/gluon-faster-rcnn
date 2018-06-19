from .vgg import *


def get_model_conv_block(name, **kwargs):
    """Returns base_model's convolution part by name
    :param name: str
        Name of the base_model
    :return: HybridBlock
        The base_model convolution part
    """
    models = {
        'vgg11': VGGConvBlock,
        'vgg13': VGGConvBlock,
        'vgg16': VGGConvBlock,
        'vgg19': VGGConvBlock
    }
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)


def get_model_rcnn_block(name, **kwargs):
    """Returns base_model's convolution part by name
    :param name: str
        Name of the base_model
    :return: HybridBlock
        The base_model convolution part
    """
    models = {
        'vgg11': VGGFastRCNNHead,
        'vgg13': VGGFastRCNNHead,
        'vgg16': VGGFastRCNNHead,
        'vgg19': VGGFastRCNNHead
    }
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
