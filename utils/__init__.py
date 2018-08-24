import random as pyrandom
import numpy as np
import mxnet as mx
from .config import vgg16_fixed_params_pattern, resnet50_v1b_fixed_params_pattern
from .logger import logger

__all__ = ['set_random_seed', 'fix_net_params']


def set_random_seed(a=None):
    """Seed the generator for python builtin random, numpy.random, mxnet.random.

    This method is to control random state for mxnet related random functions.

    Note that this function cannot guarantee 100 percent reproducibility due to
    hardware settings.

    Parameters
    ----------
    a : int or 1-d array_like, optional
        Initialize internal state of the random number generator.
        If `seed` is not None or an int or a long, then hash(seed) is used instead.
        Note that the hash values for some types are nondeterministic.

    """
    pyrandom.seed(a)
    np.random.seed(a)
    mx.random.seed(a)


def fix_net_params(net, name='vgg16'):
    """
    Fix network parameters, fixed_params defined in ./config.py
    """
    fixed_params_pattern = {
        'vgg16': vgg16_fixed_params_pattern,
        'resnet50_v1b': resnet50_v1b_fixed_params_pattern
    }
    if name not in fixed_params_pattern:
        raise ValueError(
            'Model %s does not configure fixed parameters. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(fixed_params_pattern.keys()))))

    param_dict = net.collect_params(fixed_params_pattern[name])
    for _, param in param_dict.items():
        param.grad_req = 'null'
    logger.info('Fixed such params for net:\n%s' % param_dict)
