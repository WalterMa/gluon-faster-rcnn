"""Model store information of pre-trained models."""
import os
from mxnet.gluon.utils import check_sha1

__all__ = ['get_model_file']

_model_sha1 = {name: checksum for checksum, name in [
    ('dd221b160977f36a53f464cb54648d227c707a05', 'vgg11'),
    ('ee79a8098a91fbe05b7a973fed2017a6117723a8', 'vgg11_bn'),
    ('6bc5de58a05a5e2e7f493e2d75a580d83efde38c', 'vgg13'),
    ('7d97a06c3c7a1aecc88b6e7385c2b373a249e95e', 'vgg13_bn'),
    ('e660d4569ccb679ec68f1fd3cce07a387252a90a', 'vgg16'),
    ('7f01cf050d357127a73826045c245041b0df7363', 'vgg16_bn'),
    ('ad2f660d101905472b83590b59708b71ea22b2e5', 'vgg19'),
    ('f360b758e856f1074a85abd5fd873ed1d98297c3', 'vgg19_bn'),
    ('2d9d980c990442f826f20781ed039851e78dabe3', 'resnet18_v1b'),
    ('8e16b84814e84f64d897854003f049872991eaa6', 'resnet34_v1b'),
    ('e263a9860be0a373003d011564f10701d4954fb8', 'resnet50_v1b'),
    ('c9d451fc69773007958205659bad43c9ae77e20d', 'resnet101_v1b'),
    ('e74027961d155170868f30bd3f113d7feb44f618', 'resnet152_v1b'),
]}


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name + '.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            raise AssertionError('Mismatch in the content of model file detected. Please download it again.')
    else:
        raise AssertionError('Model file: %s is not found. Please download before use it.' % file_path)
