import mxnet as mx
from mxnet import nd


__all__ = ['FasterRCNNDefaultTrainTransform', 'FasterRCNNDefaultValTransform', 'load_test']


class BaseRCNNTransform(object):

    def __call__(self, img, label):
        return NotImplementedError

    @staticmethod
    def _resize_image(img, label, size, max_size):
        h, w, _ = img.shape
        if h > w:
            scale = size / w
            if int(h * scale) > max_size:
                scale = max_size / h
        else:
            scale = size / h
            if int(w * scale) > max_size:
                scale = max_size / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = mx.image.imresize(img, new_w, new_h, interp=1)
        label[:, 0:4] = label[:, 0:4] * scale
        im_info = nd.array([new_h, new_w, scale])
        return img, im_info

    @staticmethod
    def _random_flip(img, label, p=0.5):
        if nd.random.uniform(low=0, high=1) > p:
            h, w, _ = img.shape
            img = nd.flip(img, axis=2)
            xmin = label[:, 0].copy()
            xmax = label[:, 2].copy()
            label[:, 0] = w - xmax
            label[:, 2] = w - xmin
        return img, label


class FasterRCNNDefaultTrainTransform(BaseRCNNTransform):
    """Default Faster-RCNN training transform."""

    def __init__(self, size, max_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), random_flip=True):
        super(FasterRCNNDefaultTrainTransform, self).__init__()
        self._size = size
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._is_random_flip = random_flip

    def __call__(self, img, label):
        if self._is_random_flip:
            img, label = self._random_flip(img, label)
        img, im_info = self._resize_image(img, label, self._size, self._max_size)
        # Converts an image NDArray of shape (H x W x C) in the range [0, 255]
        # to a float32 tensor NDArray of shape (C x H x W) in the range [0, 1).
        img = nd.image.to_tensor(img)
        img = nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, label, im_info


class FasterRCNNDefaultValTransform(BaseRCNNTransform):
    """Default Faster-RCNN validation transform."""

    def __init__(self, size, max_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(FasterRCNNDefaultValTransform, self).__init__()
        self._size = size
        self._max_size = max_size
        self._mean = mean
        self._std = std

    def __call__(self, img, label):
        img, im_info = self._resize_image(img, label, self._size, self._max_size)
        # Converts an image NDArray of shape (H x W x C) in the range [0, 255]
        # to a float32 tensor NDArray of shape (C x H x W) in the range [0, 1).
        img = nd.image.to_tensor(img)
        img = nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, label, im_info


def load_test(filenames, size=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    size : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray, mxnet.NDArray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, a numpy ndarray as
        original un-normalized color image for display, and a (1, 3) mxnet NDArray as im_info.
        If multiple image names are supplied, return three lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    tensors = []
    origs = []
    im_infos = []
    for f in filenames:
        img = mx.image.imread(f)
        h, w, _ = img.shape
        if h > w:
            scale = size / w
            if int(h * scale) > max_size:
                scale = max_size / h
        else:
            scale = size / h
            if int(w * scale) > max_size:
                scale = max_size / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = mx.image.imresize(img, new_w, new_h, interp=1)
        orig_img = img.asnumpy().astype('uint8')
        img = nd.image.to_tensor(img)
        img = nd.image.normalize(img, mean=mean, std=std)
        im_info = nd.array([new_h, new_w, scale])
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
        im_infos.append(im_info.expand_dims(0))
    if len(tensors) == 1:
        return tensors[0], origs[0], im_infos[0]
    return tensors, origs, im_infos
