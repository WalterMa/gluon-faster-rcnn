"""DataLoader utils."""
import numpy as np
from mxnet import nd
from mxnet import context
from mxnet.gluon.data import DataLoader


class FasterRCNNDefaultBatchify:

    def __init__(self, image_max_size, label_max_size, multiprocessing):
        if multiprocessing:
            # Use shared memory for collating data into batch
            self._ctx = context.Context('cpu_shared', 0)
        else:
            self._ctx = None
        self._image_max_size = image_max_size
        self._label_max_size = label_max_size

    def __call__(self, data):
        """Collate data into batch, images, im_info and labels are padded to same shape"""
        # 1. Collect image, label, im_info batch
        data = zip(*data)
        return [self.batchify(i) for i in data]

    def batchify(self, data):
        data_shape = len(data[0].shape)
        if not isinstance(data[0], nd.NDArray):
            tmp = []
            for i in data:
                tmp.append(nd.array(i, ctx=self._ctx))
            data = tmp
        if data_shape == 1:
            # 2. Stack im_info
            return nd.stack(*data)
        elif data_shape == 2:
            # 2. Padding label
            buf = nd.full((len(data), self._label_max_size, data[0].shape[-1]), val=-1, ctx=self._ctx)
            for i, l in enumerate(data):
                buf[i][:l.shape[0], :] = l
            return buf
        elif data_shape == 3:
            # 2. Padding image
            buf = nd.zeros((len(data), data[0].shape[0], self._image_max_size, self._image_max_size), ctx=self._ctx)
            for i, img in enumerate(data):
                buf[i][:, :img.shape[1], :img.shape[2]] = img
            return buf
        else:
            raise NotImplementedError


class DetectionDataLoader(DataLoader):
    """Data loader for detection dataset.


    It loads data batches from a dataset and then apply data
    transformations. It's a subclass of :py:class:`mxnet.gluon.data.DataLoader`,
    and therefore has very simliar APIs.

    The main purpose of the DataLoader is to pad variable length of labels from
    each image, because they have different amount of objects.

    Parameters
    ----------
    dataset : mxnet.gluon.data.Dataset or numpy.ndarray or mxnet.ndarray.NDArray
        The source dataset.
    batch_size : int
        The size of mini-batch.
    shuffle : bool, default False
        If or not randomly shuffle the samples. Often use True for training
        dataset and False for validation/test datasets
    sampler : mxnet.gluon.data.Sampler, default None
        The sampler to use. We should either specify a sampler or enable
        shuffle, not both, because random shuffling is a sampling method.
    last_batch : {'keep', 'discard', 'rollover'}, default is keep
        How to handle the last batch if the batch size does not evenly divide by
        the number of examples in the dataset. There are three options to deal
        with the last batch if its size is smaller than the specified batch
        size.

        - keep: keep it
        - discard: throw it away
        - rollover: insert the examples to the beginning of the next batch
    batch_sampler : mxnet.gluon.data.BatchSampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch.
        Defaults to :py:meth:`gluonvision.data.dataloader.default_pad_batchify_fn`::
            def default_pad_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [pad_batchify(i) for i in data]
                else:
                    data = np.asarray(data)
                    pad = max([l.shape[0] for l in data])
                    buf = np.full((len(data), pad, data[0].shape[-1]),
                                  -1, dtype=data[0].dtype)
                    for i, l in enumerate(data):
                        buf[i][:l.shape[0], :] = l
                    return nd.array(buf, dtype=data[0].dtype)
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
        If ``num_workers`` = 0, multiprocessing is disabled.
        Otherwise ``num_workers`` multiprocessing worker is used to process data.

    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None, image_max_size=1000, label_max_size=100,
                 num_workers=0):
        if batchify_fn is None:
            batchify_fn = FasterRCNNDefaultBatchify(image_max_size, label_max_size, num_workers)
        super(DetectionDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, last_batch,
            batch_sampler, batchify_fn, num_workers)
