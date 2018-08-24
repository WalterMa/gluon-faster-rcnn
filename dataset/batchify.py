from mxnet import nd
from mxnet import context


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
            pad = max([l.shape[0] for l in data] + [1, self._label_max_size])
            buf = nd.full((len(data), pad, data[0].shape[-1]), val=-1, ctx=self._ctx)
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

