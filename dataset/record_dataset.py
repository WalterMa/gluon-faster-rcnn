import os
import mxnet as mx
import numpy as np
from .base import BaseDataset


class RecordDataset(BaseDataset):

    def __init__(self, root='.data/', filename='train.rec',
                 transform=None, class_names=None):
        super(RecordDataset, self).__init__(root)
        assert isinstance(class_names, (list, tuple))
        for name in class_names:
            assert isinstance(name, str), "must provide names as str"
        self.idx_path = os.path.join(root, os.path.splitext(filename)[0] + '.idx')
        self.rec_path = os.path.join(root, filename)
        self._transform = transform
        self._classes = class_names
        self._fork()

    def __str__(self):
        return self.__class__.__name__

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._record.keys)

    def __getitem__(self, idx):
        record = self._record.read_idx(self._record.keys[idx])
        header, img = mx.recordio.unpack(record)
        if not isinstance(header.label, np.ndarray):
            label = np.array(header.label)
        else:
            label = header.label
        label_width = int(label[1])
        extra_header_width = int(label[0])
        label = label[extra_header_width:].reshape((-1, label_width)).copy()
        if self._transform is not None:
            # MXNet.image.imdecode default return RGB, not BGR in openCV
            return self._transform(mx.image.imdecode(img), label)
        return mx.image.imdecode(img), header.label

    def _fork(self):
        self._record = mx.recordio.MXIndexedRecordIO(self.idx_path, self.rec_path, 'r')

    def get_imglist(self):
        raise NotImplementedError
