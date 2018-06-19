import os
import mxnet as mx
from .base import BaseDataset


class RecordDataset(BaseDataset):

    def __init__(self, root='.data/', filename='train.rec',
                 transform=None, class_names=None):
        super(RecordDataset, self).__init__(root)
        assert isinstance(class_names, (list, tuple))
        for name in class_names:
            assert isinstance(name, str), "must provide names as str"
        idx_file = os.path.splitext(filename)[0] + '.idx'
        self._record = mx.recordio.MXIndexedRecordIO(idx_file, filename, 'r')
        self._transform = transform
        self._classes = class_names

    def __str__(self):
        return self.__class__.__name__

    @property
    def classes(self):
        return self.classes

    def __len__(self):
        return len(self._record.keys)

    def __getitem__(self, idx):
        record = self._record.read_idx(self._record.keys[idx])
        header, img = mx.recordio.unpack(record)
        if self._transform is not None:
            # MXNet.image.imdecode default return RGB, not BGR in openCV
            return self._transform(mx.image.imdecode(img), header.label)
        return mx.image.imdecode(img), header.label
