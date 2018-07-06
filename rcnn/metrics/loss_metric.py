import mxnet as mx
from mxnet import nd


class SmoothL1LossMetric(mx.metric.EvalMetric):

    def __init__(self, name='SmoothL1Loss', batch_size=256, output_names=None, label_names=None):
        super(SmoothL1LossMetric, self).__init__(name, output_names, label_names)
        self._batch_size = batch_size

    def update(self, labels=None, preds=None):

        if isinstance(preds, nd.ndarray.NDArray):
            preds = [preds]

        for pred in preds:
            # Since we normalized by divide positive instance num when compute loss,
            # we simply add here
            self.sum_metric += nd.sum(pred).asscalar()
        self.num_inst += 1


class LogLossMetric(mx.metric.EvalMetric):

    def __init__(self, name='LogLoss', batch_size=256, output_names=None, label_names=None):
        super(LogLossMetric, self).__init__(name, output_names, label_names)
        self._batch_size = batch_size

    def update(self, labels=None, preds=None):

        if isinstance(preds, nd.ndarray.NDArray):
            preds = [preds]

        for pred in preds:
            # Since we already normalized by divide rpn or roi batch size when compute loss,
            # we simply add here
            self.sum_metric += nd.sum(pred).asscalar()
        self.num_inst += 1
