from mxnet.gluon import Block
from mxnet import nd
from mxnet import autograd
import numpy as np


class RPNLoss(Block):

    def __init__(self, rpn_batch_size=256, bbox_weights=(1.0, 1.0, 1.0, 1.0), **kwargs):
        super(RPNLoss, self).__init__(**kwargs)
        self._rpn_batch_size = self.params.get_constant('rpn_batch_size', [rpn_batch_size])
        # bbox_weights shape (1, 4, 1)
        bbox_weights = np.array(bbox_weights).reshape((1, 4, 1))
        self._bbox_weights = self.params.get_constant('bbox_weights', bbox_weights)

    def forward(self, rpn_cls_pred, rpn_bbox_pred, rpn_cls_gt, rpn_bbox_gt):
        with autograd.pause():
            ctx = rpn_cls_pred.context
            batch_size = rpn_cls_pred.shape[0]
            # construct cls_mask to ignore label=-1
            cls_mask = nd.stack(rpn_cls_gt == 0, rpn_cls_gt == 1, axis=1)
            bbox_weights = (rpn_cls_gt == 1).reshape(batch_size, 1, -1) * self._bbox_weights.data(ctx)

        # reshape -> (batch_size, 2, num_anchors*feat_h*feat_w)
        rpn_cls_log = nd.log(nd.clip(rpn_cls_pred.reshape((batch_size, 2, -1)), 1e-14, 1))
        cls_log_loss = -nd.sum(rpn_cls_log * cls_mask) / self._rpn_batch_size.data(ctx)

        # reshape -> (batch_size, 4, num_anchors*feat_h*feat_w)
        rpn_bbox_smooth_l1 = nd.smooth_l1(rpn_bbox_pred.reshape((batch_size, 4, -1)) - rpn_bbox_gt, scalar=3.0)
        bbox_smooth_l1_loss = nd.sum(rpn_bbox_smooth_l1 * bbox_weights) / self._rpn_batch_size.data(ctx)

        return cls_log_loss, bbox_smooth_l1_loss


class RCNNLoss(Block):

    def __init__(self, roi_batch_size=128, bbox_weights=(1.0, 1.0, 1.0, 1.0), **kwargs):
        super(RCNNLoss, self).__init__(**kwargs)
        self._roi_batch_size = self.params.get_constant('roi_batch_size', [roi_batch_size])
        # bbox_weights shape (1, 1, 4)
        bbox_weights = np.array(bbox_weights).reshape((1, 1, 4))
        self._bbox_weights = self.params.get_constant('bbox_weights', bbox_weights)

    def forward(self, rcnn_cls_pred, rcnn_bbox_pred, rcnn_cls_gt, rcnn_bbox_gt):
        with autograd.pause():
            ctx = rcnn_cls_pred.context
            roi_num = rcnn_cls_pred.shape[0]
            roi_idx = nd.arange(roi_num, ctx=ctx).reshape(-1, 1)
            fg_bbox_mask = (rcnn_cls_gt > 0).reshape(0, 1, 1)
            bbox_weights = nd.zeros_like(rcnn_bbox_gt).reshape(0, -1, 4)
            bbox_weights[roi_idx, rcnn_cls_gt[:], :] = \
                self._bbox_weights.data(ctx).broadcast_to((roi_num, 1, 4)) * fg_bbox_mask
            bbox_weights = bbox_weights.reshape(0, -1)

        # rcnn_cls_pred.shape (roi_num, num_classes)
        rcnn_cls_log = nd.log(nd.clip(rcnn_cls_pred, 1e-14, 1))
        cls_log_loss = -nd.sum(rcnn_cls_log[roi_idx, rcnn_cls_gt]) / self._roi_batch_size.data(ctx)

        # rcnn_bbox_pred.shape (roi_num, num_classes*4)
        rcnn_bbox_smooth_l1 = nd.smooth_l1(rcnn_bbox_pred - rcnn_bbox_gt, scalar=1.0)
        bbox_smooth_l1_loss = nd.sum(rcnn_bbox_smooth_l1 * bbox_weights) / self._roi_batch_size.data(ctx)

        return cls_log_loss, bbox_smooth_l1_loss

