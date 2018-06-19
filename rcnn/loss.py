from mxnet.gluon import Block
from mxnet import nd


class RPNLoss(Block):

    def __init__(self, rpn_batch_size=256, **kwargs):
        super(RPNLoss, self).__init__(**kwargs)
        self._rpn_batch_size = self.params.get_constant('rpn_batch_size', [rpn_batch_size])

    def forward(self, rpn_cls_pred, rpn_bbox_pred, rpn_cls_gt, rpn_bbox_gt):
        ctx = rpn_cls_pred.context
        batch_size = rpn_cls_pred.shape[0]
        # reshape -> (batch_size, 2, num_anchors*feat_h*feat_w)
        rpn_cls_log = nd.log(nd.clip(rpn_cls_pred.reshape((batch_size, 2, -1)), 1e-14, 1))
        # construct cls_mask to ignore label=-1
        cls_mask = nd.stack(rpn_cls_gt == 0, rpn_cls_gt == 1, axis=1)
        cls_log_loss = -nd.sum(rpn_cls_log * cls_mask) / self._rpn_batch_size.data(ctx)

        # reshape -> (batch_size, 4, num_anchors*feat_h*feat_w)
        rpn_bbox_smooth_l1 = nd.smooth_l1(rpn_bbox_pred.reshape((batch_size, 4, -1)) - rpn_bbox_gt, scalar=3.0)
        # construct reg_mask to select label=1
        reg_mask = (rpn_cls_gt == 1).reshape(batch_size, 1, -1)
        bbox_smooth_l1_loss = nd.sum(rpn_bbox_smooth_l1 * reg_mask) / (nd.sum(reg_mask) + 1e-14)

        return cls_log_loss, bbox_smooth_l1_loss


class RCNNLoss(Block):

    def __init__(self, roi_batch_size=128, **kwargs):
        super(RCNNLoss, self).__init__(**kwargs)
        self._roi_batch_size = self.params.get_constant('roi_batch_size', [roi_batch_size])

    def forward(self, rcnn_cls_pred, rcnn_bbox_pred, rcnn_cls_gt, rcnn_bbox_gt):
        ctx = rcnn_cls_pred.context
        roi_num = rcnn_cls_pred.shape[0]
        roi_idx = nd.arange(roi_num, ctx=ctx).reshape(-1, 1)

        # rcnn_cls_pred.shape (roi_num, num_classes)
        rcnn_cls_log = nd.log(nd.clip(rcnn_cls_pred, 1e-14, 1))
        cls_log_loss = -nd.sum(rcnn_cls_log[roi_idx, rcnn_cls_gt]) / self._roi_batch_size.data(ctx)

        # rcnn_bbox_pred.shape (roi_num, num_classes*4)
        rcnn_bbox_smooth_l1 = nd.smooth_l1(rcnn_bbox_pred - rcnn_bbox_gt, scalar=1.0)
        reg_mask = nd.clip(rcnn_cls_gt, 0, 1).reshape(-1, 1)
        bbox_smooth_l1_loss = nd.sum(rcnn_bbox_smooth_l1 * reg_mask) / (nd.sum(reg_mask) + 1e-14)

        return cls_log_loss, bbox_smooth_l1_loss

