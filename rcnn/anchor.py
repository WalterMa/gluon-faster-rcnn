from mxnet.gluon import HybridBlock
from .utils import generate_anchors
from .op import anchor_target


class AnchorTargetLayer(HybridBlock):

    def __init__(self, feature_stride, scales=(8, 16, 32), ratios=(0.5, 1, 2),
                 allowed_border=0, rpn_batch_size=256, fg_fraction=0.5, positive_iou_threshold=0.7,
                 negative_iou_threshold=0.3, **kwargs):
        super(AnchorTargetLayer, self).__init__(**kwargs)
        base_anchors = generate_anchors(base_size=feature_stride, scales=scales, ratios=ratios)
        self._base_anchors = self.params.get_constant('base_anchors', base_anchors)
        self._feat_stride = self.params.get_constant('feature_stride', [feature_stride])
        self._allowed_border = self.params.get_constant('allowed_border', [allowed_border])
        self._positive_iou_th = positive_iou_threshold
        self._negative_iou_th = negative_iou_threshold
        self._rpn_batch_size = rpn_batch_size
        self._rpn_fg_num = int(fg_fraction * rpn_batch_size)
        # Initialize constant parameters here, if not later will be init by global initializer
        # self._base_anchors.initialize()
        # self._feat_stride.initialize()
        # self._allowed_border.initialize()

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, rpn_cls_score, gt_boxes, im_info, _base_anchors, _feat_stride, _allowed_border):
        labels, bbox_targets = F.Custom(rpn_cls_score, gt_boxes, im_info, _base_anchors, _feat_stride, _allowed_border,
                                        op_type='AnchorTarget', rpn_batch_size=self._rpn_batch_size,
                                        rpn_fg_num=self._rpn_fg_num, positive_iou_threshold=self._positive_iou_th,
                                        negative_iou_threshold=self._negative_iou_th)
        return labels, bbox_targets
