import numpy as np
from mxnet.gluon import HybridBlock
from .op import anchor_target


class AnchorTargetLayer(HybridBlock):

    def __init__(self, feature_stride, base_size, scales=(8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(128, 128),
                 allowed_border=0, rpn_batch_size=256, fg_fraction=0.5, positive_iou_threshold=0.7,
                 negative_iou_threshold=0.3, **kwargs):
        super(AnchorTargetLayer, self).__init__(**kwargs)
        anchors = self._generate_anchors(feature_stride, base_size, ratios, scales, alloc_size)
        self._anchors = self.params.get_constant('anchors', anchors)
        self._allowed_border = self.params.get_constant('allowed_border', [allowed_border])
        self._positive_iou_th = positive_iou_threshold
        self._negative_iou_th = negative_iou_threshold
        self._rpn_batch_size = rpn_batch_size
        self._rpn_fg_num = int(fg_fraction * rpn_batch_size)

    def _generate_anchors(self, stride, base_size, ratios, scales, alloc_size):
        """Pre-generate all anchors."""
        # generate same shapes on every location
        px, py = (base_size - 1) * 0.5, (base_size - 1) * 0.5
        base_sizes = []
        for r in ratios:
            for s in scales:
                size = base_size * base_size / r
                ws = np.round(np.sqrt(size))
                w = (ws * s - 1) * 0.5
                h = (np.round(ws * r) * s - 1) * 0.5
                base_sizes.append([px - w, py - h, px + w, py + h])
        base_sizes = np.array(base_sizes)  # (N, 4)

        # propagete to all locations by shifting offsets
        height, width = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        # broadcast_add (1, N, 4) + (M, 1, 4)
        anchors = (base_sizes.reshape((1, -1, 4)) + offsets.reshape((-1, 1, 4)))
        anchors = anchors.reshape((height, width, -1, 4)).astype(np.float32)
        return anchors

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, feat, gt_boxes, im_info, _anchors, _allowed_border):
        labels, bbox_targets = F.Custom(feat, gt_boxes, im_info, _anchors, _allowed_border,
                                        op_type='AnchorTarget', rpn_batch_size=self._rpn_batch_size,
                                        rpn_fg_num=self._rpn_fg_num, positive_iou_threshold=self._positive_iou_th,
                                        negative_iou_threshold=self._negative_iou_th)
        return labels, bbox_targets
