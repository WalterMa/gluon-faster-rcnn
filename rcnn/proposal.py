from mxnet.gluon import HybridBlock
from .op import proposal_target


class ProposalLayer(HybridBlock):

    def __init__(self, feature_stride, scales, ratios, rpn_pre_nms_top_n, rpn_post_nms_top_n,
                 rpn_nms_threshold, rpn_min_size, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.feature_stride = feature_stride
        self.scales = scales
        self.ratios = ratios
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_min_size = rpn_min_size

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, rpn_cls_prob, rpn_bbox_pred, im_info):
        # Parameters name inconsistent in mxnet v1.2.0
        # https://github.com/apache/incubator-mxnet/pull/10242
        # Return 2D array of [[batch_index, x1, y1, x2, y2]],
        # where (x1, y1) and (x2, y2) are top left and bottom
        # right corners of designated region of interest.
        # batch_index indicates the index of corresponding
        # image in the input array
        rois = F.contrib.MultiProposal(
            cls_prob=rpn_cls_prob, bbox_pred=rpn_bbox_pred, im_info=im_info,
            feature_stride=self.feature_stride, scales=tuple(self.scales),
            ratios=tuple(self.ratios),
            rpn_pre_nms_top_n=self.rpn_pre_nms_top_n, rpn_post_nms_top_n=self.rpn_post_nms_top_n,
            threshold=self.rpn_nms_threshold, rpn_min_size=self.rpn_min_size)
        return rois

    def set_nms(self, rpn_pre_nms_top_n, rpn_post_nms_top_n):
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n


class ProposalTargetLayer(HybridBlock):

    def __init__(self, num_classes, image_batch_size, roi_batch_size, roi_fg_fraction, roi_fg_threshold,
                 roi_bg_threshold_hi, roi_bg_threshold_lo, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(ProposalTargetLayer, self).__init__(**kwargs)
        assert roi_batch_size % image_batch_size == 0, \
            'Image batch size {} must divide ROI batch size {}.'.format(image_batch_size, roi_batch_size)
        self._roi_per_img = int(roi_batch_size / image_batch_size)
        self._fg_threshold = roi_fg_threshold
        self._bg_threshold_hi = roi_bg_threshold_hi
        self._bg_threshold_lo = roi_bg_threshold_lo
        self._fg_per_img = int(self._roi_per_img * roi_fg_fraction)
        self._num_classes = num_classes
        self._bbox_mean = self.params.get_constant('bbox_mean', bbox_mean)
        self._bbox_std = self.params.get_constant('bbox_std', bbox_std)
        # Initialize constant parameters here, if not later will be init by global initializer
        # self._bbox_mean.initialize()
        # self._bbox_std.initialize()

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, rois, gt_boxes, _bbox_mean, _bbox_std):
        rois, labels, bbox_targets = F.Custom(rois, gt_boxes, _bbox_mean, _bbox_std, op_type='ProposalTarget',
                                              num_classes=self._num_classes, roi_per_img=self._roi_per_img,
                                              fg_per_img=self._fg_per_img, roi_fg_threshold=self._fg_threshold,
                                              roi_bg_threshold_hi=self._bg_threshold_hi,
                                              roi_bg_threshold_lo=self._bg_threshold_lo)
        return rois, labels, bbox_targets
