import warnings
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from .rpn import RPN
from .anchor import AnchorTargetLayer
from .proposal import ProposalLayer, ProposalTargetLayer
from .base_model import get_model_features
from .decoder import NormalizedBoxDecoder, MultiPerClassDecoder


class FasterRCNN(HybridBlock):

    def __init__(self, network, pretrained_base, num_classes, batch_size, scales, ratios, feature_stride, base_size,
                 allowed_border, rpn_channels, rpn_batch_size, rpn_fg_fraction, rpn_positive_threshold,
                 rpn_negative_threshold, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_threshold, rpn_min_size,
                 roi_mode, roi_size, roi_batch_size, roi_fg_fraction, roi_fg_threshold,
                 roi_bg_threshold_hi, roi_bg_threshold_lo, bbox_nms_threshold, bbox_nms_top_n,
                 bbox_mean=(0.0, 0.0, 0.0, 0.0), bbox_std=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)
        # Ignore not used params/data warning when call forward,
        # since network inputs and outputs are different in train and test stage
        warnings.filterwarnings('ignore', message='.*not used.*', module=r'mxnet\.gluon\.block', append=True)
        num_anchors = len(scales) * len(ratios)
        num_classes = num_classes + 1  # Add 1 for background class
        self.num_classes = num_classes
        self.feature_stride = feature_stride
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self.roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self.roi_size = roi_size
        self.bbox_nms_thresh = bbox_nms_threshold
        self.bbox_nms_top_n = bbox_nms_top_n
        features, top_features = get_model_features(name=network, pretrained=pretrained_base)
        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.anchor_target = AnchorTargetLayer(feature_stride=feature_stride, base_size=base_size,
                                                   scales=scales, ratios=ratios,
                                                   allowed_border=allowed_border, rpn_batch_size=rpn_batch_size,
                                                   fg_fraction=rpn_fg_fraction,
                                                   positive_iou_threshold=rpn_positive_threshold,
                                                   negative_iou_threshold=rpn_negative_threshold)
            self.proposal = ProposalLayer(feature_stride=feature_stride, scales=scales, ratios=ratios,
                                          rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                                          rpn_post_nms_top_n=rpn_post_nms_top_n,
                                          rpn_nms_threshold=rpn_nms_threshold,
                                          rpn_min_size=rpn_min_size)
            self.proposal_target = ProposalTargetLayer(num_classes=num_classes, image_batch_size=batch_size,
                                                       roi_batch_size=roi_batch_size, roi_fg_fraction=roi_fg_fraction,
                                                       roi_fg_threshold=roi_fg_threshold,
                                                       roi_bg_threshold_hi=roi_bg_threshold_hi,
                                                       roi_bg_threshold_lo=roi_bg_threshold_lo,
                                                       bbox_mean=bbox_mean, bbox_std=bbox_std)
            self.rpn = RPN(channels=rpn_channels, num_anchors=num_anchors)
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.class_predictor = nn.Dense(num_classes, weight_initializer=mx.init.Normal(0.01))
            self.box_predictor = nn.Dense(num_classes * 4, weight_initializer=mx.init.Normal(0.001))
            self.bbox_decoder = NormalizedBoxDecoder(num_classes=num_classes, means=bbox_mean, stds=bbox_std)
            self.cls_decoder = MultiPerClassDecoder(num_classes=num_classes, thresh=0.01)

    def initialize(self, init=None, ctx=None, verbose=False, force_reinit=False):
        if force_reinit:
            self.collect_params().initialize(init=init, ctx=ctx, verbose=verbose, force_reinit=True)
        else:
            # ignore pretrained network
            for param in self.collect_params().values():
                if param._data is not None:
                    param.reset_ctx(ctx)
                else:
                    param.initialize(init=init, ctx=ctx, force_reinit=False)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, im_info, gt_boxes=None):
        feat = self.features(x)
        rpn_cls_prob, rpn_bbox_pred = self.rpn(feat)
        rois = self.proposal(rpn_cls_prob, rpn_bbox_pred, im_info)

        # Generate and sample targets
        if autograd.is_training():
            with autograd.pause():
                rois = F.stop_gradient(rois)
                # When training, get and output anchor/proposal targets for loss computation
                rpn_label, rpn_bbox_target = self.anchor_target(feat, gt_boxes, im_info)
                # Add 1 to cls_id for background class
                gt_boxes = F.concat(gt_boxes.slice_axis(axis=-1, begin=0, end=4),
                                    gt_boxes.slice_axis(axis=-1, begin=4, end=5) + 1, dim=-1)
                rois, rois_label, rois_bbox_target = self.proposal_target(rois, gt_boxes)

        # ROI features
        if self.roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rois, self.roi_size, 1. / self.feature_stride)
        elif self.roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, rois, self.roi_size, 1. / self.feature_stride,
                                             sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self.roi_mode))

        # RCNN prediction
        top_feat = self.top_features(pooled_feat)
        avg_feat = self.global_avg_pool(top_feat)
        rcnn_cls_score = self.class_predictor(avg_feat)
        rcnn_cls_prob = F.softmax(data=rcnn_cls_score)
        rcnn_bbox_pred = self.box_predictor(avg_feat)

        if autograd.is_training():
            return rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target, \
                   rcnn_cls_prob, rcnn_bbox_pred, rois_label, rois_bbox_target

        rois = rois.reshape((-1, self.proposal.rpn_post_nms_top_n, 5))
        rcnn_cls_prob = rcnn_cls_prob.reshape((-1, self.proposal.rpn_post_nms_top_n, self.num_classes))
        rcnn_bbox_pred = rcnn_bbox_pred.reshape((-1, self.proposal.rpn_post_nms_top_n, self.num_classes * 4))
        # background pred values will be removed after decode
        bboxes = self.bbox_decoder(rcnn_bbox_pred, rois, im_info)
        # background cls will be removed after decode, cls_id still starts from 0
        cls_ids, scores = self.cls_decoder(rcnn_cls_prob)
        results = []
        for i in range(self.num_classes - 1):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = scores.slice_axis(axis=-1, begin=i, end=i + 1)
            bbox = bboxes.slice_axis(axis=-1, begin=4 * i, end=4 * (i + 1))
            # per class results
            per_result = F.concat(*[cls_id, score, bbox], dim=-1)
            if (self.bbox_nms_thresh > 0) and (self.bbox_nms_thresh < 1):
                per_result = F.contrib.box_nms(
                    per_result, overlap_thresh=self.bbox_nms_thresh, topk=self.bbox_nms_top_n,
                    id_index=0, score_index=1, coord_start=2)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        cls = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return cls, scores, bboxes
