import warnings
from mxnet import autograd
from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from mxnet.gluon.model_zoo.vision import get_model

from .base_model import get_model_conv_block
from .anchor import AnchorTargetLayer


class RPNHead(HybridBlock):
    """Region Proposal Network Detection Head
    Parameters
    ----------
    num_anchors: The number of anchors this RPN should predict.
    """

    def __init__(self, num_anchors, **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        with self.name_scope():
            self.rpn_conv = nn.Conv2D(channels=512, kernel_size=(3, 3), padding=(1, 1), activation='relu',
                                      weight_initializer=initializer.Normal(0.01))
            self.rpn_cls_score = nn.Conv2D(channels=2 * num_anchors, kernel_size=(1, 1), padding=(0, 0),
                                           weight_initializer=initializer.Normal(0.01))
            self.rpn_bbox_pred = nn.Conv2D(channels=4 * num_anchors, kernel_size=(1, 1), padding=(0, 0),
                                           weight_initializer=initializer.Normal(0.01))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rpn_conv(x)
        rpn_cls_score = self.rpn_cls_score(x)
        rpn_cls_prob = F.softmax(data=rpn_cls_score.reshape((0, 2, -1, 0)), axis=1)
        rpn_cls_prob = rpn_cls_prob.reshape((0, 2*self.num_anchors, -1, 0))
        rpn_bbox_pred = self.rpn_bbox_pred(x)
        return rpn_cls_prob, rpn_bbox_pred


class RPN(HybridBlock):

    def __init__(self, network, pretrained_base, scales, ratios, feature_stride,
                 allowed_border, rpn_batch_size, rpn_fg_fraction, rpn_positive_threshold,
                 rpn_negative_threshold, **kwargs):
        super(RPN, self).__init__(**kwargs)
        # Ignore not used params/data warning when call forward,
        # since network inputs and outputs are different in train and test stage
        warnings.filterwarnings('ignore', message='.*not used.*', module=r'mxnet\.gluon\.block', append=True)
        base_model = get_model(name=network, pretrained=pretrained_base)
        num_anchors = len(scales) * len(ratios)
        with self.name_scope():
            self.feature = get_model_conv_block(name=network, base_model=base_model)
            self.rpn = RPNHead(num_anchors=num_anchors)
            self.anchor_target = AnchorTargetLayer(feature_stride=feature_stride, scales=scales, ratios=ratios,
                                                   allowed_border=allowed_border, rpn_batch_size=rpn_batch_size,
                                                   fg_fraction=rpn_fg_fraction,
                                                   positive_iou_threshold=rpn_positive_threshold,
                                                   negative_iou_threshold=rpn_negative_threshold)

    def initialize(self, init=None, ctx=None, verbose=False, force_reinit=False):
        if force_reinit:
            self.collect_params().initialize(init=init, ctx=ctx, verbose=verbose, force_reinit=force_reinit)
        else:
            # ignore pretrained network
            for param in self.collect_params().values():
                if param._data is not None:
                    param.reset_ctx(ctx)
                else:
                    param.initialize(init=init, ctx=ctx, force_reinit=force_reinit)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, im_info=None, gt_boxes=None):
        x = self.feature(x)
        rpn_cls_prob, rpn_bbox_pred = self.rpn(x)
        if autograd.is_training():
            # When training, get and output anchor targets for loss computation
            rpn_label, rpn_bbox_target = self.anchor_target(rpn_cls_prob, gt_boxes, im_info)
            return rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target
        else:
            return rpn_cls_prob, rpn_bbox_pred
