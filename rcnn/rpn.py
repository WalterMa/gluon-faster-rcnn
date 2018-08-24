from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock


class RPN(HybridBlock):
    """Region Proposal Network Detection Block
    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    num_anchors:
        Number of anchors this RPN should predict.
    """

    def __init__(self, channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        with self.name_scope():
            self.rpn_conv = nn.Conv2D(channels=channels, kernel_size=(3, 3), padding=(1, 1), activation='relu',
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
