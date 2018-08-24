import mxnet as mx
from mxnet import nd
from rcnn.utils import bbox_overlaps, bbox_transform


class AnchorTargetOp(mx.operator.CustomOp):

    def __init__(self, rpn_batch_size, rpn_fg_num, positive_iou_threshold, negative_iou_threshold):
        super(AnchorTargetOp, self).__init__()
        self._rpn_batch_size = rpn_batch_size
        self._rpn_fg_num = rpn_fg_num
        self._positive_iou_th = positive_iou_threshold
        self._negative_iou_th = negative_iou_threshold

    def forward(self, is_train, req, in_data, out_data, aux):
        # im_info.shape(batch_size, 3)
        feat = in_data[0]
        gt_boxes = in_data[1]
        im_info = in_data[2]
        anchors = in_data[3]
        allowed_border = in_data[4]

        ctx = feat.context
        batch_size = feat.shape[0]
        feat_height, feat_width = feat.shape[-2:]
        A = anchors.shape[-2]
        K = feat_height * feat_width
        N = K * A

        # slice to feat size and broadcast along batch axis
        all_anchors = anchors.slice(begin=(0, 0), end=(feat_height, feat_width))\
            .reshape((1, -1, 4)).broadcast_to((batch_size, N, 4))
        # keep only inside anchors, set outside anchors coordinate = (-1, -1, -1, -1)
        inside_bool_mask = (all_anchors[:, :, 0] >= -allowed_border) * \
                           (all_anchors[:, :, 1] >= -allowed_border) * \
                           (all_anchors[:, :, 2] < (im_info[:, 1] + allowed_border).reshape(0, 1)) * \
                           (all_anchors[:, :, 3] < (im_info[:, 0] + allowed_border).reshape(0, 1))
        all_anchors[:] = inside_bool_mask.reshape(batch_size, -1, 1) * (all_anchors + 1) - 1

        overlaps = bbox_overlaps(gt_boxes, all_anchors)
        # get max iou anchor for each gt_boxes
        gt_max_overlaps = overlaps.max(axis=2)
        gt_argmax_overlaps = overlaps.argmax(axis=2)
        # get max iou for each anchors
        max_overlaps = overlaps.max(axis=1)
        argmax_overlaps = overlaps.argmax(axis=1)
        # set positive anchor label=1, other=0
        labels = max_overlaps >= self._positive_iou_th
        # set neither positive nor negative anchor label = -1
        labels[:] = labels - ((max_overlaps > self._negative_iou_th) * (max_overlaps < self._positive_iou_th))
        # set max iou anchor for each gt_boxes label >=  1 (<=3) and ignore padded gt_box
        batch_idx = nd.arange(batch_size, ctx=ctx).reshape(-1, 1)
        labels[batch_idx, gt_argmax_overlaps] = labels[batch_idx, gt_argmax_overlaps] + 2 * (gt_max_overlaps > 0)
        # set outside anchor label <= -1
        # then remain label=0 is negative samples
        labels[:] = labels - 4 * (1 - inside_bool_mask)
        # clip label values to -1, 0, 1
        labels[:] = nd.clip(labels, -1, 1)

        # random choice labels
        labels_with_idx = nd.concat(labels.transpose(), nd.arange(N, ctx=ctx).reshape(-1, 1), dim=1)
        # column 0:batch_size is label, column batch_size is labels original index
        rand_labels_with_idx = nd.random.shuffle(labels_with_idx)
        # may include some bg_label if labels==1 num < num_fg
        fg_rand_labels_idx = rand_labels_with_idx[:, :batch_size].argsort(axis=0, is_ascend=0)[:self._rpn_fg_num]
        # use abs() to invert all label=-1, so that label=0 will at top after ascend sort
        abs_rand_labels = nd.abs(rand_labels_with_idx[:, :batch_size])
        # set fg_label=-1 to let it at top after ascend sort
        abs_rand_labels[fg_rand_labels_idx, batch_idx.transpose()] = -1
        # select rand labels idx that will be excluded
        exclude_rand_labels_idx = abs_rand_labels.argsort(axis=0, is_ascend=1)[self._rpn_batch_size:]
        # get original label index
        exclude_labels_idx = rand_labels_with_idx[exclude_rand_labels_idx, batch_size]
        # set exclude label = -1

        labels[batch_idx, exclude_labels_idx.transpose()] = -1

        # assign gt_boxes to anchor, anchor box_target is its max iou gt_box
        bbox_targets = nd.empty((batch_size, N, 4), ctx=ctx)
        bbox_targets[:] = bbox_transform(all_anchors, gt_boxes[batch_idx, argmax_overlaps, :4])

        labels = labels.reshape((batch_size, feat_height, feat_width, A)).transpose(axes=(0, 3, 1, 2))
        labels = labels.reshape((batch_size, A * feat_height * feat_width))
        bbox_targets = bbox_targets.reshape((batch_size, feat_height, feat_width, A, 4)).transpose(axes=(0, 4, 3, 1, 2))
        bbox_targets = bbox_targets.reshape((batch_size, 4, A * feat_height * feat_width))

        out_data[0][:] = labels
        out_data[1][:] = bbox_targets

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register('AnchorTarget')
class AnchorTargetProp(mx.operator.CustomOpProp):
    def __init__(self, rpn_batch_size, rpn_fg_num, positive_iou_threshold, negative_iou_threshold):
        super(AnchorTargetProp, self).__init__(need_top_grad=False)
        # All arguments are in string format so we need to convert them
        self._rpn_batch_size = int(rpn_batch_size)
        self._rpn_fg_num = int(rpn_fg_num)
        self._positive_iou_th = float(positive_iou_threshold)
        self._negative_iou_th = float(negative_iou_threshold)

    def list_arguments(self):
        return ['feat', 'gt_boxes', 'im_info', 'anchors', 'allowed_border']

    def list_outputs(self):
        return ['labels', 'bbox_targets']

    def infer_shape(self, in_shape):
        feat_shape = in_shape[0]
        anchors_shape = in_shape[3]

        batch_size = feat_shape[0]
        A = anchors_shape[-2]
        feat_height, feat_width = feat_shape[-2:]

        total_anchor_num = A * feat_height * feat_width

        labels_shape = (batch_size, total_anchor_num)
        bbox_targets_shape = (batch_size, 4, total_anchor_num)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return in_shape, (labels_shape, bbox_targets_shape), ()

    def create_operator(self, ctx, shapes, dtypes):
        return AnchorTargetOp(rpn_batch_size=self._rpn_batch_size, rpn_fg_num=self._rpn_fg_num,
                              positive_iou_threshold=self._positive_iou_th,
                              negative_iou_threshold=self._negative_iou_th)
