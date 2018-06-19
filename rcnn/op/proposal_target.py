import mxnet as mx
from mxnet import nd
from rcnn.utils import bbox_overlaps, bbox_transform


class ProposalTargetOp(mx.operator.CustomOp):

    def __init__(self, num_classes, roi_per_img, fg_per_img, roi_fg_threshold,
                 roi_bg_threshold_hi, roi_bg_threshold_lo):
        super(ProposalTargetOp, self).__init__()
        self._roi_per_img = roi_per_img
        self._fg_threshold = roi_fg_threshold
        self._bg_threshold_hi = roi_bg_threshold_hi
        self._bg_threshold_lo = roi_bg_threshold_lo
        self._fg_per_img = fg_per_img
        self._num_classes = num_classes

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0]
        gt_boxes = in_data[1]
        ctx = gt_boxes.context
        batch_size = gt_boxes.shape[0]  # may be different with image_batch_size when using multi-gpu
        batch_idx = nd.arange(batch_size, ctx=ctx).reshape(-1, 1)
        batch_idx_prefix = batch_idx.reshape((0, 1, 1)).broadcast_to((batch_size, gt_boxes.shape[1], 1))
        gt_rois = nd.concat(batch_idx_prefix, gt_boxes[:, :, :4], dim=-1)
        # reshape roi and append gt_boxes
        rois = nd.concat(rois.reshape(batch_size, -1, 5), gt_rois, dim=1)
        bbox_mean = in_data[2]
        bbox_std = in_data[3]

        # compute overlaps
        overlaps = bbox_overlaps(gt_boxes, rois[:, :, 1:])
        roi_num = overlaps.shape[2]
        # get max iou for each anchors
        max_overlaps = overlaps.max(axis=1)
        argmax_overlaps = overlaps.argmax(axis=1)
        # get gt label for each rois
        labels = gt_boxes[batch_idx, argmax_overlaps, 4]
        true_labels_mask = rois[:, :, 1] >= 0
        # set foreground labels_mask=1, background labels_mask=0, others=-1
        labels_mask = nd.full((batch_size, roi_num), val=-1, ctx=ctx)
        labels_mask[:] = labels_mask + 2 * (max_overlaps >= self._fg_threshold) * true_labels_mask
        labels_mask[:] = labels_mask + (max_overlaps >= self._bg_threshold_lo) * \
                         (max_overlaps < self._bg_threshold_hi) * true_labels_mask

        # random choice labels
        masks_with_idx = nd.concat(labels_mask.transpose(), nd.arange(roi_num, ctx=ctx).reshape(-1, 1), dim=1)
        # column 0:batch_size is labels_mask, colum -1 is labels_mask original index
        rand_masks_with_idx = nd.random.shuffle(masks_with_idx)
        # may include some bg labels_mask if labels_mask==1 num < num_fg_per_img
        fg_rand_masks_idx = rand_masks_with_idx[:, :batch_size].argsort(axis=0, is_ascend=0)[:self._fg_per_img]
        # use abs() to invert all labels_mask=-1, so that labels_mask=0 will at top after ascend sort
        abs_rand_masks = nd.abs(rand_masks_with_idx[:, :batch_size])
        # set fg_label=-1 to let it at top after ascend sort
        abs_rand_masks[fg_rand_masks_idx, batch_idx.transpose()] = -1
        # select rand labels_mask idx that we need
        selected_rand_labels_idx = abs_rand_masks.argsort(axis=0, is_ascend=1)[:self._roi_per_img]

        # get original labels_mask index
        selected_labels_idx = (rand_masks_with_idx[selected_rand_labels_idx, batch_size]).transpose()
        # select and clip labels_mask value to 0,1
        selected_labels_mask = nd.clip(labels_mask[batch_idx, selected_labels_idx], 0, 1)
        # select and set bg labels value=0
        selected_labels = selected_labels_mask * labels[batch_idx, selected_labels_idx]
        selected_rois = rois[batch_idx, selected_labels_idx]
        selected_gt_boxes = gt_boxes[batch_idx, argmax_overlaps[batch_idx, selected_labels_idx], :4]

        # assign gt_boxes to selected_rois
        bbox_targets = nd.empty((batch_size, self._roi_per_img, 4), ctx=ctx)
        bbox_targets[:] = bbox_transform(selected_rois[:, :, 1:], selected_gt_boxes)

        selected_rois = selected_rois.reshape(-1, 5)
        selected_labels = selected_labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, 4)

        # Normalize bbox targets
        bbox_targets[:] = (bbox_targets - bbox_mean) / bbox_std

        # expand bbox_targets to[-1, num_classes * 4]
        # only the right class has non-zero bbox regression targets
        selected_label_num = selected_labels.shape[0]
        selected_label_idx = nd.arange(selected_label_num, ctx=ctx).reshape(-1, 1)
        expanded_bbox_targets = nd.zeros((selected_label_num, self._num_classes, 4), ctx=ctx)
        expanded_bbox_targets[selected_label_idx, selected_labels[:], :] = bbox_targets.reshape(-1, 1, 4)
        expanded_bbox_targets = expanded_bbox_targets.reshape(-1, self._num_classes * 4)

        out_data[0][:] = selected_rois
        out_data[1][:] = selected_labels
        out_data[2][:] = expanded_bbox_targets

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register('ProposalTarget')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, roi_per_img, fg_per_img, roi_fg_threshold,
                 roi_bg_threshold_hi, roi_bg_threshold_lo):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        # All arguments are in string format so we need to convert them
        self._roi_per_img = int(roi_per_img)
        self._fg_per_img = int(fg_per_img)
        self._num_classes = int(num_classes)
        self._fg_threshold = float(roi_fg_threshold)
        self._bg_threshold_hi = float(roi_bg_threshold_hi)
        self._bg_threshold_lo = float(roi_bg_threshold_lo)

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'bbox_mean', 'bbox_std']

    def list_outputs(self):
        return ['rois', 'labels', 'bbox_targets']

    def infer_shape(self, in_shape):
        gt_boxes_shape = in_shape[1]

        batch_size = gt_boxes_shape[0]
        output_roi_num = batch_size * self._roi_per_img

        output_rois_shape = (output_roi_num, 5)
        labels_shape = (output_roi_num, 1)
        bbox_targets_shape = (output_roi_num, self._num_classes * 4)
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return in_shape, (output_rois_shape, labels_shape, bbox_targets_shape), ()

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOp(num_classes=self._num_classes, roi_per_img=self._roi_per_img,
                                fg_per_img=self._fg_per_img, roi_fg_threshold=self._fg_threshold,
                                roi_bg_threshold_hi=self._bg_threshold_hi, roi_bg_threshold_lo=self._bg_threshold_lo)
