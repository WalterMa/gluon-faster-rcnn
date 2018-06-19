from mxnet.gluon import HybridBlock


class NormalizedBoxDecoder(HybridBlock):
    """Decode bounding boxes training target with normalized center offsets.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).

    """
    def __init__(self, num_classes, means=(0.0, 0.0, 0.0, 0.0), stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._fg_class = num_classes - 1
        self._stds = stds
        self._means = means

    def hybrid_forward(self, F, x, rois, im_info):
        # x shape(batch_size, roi_num, num_classes*4)
        # rois shape(batch_size, roi_num, 5)
        # im_info shape(batch_size, 3)
        rois = rois.slice_axis(axis=-1, begin=1, end=5)  # remove batch_idx in rois first column
        a = rois.split(axis=-1, num_outputs=4)
        a_w = a[2] - a[0] + 1
        a_h = a[3] - a[1] + 1
        a_ctr_x = a[0] + 0.5 * (a_w - 1)
        a_ctr_y = a[1] + 0.5 * (a_h - 1)
        im_width = im_info.slice(begin=(None, 1), end=(None, 2)).reshape((0, 1, 1))
        im_height = im_info.slice(begin=(None, 0), end=(None, 1)).reshape((0, 1, 1))
        cls_bbox = []
        for i in range(self._fg_class):
            p = F.slice(x, begin=(None, None, 4*(i+1)), end=(None, None, 4*(i+2)))
            p = p.split(axis=-1, num_outputs=4)
            ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0] + self._means[0], a_w), a_ctr_x)
            oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1] + self._means[1], a_h), a_ctr_y)
            ow = F.broadcast_mul(F.exp(p[2] * self._stds[2] + self._means[2]), a_w) / 2
            oh = F.broadcast_mul(F.exp(p[3] * self._stds[3] + self._means[3]), a_h) / 2
            x1 = F.maximum(F.broadcast_minimum(ox - ow, im_width - 1), 0)
            y1 = F.maximum(F.broadcast_minimum(oy - oh, im_height - 1), 0)
            x2 = F.maximum(F.broadcast_minimum(ox + ow, im_width - 1), 0)
            y2 = F.maximum(F.broadcast_minimum(oy + oh, im_height - 1), 0)
            cls_bbox.append(F.concat(x1, y1, x2, y2, dim=-1))
        cls_bbox = F.concat(*cls_bbox, dim=-1)  # b, N, fg_class*4
        return cls_bbox


class MultiPerClassDecoder(HybridBlock):
    """Decode classification results.

    For each position(anchor boxes), each foreground class can have their own
    results, rather than enforced to be the best one.
    For example, for a 5-class prediction with background(totaling 6 class), say
    (0.5, 0.1, 0.2, 0.1, 0.05, 0.05) as (bg, apple, orange, peach, grape, melon),
    `MultiClassDecoder` produce only one class id and score, that is  (orange-0.2).
    `MultiPerClassDecoder` produce 5 results individually:
    (apple-0.1, orange-0.2, peach-0.1, grape-0.05, melon-0.05).

    Parameters
    ----------
    num_classes : int
        Number of classes including background.
    axis : int
        Axis of class-wise results.
    thresh : float
        Confidence threshold for the post-softmax scores.
        Scores less than `thresh` are marked with `0`, corresponding `cls_id` is
        marked with invalid class id `-1`.

    """
    def __init__(self, num_classes, axis=-1, thresh=0.01):
        super(MultiPerClassDecoder, self).__init__()
        self._fg_class = num_classes - 1
        self._axis = axis
        self._thresh = thresh

    def hybrid_forward(self, F, x):
        # x shape(batch_size, roi_num, num_class)
        scores = x.slice_axis(axis=self._axis, begin=1, end=None)  # b x N x fg_class
        template = F.zeros_like(x.slice_axis(axis=-1, begin=0, end=1))
        cls_ids = []
        for i in range(self._fg_class):
            cls_ids.append(template + i)  # b x N x 1
        cls_id = F.concat(*cls_ids, dim=-1)  # b x N x fg_class
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores
