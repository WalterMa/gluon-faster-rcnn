import numpy as np
from mxnet import nd


__all__ = ['bbox_overlaps', 'bbox_transform', 'generate_anchors']


def bbox_overlaps(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes, ignore coordinate=-1
    :param boxes: batch_size * n * 4 bounding boxes
    :param query_boxes: batch_size * k * 4 bounding boxes
    :return: overlaps: batch_size * n * k overlaps
    """
    query_boxes_mask = (query_boxes[:, :, 0] >= 0).reshape((0, 1, -1))  # shape b, 1, k
    boxes_mask = (boxes[:, :, 0] >= 0).reshape((0, 0, 1))  # shape b, n, 1
    query_boxes_area = (query_boxes[:, :, 2:3] - query_boxes[:, :, 0:1] + 1) * (
            query_boxes[:, :, 3:4] - query_boxes[:, :, 1:2] + 1)
    boxes_area = (boxes[:, :, 2:3] - boxes[:, :, 0:1] + 1) * (boxes[:, :, 3:4] - boxes[:, :, 1:2] + 1)

    n = boxes.shape[1]
    k = query_boxes.shape[1]

    # tile and repeat to match shape(batch_size, n*k, 4)
    query_boxes = nd.tile(query_boxes, reps=(1, n, 1))
    boxes = nd.repeat(boxes, k, axis=1)
    query_boxes_area = nd.tile(query_boxes_area, reps=(1, n, 1))
    boxes_area = nd.repeat(boxes_area, k, axis=1)

    # get max xmin, max ymin
    start = nd.where(boxes[:, :, 0:2] < query_boxes[:, :, 0:2], query_boxes[:, :, 0:2], boxes[:, :, 0:2])
    # get min xmax, min ymax
    end = nd.where(boxes[:, :, 2:4] > query_boxes[:, :, 2:4], query_boxes[:, :, 2:4], boxes[:, :, 2:4])
    wh = end - start + 1
    wh = nd.clip(wh, 0, float("inf"))  # shape(batch_size, n * k, 2)

    i_area = wh[:, :, 0:1] * wh[:, :, 1:2]
    overlaps = i_area / (boxes_area + query_boxes_area - i_area)
    return overlaps.reshape(0, n, k) * query_boxes_mask * boxes_mask


def bbox_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [batch_size, N, 4]
    :param gt_rois: [batch_size, N, 4]
    :return: [batch_size, N, 4]
    """

    ex_widths = ex_rois[:, :, 2:3] - ex_rois[:, :, 0:1] + 1.0
    ex_heights = ex_rois[:, :, 3:4] - ex_rois[:, :, 1:2] + 1.0
    ex_ctr_x = ex_rois[:, :, 0:1] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, :, 1:2] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, :, 2:3] - gt_rois[:, :, 0:1] + 1.0
    gt_heights = gt_rois[:, :, 3:4] - gt_rois[:, :, 1:2] + 1.0
    gt_ctr_x = gt_rois[:, :, 0:1] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, :, 1:2] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = nd.log(gt_widths / ex_widths)
    targets_dh = nd.log(gt_heights / ex_heights)

    targets = nd.concat(targets_dx, targets_dy, targets_dw, targets_dh, dim=-1)
    return targets


def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    ratios = np.array(ratios)
    scales = np.array(scales)
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


if __name__ == '__main__':
    base_anchors = generate_anchors()
    print(base_anchors)
