import argparse
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
from matplotlib import pyplot as plt
from rcnn import FasterRCNN
from utils.viz import plot_bbox
from utils.config import default, generate_config
from dataset.transforms import load_test


def parse_args():
    parser = argparse.ArgumentParser(description='Demo Faster RCNN')
    parser.add_argument('--network', type=str, default=default.network,
                        help='network name')
    parser.add_argument('--image-path', type=str, default='./data/007944.jpg',
                        help='dataset name')
    parser.add_argument('--model-params', type=str, default=default.model_params,
                        help='model params path')
    return parser.parse_args()


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = mx.nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


if __name__ == '__main__':
    args = parse_args()
    cfg = generate_config(vars(args))

    # demo contexts
    ctx = try_gpu()
    num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

    # provide classes info here
    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(classes)

    net = FasterRCNN(network=cfg.network, pretrained_base=False, batch_size=cfg.batch_size, num_classes=num_classes,
                     scales=cfg.anchor_scales, ratios=cfg.anchor_ratios, feature_stride=cfg.feature_stride,
                     base_size=cfg.base_size, allowed_border=cfg.allowed_border, rpn_batch_size=cfg.rpn_batch_size,
                     rpn_channels=cfg.rpn_channels, roi_mode=cfg.roi_mode, roi_size=cfg.roi_size,
                     rpn_fg_fraction=cfg.rpn_fg_fraction, rpn_positive_threshold=cfg.rpn_positive_threshold,
                     rpn_negative_threshold=cfg.rpn_negative_threshold,
                     rpn_pre_nms_top_n=cfg.rpn_test_pre_nms_top_n, rpn_post_nms_top_n=cfg.rpn_test_post_nms_top_n,
                     rpn_nms_threshold=cfg.rpn_nms_threshold,
                     rpn_min_size=cfg.rpn_min_size, roi_batch_size=cfg.roi_batch_size,
                     roi_fg_fraction=cfg.roi_fg_fraction, roi_fg_threshold=cfg.roi_fg_threshold,
                     roi_bg_threshold_hi=cfg.roi_bg_threshold_hi, roi_bg_threshold_lo=cfg.roi_bg_threshold_lo,
                     bbox_nms_threshold=cfg.bbox_nms_threshold, bbox_nms_top_n=cfg.bbox_nms_top_n,
                     bbox_mean=cfg.bbox_mean, bbox_std=cfg.bbox_std)

    net.load_parameters(cfg.model_params.strip(), ctx=ctx)

    data, orig_img, im_info = load_test(cfg.image_path, size=cfg.image_size, max_size=cfg.image_max_size,
                                        mean=cfg.image_mean, std=cfg.image_std)

    # get prediction results
    cls, scores, bboxes = net(data.as_in_context(ctx), im_info.as_in_context(ctx))

    ax = plot_bbox(orig_img, bboxes[0], scores[0], cls[0], class_names=classes)

    ax.set_axis_off()

    plt.show()
