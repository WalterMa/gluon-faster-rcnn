import argparse
import mxnet as mx
from mxnet import gluon
from rcnn import FasterRCNN
from rcnn.metrics.voc_detection import VOC07MApMetric
from dataset import VOCDetection
from utils.logger import logger
from utils.config import default, generate_config
from dataset.dataloader import DetectionDataLoader
from rcnn.transforms import FasterRCNNDefaultValTransform
import os
import logging
from tqdm import tqdm


def test_faster_rcnn(net, test_data, cfg):
    """Test on dataset."""

    logger.info('Config for testing FasterRCNN:\n%s' % cfg)
    if cfg.hybridize:
        net.hybridize()

    metric = VOC07MApMetric(iou_thresh=0.5, class_names=cfg.classes)

    with tqdm(total=cfg.dataset_size) as pbar:
        for batch in test_data:
            pred_bboxes = []
            pred_cls = []
            pred_scores = []
            gt_bboxes = []
            gt_cls = []
            gt_difficults = []
            # Split and load data for multi-gpu
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            gt_box_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            im_info_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            for data, gt_box, im_info in zip(data_list, gt_box_list, im_info_list):
                # get prediction results
                cls, scores, bboxes = net(data, im_info)
                pred_cls.append(cls)
                pred_scores.append(scores)
                pred_bboxes.append(bboxes)
                # split ground truths
                gt_cls.append(gt_box.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(gt_box.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(gt_box.slice_axis(axis=-1, begin=5, end=6) if gt_box.shape[-1] > 5 else None)

            # update metric
            metric.update(pred_bboxes, pred_cls, pred_scores, gt_bboxes, gt_cls, gt_difficults)
            pbar.update(batch[0].shape[0])

    return metric.get()


def get_dataset(dataset, dataset_path):
    if dataset.lower() == 'voc':
        dataset = VOCDetection(splits=[(2007, 'test')],
                               transform=FasterRCNNDefaultValTransform(cfg.image_size, cfg.image_max_size,
                                                                       cfg.image_mean, cfg.image_std),
                               root=dataset_path, preload_label=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return dataset


def get_dataloader(dataset, cfg):
    """Get dataloader."""
    loader = DetectionDataLoader(dataset, cfg.batch_size, False, last_batch='keep',
                                 num_workers=cfg.num_workers)
    return loader


def parse_args():
    parser = argparse.ArgumentParser(description='Test Faster RCNN')
    parser.add_argument('--network', type=str, default=default.network,
                        help='network name')
    parser.add_argument('--dataset', type=str, default=default.dataset,
                        help='dataset name')
    parser.add_argument('--dataset-path', default=default.dataset_path, type=str,
                        help='dataset path')
    parser.add_argument('--model-params', type=str, default=default.model_params,
                        help='model params path')
    parser.add_argument('--gpus', nargs='*', type=int, default=default.gpus,
                        help='testing with GPUs, such as --gpus 0 1 ')
    return parser.parse_args()


if __name__ == '__main__':
    # set 0 to disable Running performance tests
    # cmd: set MXNET_CUDNN_AUTOTUNE_DEFAULT=0

    args = parse_args()
    cfg = generate_config(vars(args))

    log_file_path = cfg.save_prefix + '_test.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in cfg.gpus]
    ctx = ctx if ctx else [mx.cpu()]
    num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

    test_dataset = get_dataset(cfg.dataset, cfg.dataset_path)
    test_data = get_dataloader(test_dataset, cfg)
    cfg.dataset_size = len(test_dataset)

    cfg.num_classes = len(test_dataset.classes)
    cfg.classes = test_dataset.classes

    net = FasterRCNN(network=cfg.network, pretrained_base=False, batch_size=cfg.batch_size, num_classes=cfg.num_classes,
                     scales=cfg.anchor_scales, ratios=cfg.anchor_ratios, feature_stride=cfg.feature_stride,
                     allowed_border=cfg.allowed_border, rpn_batch_size=cfg.rpn_batch_size,
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

    map_name, mean_ap = test_faster_rcnn(net, test_data, cfg)
    result_msg = '\n'.join(['%s=%f' % (k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Done] Test Results: \n%s' % result_msg)
