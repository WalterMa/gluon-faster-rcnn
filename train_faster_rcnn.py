import argparse
import logging
import time
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon.data import DataLoader
from rcnn import FasterRCNN
from rcnn.loss import RPNLoss, RCNNLoss
from rcnn.metrics.loss_metric import LogLossMetric, SmoothL1LossMetric
from rcnn.metrics.voc_detection import VOC07MApMetric
from dataset import VOCDetection, RecordDataset
from utils import set_random_seed, fix_net_params
from utils.logger import logger
from utils.config import default, generate_config
from dataset.transforms import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform
from dataset.batchify import FasterRCNNDefaultBatchify

def train_faster_rcnn(net, train_data, val_data, cfg):
    """Training pipeline"""
    rpn_loss = RPNLoss(cfg.rpn_batch_size)
    rcnn_loss = RCNNLoss(cfg.roi_batch_size)
    rpn_loss.initialize(ctx=ctx)
    rcnn_loss.initialize(ctx=ctx)

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': cfg.lr, 'wd': cfg.wd, 'momentum': cfg.momentum, 'clip_gradient': 5})

    # lr decay policy
    lr_decay = float(cfg.lr_decay)
    lr_steps = sorted(cfg.lr_decay_epochs)

    # Create Metrics
    rpn_log_metric = LogLossMetric(name='RPNLogLoss', batch_size=cfg.rpn_batch_size)
    rpn_smoothl1_metric = SmoothL1LossMetric(name='RPNSmoothL1Loss', batch_size=cfg.rpn_batch_size)
    rcnn_log_metric = LogLossMetric(name='RCNNLogLoss', batch_size=cfg.roi_batch_size)
    rcnn_smoothl1_metric = SmoothL1LossMetric(name='RCNNSmoothL1Loss', batch_size=cfg.roi_batch_size)
    # New list to store loss and label for backward and update metric
    rpn_cls_loss_list = []
    rpn_bbox_loss_list = []
    rcnn_cls_loss_list = []
    rcnn_bbox_loss_list = []

    logger.info('Config for end to end training FasterRCNN:\n%s' % cfg)
    logger.info('Start training from [Epoch %d]' % args.start_epoch)
    best_map = [0]
    for epoch in range(cfg.start_epoch, cfg.end_epoch):
        # When hybridize is true, set network to train mode, reset proposal nms params
        # then clear and cache new compute graph
        net.proposal.set_nms(cfg.rpn_pre_nms_top_n, cfg.rpn_post_nms_top_n)
        if cfg.hybridize:
            autograd.set_training(train_mode=True)
            net.hybridize()

        # Check and update learning rate
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        # Refresh time and metrics
        tic = time.time()
        btic = time.time()
        rpn_log_metric.reset()
        rpn_smoothl1_metric.reset()
        rcnn_log_metric.reset()
        rcnn_smoothl1_metric.reset()

        for i, batch in enumerate(train_data):
            # Empty lists
            rpn_cls_loss_list[:] = []
            rpn_bbox_loss_list[:] = []
            rcnn_cls_loss_list[:] = []
            rcnn_bbox_loss_list[:] = []
            # Split and load data for multi-gpu
            batch_size = batch[0].shape[0]
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            gt_box_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            im_info_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            # Network Forward
            with autograd.record():
                for data, gt_box, im_info in zip(data_list, gt_box_list, im_info_list):
                    rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target, \
                    rcnn_cls_prob, rcnn_bbox_pred, rcnn_label, rcnn_bbox_target = net(data, im_info, gt_box)
                    rpn_cls_loss, rpn_bbox_loss = \
                        rpn_loss(rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target)
                    rcnn_cls_loss, rcnn_bbox_loss = \
                        rcnn_loss(rcnn_cls_prob, rcnn_bbox_pred, rcnn_label, rcnn_bbox_target)
                    rpn_cls_loss_list.append(rpn_cls_loss)
                    rpn_bbox_loss_list.append(rpn_bbox_loss)
                    rcnn_cls_loss_list.append(rcnn_cls_loss)
                    rcnn_bbox_loss_list.append(rcnn_bbox_loss)
            # Backward and update parameters and metrics
            autograd.backward(rpn_cls_loss_list + rpn_bbox_loss_list + rcnn_cls_loss_list + rcnn_bbox_loss_list)
            trainer.step(1)
            rpn_log_metric.update(preds=rpn_cls_loss_list)
            rpn_smoothl1_metric.update(preds=rpn_bbox_loss_list)
            rcnn_log_metric.update(preds=rcnn_cls_loss_list)
            rcnn_smoothl1_metric.update(preds=rcnn_bbox_loss_list)

            # Log training states
            if cfg.log_interval and not (i + 1) % cfg.log_interval:
                name1, loss1 = rpn_log_metric.get()
                name2, loss2 = rpn_smoothl1_metric.get()
                name3, loss3 = rcnn_log_metric.get()
                name4, loss4 = rcnn_smoothl1_metric.get()
                logger.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f, %s=%f'
                            % (epoch, i, batch_size / (time.time() - btic), name1, loss1, name2, loss2,
                               name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = rpn_log_metric.get()
        name2, loss2 = rpn_smoothl1_metric.get()
        name3, loss3 = rcnn_log_metric.get()
        name4, loss4 = rcnn_smoothl1_metric.get()
        logger.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f, %s=%f, %s=%f' % (
            epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))

        map_name, mean_ap = validate_faster_rcnn(net, val_data, cfg)
        val_msg = '\n'.join(['%s=%f' % (k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Epoch %d] Validation: \n%s' % (epoch, val_msg))
        save_params(net, best_map, mean_ap[-1], epoch, cfg.save_interval, cfg.save_prefix)


def validate_faster_rcnn(net, val_data, cfg):
    """Test on validation dataset."""
    # When hybridize is true, set network to test mode, set proposal nms test params
    # then clear and cache new compute graph
    # FIXME Will raise deferred init error if call hybridized net in test mode first
    net.proposal.set_nms(cfg.rpn_test_pre_nms_top_n, cfg.rpn_test_post_nms_top_n)
    if cfg.hybridize:
        autograd.set_training(train_mode=False)
        net.hybridize()

    metric = VOC07MApMetric(iou_thresh=0.5, class_names=cfg.classes)

    for batch in val_data:
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

    return metric.get()


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def get_faster_rcnn(pretrained_base, cfg):
    return FasterRCNN(network=cfg.network, pretrained_base=pretrained_base, batch_size=cfg.batch_size,
                      num_classes=cfg.num_classes,
                      scales=cfg.anchor_scales, ratios=cfg.anchor_ratios, feature_stride=cfg.feature_stride,
                      base_size=cfg.base_size, allowed_border=cfg.allowed_border, rpn_batch_size=cfg.rpn_batch_size,
                      rpn_channels=cfg.rpn_channels, roi_mode=cfg.roi_mode, roi_size=cfg.roi_size,
                      rpn_fg_fraction=cfg.rpn_fg_fraction, rpn_positive_threshold=cfg.rpn_positive_threshold,
                      rpn_negative_threshold=cfg.rpn_negative_threshold,
                      rpn_pre_nms_top_n=cfg.rpn_pre_nms_top_n, rpn_post_nms_top_n=cfg.rpn_post_nms_top_n,
                      rpn_nms_threshold=cfg.rpn_nms_threshold,
                      rpn_min_size=cfg.rpn_min_size, roi_batch_size=cfg.roi_batch_size,
                      roi_fg_fraction=cfg.roi_fg_fraction, roi_fg_threshold=cfg.roi_fg_threshold,
                      roi_bg_threshold_hi=cfg.roi_bg_threshold_hi, roi_bg_threshold_lo=cfg.roi_bg_threshold_lo,
                      bbox_nms_threshold=cfg.bbox_nms_threshold, bbox_nms_top_n=cfg.bbox_nms_top_n,
                      bbox_mean=cfg.bbox_mean, bbox_std=cfg.bbox_std)


def get_dataset(dataset, dataset_path):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=FasterRCNNDefaultTrainTransform(cfg.image_size, cfg.image_max_size,
                                                                               cfg.image_mean, cfg.image_std,
                                                                               random_flip=True),
                                     root=dataset_path, preload_label=True)
        val_dataset = VOCDetection(splits=[(2007, 'test')],
                                   transform=FasterRCNNDefaultValTransform(cfg.image_size, cfg.image_max_size,
                                                                           cfg.image_mean, cfg.image_std),
                                   root=dataset_path, preload_label=True)
    elif dataset.lower() == 'rec':
        class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

        train_dataset = RecordDataset(root='./data', filename='voc_0712_trainval.rec',
                                      transform=FasterRCNNDefaultTrainTransform(cfg.image_size, cfg.image_max_size,
                                                                                cfg.image_mean, cfg.image_std,
                                                                                random_flip=True),
                                      class_names=class_names)
        val_dataset = RecordDataset(root='./data', filename='voc_07_test.rec',
                                    transform=FasterRCNNDefaultValTransform(cfg.image_size, cfg.image_max_size,
                                                                            cfg.image_mean, cfg.image_std),
                                    class_names=class_names)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, cfg):
    """Get dataloader."""
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, last_batch='rollover',
                              batchify_fn=FasterRCNNDefaultBatchify(cfg.image_max_size, cfg.label_max_size,
                                                                    cfg.num_workers), num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, cfg.batch_size, False, last_batch='keep',
                            batchify_fn=FasterRCNNDefaultBatchify(cfg.image_max_size, cfg.label_max_size,
                                                                  cfg.num_workers), num_workers=cfg.num_workers)
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
    parser.add_argument('--network', type=str, default=default.network,
                        help='network name')
    parser.add_argument('--dataset', type=str, default=default.dataset,
                        help='dataset name')
    parser.add_argument('--dataset-path', default=default.dataset_path, type=str,
                        help='dataset path')
    parser.add_argument('--lr', type=float, default=default.lr,
                        help='base learning rate')
    parser.add_argument('--lr-decay', type=float, default=default.lr_decay,
                        help='decay rate of learning rate.')
    parser.add_argument('--lr-decay-epochs', nargs='*', type=int, default=default.lr_decay_epochs,
                        help='epoches at which learning rate decays, such as --lr-decay-epochs 20 40.')
    parser.add_argument('--resume', type=bool, default=default.resume,
                        help='resume from model params path if True.')
    parser.add_argument('--model-params', type=str, default=default.model_params,
                        help='model params path')
    parser.add_argument('--start-epoch', type=int, default=default.start_epoch,
                        help='start epoch of training')
    parser.add_argument('--end-epoch', type=int, default=default.end_epoch,
                        help='end epoch of training')
    parser.add_argument('--save-prefix', type=str, default=default.save_prefix,
                        help='save model prefix')
    parser.add_argument('--gpus', nargs='*', type=int, default=default.gpus,
                        help='training with GPUs, such as --gpus 0 1 ')
    return parser.parse_args()


if __name__ == '__main__':
    # set random seed for python, mxnet and numpy
    set_random_seed(233)

    args = parse_args()
    cfg = generate_config(vars(args))

    log_file_path = cfg.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in cfg.gpus]
    ctx = ctx if ctx else [mx.cpu()]
    num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

    train_dataset, val_dataset = get_dataset(cfg.dataset, cfg.dataset_path)
    train_data, val_data = get_dataloader(train_dataset, val_dataset, cfg)

    cfg.num_classes = len(train_dataset.classes)
    cfg.classes = train_dataset.classes

    if cfg.resume:
        net = get_faster_rcnn(pretrained_base=False, cfg=cfg)
        net.load_parameters(cfg.model_params.strip(), ctx=ctx)
    else:
        net = get_faster_rcnn(pretrained_base=True, cfg=cfg)
        net.initialize(ctx=ctx)

    fix_net_params(net, cfg.network)

    train_faster_rcnn(net, train_data, val_data, cfg)
