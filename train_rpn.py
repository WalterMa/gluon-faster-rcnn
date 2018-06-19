import argparse
import time
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from rcnn import RPN
from rcnn.loss import RPNLoss
from rcnn.metrics.loss_metric import LogLossMetric, SmoothL1LossMetric
from dataset import VOCDetection
from utils import set_random_seed, fix_net_params
from utils.logger import logger
from utils.config import default, generate_config
from dataset.dataloader import DetectionDataLoader
from rcnn.transforms import FasterRCNNDefaultTrainTransform
import os


def train_rpn(net, train_data, cfg):
    """Training pipeline"""
    rpn_loss = RPNLoss(cfg.rpn_batch_size)
    rpn_loss.initialize(ctx=ctx)

    if cfg.hybridize:
        autograd.set_training(train_mode=True)
        net.hybridize()

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': cfg.lr, 'wd': cfg.wd, 'momentum': cfg.momentum, 'clip_gradient': 5})

    # lr decay policy
    lr_decay = float(cfg.lr_decay)
    lr_steps = sorted(cfg.lr_decay_epochs)

    # Create Metrics
    log_metric = LogLossMetric(name='LogLoss', batch_size=cfg.rpn_batch_size)
    smoothl1_metric = SmoothL1LossMetric(name='SmoothL1Loss', batch_size=cfg.rpn_batch_size)

    logger.info('Config for training RPN:\n%s' % cfg)
    logger.info('Start training from [Epoch %d]' % args.start_epoch)

    for epoch in range(cfg.start_epoch, cfg.end_epoch):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        tic = time.time()
        btic = time.time()
        log_metric.reset()
        smoothl1_metric.reset()

        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            gt_box_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            im_info_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_loss_list = []
                bbox_loss_list = []
                label_list = []
                for data, gt_box, im_info in zip(data_list, gt_box_list, im_info_list):
                    rpn_cls_prob, rpn_bbox_pred, labels, bbox_targets = net(data, gt_box, im_info)
                    cls_loss, bbox_loss = rpn_loss(rpn_cls_prob, rpn_bbox_pred, labels, bbox_targets)
                    cls_loss_list.append(cls_loss)
                    bbox_loss_list.append(bbox_loss)
                    label_list.append(labels)
            autograd.backward(cls_loss_list + bbox_loss_list)
            trainer.step(batch_size)
            log_metric.update(label_list, cls_loss_list)
            smoothl1_metric.update(label_list, bbox_loss_list)
            if cfg.log_interval and not (i + 1) % cfg.log_interval:
                name1, loss1 = log_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f' % (
                    epoch, i, batch_size / (time.time() - btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = log_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f' % (
            epoch, (time.time() - tic), name1, loss1, name2, loss2))
        save_params(net, epoch, cfg.save_interval, cfg.save_prefix)


def save_params(net, epoch, save_interval, prefix):
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}.params'.format(prefix, epoch))


def get_rpn(pretrained_base, cfg):
    return RPN(network=cfg.network, pretrained_base=pretrained_base, feature_stride=cfg.feature_stride,
               scales=cfg.anchor_scales, ratios=cfg.anchor_ratios, allowed_border=cfg.allowed_border,
               rpn_batch_size=cfg.rpn_batch_size, rpn_fg_fraction=cfg.rpn_fg_fraction,
               rpn_positive_threshold=cfg.rpn_positive_threshold,
               rpn_negative_threshold=cfg.rpn_negative_threshold)


def get_dataset(dataset, dataset_path):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')],
                                     transform=FasterRCNNDefaultTrainTransform(cfg.image_size, cfg.image_max_size,
                                                                               cfg.image_mean, cfg.image_std,
                                                                               random_flip=True),
                                     root=dataset_path, preload_label=False)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset


def get_dataloader(train_dataset, cfg):
    """Get dataloader."""
    train_loader = DetectionDataLoader(train_dataset, cfg.batch_size, shuffle=True, last_batch='rollover',
                                       num_workers=cfg.num_workers)
    return train_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
    parser.add_argument('--network', type=str, default=default.network,
                        help='network name')
    parser.add_argument('--dataset', type=str, default=default.dataset,
                        help='dataset name')
    parser.add_argument('--dataset-path', default=default.dataset_path, type=str,
                        help='dataset path')
    parser.add_argument('--lr', type=float, default=default.rpn.lr,
                        help='base learning rate')
    parser.add_argument('--lr-decay', type=float, default=default.rpn.lr_decay,
                        help='decay rate of learning rate.')
    parser.add_argument('--lr-decay-epochs', nargs='*', type=int, default=default.rpn.lr_decay_epochs,
                        help='epoches at which learning rate decays, such as --lr-decay-epochs 20 40.')
    parser.add_argument('--resume', type=bool, default=default.rpn.resume,
                        help='resume from model params path if True.')
    parser.add_argument('--model-params', type=str, default=default.rpn.model_params,
                        help='model params path')
    parser.add_argument('--start-epoch', type=int, default=default.rpn.start_epoch,
                        help='start epoch of training')
    parser.add_argument('--end-epoch', type=int, default=default.rpn.end_epoch,
                        help='end epoch of training')
    parser.add_argument('--save-prefix', type=str, default=default.rpn.save_prefix,
                        help='save model prefix')
    parser.add_argument('--gpus', nargs='*', type=int, default=default.gpus,
                        help='training with GPUs, such as --gpus 0 1 ')
    return parser.parse_args()


if __name__ == '__main__':
    # set 0 to disable Running performance tests
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    # set random seed for python, mxnet and numpy
    set_random_seed(2333)

    args = parse_args()
    cfg = generate_config(vars(args))

    # training contexts
    ctx = [mx.gpu(int(i)) for i in cfg.gpus]
    ctx = ctx if ctx else [mx.cpu()]
    if cfg.resume:
        net = get_rpn(pretrained_base=False, cfg=cfg)
        net.load_params(cfg.model_params.strip(), ctx=ctx)
    else:
        net = get_rpn(pretrained_base=True, cfg=cfg)
        net.initialize(ctx=ctx)

    fix_net_params(net, cfg.network)

    train_dataset = get_dataset(cfg.dataset, cfg.dataset_path)
    train_data = get_dataloader(train_dataset, cfg)

    train_rpn(net, train_data, cfg)
