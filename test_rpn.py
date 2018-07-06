import argparse
import time
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from rcnn import RPN
from rcnn.loss import RPNLoss
from rcnn.proposal import ProposalLayer
from dataset import VOCDetection
from utils.logger import logger
from utils.config import default, generate_config
from dataset.dataloader import DetectionDataLoader
from rcnn.transforms import FasterRCNNDefaultValTransform
from rcnn.metrics.loss_metric import LogLossMetric, SmoothL1LossMetric
import os


def test_rpn(net, test_data, cfg):
    """Testing and saving proposals pipeline"""
    proposal_layer = ProposalLayer(feature_stride=cfg.feature_stride, scales=cfg.anchor_scales,
                                   ratios=cfg.anchor_ratios, rpn_pre_nms_top_n=cfg.rpn_pre_nms_top_n,
                                   rpn_post_nms_top_n=cfg.rpn_post_nms_top_n,
                                   rpn_nms_threshold=cfg.rpn_nms_threshold, rpn_min_size=cfg.rpn_min_size)
    proposal_layer.initialize(ctx=ctx)

    if cfg.hybridize:
        net.hybridize()

    logger.info('Config for testing RPN:\n%s' % cfg)
    logger.info('Save proposals: %d' % args.save_proposal)

    tic = time.time()
    btic = time.time()
    proposals = None

    for i, batch in enumerate(test_data):
        batch_size = batch[0].shape[0]
        data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        gt_box_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        im_info_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

        for data, gt_box, im_info in zip(data_list, gt_box_list, im_info_list):
            rpn_cls_prob, rpn_bbox_pred = net(data)
            proposal = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info)
            # save predicted proposals in cpu memory
            # TODO save proposals by h5py
            if proposals:
                proposals = nd.concat(proposals,
                                      proposal[:, 1:].resahpe(-1, cfg.rpn_post_nms_top_n, 4).as_in_context(mx.cpu()),
                                      dim=0)
            else:
                proposals = proposal[:, 1:].resahpe(-1, cfg.rpn_post_nms_top_n, 4).as_in_context(mx.cpu())
        if cfg.log_interval and not (i + 1) % cfg.log_interval:
            logger.info('[Batch %d], Speed: %f samples/sec' % (
                 i, batch_size / (time.time() - btic)))
        btic = time.time()

    save_proposals()
    logger.info('[Done] Testing cost: %f' % (time.time() - tic))


def save_proposals():
    # TODO save proposals by h5py
    raise NotImplementedError


def get_dataset(dataset_name, dataset_path):
    if dataset_name.lower() == 'voc':
        dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')],
                               transform=FasterRCNNDefaultValTransform(cfg.image_size, cfg.image_max_size,
                                                                       cfg.image_mean, cfg.image_std),
                               root=dataset_path, preload_label=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset_name))
    return dataset


def get_dataloader(dataset, cfg):
    """Get dataloader."""
    data_loader = DetectionDataLoader(dataset, cfg.batch_size, shuffle=False, last_batch='keep',
                                      image_max_size=cfg.image_max_size, label_max_size=20,
                                      num_workers=cfg.num_workers)
    return data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Trained Region Proposal Network')
    parser.add_argument('--network', type=str, default=default.network,
                        help='network name')
    parser.add_argument('--dataset', type=str, default=default.dataset,
                        help='dataset name')
    parser.add_argument('--dataset-path', default=default.dataset_path, type=str,
                        help='dataset path')
    parser.add_argument('--model-params', type=str, default=default.rpn.model_params,
                        help='loaded model params path')
    parser.add_argument('--save-proposal', type=bool, default=default.rpn.save_proposal,
                        help='whether to save predicted proposals')
    parser.add_argument('--proposal-path', default=default.proposal_path, type=str,
                        help='proposal will be saved in this path')
    parser.add_argument('--gpus', nargs='*', type=int, default=default.gpus,
                        help='testing with GPUs, such as --gpus 0 1 ')
    return parser.parse_args()


if __name__ == '__main__':
    # set 0 to disable Running performance tests
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    args = parse_args()
    cfg = generate_config(vars(args))

    # testing contexts
    ctx = [mx.gpu(int(i)) for i in cfg.gpus]
    ctx = ctx if ctx else [mx.cpu()]
    num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

    net = RPN(network=cfg.network, pretrained_base=False, feature_stride=cfg.feature_stride, scales=cfg.anchor_scales,
              ratios=cfg.anchor_ratios, allowed_border=cfg.allowed_border, rpn_batch_size=cfg.rpn_batch_size,
              rpn_fg_fraction=cfg.rpn_fg_fraction, rpn_positive_threshold=cfg.rpn_positive_threshold,
              rpn_negative_threshold=cfg.rpn_negative_threshold)
    net.load_parameters(cfg.model_params.strip(), ctx=ctx)

    test_dataset = get_dataset(cfg.dataset, cfg.dataset_path)
    test_data = get_dataloader(test_dataset, cfg)

    test_rpn(net, test_data, cfg)
