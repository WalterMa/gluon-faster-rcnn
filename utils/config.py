from easydict import EasyDict as edict

# default settings
default = edict()

# network settings
default.network = 'resnet50_v1b' # optional: vgg16, resnet50_v1b
default.feature_stride = 16
default.hybridize = True
# dataset settings
default.dataset = 'voc'  # optional: voc, rec
default.dataset_path = '~/data/VOCdevkit'
# data pre-processing settings
default.image_size = 600
default.image_max_size = 1000
default.label_max_size = -1  # padding label to same size, set -1 to disable
# MXNet.image.imread default return RGB, not BGR in openCV
default.image_mean = (0.485, 0.456, 0.406)
default.image_std = (0.229, 0.224, 0.225)
# Bounding box normalization
default.bbox_mean = (0.0, 0.0, 0.0, 0.0)
default.bbox_std = (0.1, 0.1, 0.2, 0.2)
default.bbox_nms_threshold = 0.3
default.bbox_nms_top_n = 100
# anchor settings
default.base_size = 16
default.anchor_scales = (2, 4, 8, 16, 32)
default.anchor_ratios = (0.5, 1, 2)
default.allowed_border = 0
# RPN proposal settings
default.rpn_channels = 1024
default.rpn_batch_size = 256
default.rpn_fg_fraction = 0.5
default.rpn_positive_threshold = 0.7
default.rpn_negative_threshold = 0.3
default.rpn_nms_threshold = 0.7
default.rpn_pre_nms_top_n = 20000
default.rpn_post_nms_top_n = 2000
default.rpn_test_pre_nms_top_n = 6000
default.rpn_test_post_nms_top_n = 300
default.rpn_min_size = default.feature_stride
# ROI Pooling params
default.roi_mode = 'align'
default.roi_size = (14, 14)
# RCNN rois sampling params
default.roi_batch_size = 128
default.roi_fg_fraction = 0.25
default.roi_fg_threshold = 0.5
default.roi_bg_threshold_hi = 0.5
default.roi_bg_threshold_lo = 0.0
# Hyper Parameters
default.batch_size = 2
default.start_epoch = 0
default.end_epoch = 20
default.lr = 0.001
default.lr_decay = 0.1
default.lr_decay_epochs = (14,)
default.wd = 0.0005
default.momentum = 0.9
# log and saved path
default.resume = False
default.model_params = 'model/faster_rcnn_resnet50_v1b_best.params'
default.save_prefix = 'model/faster-rcnn_resnet50_v1b'
default.save_interval = 1
default.log_interval = 10
default.num_workers = 4
default.gpus = (0,)

# Fixed Parameters
# Fixed layers before conv3_1 for the VGG net
vgg16_fixed_params_pattern = 'vgg0_conv(0|1|2|3)_'
resnet50_v1b_fixed_params_pattern = 'resnetv1b0_conv0|resnetv1b0_layers1|resnetv1b0_down1|resnetv1b0.*batchnorm'


def generate_config(config_dict=''):
    config = default.copy()
    config.update(config_dict)
    return edict(config)
