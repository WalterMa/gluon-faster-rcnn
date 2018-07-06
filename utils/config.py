from easydict import EasyDict as edict

# default settings
default = edict()

# network settings
default.network = 'vgg16'
default.feature_stride = 16
default.hybridize = True
# dataset settings
default.dataset = 'voc'  # optional: voc, rec
default.dataset_path = './data/VOCdevkit'
default.proposal_path = 'model/rpn_vgg16'
# data pre-processing settings
default.image_size = 600
default.image_max_size = 1000
# MXNet.image.imread default return RGB, not BGR in openCV
default.image_mean = (0.485, 0.456, 0.406)
default.image_std = (0.229, 0.224, 0.225)
# Bounding box normalization
default.bbox_mean = (0.0, 0.0, 0.0, 0.0)
default.bbox_std = (0.1, 0.1, 0.2, 0.2)
default.bbox_nms_threshold = 0.3
default.bbox_nms_top_n = -1
# anchor settings
default.anchor_scales = (8, 16, 32)
default.anchor_ratios = (0.5, 1, 2)
default.allowed_border = 0
# RPN proposal settings
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
# RCNN rois sampling params
default.roi_batch_size = 128
default.roi_fg_fraction = 0.25
default.roi_fg_threshold = 0.5
default.roi_bg_threshold_hi = 0.5
default.roi_bg_threshold_lo = 0.0
# Hyper Parameters
default.batch_size = 2
default.start_epoch = 0
default.end_epoch = 10
default.lr = 0.001
default.lr_decay = 0.1
default.lr_decay_epochs = (7,)
default.wd = 0.0005
default.momentum = 0.9
# log and saved path
default.resume = False
default.model_params = 'model/faster-rcnn_vgg16_best.params'
default.save_prefix = 'model/faster-rcnn_vgg16'
default.save_interval = 1
default.log_interval = 10
default.num_workers = 4
default.gpus = (0,)

# RPN specific default settings
default.rpn = edict()
# RPN Hyper Parameters
default.rpn.start_epoch = 0
default.rpn.end_epoch = 8
default.rpn.lr = 0.001
default.rpn.lr_decay = 0.1
default.rpn.lr_decay_epochs = (6,)
default.rpn.wd = 0.0005
default.rpn.momentum = 0.9
# RPN log and saved path
default.rpn.resume = False
default.rpn.model_params = 'model/rpn_vgg16_0000.params'
default.rpn.save_prefix = 'model/rpn_vgg16'
default.rpn.save_interval = 1
default.rpn.save_proposal = True  # Only used in test stage
default.rpn.log_interval = 10
default.rpn.num_workers = 4

# RCNN specific default settings
default.rcnn = edict()
# RCNN Hyper Parameters
default.rcnn.start_epoch = 0
default.rcnn.end_epoch = 30
default.rcnn.lr = 0.001
default.rcnn.lr_decay = 0.1
default.rcnn.lr_decay_epochs = (10, 20)
default.rcnn.wd = 0.0005
default.rcnn.momentum = 0.9
# RCNN log and saved path
default.rcnn.resume = False
default.rcnn.model_params = 'model/rcnn_vgg16_0000.params'
default.rcnn.save_prefix = 'model/rcnn_vgg16'
default.rcnn.save_interval = 1
default.rcnn.log_interval = 10
default.rcnn.num_workers = 4

# Fixed Parameters
# Fixed layers before conv3_1 for the VGG net to conserve memory
vgg16_fixed_params = ['vgg0_conv0_weight', 'vgg0_conv0_bias',
                      'vgg0_conv1_weight', 'vgg0_conv1_bias',
                      'vgg0_conv2_weight', 'vgg0_conv2_bias',
                      'vgg0_conv3_weight', 'vgg0_conv3_bias']


def generate_config(config_dict=''):
    config = default.copy()
    config.update(config_dict)
    return edict(config)
