from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock


class VGGConvBlock(HybridBlock):

    def __init__(self, base_model, **kwargs):
        super(VGGConvBlock, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        # Exclude last 5 vgg feature layers (1 max pooling + 2 * (fc + dropout))
        for layer in base_model.features[:-5]:
            self.features.add(layer)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        return x


class VGGFastRCNNHead(HybridBlock):

    def __init__(self, base_model, num_classes, feature_stride, **kwargs):
        super(VGGFastRCNNHead, self).__init__(**kwargs)
        self.feature_stride = feature_stride
        self.fc_layers = nn.HybridSequential()
        # Include last 4 vgg feature layers (2 * (fc + dropout))
        for layer in base_model.features[-4:]:
            self.fc_layers.add(layer)
        self.cls_score = nn.Dense(in_units=4096, units=num_classes, weight_initializer=initializer.Normal(0.01))
        self.bbox_pred = nn.Dense(in_units=4096, units=num_classes * 4, weight_initializer=initializer.Normal(0.001))

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, feature_map, rois):
        x = F.ROIPooling(data=feature_map, rois=rois, pooled_size=(7, 7), spatial_scale=1.0/self.feature_stride)
        x = F.flatten(data=x)  # shape(roi_num, 512*7*7)
        x = self.fc_layers(x)
        cls_score = self.cls_score(x)
        cls_prob = F.softmax(data=cls_score)  # shape(roi_num, num_classes)
        bbox_pred = self.bbox_pred(x)  # shape(roi_num, num_classes*4)
        return cls_prob, bbox_pred



