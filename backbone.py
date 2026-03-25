import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import efficientnet_b3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

from config import NUM_CLASSES


class EfficientNetFPNBackbone(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super().__init__()
        efficientnet = efficientnet_b3(weights=weights)
        return_nodes = {
            'features.2': 'feat1',
            'features.4': 'feat2',
            'features.6': 'feat3',
            'features.8': 'feat4',
        }
        self.feature_extractor = create_feature_extractor(efficientnet, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[32, 96, 232, 1536],
            out_channels=256,
        )
        self.out_channels = 256

    def forward(self, x):
        features = self.feature_extractor(x)
        ordered_features = OrderedDict((k, features[k]) for k in ['feat1', 'feat2', 'feat3', 'feat4'])
        return self.fpn(ordered_features)


def build_model(num_classes=NUM_CLASSES, weights='IMAGENET1K_V1'):
    backbone = EfficientNetFPNBackbone(weights=weights)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
    )
    return model
