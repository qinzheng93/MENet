import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = [
    'mobilenet',
    'shallow_mobilenet',
]

__mobilenet_shallow_channels = [64, 128, 128, 256, 256, 512, 1024, 1024]
__mobilenet_shallow_strides = [1, 2, 1, 2, 1, 2, 2, 1]
__mobilenet_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
__mobilenet_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

class MobileNet(nn.Module):
    def __init__(self, init_features, channels, strides):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv_0', nn.Conv2d(3, init_features, 3, stride=2, padding=1, bias=False)),
            ('norm_0', nn.BatchNorm2d(init_features)),
            ('relu_0', nn.ReLU(inplace=True)),
        ]))
        in_c = init_features
        for _, (out_c, stride) in enumerate(zip(channels, strides)):
            self.features.add_module('dw_conv_{}'.format(_), nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False))
            self.features.add_module('dw_norm_{}'.format(_), nn.BatchNorm2d(in_c))
            self.features.add_module('dw_relu_{}'.format(_), nn.ReLU(inplace=True))
            self.features.add_module('pw_conv_{}'.format(_), nn.Conv2d(in_c, out_c, 1, bias=False))
            self.features.add_module('pw_norm_{}'.format(_), nn.BatchNorm2d(out_c))
            self.features.add_module('pw_relu_{}'.format(_), nn.ReLU(inplace=True))
            in_c = out_c
        self.pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(in_c, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenet(model_config):
    width_mul = model_config['width_mul']
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in __mobilenet_channels]
    return MobileNet(init_features, channels, __mobilenet_strides)

def shallow_mobilenet(model_config):
    width_mul = model_config['width_mul']
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in __mobilenet_shallow_channels]
    return MobileNet(init_features, channels, __mobilenet_shallow_strides)
