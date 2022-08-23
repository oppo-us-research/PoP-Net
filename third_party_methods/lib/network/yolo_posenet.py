"""
A rtpose network with new 3D branch and 2D align branch :
Modification from rtpose:
    1. preprocess trained_model uses residual modules rather than vgg. The resolution reduction is tuned to be 1/8
    2. only uses two stages in the prediction
    3. New 3D branch is developed and integrated
    4. New 2D align branch is developed and integrated

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: April 2020
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init
from thop import profile, clever_format
from lib.network import resnet


class ResNetBackBone(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNetBackBone, self).__init__()

        model_resnet = resnet.resnet34(pretrained=False)

        # redefine the initial layer for depth
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        # self.model = modelPreTrain50

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)

        return x2


def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.BatchNorm2d(v[1]), nn.LeakyReLU(0.1, inplace=True)]

    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    # ATTENTION: the last BN is critical to avoid inf grad caused by outlier
    layers += [conv2d]
    return nn.Sequential(*layers)


class YoloPoseNet(nn.Module):
    def __init__(self, num_parts=15, input_dim=3, anchors=[(6., 3.), (12., 6.)]):
        super(YoloPoseNet, self).__init__()
        self.num_parts = num_parts
        self.anchors = anchors

        # construct model dict
        model_dict = {}
        # block0 is the preprocessing stage
        model_dict['block0'] = ResNetBackBone(input_dim)

        blocks = {}

        # intermediate layers
        blocks['block1'] = [{'conv1_H': [128, 256, 3, 1, 1]},
                              {'conv2_H': [256, 256, 3, 1, 1]},
                              {'conv3_H': [256, 256, 3, 1, 1]},
                              {'conv4_H': [256, 256, 3, 1, 1]},
                              {'conv5_H': [256, 256, 3, 1, 1]}]

        for k, v in blocks.items():
            model_dict[k] = make_stages(list(v))

        # construct the actual model (hard coded for 2 stages)
        self.model0 = model_dict['block0']

        self.model1 = model_dict['block1']

        self.model2_1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.MaxPool2d(2, 2))
        self.model2_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.model2_3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.model2_4 = nn.Sequential(
            nn.Conv2d(128, len(self.anchors) * (5 + 3*self.num_parts), 3, 1, 1, bias=False))

        self._initialize_weights_norm()
        # self._initialize_weights_xavier()

    def forward(self, x):
        saved_for_loss = []
        out0 = self.model0(x)

        out1 = self.model1(out0)

        # out2_1 = self.model2_1(out2)
        out2 = self.model2_1(out1)
        out2 = self.model2_2(out2)
        out2 = self.model2_3(out2)
        out2 = self.model2_4(out2)

        # cast output range
        num_anchor_feat = 5 + 3 * self.num_parts

        for i in range(len(self.anchors)):
            # boxes (dx dy) from anchor center in (-1, 1), conf in (0, 1) range; (width, height) cast in (0, 2) range
            out2[:, i * num_anchor_feat:i * num_anchor_feat + 2] = \
                (out2[:, i * num_anchor_feat:i * num_anchor_feat + 2].sigmoid() - 0.5) * 2
            out2[:, i * num_anchor_feat + 2: i * num_anchor_feat + 4] = \
                out2[:, i * num_anchor_feat + 2: i * num_anchor_feat + 4].sigmoid() * 2
            out2[:, i * num_anchor_feat + 4] = \
                out2[:, i * num_anchor_feat + 4].sigmoid()
            # x2d, y2d, Z3d cast in [-2, 2] range
            out2[:, i*num_anchor_feat + 5: i*num_anchor_feat + 5 + 3*self.num_parts] = \
                (out2[:, i*num_anchor_feat + 5: i*num_anchor_feat + 5 + 3*self.num_parts].sigmoid() - 0.5) * 4

        return out2

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0, std=0.01)

    def _initialize_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)


if __name__ == "__main__":
    num_parts = 15
    num_stages = 2
    model = YoloPoseNet(num_parts, input_dim=1)
    print(model)

    input = torch.randn(1, 1, 224, 224)
    output = model(input)
    macs, params = profile(model, inputs=(input,))

    print("Params(M): {:.3f}, MACs(G): {:.3f}".format(params/10**6, macs/10**9))
