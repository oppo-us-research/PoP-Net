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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResPreprocessNet(nn.Module):

    def __init__(self, block, layers, input_dim=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResPreprocessNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.conv2 = conv1x1(128, 128, 1)
        self.bn2 = norm_layer(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # res module 1
        x = self.avgpool1(x)

        x = self.layer2(x)  # res module 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


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
            # TODO: the normalized branch requires to apply BN?
            elif '_D' in k or '_A' in k:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                # add batchNorm to avoid vanishing gradient
                layers += [conv2d, nn.BatchNorm2d(v[1]), nn.LeakyReLU(0.1, inplace=True)]
            # TODO: the physical value branch should not apply BN?
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                # add batchNorm to avoid vanishing gradient
                # layers += [conv2d, nn.BatchNorm2d(v[1]), nn.LeakyReLU(0.1, inplace=True)]
                layers += [conv2d, nn.LeakyReLU(0.1, inplace=True)]

    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


class rtpose_align3d(nn.Module):
    def __init__(self, num_parts=18, num_limbs=19, num_stages=2, input_dim=3):
        super(rtpose_align3d, self).__init__()
        self.num_parts = num_parts
        self.num_stages = num_stages

        # construct model dict
        model_dict = {}
        # block0 is the preprocessing stage
        model_dict['block0'] = ResPreprocessNet(BasicBlock, [2, 1], input_dim)

        blocks = {}
        # Stage 1
        # paf branch
        blocks['block1_1'] = [{'conv5_1_CPM_L': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_L': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_L': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L': [512, 2 * num_limbs, 1, 1, 0]}]
        # heatmap branch
        blocks['block1_2'] = [{'conv5_1_CPM_S': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_S': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_S': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_S': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_S': [512, num_parts + 1, 1, 1, 0]}]

        # Depth branch (this design is only for depth data for now)
        # TODO: why higher dims leads to vanishing gradient?
        blocks['block1_3'] = [{'conv5_1_CPM_D': [128, 64, 3, 1, 1]},
                              {'conv5_2_CPM_D': [64, 32, 3, 1, 1]},
                              {'conv5_3_CPM_D': [32, 32, 1, 1, 0]},
                              {'conv5_4_CPM_D': [32,  num_parts, 1, 1, 0]}]

        # 2D Align branch
        blocks['block1_4'] = [{'conv5_1_CPM_A': [128, 256, 3, 1, 1]},
                              {'conv5_2_CPM_A': [256, 256, 3, 1, 1]},
                              {'conv5_3_CPM_A': [256, 256, 3, 1, 1]},
                              {'conv5_4_CPM_A': [256, 128, 1, 1, 0]},
                              {'conv5_5_CPM_A': [128, 2 * num_parts, 1, 1, 0]}]

        # Stages 2
        for i in range(2, num_stages + 1):
            # paf branch:
            # TODO: the large kernel size is to handle the limb-wise context.
            #  However, this is costly and sensitive to scale.
            blocks['block%d_1' % i] = [
                {'Mconv1_stage%d_L' % i: [128 + num_limbs * 2 + num_parts + 1 + 3 * num_parts, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L' % i: [128, 2 * num_limbs, 1, 1, 0]}
            ]

            # heatmap branch
            blocks['block%d_2' % i] = [
                {'Mconv1_stage%d_S' % i: [128 + num_limbs * 2 + num_parts + 1 + 3 * num_parts, 128, 3, 1, 1]},
                {'Mconv2_stage%d_S' % i: [128, 128, 3, 1, 1]},
                {'Mconv3_stage%d_S' % i: [128, 128, 3, 1, 1]},
                {'Mconv4_stage%d_S' % i: [128, 128, 3, 1, 1]},
                {'Mconv5_stage%d_S' % i: [128, 128, 3, 1, 1]},
                {'Mconv6_stage%d_S' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_S' % i: [128, num_parts + 1, 1, 1, 0]}
            ]

            # Depth branch (this design is only for depth data)
            # TODO: why higher dims leads to vanishing gradient?
            blocks['block%d_3' % i] = [
                {'Mconv1_stage%d_D' % i: [128 + num_limbs * 2 + num_parts + 1 + 3 * num_parts, 128, 3, 1, 1]},
                {'Mconv2_stage%d_D' % i: [128, 64, 3, 1, 1]},
                {'Mconv3_stage%d_D' % i: [64, 32, 3, 1, 1]},
                {'Mconv4_stage%d_D' % i: [32, 32, 1, 1, 0]},
                {'Mconv5_stage%d_D' % i: [32,  num_parts, 1, 1, 0]}
            ]

            # pose align branch
            blocks['block%d_4' % i] = [
                {'Mconv1_stage%d_A' % i: [128 + num_limbs * 2 + num_parts + 1 + 3 * num_parts, 128, 3, 1, 1]},
                {'Mconv2_stage%d_A' % i: [128, 256, 3, 1, 1]},
                {'Mconv3_stage%d_A' % i: [256, 256, 3, 1, 1]},
                {'Mconv4_stage%d_A' % i: [256, 256, 3, 1, 1]},
                {'Mconv5_stage%d_A' % i: [256, 128, 1, 1, 0]},
                {'Mconv6_stage%d_A' % i: [128, 2 * num_parts, 1, 1, 0]}
            ]

        for k, v in blocks.items():
            model_dict[k] = make_stages(list(v))

        # construct the actual model (hard coded for 2 stages)
        self.model0 = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model1_2 = model_dict['block1_2']
        self.model1_3 = model_dict['block1_3']
        self.model1_4 = model_dict['block1_4']

        self.model2_1 = model_dict['block2_1']
        self.model2_2 = model_dict['block2_2']
        self.model2_3 = model_dict['block2_3']
        self.model2_4 = model_dict['block2_4']

        self._initialize_weights_norm()

    def forward(self, x):

        saved_for_loss = []
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out1_3 = self.model1_3(out1)
        out1_4 = self.model1_4(out1)

        out2 = torch.cat([out1_1, out1_2, out1_3, out1_4, out1], 1)
        saved_for_loss.append(out1_1)
        saved_for_loss.append(out1_2)
        saved_for_loss.append(out1_3)
        saved_for_loss.append(out1_4)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out2_3 = self.model2_3(out2)
        out2_4 = self.model2_4(out2)
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)
        saved_for_loss.append(out2_3)
        saved_for_loss.append(out2_4)

        return (out2_1, out2_2, out2_3, out2_4), saved_for_loss

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0, std=0.01)


if __name__ == "__main__":
    num_parts = 15
    num_limbs = 14
    num_stages = 2
    model = rtpose_align3d(num_parts, num_limbs, num_stages, input_dim=1)
    print(model)

    input = torch.randn(1, 1, 224, 224)
    _, saved_for_loss = model(input)
    macs, params = profile(model, inputs=(input,))

    print("Params(M): {:.3f}, MACs(G): {:.3f}".format(params/10**6, macs/10**9))
