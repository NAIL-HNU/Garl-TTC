# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(
            self, 
            inplanes, 
            planes, 
            padding=1, 
            kernel_size=3, 
            stride=1, 
            with_nonlinearity=True
            ):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of 
    Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
            self, 
            inplanes, 
            planes, 
            up_conv_inplanes=None, 
            up_conv_planes=None,
            upsampling_method="conv_transpose"
            ):
        super().__init__()

        if up_conv_inplanes is None:
            up_conv_inplanes = inplanes
        if up_conv_planes is None:
            up_conv_planes = planes

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_inplanes, 
                up_conv_planes, 
                kernel_size=2, 
                stride=2
                )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(inplanes, planes)
        self.conv_block_2 = ConvBlock(planes, planes)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class ResNetBackbone(nn.Module):

    def __init__(
            self, 
            block, 
            layers, 
            cfg, 
            **kwargs
            ):
        input_feat_num = cfg['model']['input_feat_num']
        input_feat_num = kwargs.get('input_feat_num', input_feat_num)
        
        self.inplanes = 64
        
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(input_feat_num, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.with_decoder = 'with_decoder' in cfg['model'] and cfg['model']['with_decoder']
        if self.with_decoder:            
            layers_up = cfg['model']['decoder_layers']
            self.up_block1, self.up_layer1 = self._make_upsample_layer(2048 + 1024, 2048, 1024, layers_up[0])
            self.up_block2, self.up_layer2 = self._make_upsample_layer(1024 + 512, 1024, 512, layers_up[1])
            self.up_block3, self.up_layer3 = self._make_upsample_layer(512 + 256, 512, 256, layers_up[2])
            self.up_block4, self.up_layer4 = self._make_upsample_layer(256 + 64, 256, 64, layers_up[3])

            self.decoder_out1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.decoder_out2 = nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_upsample_layer(
            self, 
            inplanes,
            up_conv_inplanes,
            planes,             
            blocks,             
            stride=1,
            block=BasicBlock
            ):
        up_block = UpBlock(
                inplanes, 
                planes,
                up_conv_inplanes=up_conv_inplanes,
                up_conv_planes=up_conv_inplanes
                )
            
        layers = []
        
        for i in range(1, blocks):
            layers.append(block(planes, planes))
            
        self.inplanes = planes * block.expansion            

        return up_block, nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        before_pool = self.relu(x)
        after_pool = self.maxpool(before_pool)

        feat1 = self.layer1(after_pool)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        
        if self.with_decoder:
            upfeat4 = self.up_layer1(self.up_block1(feat4, feat3))
            upfeat3 = self.up_layer2(self.up_block2(upfeat4, feat2))
            upfeat2 = self.up_layer3(self.up_block3(upfeat3, feat1))
            upfeat1 = self.up_layer4(self.up_block4(upfeat2, before_pool))
            decode_out = self.decoder_out2(self.decoder_out1(upfeat1))
            return feat4, decode_out
        else:
            decode_out = None

        return feat4, decode_out

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}
