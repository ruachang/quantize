# coding=utf-8
'''
mfcnn的网络部分
'''
from audioop import avg
import os
from turtle import forward
from scipy import spatial
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from dataset import TenFoldsDataset
# from torchsummary import summary
from torch.autograd import Variable
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

# from fusion import iAFF, ASFF, ASFF_16

# * 定义新的量化卷积层
def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            padding: int = 0,
            quantize: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if quantize:
        return quant_nn.QuantConv2d(in_planes,
                                    out_planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    groups=groups,
                                    bias=False)
    else:
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=3,
                         stride=stride,
                         padding=padding,
                         groups=groups,
                         bias=False)

def conv7x7(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            padding: int = 0,
            quantize: bool = False) -> nn.Conv2d:
    """7x7 convolution with padding"""
    if quantize:
        return quant_nn.QuantConv2d(in_planes,
                                    out_planes,
                                    kernel_size=7,
                                    stride=stride,
                                    padding=padding,
                                    groups=groups,
                                    bias=False)
    else:
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=7,
                         stride=stride,
                         padding=padding,
                         groups=groups,
                         bias=False)
    
def conv1x1(in_planes: int, out_planes: int, stride: int = 1, quantize: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    if quantize:
        return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def maxPool2d(kernel_size: int, quantize: bool = False, **kwargs) -> nn.MaxPool2d:
    """2d MaxPool"""
    if quantize:
        return quant_nn.QuantMaxPool2d(kernel_size=kernel_size, **kwargs)
    else:
        return nn.MaxPool2d(kernel_size=kernel_size)

class ChannelAttention(nn.Module):
# * 神奇的注意力模块, 只在神经网络搭建的时候用了
# * Channel 注意力
    def __init__(self, in_planes, ratio=16, quantize=False):
        super(ChannelAttention, self).__init__()
        self._quantize = quantize
        # if quantize:
        #     self.avg_pool = quant_nn.AdaptiveAvgPool2d(1)
        #     self.max_pool = quant_nn.AdaptiveAvgPool2d(1)
        # else:
        #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #     self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(conv1x1(in_planes, in_planes // 16, quantize=quantize),
                               nn.ReLU(),
                               conv1x1(in_planes // 16, in_planes, quantize=quantize))
        self.sigmoid = nn.Sigmoid()
        if self._quantize:
            self.channel_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
 
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        if self._quantize:
            out = self.channel_quantizer(avg_out) + self.channel_quantizer(max_out)
            out = self.sigmoid(self.channel_quantizer(out))
        else: 
            out = max_out + avg_out
            out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
# * 神奇的注意力集中模块堂堂再登场! 同样只在神经网络搭建时用到
# * Spatial 注意力
    def __init__(self, kernel_size=7, quantize=False):
        super(SpatialAttention, self).__init__()
        self._quantize = quantize
        self.conv1 = conv7x7(2, 1, padding=kernel_size//2, quantize=quantize)
        self.sigmoid = nn.Sigmoid()
        if self._quantize:
            self.spatial_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
    def forward(self, x):
        # if self._quantize:
        #     avg_out = torch.mean(self.spatial_quantizer(x), dim=1, keepdim=True)
        #     max_out, _ = torch.max(self.spatial_quantizer(x), dim=1, keepdim=True)
        # else:
        #     avg_out = torch.mean(x, dim=1, keepdim=True)
        #     max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        if self._quantize:
            return self.sigmoid(self.spatial_quantizer(x))
        else:
            return self.sigmoid(x)

class MS_CAM(nn.Module):
# * 尝试新的注意力模块
    def __init__(self, in_channels, ratio=16) -> None:
        super(MS_CAM, self).__init__()
        mid_channels = in_channels // ratio
        self.glob_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)   
        )

        self.local_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_att = self.glob_attention(x)
        local_att  = self.local_attention(x)
        attention = self.sigmoid(local_att + global_att.expand_as(local_att))
        return attention * x
    
class CNN(nn.Module):
    def __init__(self, kind, quantize=False):
        super(CNN, self).__init__()
        self.kind = kind
        self._quantize = False
        if quantize:
            # 这里的设置和主文件保持了一致
            self.mul_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
            ptq_quant_desc = QuantDescriptor(calib_method="histogram", num_bits=16, fake_quant=True, axis=None)
            # self.ptq_quant = quant_nn.TensorQuantizer(ptq_quant_desc)
        self.conv1 = nn.Sequential(
            # * 输入通道数1, 输出图像长宽-2
            conv3x3(in_planes=1, out_planes=64, stride=1, padding=0, quantize=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # * 输入通道数64, 输出图像长宽-2
            conv3x3(in_planes=64, out_planes=64, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # * 输出图像大小变成原来的一半
            maxPool2d(kernel_size=2, quantize=quantize),
        )
        self.conv2 = nn.Sequential(
            # * 输入通道数64, 输出图像长宽-2
            conv3x3(in_planes=64, out_planes=128, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # * 输入通道数128, 输出图像长宽-2
            conv3x3(in_planes=128, out_planes=128, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # * 输出图像大小变成原来一半
            maxPool2d(kernel_size=2, quantize=quantize),
        )
        self.conv3 = nn.Sequential(
            # * 输入通道数128, 输出图像长宽-2
            conv3x3(in_planes=128, out_planes=256, stride=1, padding=0, quantize=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # * 输入通道数256, 输出图像长宽-2
            conv3x3(in_planes=256, out_planes=256, stride=1, padding=0, quantize=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # * 输出图像大小变成原来一半
            maxPool2d(kernel_size=2, quantize=quantize),
        )
        # * 使用两种注意力集中模块依次作用
        self.attention_spatial = SpatialAttention(quantize=False)
        self.attention_channel = ChannelAttention(384, quantize=quantize)

        # TODO 是否初始化MS_CAM
        # ? 暂时不考虑global的了, 有点麻烦
        # self.global_asff = ASFF(1)
        # self.ms_cam = MS_CAM(384)
        if quantize:
            self.pool1 = quant_nn.AdaptiveAvgPool2d(1)
        else:
            self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.2)
        if quantize:
            self.classfier1 = quant_nn.QuantLinear(384, self.kind)
        else:
            self.classfier1 = nn.Linear(384, self.kind)
        # * conv4和conv1的结构差不多
        self.conv4 = nn.Sequential(
            conv3x3(in_planes=1, out_planes=64, stride=1, padding=0, quantize=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv3x3(in_planes=64, out_planes=64, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            maxPool2d(kernel_size=2, quantize=quantize),
        )
        # * conv5和conv2的结构差不多
        self.conv5 = nn.Sequential(
            conv3x3(in_planes=64, out_planes=128, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(in_planes=128, out_planes=128, stride=1, padding=0, quantize=quantize),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            maxPool2d(kernel_size=2, quantize=False),
        )
        # * 两边使用的注意力集中模块不太一样, 维度不太一样
        self.patchattention_spatial = SpatialAttention(quantize=False)
        self.patchattention_channel = ChannelAttention(128, quantize=False)

        # TODO 是否初始化MS_CAM
        # self.patch_ms_cam = MS_CAM(128)
        if self._quantize:
            self.pool2 = quant_nn.AdaptiveAvgPool2d(1)
        else:
            self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.dropout2 = nn.Dropout(0.2)

        # TODO 取决于是否使用iaff, 使用的话这里会采用128
        if self._quantize:
            self.classfier2 = quant_nn.QuantLinear(128 * 16, self.kind)
        else:
            self.classfier2 = nn.Linear(128 * 16, self.kind)
        # TODO 是否初始化iaff融合
        # self.iaff = iAFF(channels=128)
        # TODO 是否初始化ASFF融合
        # self.asff_16 = ASFF_16()
        # * 上面的是每一个神经网络block的构成, 看起来都是普通的神经网络, 除了层数较深以外没有什么特别的
        # * 下面的是每一个神经网络block的初始化, 针对不同层有不同的初始化方式
        for m in self.modules():
            # * 如果目前处于的层是..., 则使用...初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x_mid = nn.functional.interpolate(x, size=[4, 4])
        x = self.conv3(x)
        # * 把低层特征和中层特征融合
        # TODO 融合特征1: 是否使用ASFF进行融合
        x_concate = torch.cat(tensors=(x, x_mid), dim=1)
        # x_concate = self.global_asff(x_mid, x)
        # * 对融合后特征做"集中注意力"处理
        # torch.mul是矩阵相乘
        # TODO 特征模块: 使用CBAM还是MA-CAM
        # x_concate = torch.mul(x_concate, self.attention_spatial(x_concate))
        # if self._quantize:
        if True:
            x_concate = (torch.mul((x_concate), (self.attention_spatial(x_concate))))
            x_concate = torch.mul(self.mul_quantizer(x_concate), self.mul_quantizer(self.attention_channel(x_concate)))
        else:
            x_concate = torch.mul(x_concate, self.attention_spatial(x_concate))
            x_concate = torch.mul(x_concate, self.attention_channel(x_concate))
        # x_concate = self.ms_cam(x_concate)

        x_concate = self.pool1(x_concate)
        x_concate = self.dropout1(x_concate)
        x_concate = x_concate.view(len(x_concate), -1)
        output1 = self.classfier1(x_concate)
        
        # * 这里是对另一支进行的计算, 也即对16分割之后的图像进行操作, 最终和主枝合并, 视为局部特征
        for i in range(16):
            idx_x = i % 4
            idx_y = i // 4
            if i == 0:
                # * 当处于第一个图像块的时候, 算第一个channel的结果
                x_patch_vector = self.conv4(input[:, :, idx_x*12:idx_x*12+24, idx_y*12:idx_y*12+24])
                x_patch_vector = self.conv5(x_patch_vector)
                # # * 加入fp16量化
                # x_patch_vector = self.ptq_quant(x_patch_vector)
                # TODO 每个小块attentionunit使用MS-CAM
                if self._quantize:
                    x_patch_vector = (torch.mul((x_patch_vector), (self.patchattention_spatial(x_patch_vector))))
                    x_patch_sum = torch.mul(self.mul_quantizer(x_patch_vector), self.mul_quantizer(self.patchattention_channel(x_patch_vector)))
                else:
                    x_patch_vector = torch.mul(x_patch_vector, self.patchattention_spatial(x_patch_vector))
                    x_patch_sum = torch.mul(x_patch_vector, self.patchattention_channel(x_patch_vector))
                # x_patch_sum = self.patch_ms_cam(x_patch_vector)
            else:
                # * 当不是第一个图像块的时候, 把后面每一个块算出来的结果拼接起来
                x_patch_vector = self.conv4(input[:, :, idx_x*12:idx_x*12+24, idx_y*12:idx_y*12+24])
                x_patch_vector = self.conv5(x_patch_vector)
                # TODO 每个小块都加了一个CBAM
                if self._quantize: 
                    x_patch_vector = torch.mul(x_patch_vector, self.patchattention_spatial(x_patch_vector))
                    x_patch_vector = torch.mul(self.mul_quantizer(x_patch_vector), self.mul_quantizer(self.patchattention_channel(x_patch_vector)))
                else:
                    x_patch_vector = torch.mul(x_patch_vector, self.patchattention_spatial(x_patch_vector))
                    x_patch_vector = torch.mul(x_patch_vector, self.patchattention_channel(x_patch_vector))
                # x_patch_vector = self.patch_ms_cam(x_patch_vector)
                # TODO 如何进行特征融合: ASFF or IAFF or concate
                x_patch_sum = torch.cat(tensors=(x_patch_sum, x_patch_vector), dim=1)
                # x_patch_sum = self.iaff(x_patch_vector, x_patch_sum)
        # x_patch_sum = self.asff_16(x_patch_sum)
        x_patch_sum = self.pool2(x_patch_sum)
        x_patch_sum = self.dropout2(x_patch_sum)
        x_patch_sum = x_patch_sum.view(len(x_patch_sum), -1)
        output2 = self.classfier2(x_patch_sum)

        # * 输出的是中低层特征以及加工后的特征图映射后的全连接层
        return output1, output2

