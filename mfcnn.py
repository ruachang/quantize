# coding=utf-8
'''
mfcnn的网络部分
'''
import os
from turtle import forward
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
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
# from fusion import iAFF, ASFF, ASFF_16
class ChannelAttention(nn.Module):
# * 神奇的注意力模块, 只在神经网络搭建的时候用了
# * Channel 注意力
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
# * 神奇的注意力集中模块堂堂再登场! 同样只在神经网络搭建时用到
# * Spatial 注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
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
    def __init__(self, kind):
        super(CNN, self).__init__()
        self.kind = kind
        self.conv1 = nn.Sequential(
            # * 输入通道数1, 输出图像长宽-2
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # * 输入通道数64, 输出图像长宽-2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # * 输出图像大小变成原来的一半
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            # * 输入通道数64, 输出图像长宽-2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # * 输入通道数128, 输出图像长宽-2
           nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # * 输出图像大小变成原来一半
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            # * 输入通道数128, 输出图像长宽-2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # * 输入通道数256, 输出图像长宽-2
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # * 输出图像大小变成原来一半
            nn.MaxPool2d(kernel_size=2),
        )
        # * 使用两种注意力集中模块依次作用
        self.attention_spatial = SpatialAttention()
        self.attention_channel = ChannelAttention(384)

        # TODO 是否初始化MS_CAM
        # ? 暂时不考虑global的了, 有点麻烦
        # self.global_asff = ASFF(1)
        # self.ms_cam = MS_CAM(384)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.2)
        self.classfier1 = nn.Linear(384, self.kind)
        # * conv4和conv1的结构差不多
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # * conv5和conv2的结构差不多
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # * 两边使用的注意力集中模块不太一样, 维度不太一样
        self.patchattention_spatial = SpatialAttention()
        self.patchattention_channel = ChannelAttention(128)

        # TODO 是否初始化MS_CAM
        # self.patch_ms_cam = MS_CAM(128)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.dropout2 = nn.Dropout(0.2)

        # TODO 取决于是否使用iaff, 使用的话这里会采用128
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
                # TODO 每个小块attentionunit使用MS-CAM
                x_patch_vector = torch.mul(x_patch_vector, self.patchattention_spatial(x_patch_vector))
                x_patch_sum = torch.mul(x_patch_vector, self.patchattention_channel(x_patch_vector))
                # x_patch_sum = self.patch_ms_cam(x_patch_vector)
            else:
                # * 当不是第一个图像块的时候, 把后面每一个块算出来的结果拼接起来
                x_patch_vector = self.conv4(input[:, :, idx_x*12:idx_x*12+24, idx_y*12:idx_y*12+24])
                x_patch_vector = self.conv5(x_patch_vector)
                # TODO 每个小块都加了一个CBAM
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

