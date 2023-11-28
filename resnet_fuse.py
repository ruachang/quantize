'''
ResNet in PyTorch. 基本模板

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn

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
        
def conv1x1(in_planes: int, out_planes: int, stride: int = 1, quantize: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    if quantize:
        return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, quantize=False):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(in_planes, planes, stride=stride, padding=1, quantize=quantize)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=1, padding=1, quantize=quantize)
        self.bn2 = nn.BatchNorm2d(planes)
        self._quantize = quantize
        if self._quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride, quantize=quantize),
                
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        if self._quantize:
            out += self.residual_quantizer(shortcut)
        else:
            out += shortcut
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, quantize=False):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x1(in_planes, planes, quantize=quantize)
        
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=1, quantize=quantize)
        
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.conv3 = conv1x1(planes, self.expansion*planes, quantize=quantize)
        
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride, quantize=quantize),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self._quantize = quantize
        if self._quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x)
        if self._quantize:
            out += self.residual_quantizer(shortcut)
        else:
            out += shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7, quantize=False):
        super(ResNet, self).__init__()
        self._quantize = quantize
        
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv3x3(3, 64, stride=1, padding=1, quantize=quantize)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, quantize=quantize)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, quantize=quantize)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, quantize=quantize)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, quantize=quantize)
        # self.last_layer=nn.Conv2d(512,num_classes,kernel_size=1)
        if quantize:
            self.linear = quant_nn.QuantLinear(512, num_classes)
        else:
            self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, quantize):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, quantize))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out=self.last_layer(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        return out

def ResNet18(num_class=8, num_blocks=[2,2,2,2], quantize=False):
    return ResNet(BasicBlock, num_blocks, num_classes=num_class, quantize=quantize)