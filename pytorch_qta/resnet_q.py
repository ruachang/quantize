'''
pytorch quantization
ResNet in PyTorch : quantized modified

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization.fuse_modules as fuse_modules
from torch.autograd import Variable

class QuantizableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(QuantizableBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = self.skip_add.add(out, self.shortcut(x))
        out = F.relu(out)
        return out
    def fuse_model(self):
        for m in self.named_modules():
            if m[0] == "conv1":
                fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
            elif m[0] == "conv2":
                fuse_modules(self, ['conv2', 'bn2'], inplace=True)
            elif m[0] == "shortcut":
                if len(m[1]) != 0:
                    fuse_modules(m[1], ['0', '1'], inplace=True)
            else:
                continue
    
                

class QuantizableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(QuantizableBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = self.skip_add(out, self.shortcut(x))
        out = F.relu(out)
        return out


class QuantizableResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(QuantizableResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.last_layer=nn.Conv2d(512,num_classes,kernel_size=1)
        self.linear = nn.Linear(512, num_classes)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out=self.last_layer(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        out = self.dequant(out)
        return out
    def fuse_model(self):
        # for m in self.named_children():
            # print(m[0])
        for m in self.named_children():
            if (m[0]) == 'conv1':
                fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
            elif "layer" in (m[0]):
                for i in m[1]:
                    i.fuse_model()
            else:
                continue
            

def QuantizableResNet18(num_class=8, num_blocks=[2,2,2,2]):
    return QuantizableResNet(QuantizableBasicBlock, num_blocks, num_classes=num_class)