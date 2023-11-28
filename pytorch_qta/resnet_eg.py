'''
pytorch 量化
官方示例: 使用MobileNet的官方修改QAT示例
'''

import torch
from torch.quantization import prepare_qat, get_default_qat_qconfig, convert
from torchvision.models import quantization
import torch.quantization.fuse_modules as fuse_modules
# Step1：修改模型
# 这里直接使用官方修改好的MobileNet V2，下文会对修改点进行介绍
model = quantization.mobilenet_v2()
print("original model:")
print(model)

# Step2：折叠算子
# fuse_model()在training或evaluate模式下算子折叠结果不同，
# 对于QAT，需确保在training状态下进行算子折叠
assert model.training
model.fuse_model()
print("fused model:")
print(model)

# Step3:指定量化方案
# 通过给模型实例增加一个名为"qconfig"的成员变量实现量化方案的指定
# backend目前支持fbgemm和qnnpack
BACKEND = "fbgemm"
model.qconfig = get_default_qat_qconfig(BACKEND)

# Step4：插入伪量化模块
prepare_qat(model, inplace=True)
print("model with observers:")
print(model)

# 正常的模型训练，无需修改代码

# Step5：实施量化
model.eval()
# 执行convert函数前，需确保模型在evaluate模式
model_int8 = convert(model)
print("quantized model:")
print(model_int8)

# Step6：int  8模型推理
# 指定与qconfig相同的backend，在推理时使用正确的算子
torch.backends.quantized.engine = BACKEND
# 目前Pytorch的int8算子只支持CPU推理,需确保输入和模型都在CPU侧
# 输入输出仍为浮点数
fp32_input = torch.randn(1, 3, 224, 224)
y = model_int8(fp32_input)
print("output:")
print(y)

####################################################################
# STEP1 修改模型加入量化伪量化节点
class QuantizableMobileNetV2(MobileNetV2):
  def __init__(self, *args, **kwargs):
    """
    MobileNet V2 main class
    Args:
    Inherits args from floating point MobileNetV2
    """
    super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
    self.quant = QuantStub()
    self.dequant = DeQuantStub()

  def forward(self, x):
    x = self.quant(x)
    x = self._forward_impl(x)
    x = self.dequant(x)
    return x

class QuantizableInvertedResidual(InvertedResidual):
  def __init__(self, *args, **kwargs):
    super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
    # 加法的伪量化节点需要记录所经过该节点的数值的范围，因此需要实例化一个对象
    self.skip_add = nn.quantized.FloatFunctional()

  def forward(self, x):
      if self.use_res_connect:
          # 普通版本MobileNet V2的加法
          # return x + self.conv(x)
          # 量化版本MobileNet V2的加法
          return self.skip_add.add(x, self.conv(x))
      else:
          return self.conv(x)
# STEP2 模型的融合

def fuse_model(self):
    # 遍历模型内的每个子模型，判断类型并进行相应的算子折叠
    for m in self.modules():
        # 两种融合方式, 直接指定带融合的类型, 并且指定融合的标号
        if type(m) == ConvBNReLU:
            fuse_modules(m, ['0', '1', '2'], inplace=True)
        # 调用子模块中自带的融合
        if type(m) == QuantizableInvertedResidual:
            # 调用子模块实现的fuse_model()，间接调用fuse_modules()
            m.fuse_model()

