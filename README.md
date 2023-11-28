# 量化训练PyTorch实现

## [大致讲解](https://zhuanlan.zhihu.com/p/299108528)

***整个基于pytorch包的实现就无法使用onnx进行推理， 也不支持在任何硬件平台上部署*** 只能从理论上加速

* 对tensor的量化: 如论文所述, 对原始的数据进行 ***线性投影***, 使用的函数为`torch.quantize_per_tensor`, 按照参数中的scale和zero point进行投影和缩放, 反量化为`dequantize`
* 量化的两种模式, 指的是在多大范围内使用的量化参数是一样的
  * per tensor: 一个tensor内量化
  * per channel: tensor中一个维度的内量化

目前PyTorch中实现的量化方法有三种, 其中两种是基于训练好的模型直接定标, 该参数, 一种是训练的时候引入伪量化节点, 帮助提升量化的精度

### ptq(dynamic)

适用于RNN和Linear变种的量化方式, 把weight的参数全部量化, 而input的值在 ***推理的过程中***, 根据保存的量化参数量化运算, 在输出时在把结果 ***量化回去***

### ptq(static)

使用数据集对网络做calibration, 根据数据集分布特性来计算量化参数. 网络包含forward和activation, 可以将网络的输入调整成需要的模式.

需要的步骤同样可以分为五步

#### fuse model

如下QAT所述相同, 不在此赘述, 但是在ptq模式融合带有Batchnorm的层会无法量化, 所以实际上不太可以融合

#### 设置qconfig

设置模型的运行平台, 和下面的一样

#### prepare

为网络中插入`Observer`, 收集数据, 方便之后进行定标, 插入`QuantStub`和`DeQuantStub`

`QuantStub`和`DeQuantStub`会在`observer`中记录参数值, 相关模块会被替换成pytorch中定义的量化模块`nnq.Quantize`和`nnq.DeQuantize`, 同样在实际推理的时候会经过 ***量化和反量化***

#### 放入数据定标

#### 模型转换

该函数会对模型中的可以量化的模块进行替换

### QAT(Quantization Aware Training)

主要的是步骤可以分为五步
[参考链接](my.oschina.net/u/4580321/blog/4750607)
#### 修改模型

训练模式 + 网络设置, 主要是为了方便模型之后在量化

#### fuse module

在量化时, 插入的伪量化节点越少, 对最后的精度影响就越小, 因此通过将层进行融合, 可以减少插入的节点

使用函数`torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)`可以对网络做融合, 可以对原始网络做替换, 变为`融合模块 + Identity(占位符)`的形式

* 对于在Sequential中的模块, 可以迭代的融合
* 可以融合的只有特定的模块顺序, 在`DEFAULT_OP_LIST_TO_FUSER_METHOD`中有定义
* 必须手动的定义

#### 插入量化节点和伪量化节点

* 对每种operation设置qconfig
  * 加法需要使用量化运算
  * 模型中只有加了`qconfig`的层才会进行量化, 如果直接对整个模型设定, 则对全模型量化
* 将PyTorch中运行量化的module整个换成预定义好的量化module
  * 在模型的输入前加入`QuantStub`, 模型的输出后加入`DeQuantStub`
    * 加入量化的会在计算的时候量化为, 反量化模块的地方会在计算时反量化

在设置完成后使用`prepare_qat()`为每层增加`Observer`模块, 方便之后对模型进行量化

#### 训练

#### 转化

经过训练后得到每个操作中量化时需要用来投影的参数组, 通过`torch.quantization.convert`对模型进行转化, 所有的operation被替换为了量化后的节点

### 实验结果

转化和融合模型成功了, 但是没有办法导出onnx, 似乎目前使用QAT的来进行化简的方式使用的伪量化节点没有办法导出onnx, 显示`tensor has no detach`, 应该是pytorch还不支持.

### ptq(static) 和 QAT 差别

* qconfig: 
  * ptq: 两个observer, 只根据样本观察weight和activation
  * QAT: Fakequantize, 伪量化节点, 求解最值的方法不同, 包括滑动窗法和直接求
* 对模型做整体的模块替换
  * ptq: prepare, 插入量化
  * QAT: prepare_qat, 插入量化, 转换子module, 把某些模块换掉
* 下一步
  * ptq: 校准
  * QAT: 训练

## 官方文档(PyTorch-quantization)

pytorch-quantization 是使用TensorRt部署训练时量化的官方方法, 其适用于`int8`模型的训练, 可以使用`qat`, 也可以使用`ptq`.

由于其在训练时插入了量化节点和反量化节点, 在由onnx转化为trt时, 只有TensorRt8以上的才可以直接转化, 8以下的版本需要手写量化节点的转化(没学)

注: 该量化不支持`fp16`模式

### 2.0.0

文档里面什么都没有...

#### notebook中的两个例子

理论上允许伪量化, 相当于对节点量化在反量化, 将代表的float的bit数减少来进行量化

如果使用已有的模型进行量化的话, 相当于使用`PTQ`, 对网络中的`Convnd`和`Linear`进行取代, 变为量化版的. 量化版和原版的差别是
在前推和后推的时候会用模型中保存的max计算量化参数并在推理时进行量化, 因此实际不会保存网络的量化参数

如果使用`PTQ`方法, 需要在定义好量化模型后对模型进行校准, 并且量化模式. 在做好基本设定后, 只要遵循`enable calibration, disable quantization => calibration => load, enable quantization`即可.

在使用`PTQ`后, 可以进一步重新对模型做finetune进一步提升准确率, 只要一开始, 在模型声明之前将当前环境初始化为`pytorch-quantization`中定义的`quant_modules.initialize()`, 就可以将所有模型中的`Convnd`和`Linear`一次替换, 可以在calibration之后直接做finetune, 加载模型的时候直接设置即可

可以直接导出onnx, 导出的ONNX会有QuantizeLinear和DequantizeLinear两个算子, 而TensorRT8版本以下的不支持直接载入，需要手动去赋值MAX阈值


"
ONNX’s QuantizeLinear and DequantizeLinear operators are mapped to these new layers which enables the support for networks trained using Quantization-Aware Training (QAT) methodology. For more information, refer to the Explicit-Quantization, IQuantizeLayer, and IDequantizeLayer sections in the TensorRT Developer Guide and Q/DQ Fusion in the Best Practices For TensorRT Performance guide.
"

对于Quantization Aware Training, 理论上讲是可以的, 需要在 ***校准之后***, 开始进行重新训练, 对模型进行finetune, 理论上还可以导出onnx

#### classification flow

STEP 

* 设定量化模式, 对weight和activation的量化方式初始化, 最后初始化
  * 分为per-channel和per-layer: 每个filter每个channel的初始化方式是否相同
* 定义模型, 加载参数, 加载数据
* 校准数据
  * 先对模型中插入的 ***量化版*** 节点进行模式的设定: 量化模式 => 校准模式
  * 放进去测试数据做校准: 也即直接投喂数据收集weight和activation的直方图
  * 再把模式调回去: 校准模式 => 量化模式
  * 依次对各层遍历, 计算每一个需要量化节点的最值, 保存为`amax`
  * validate, 确认模型在量化后的准确率
* 对模型进行进一步的finetune
* 导出onnx模型
  * 进一步对onnx模型做验证: 使用`onnxruntime`来验证

### 2.1.2

#### 实现QAT

QAT和PTQ的差别只在是否进行后面的训练, 以及插入的是量化节点还是伪量化节点. 伪量化节点输出仍然是浮点数, 量化节点输出是整型数

对于待量化的部分, 可以选择手动插入, 也可以自动替换
* 自动替换的范围是所有有量化定义的module, 其它的运算无法进行量化
* 手动插入需要在目标前面加入`quant_nn.TensorQuantizer`类的量化器, 在之后进行运算时对目标进行量化后在进行后续的运算


文档里面只写了如何手动构建一个进行量化的模块, 这样的模块有三种, 分别是

* 没有参数, 只量化输入
  * 设定新的量化类, 继承原本的类, 并且新加一个`_utils`的变量
  * 为该类初始化量化的设定和`TensorQuantizer`(和自动量化一致的步骤)
  * 定义`forward`, 并且继承原本的前推和定义还的`_input_quantizer`, 对输入做量化
* 需要同时量化输出和weight, 步骤和上述大致相同
  * 设定新的变量类, 继承原来的类, 新加一个`_utils`类用来定义量化的weight和input
  * 为该类初始化量化以及设定input和weight的量化方式
  * 在forward中定义对weight和input的量化并计算输出
* 直接在网络的运算图中修改
  * 在目标量化的操作之前加入`quant_nn.TensorQuantizer`, 在输入前量化
  * 如果对于多输入的操作想要进行量化, 需要保证输入到目标节点的都是同样的精度
  * 类似在`forward`的时候改

### 复现

成功了!!至少可以完成由`quantization => onnx => tensorrt`的转化

现在需要测试转化的性能, 需要测试
resnet
* 现在的量化时16位的有问题的量化
  模型类型|original|exampled quantization|original + TensorRT --fp16| Quantization + TensorRT --fp16 |
|--|:----------:|:--------------:|:----------:|:-----------:|
是否复现     |  v  | V | X | V | 
模型大小     |     |
运行时间     | [![直接转化模型](https://s1.ax1x.com/2022/12/21/zXAjJ0.png)](https://imgse.com/i/zXAjJ0) | [![直接转化量化模型](https://s1.ax1x.com/2022/12/22/zXmC1x.png)](https://imgse.com/i/zXmC1x) | [![fp16转化模型](https://s1.ax1x.com/2022/12/22/zXMAxI.png)](https://imgse.com/i/zXMAxI) | [![fp16量化模型](https://s1.ax1x.com/2022/12/22/zXMvWj.png)](https://imgse.com/i/zXMvWj)

mfcnn

  模型类型|original|exampled quantization|original + TensorRT --fp16| Quantization + TensorRT --fp16 |
|--|:----------:|:--------------:|:----------:|:-----------:|
是否复现     |  v  | V | X | V | 
模型大小     |     |
运行时间     | [![直接转化模型](https://s1.ax1x.com/2022/12/21/zXAjJ0.png)](https://imgse.com/i/zXAjJ0) | [![直接转化量化模型](https://s1.ax1x.com/2022/12/22/zXmC1x.png)](https://imgse.com/i/zXmC1x) | [![fp16转化模型.png](https://s1.ax1x.com/2023/02/20/pSXQadJ.png)](https://imgse.com/i/pSXQadJ)| [![fp16量化.png](https://s1.ax1x.com/2023/02/20/pSXQFPI.png)](https://imgse.com/i/pSXQFPI)

./trtexec --onnx=/home/developers/liuchang/quantize/quantized.onnx --saveEngine=/home/developers/liuchang/quantize/py-quant-ptq/fp32_q_resnet.trt --workspace=100 --verbose=True --exportProfile=/home/developers/liuchang/quantize/py-quant-ptq/fp32_q_resnet.json

## trtexec直接转化

最直接简单的转化法, 该方法同样可以通过简单的手写代码实现

## 新找到的东西
[官方解说](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html)最后一部分

For efficient inference on TensorRT, we need know more details about the runtime optimization. TensorRT supports fusion of quantizing convolution and residual add. The new fused operator has two inputs. Let us call them conv-input and residual-input. Here the fused operator’s output precision must match the residual input precision. When there is another quantizing node after the fused operator, we can insert a pair of quantizing/dequantizing nodes between the residual-input and the Elementwise-Addition node, so that quantizing node after the Convolution node is fused with the Convolution node, and the Convolution node is completely quantized with INT8 input and output. We cannot use automatic monkey-patching to apply this optimization and we need to manually insert the quantizing/dequantizing nodes.

针对ResNet的相加部分, 认为可以额外增加量化部分, [代码链接](https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/torchvision/models/classification/resnet.py)

经过实践, 这一部分指的是不依靠自动插入伪量化节点, 对某些运算符进行量化, 如果量化的Operator允许量化, 可能可以有新的可融合节点, 或者省下一部分原先的reformate层来