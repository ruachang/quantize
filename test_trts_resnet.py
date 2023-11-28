'''
作用: 用来测试所有模型引擎在某个数据集上面的运行时间和精确度(for resnet)
需要输入的值有:
* 数据集的txt路径
* 数据集路径
* 模型的trt路径
* 数据记录的路径

如果要和原来的对比, 就保留STEP4之后的, 并且输入生成模型的路径
'''
from collections import OrderedDict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from sklearn.utils import shuffle
import torch
import torch.utils.data as Data
import torchvision
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import time
import cv2
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime
import tensorrt as trt 
import onnx

from Load_face_single import dataset

BATCH_SIZE = 1
KIND = 6
device_id = [0]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# STEP1 修改模型和数据集路径
TRT_LOGGER = trt.Logger()
runtime = trt.Runtime(TRT_LOGGER)
engine_file_path = ["py-quant-ptq/tensor8/fp32/ori_resnet.trt", "py-quant-ptq/tensor8/fp32/ori_resnet_sim.trt", 
                    "py-quant-ptq/tensor8/fp16/ptq_fp16_resnet.trt", "py-quant-ptq/tensor8/int8/ptq_int8_resnet.trt",
                    "py-quant-ptq/tensor8/int8/qat_int8_resnet.trt"]
engine_file_path = ['py-quant-ptq/tensor8/int8/qat_int8_resnet.trt', 'py-quant-ptq/conf_use/model/resnet_fuse/qta_int8_resnet.onnx.engine']
txt_path ="/home/developers/liuchang/FER数据集/CK+/CK+100cropped/3_FalseHaveNormal.txt"
img_path = "/home/developers/liuchang/FER数据集"
record_path = "py-quant-ptq/conf_use/model/resnet_fuse/fuse_record.txt"
if os.path.isdir(os.path.dirname(record_path)) is False:
    os.mkdir(os.path.dirname(record_path))
# * 定义记录文件record_path
file = open(record_path, 'w')

# STEP2 载入数据集
img_labels = pd.read_csv(txt_path, delimiter=" ", header=None)
img_labels = shuffle(img_labels)
reason_time = 0
img_num, _ = img_labels.shape


def read_single_image(img_labels, index):
    # * 读取图片
    # 此处默认第一列为名字, 第二列为标签
    img_path_body = img_labels.iloc[index, 0]
    # img_path_head = img_path_body.split('_')
    pic_path = os.path.join(img_path, img_path_body)
    # [拼出文件夹的名字]
    # img_lst = np.zeros((1, 60, 60))    
    # img_tmp = cv2.imread(pic_path)
    image = cv2.imread(pic_path)
    image = Image.fromarray(image)
    transform_Fesim=torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            torchvision.transforms.Resize((48, 48)),
            # torchvision.transforms.Grayscale(1),
            torchvision.transforms.ToTensor(),
                                        ])
    
    image_label = int(img_labels.iloc[index, 1]) - 1
    image = transform_Fesim(image)
    image = image.unsqueeze(0).numpy()
    imgs = np.array(image, dtype= np.float32)
    # print(img_seq.shape)
    sample = {"image": imgs,"labels": image_label}
    return sample      

def confusion_matrix(val_loader, model, kind):
    # * 绘制7类预测的混淆矩阵
    model.eval()
    reason_time = 0
    confusion_matrix = np.zeros(shape=(kind, kind), dtype=np.int)
    num_sample = np.zeros(shape=(1, kind), dtype=np.int)
    # 针对验证集求
    for i, sample_batched in enumerate(val_loader):
        target = sample_batched["labels"]
        input = sample_batched["image"]
        target = target.to(device)
        input = input.to(device)
        # 沿着第二维计算最大的一个
        start = time.time()
        output = model(input)
        end = time.time()
        reason_time += end - start 
        _, pred = output.topk(1, 1, True, True)
        
        for i in range(len(target)):
            num_sample[0, target[i].data] += 1
            confusion_matrix[target[i].data, pred[i].data] += 1   
              
    return num_sample, confusion_matrix, reason_time

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()
    
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
# STEP3 载入模型引擎
def trt_infer(engine_file_path):
    f = open(engine_file_path, "rb")
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context() 
    # with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime,\
    # runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    buffers = allocate_buffers(engine)
    inputs, outputs, bindings, stream = buffers
    reason_time = 0
    num_sample_val = np.zeros(shape=(1, KIND), dtype=np.int)
    confuse_mat = np.zeros(shape=(KIND, KIND), dtype=np.int)
    for i in range(img_num):
        sample_batched = read_single_image(img_labels, i)
        target = sample_batched["labels"]
        # Batch_size * C * H * W
        input = sample_batched["image"]
        # target = target.to(device)
        # input = input.to(device)
        inputs[0].host = input
        start = time.time() 
        trt_outputs =do_inference_v2(context, bindings=bindings, \
	    inputs=inputs, outputs=outputs, stream=stream)
        end = time.time()
        reason_time += end - start
        output = trt_outputs[0]
        if len(trt_outputs) != 1:
            for i in range(1, len(trt_outputs)):
                output += trt_outputs[i]
        pred = output.argmax()        
    
        num_sample_val[0, target] += 1
        confuse_mat[target, pred] += 1   
    print("In trt model {}: ".format(os.path.basename(engine_file_path).split(".")[0]))
    print("Number of samples")
    print(num_sample_val)
    print("average reason time: ", reason_time / img_num)
    print("Confusion matrix")
    print(confuse_mat)
    correct = 0
    for i in range(KIND):
        for j in range(KIND):
            if i==j:
                correct += confuse_mat[i,j]

    accuracy_expression = correct / img_num
    file.write("In trt model {}: ".format(os.path.basename(engine_file_path).split(".")[0]))
    file.write("Final accuracy of expression: {:.4f}, Average reasoning time: {:.7f}\nNumber of samples: {}, confusion matrix: \n {} \n".format(
        accuracy_expression, reason_time / img_num, 
        num_sample_val, confuse_mat))
    file.flush()    
    f.close()

for i in engine_file_path:
    trt_infer(i)
