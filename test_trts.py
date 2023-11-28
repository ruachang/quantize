'''
作用: 用来测试所有模型引擎在某个数据集上面的运行时间和精确度
需要输入的值有:
* 数据集的txt路径
* 数据集路径
* 模型的trt路径
* 数据记录的路径

如果要和原来的对比, 就保留STEP4之后的, 并且输入生成模型的路径
'''
from collections import OrderedDict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

cuda.init()
cfx = cuda.Device(0).make_context()
# STEP1 修改模型和数据集路径
TRT_LOGGER = trt.Logger()
runtime = trt.Runtime(TRT_LOGGER)
engine_file_path = ["py-quant-ptq/conf_use/model/mffcnn_fuse/PWN/PWN_mix_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/model/mffcnn_fuse/channel_attention_q/qat8_histogram_maxpoolq_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/model/mffcnn_fuse/maxpool_q/qat8_histogram_mix_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/test2/q_the_10k/q_the_10k_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/test2/q_the_100k/q_the_100k_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/test2/q_the_1M/q_the_1M_mffcnn.onnx.engine",
                    "py-quant-ptq/conf_use/ori/ori_mfcnn.onnx.engine"
                    
                    ]
txt_path ="/home/developers/liuchang/FER数据集/CK+/CK+100cropped/3_FalseHaveNormal.txt"
# txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/{}_10_fold_val.txt".format(7)
# txt_path = "/home/developers/liuchang/FER数据集/cam_data/cropped_record/1_FalseHaveNormal_pick.txt"
img_path = "/home/developers/liuchang/FER数据集"
record_path = "py-quant-ptq/conf_use/mffcnn_fuse/record_all.txt"
# 是否在原来的txt上标注识别的结果
flag_label = False
labeled_path = "py-quant-ptq/doc_use/mffcnn_real_label.txt"
if os.path.isdir(os.path.dirname(record_path)) is False:
    os.mkdir(os.path.dirname(record_path))
# * 定义记录文件
file = open(record_path, 'w')
if flag_label:
    label_file = open(labeled_path, 'w')
else:
    label_file = None
# STEP2 载入数据集
img_labels = pd.read_csv(txt_path, delimiter=" ", header=None)
# img_labels = shuffle(img_labels)
reason_time = 0
img_num, _ = img_labels.shape


def read_single_image(img_labels, index, label_file=None):
    # * 读取图片
    # 此处默认第一列为名字, 第二列为标签
    img_path_body = img_labels.iloc[index, 0]
    # img_path_head = img_path_body.split('_')
    pic_path = os.path.join(img_path, img_path_body)
    # [拼出文件夹的名字]
    # img_lst = np.zeros(/(1, 60, 60))    
    # img_tmp = cv2.imread(pic_path)
    image = cv2.imread(pic_path)
    data2 = torch.randn((1, 60, 60))
    # image = Image.fromarray(image)
    transform_Fesim=torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((60, 60)),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.ToTensor(),
                                        ])
    
    image_label = int(img_labels.iloc[index, 1]) - 1
    image = transform_Fesim(image)
    image = image.unsqueeze(0).numpy()
    imgs = np.array(image, dtype= np.float32)
    # print(img_seq.shape)
    sample = {"image": imgs,"labels": image_label}
    
    if flag_label:
        line = img_labels.iloc[index, 0] + " " + str(img_labels.iloc[index, 1])
        label_file.write(line)
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
        output1, output2 = model(input)
        end = time.time()
        reason_time += end - start 
        output3 = torch.add(output1, output2)
        _, pred = output3.topk(1, 1, True, True)
        
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
    reason_time2 = 0
    
    num_sample_val = np.zeros(shape=(1, KIND), dtype=np.float32)
    confuse_mat = np.zeros(shape=(KIND, KIND), dtype=np.float32)
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(20):
        sample_batched = read_single_image(img_labels, i, label_file)
        target = sample_batched["labels"]
        # Batch_size * C * H * W
        input = sample_batched["image"]
        # target = target.to(device)
        # input = input.to(device)
        inputs[0].host = input
        trt_outputs =do_inference_v2(context, bindings=bindings, \
	    inputs=inputs, outputs=outputs, stream=stream)
    
    for i in range(img_num):
        sample_batched = read_single_image(img_labels, i, label_file)
        target = sample_batched["labels"]
        # Batch_size * C * H * W
        input = sample_batched["image"]
        # target = target.to(device)
        # input = input.to(device)
        inputs[0].host = input
        start = time.time() 
        # starter.record()
        cfx.push()
        trt_outputs =do_inference_v2(context, bindings=bindings, \
	    inputs=inputs, outputs=outputs, stream=stream)
        cfx.pop()
        end = time.time()
        # ender.record()
        torch.cuda.synchronize()
        # reason_time2 += starter.elapsed_time(ender)
        reason_time += end - start
        output = trt_outputs[0]
        if len(trt_outputs) != 1:
            for i in range(1, len(trt_outputs)):
                output += trt_outputs[i]
        pred = output.argmax()        

        if flag_label:
            label_file.write(" {} {} \n".format(pred, str(pred == target)))
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
        confuse_mat[i, ...] /= num_sample_val[0, i]

    accuracy_expression = correct / img_num
    file.write("In trt model {}: \n".format(os.path.basename(engine_file_path).split(".")[0]))
    file.write("Final accuracy of expression: {:.4f}, Average reasoning time: {:.7f}(qps: {}), Average infer time: {:.7f}(qps: {})\nNumber of samples: {}, confusion matrix: \n {} \n\n".format(
        accuracy_expression, reason_time / img_num, img_num / reason_time, reason_time2 / img_num, img_num / reason_time, 
        num_sample_val, confuse_mat))
    file.flush()   
    
    f.close()

for models in engine_file_path:
    trt_infer(models)
file.close()
cfx.pop()

if flag_label:
    label_file.close()