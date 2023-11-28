'''
检测QAT生成的能否导出onnx: 导不出
并测试运行时间
'''
from collections import OrderedDict
import os
import time
import torch

from Load_face_single import dataset
import torch.utils.data as Data
import torchvision
from torch.quantization import prepare, get_default_qconfig, prepare_qat, get_default_qat_qconfig, convert
from mfcnn import CNN
from resnet import ResNet18
from thop import profile
from pytorch_model_summary import summary
from resnet_q import QuantizableResNet18
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

SPLIT = 10


txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt".format(SPLIT)
# txt_path ="/home/developers/liuchang/FER数据集/RAF-DB/dataset_train_adj.txt"
# txt_path2 ="/home/developers/liuchang/FER数据集/RAF-DB/dataset_test_adj.txt"
# txt_path ="/home/developers/liuchang/FER数据集/Oulu-CASIA/OriginalImg/NI/{}_fold_txt_xSurprise".format(SPLIT)
img_path = "/home/developers/liuchang/FER数据集"
model_dir = "[12-20]/[12-20]ptq_oribest_model_for_split_0.pth.tar"
mode = "ptq"
test_dataset = dataset(annotation_file= os.path.join(txt_path, "{}_{}_fold_val.txt").format(0, SPLIT),
                                        img_dir= img_path,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((48, 48)),
                                        #    torchvision.transforms.Grayscale(1),
                                           torchvision.transforms.ToTensor(),
                                       ])
                                       )

test_dataloader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=80,
    shuffle=True,
)

def run_benchmark(model, img_loader):
    elapsed = 0
    # checkpoint = torch.load(model_file)['state_dict']
    model.eval()
    num_batches = 15
    # Run the scripted model on a few batches of images
    for i, sample_batched in enumerate(img_loader):
        if i == 0:
            img = sample_batched["image"][0]
        target = sample_batched["labels"]
        images = sample_batched["image"]
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break

    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f us' % (elapsed/num_images*1000000))
    return elapsed, img.unsqueeze(0)

model_dir = "/home/developers/liuchang/quantize/[12-15]/[12-15]oribest_model_for_split_0.pth.tar"
ori_model = ResNet18(num_class=6, num_blocks=[2, 2, 2, 2])
# ori_model = CNN(kind=6)
ori_model.eval()
checkpoint = torch.load(model_dir)#['state_dict']
prec = torch.load(model_dir)#['best_prec']
# print("modified float model:\nbest prec@", prec)
ori_model.load_state_dict(checkpoint)

# checkpoint = torch.load(model_dir)
# new_state_dict = OrderedDict() 
# for key, value in checkpoint.items():
    # name = key[7:]
        # name = key
    # new_state_dict[name] = value 
# ori_model.load_state_dict(new_state_dict)

model = QuantizableResNet18(num_class=6, num_blocks=[2, 2, 2, 2])
BACKEND = "fbgemm"

if mode == "qat":
    assert model.training
    model.fuse_model()
    model.qconfig = get_default_qat_qconfig(BACKEND)
    model = prepare_qat(model, inplace=True)
elif mode == "ptq":
    model.qconfig = get_default_qconfig(BACKEND)
    model = prepare(model, inplace=True)
ori_model.eval()
model.eval()
# run_benchmark(ori_model , test_dataloader)

checkpoint = torch.load(model_dir)['state_dict']
prec = torch.load(model_dir)['prec']
print("modified float model:\nbest prec@", prec)
# model.load_state_dict(checkpoint)
# run_benchmark(model , test_dataloader)

model_int8 = convert(model, inplace=True)
model_dir = "[12-20]/[12-20]ptq_quantizedbest_model_for_split_0.pth.tar"
checkpoint = torch.load(model_dir)['state_dict']
model_int8.load_state_dict(checkpoint)
prec = torch.load(model_dir)['prec']
print("modified quantized model:\nbest prec@", prec)
_, img = run_benchmark(model_int8, test_dataloader)

# print(summary(model, img, show_input=False, show_hierarchical=False))
# flops, params = profile(model, (img,))
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# print(summary(model_int8, img, show_input=False, show_hierarchical=False))
# flops, params = profile(model_int8, (img,))
# print('flops: %.2f M, par
# quant_nn.TensorQuantizer.use_fb_fake_quant = True

# quant_modules.initialize()
input_name = ["input"]
output_name = ["output"]
input = img
dummy_input = torch.randn(1, 1, 60, 60) #TODO: switch input dims by model

pred2 = ori_model(dummy_input)
torch.onnx.export(ori_model, dummy_input, "py-quant-ptq/ori_mfcnn.onnx", input_names=input_name, 
                output_names=output_name, do_constant_folding=True,verbose=True,opset_version=12, enable_onnx_checker=False)
# pred = model_int8(img)
# torch.onnx.export(model_int8, input, "quantized.onnx", input_names=input_name, 
#                 output_names=output_name, do_constant_folding=True,verbose=True,opset_version=12, enable_onnx_checker=False)