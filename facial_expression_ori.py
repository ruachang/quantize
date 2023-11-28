# coding=utf-8
'''
保存用来训练和测试的函数
'''
import os
from statistics import mode
from this import d
import warnings

from module_half import network_to_half
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from turtle import forward
import torch
import torch.nn as nn
import torch.utils.data as Data
from pytorch_quantization import nn as quant_nn
import onnxruntime

import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from Load_face_single import dataset
# from pytorch_quantize.mfcnn_quantize import export_onnx
# ? 一个常见的用来管理变量更新和迭代的类, 该类的定义可以计算变量的平均值和和
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train(train_loader, val_loader, device, model, criterion, optimizer, epochs, lr, record_flag=False, record_file=None,
          **kwargs):
    losses = AverageMeter()
    losses_image = AverageMeter()
    losses_patch = AverageMeter()
    top1_expression = AverageMeter()
    model.train()
    best_prac, _ = validate(val_loader, device, model, criterion, 0, 50)
    # ! 方便测试每次生成onnx改一下
    best_prac = 0
    onnx_flag = False
    for  epoch in range(epochs):
        if (epoch + 1) % 30 == 0:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for i, sample_batched in enumerate(train_loader):
            target = sample_batched["labels"]
            input = sample_batched["image"]
            target = target.to(device)
            input = input.to(device)

            output1, output2 = model(input)
            # * 对global和local做合并
            output3 = output1 + output2
            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            loss3 = criterion(output3, target)
            # TEST 这里有一组可以进行设置的超参数  ??如何调参 
            loss = loss1 + loss2 + 0.5 * loss3

            # ? 根据对AverageMeter的定义, 会自动累积计算loss的和, 平均
            losses.update(loss.item(), input.size(0))
            losses_image.update(loss1.item(), input.size(0))
            losses_patch.update(loss2.item(), input.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\n'
                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), losses=losses, lr=lr))
            print(output)
            if record_flag == True and i % 50 == 0:
                record_file.write("Epoch {}: {}".format(epoch, output))
        test_prec, test_loss = validate(val_loader, device, model, criterion, best_prac, 50)
        if best_prac <= test_prec:
            onnx_flag = export_onnx(model, device, **kwargs)
            best_prac = test_prec
        if record_flag == True:
            record_file.write("Epoch {}: test loss {:.4f}, test avg accuracy: {:.4f}\n".format(
                epoch, test_loss, test_prec
            ))
    return losses.avg, onnx_flag
    
            
def validate(val_loader, device, model, criterion, best_prec1, print_freq):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    for i, sample_batched in enumerate(val_loader):
        
        target = sample_batched["labels"]
        input = sample_batched["image"]
        input = input.to(device)
        target = target.to(device)

        output1, output2 = model(input)
        output3 = output1 + output2
        # * 这里应该是一个自定义的损失函数
        loss = criterion(output3, target)
        # * 这里是一个自定义的精确度计算
        prec1, _ = accuracy(output3.data, target, topk=(1,5))
        # * 对损失和精确度累积求和
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        if (i + 1) % print_freq == 0:
            output = ('Test: [{0}/{1}]\n'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), losses=losses, top1=top1))
            print(output)

    output = ('Testing Results:\n'
          'Prec@1 {top1.avg:.3f}\n'
          'Loss {losses.avg:.5f}\n'.format(
          top1=top1, losses=losses))
          
    print(output)
    
    output_best = 'Best Prec@1: %.3f'%(best_prec1)
    print(output_best)
    return top1.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # ? topk表示的是降序排列后的前k个
    # ? torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) 
    # ? A namedtuple of (values, indices) is returned with the values and indices 
    # ? of the largest k elements of each row of the input tensor in the given dimension dim.

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def export_onnx(model, device, onnx_filename, batch_onnx, per_channel_quantization):
    model.eval()
    # if fake_node:
    quant_nn.TensorQuantizer.use_fb_fake_quant = True # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    if per_channel_quantization:
        # Before opset13 is supported by Pytorch and ONNX runtime, we'll have to use the work around that
        # pretends opset12 supports per channel.
        # Use scripts/patch_onnx_export.sh to patch pytorch.
        opset_version = 12
    else:
        opset_version = 13

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    input_name = ["input"]
    output_name = ['output']
    dummy_input = torch.randn(batch_onnx, 1, 60, 60).to(device) #TODO: switch input dims by model
    output = model(dummy_input)
    try:
        torch.onnx.export(model, dummy_input, onnx_filename, input_names=input_name, 
                output_names=output_name, verbose=False, opset_version=opset_version, enable_onnx_checker=False, do_constant_folding=True)
    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    return True

def evaluate_onnx(onnx_filename, data_loader, criterion):
    """Evaluate accuracy on the given ONNX file using the provided data loader and criterion.
       The method returns the average top-1 accuracy on the given dataset.
    """
    print("Loading ONNX file: ", onnx_filename)
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    loss_logger = AverageMeter()
    top1_logger = AverageMeter()
    with torch.no_grad():
        with torch.no_grad():
            for i, sample_batched in enumerate(data_loader):
                target = sample_batched['labels']
                image = sample_batched['image']
                image = image.to("cpu", non_blocking=True)
                image_data = np.array(image)
                input_data = image_data

                # run the data through onnx runtime instead of torch model
                input_name = ort_session.get_inputs()[0].name
                raw_result = ort_session.run([], {input_name: input_data})
                output0, output1 = torch.tensor((raw_result))
                output = output0 + output1
                output = output.float()
                loss = criterion(output, target)
                acc1, _ = accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                loss_logger.update(loss.item(), batch_size)
                top1_logger.update(acc1.item(), batch_size)
        print('  ONNXRuntime: Acc@1 {top1.avg:.3f} Loss {loss.avg:.3f}'
            .format(top1=top1_logger, loss=loss_logger))
        return loss_logger.avg, top1_logger.avg