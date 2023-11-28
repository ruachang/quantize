# coding=utf-8
'''
针对resnet尝试使用pytorch进行量化
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.quantization import prepare_qat, prepare, get_default_qconfig, get_default_qat_qconfig, convert

import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

from Load_face_single import dataset
from resnet import ResNet, ResNet18
from resnet_q import QuantizableResNet18

BATCH_SIZE = 4      
EPOCH = 1         
SPLIT = 10  
KIND = 6

mode = "ptq"

device_id = [0]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# TEST: txt的路径
txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt".format(SPLIT)
# txt_path ="/home/developers/liuchang/FER数据集/RAF-DB/dataset_train_adj.txt"
# txt_path2 ="/home/developers/liuchang/FER数据集/RAF-DB/dataset_test_adj.txt"
# txt_path ="/home/developers/liuchang/FER数据集/Oulu-CASIA/OriginalImg/NI/{}_fold_txt_xSurprise".format(SPLIT)
img_path = "/home/developers/liuchang/FER数据集"
# ? 一个常见的用来管理变量更新和迭代的类, 该类的定义可以计算变量的平均值和
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
        
def train(train_loader, model, criterion, optimizer, epoch, lr):
    model.to(device)
    losses = AverageMeter()
    top1_expression = AverageMeter()
    model.train()
        
    for i, sample_batched in enumerate(train_loader):
        target = sample_batched["labels"]
        input = sample_batched["image"]
        target = target.to(device)
        input = input.to(device)
        
        output1 = model(input)
        # * 对global和local做合并
        
        loss = criterion(output1, target)

        # ? 根据对AverageMeter的定义, 会自动累积计算loss的和, 平均
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\n'
                'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), losses=losses, lr=lr))
        print(output)
    return losses.avg
    
def validate(val_loader, model, criterion, best_prec1):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    model.to(torch.device('cpu'))
    model_int8 = convert(model, inplace=True)
    model_int8.to(torch.device('cpu'))
    # model_int8.to(device)

    for i, sample_batched in enumerate(val_loader):
        
        target = sample_batched["labels"]
        input = sample_batched["image"]
        # input = input.to(device)
        # target = target.to(device)

        output1 = model_int8(input)
        # * 这里应该是一个自定义的损失函数
        loss = criterion(output1, target)
        # * 这里是一个自定义的精确度计算
        prec1, _ = accuracy(output1.data, target, topk=(1,5))
        # * 对损失和精确度累积求和
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

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
    return top1.avg, losses.avg, model_int8

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
    
def save_best_model(state, is_best, path, split, time_str):
    filename = time_str + "best_model_for_split_" + str(split) + ".pth.tar"
    save_path = os.path.join(path, filename)
    if is_best:
        torch.save(state, save_path)

def confusion_matrix(val_loader, model, kind):
    # * 绘制6类预测的混淆矩阵
    model.eval()
    confusion_matrix = np.zeros(shape=(kind, kind), dtype=np.int)
    num_sample = np.zeros(shape=(1, kind), dtype=np.int)
    # 针对验证集求
    for i, sample_batched in enumerate(val_loader):
        target = sample_batched["labels"]
        input = sample_batched["image"]
        # target = target.to(device)
        # input = input.to(device)
        # 沿着第二维计算最大的一个
        output1 = model(input)
        _, pred = output1.topk(1, 1, True, True)
        
        for i in range(len(target)):
            num_sample[0, target[i].data] += 1
            confusion_matrix[target[i].data, pred[i].data] += 1   
              
    return num_sample, confusion_matrix
    
        
def main(time_str):
    num_ten_folds = np.zeros(shape=(1, KIND), dtype=np.int)
    pred_ten_folds = np.zeros(shape=(KIND, KIND), dtype=np.int)
    # TEST 记录txt名字
    # record_path = "/home/developers/liuchang/emo/face1/CASIA_record/{}_test".format(time_str)
    # if os.path.isdir(record_path) is False:
    #     os.mkdir(record_path)
    # record_name = os.path.basename(record_path)
    # file = open(os.path.join(record_path, record_name + ".txt"), 'w')

    for split in range(SPLIT):
        # loss_file = open(os.path.join(record_path, record_name + "_{}_loss.txt".format(split)), 'w')
        print('Split: ', split + 1,)
        best_prec1 = 0
        # TEST 模型保存名字
        path = os.path.join("/home/developers/liuchang/quantize/", time_str)
        if not os.path.exists(path):
            os.makedirs(path)
        # TEST 学习率
        lr = 0.002
        # ! 要求: 将人脸统一到 60 * 60 大小, 对人脸以0.5的概率随机翻转, 将图像调整为灰度, 使用了Viola-Jones人脸检测器
        # TEST 修改文件夹下面每个txt文件的名字
        train_dataset = dataset(annotation_file= os.path.join(txt_path, "{}_{}_fold_train.txt").format(split, SPLIT),
                                        img_dir = img_path,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((48, 48)),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            # torchvision.transforms.Grayscale(1),
                                            torchvision.transforms.ToTensor(),
                                        ])
                                        ) 
        test_dataset = dataset(annotation_file= os.path.join(txt_path, "{}_{}_fold_val.txt").format(split, SPLIT),
                                        img_dir= img_path,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((48, 48)),
                                        #    torchvision.transforms.Grayscale(1),
                                           torchvision.transforms.ToTensor(),
                                       ])
                                       )

        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    
    
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        
        calibe_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size = BATCH_SIZE * 4,
            shuffle=True
        )
        model = QuantizableResNet18(num_blocks=[2, 2, 2, 2],num_class=KIND)
        # model = torch.nn.DataParallel(model, device_ids=device_id).cuda()
        
        assert model.training
        
        BACKEND = "fbgemm"
        num = np.zeros(shape=(1, KIND), dtype=np.int)
        pred = np.zeros(shape=(KIND, KIND), dtype=np.int)
        criterion = torch.nn.CrossEntropyLoss()#.cuda()
        if mode == "qat":
            model.fuse_model()
            model.eval()
            model.qconfig = get_default_qat_qconfig(BACKEND)
            model = prepare_qat(model, inplace=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
            for epoch in range(EPOCH):
                if((epoch + 1 ) % 50 == 0):
                    lr = lr * 0.1
                # * 训练
                losses_avg = train(train_loader, model, criterion, optimizer, epoch, lr)
                # * 验证, 计算精确度
                prec1, losses, model_int8 = validate(test_loader, model, criterion, best_prec1)
                # loss_file.write("EPOCH {}: train avg loss: {:.4f}, train global loss: {:.4f}, train local loss: {:.4f},".format(
                #     epoch, losses_avg, losses_img, losses_patch
                # ))
                # loss_file.write("test loss: {:.4f}, test prec: {:.4f}\n".format(losses, prec1))
                # loss_file.flush()
                is_best = prec1 >= best_prec1
                best_prec1 = max(prec1, best_prec1)
                # * 保存最近模型(每个epoch)
                save_best_model({
                    'epoch': EPOCH,
                    'best_prec': best_prec1,
                    'state_dict':model_int8.state_dict()}, 
                is_best, path, split, time_str+"qat_quantized")
                save_best_model({
                    'epoch': EPOCH,
                    'best_prec': best_prec1,
                    'state_dict':model.state_dict()}, 
                is_best, path, split, time_str+"qat_ori")
                if is_best:
                    num, pred = confusion_matrix(test_loader, model_int8, KIND)
            print("Number of samples:")
            print(num)
            print("Confusion matrix:")
            print(pred)
            # loss_file.close()
        elif mode == "ptq":
            model.eval()
            model.qconfig = get_default_qconfig(BACKEND)
            model = prepare(model, inplace=True)
            for i, sample_batched in enumerate(calibe_loader):
                pic = sample_batched['image']
                model(pic)
            prec1, losses, model_int8 = validate(test_loader, model, criterion, best_prec1)
            save_best_model({
                'prec': prec1,
                'state_dict':model_int8.state_dict()}, 
            True, path, split, time_str+"ptq_quantized")
            save_best_model({
                'epoch': EPOCH,
                'prec': prec1,
                'state_dict':model.state_dict()}, 
            True, path, split, time_str+"ptq_ori")
            num, pred = confusion_matrix(test_loader, model_int8, KIND)
            
        sum = 0
        correct = 0
        for i in range(KIND):
            sum += num[0,i]
            for j in range(KIND):
                if i==j:
                    correct += pred[i,j]
        
        accuracy_expression = correct / sum
        # file.write("best model for split {}: {}%\n".format(split, accuracy_expression * 100))
        # file.write("Number of samples: {}, confusion matrix: \n {} \n".format(num, pred))
        # file.flush()
        # * 记录判断到的和总张数
        num_ten_folds += num
        pred_ten_folds += pred
        

    sum = 0
    correct = 0
    for i in range(KIND):
        sum += num_ten_folds[0,i]
        for j in range(KIND):
            if i==j:
                correct += pred_ten_folds[i,j]
    # * 计算比率
    final_accuracy_expression = correct/sum
    # * 最后将精确度print出来
    print("Final accuracy of expression is {}".format(final_accuracy_expression))
    print("Number of samples:")
    print(num_ten_folds)
    print("Confusion matrix:")
    print(pred_ten_folds)
    
    # file.write("Final accuracy of expression: {}, Number of samples: {}, Confusion matrix: \n{} \n".format(
    #     final_accuracy_expression, num_ten_folds, pred_ten_folds
    # ))
    # file.close()
    
if __name__ == '__main__':
    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]")
    main(time_str)
