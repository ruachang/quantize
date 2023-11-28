'''
demo resnet 的读取数据的函数
'''
import torch 
from torch import nn, no_grad
import os 
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import cv2 

class dataset(Dataset):
    def __init__(self, annotation_file, img_dir, test = False, transform=None, target_transform=None):
        # super(EyeData).__init__()
        self.img_labels = pd.read_csv(annotation_file, delimiter=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.test= test
    def __len__(self):
        return(len(self.img_labels))

    def __getitem__(self, index):
        # 此处默认第一列为名字, 第二列为标签
        img_path = self.img_labels.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_path)
        # TEST 打印读取图片
        # print(img_path)
        img_tmp = cv2.imread(img_path)
        img_tmp = Image.open(img_path)
        # img_tmp = cv2.resize(img_tmp, (48, 48))
        # img_tmp = img_tmp.transpose((2, 0, 1))
        if self.transform:
            img_tmp = self.transform(img_tmp)
            # img_tmp = img_tmp.to(torch.float32)
            
        image_label = int(self.img_labels.iloc[index, 1]) - 1
        if self.target_transform:
            image_label = self.target_transform(image_label)
        img = np.array(img_tmp, dtype='float32')
        # print(img_seq.shape)
        sample = {"image": img,"labels": image_label}

        return sample       
