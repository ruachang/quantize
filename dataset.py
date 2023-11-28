'''
mfcnn 的载入数据集函数
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

class TenFoldsDataset(Dataset):
    def __init__(self, txt_root, img_dir, transform=None):
        # super(EyeData).__init__()
        self.img_labels = pd.read_csv(txt_root, delimiter=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return(len(self.img_labels))

    def __getitem__(self, index):
        # 此处默认第一列为名字, 第二列为标签
        img_path_body = self.img_labels.iloc[index, 0]
        # img_path_head = img_path_body.split('_')
        pic_path = os.path.join(self.img_dir, img_path_body)

        # [拼出文件夹的名字]
        img_lst = np.zeros((1, 60, 60))    
        # img_tmp = cv2.imread(pic_path)
        img_tmp = Image.open(pic_path)
        # img_tmp = cv2.resize(img_tmp, (60, 60))
        # img_tmp = img_tmp.transpose((2, 0, 1))
        if self.transform:
            img_tmp = self.transform(img_tmp)
            # img_tmp = img_tmp.to(torch.float32)
        img_lst = img_tmp
        image_label = int(self.img_labels.iloc[index, 1]) - 1
        imgs = np.array(img_lst, dtype='float32')
        # print(img_seq.shape)
        sample = {"image": imgs,"labels": image_label}

        return sample        
