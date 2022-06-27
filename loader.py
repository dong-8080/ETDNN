import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchaudio.transforms as T
import librosa
from tqdm import tqdm
import time

import numpy as np
import pandas as pd

from mask_aug import MaskAug
from SeResNet import se_resnet50

SAMPLE_RATE = 16000
N_MFCC=80

CROP_LENGTH = 60

class TrainSet(Dataset):

    def __init__(self, index_path, type="BIN"):
        file_index = pd.read_csv(index_path)
        self.type = type
        # 三分类，做标签平衡
        self.filelist = file_index["path"].to_list()
        self.labellist = file_index["depression"].to_list()


    def __getitem__(self, index):
        mfcc_path = self.filelist[index]
        mfcc = torch.load(mfcc_path)
        label = self.labellist[index]
        if self.type == "BIN":
            if np.isnan(label):
                label = 0
            else:
                label = 1
        elif self.type == "MLT":
            label = int(label-1)
        return mfcc, label

    def __len__(self):
        return len(self.filelist)

    def downsample(self, data):
        data_1 = data[data["depression"]==1]
        data_2 = data[data["depression"]==2]
        data_3 = data[data["depression"]==3]
        data_2 = data_2.sample(frac=0.25)
        df_concat = pd.concat([data_1, data_2, data_3], axis=0)
        print(df_concat.shape)
        return df_concat

class TestSet(Dataset):

    def __init__(self, index_path, type="BIN"):
        file_index = pd.read_csv(index_path)
        self.type = type
        self.filelist = file_index["path"].to_list()
        self.labellist = file_index["depression"].to_list()


    def __getitem__(self, index):
        mfcc_path = self.filelist[index]
        mfcc = torch.load(mfcc_path)
        label = self.labellist[index]

        if self.type == "BIN":
            if np.isnan(label):
                label = 0
            else:
                label = 1
        elif self.type == "MLT":
            # 从零开始的异世界
            label = int(label-1)
        return mfcc, label

    def __len__(self):
        return len(self.filelist)

def collate_func(batch_data):
    length = [len(n[0]) for n in batch_data]
    mfcc_raw = [torch.Tensor(n[0]) for n in batch_data]
    # 补充的元素为-1或0有待实验验证
    mfcc_pad = pad_sequence(mfcc_raw, batch_first=True, padding_value=-1)

    label = [i[1] for i in batch_data]

    return mfcc_pad, label, length

