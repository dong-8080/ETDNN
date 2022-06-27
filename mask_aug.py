    import torch
import torch.nn as nn

import librosa
import librosa.display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class MaskAug():
    def __init__(self, freq_mask_width = (0,8), time_mask_width=(0,10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = time_mask_width
        
    def mask_along_axis(self, x, dim):
        origin_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        elif dim == 2:
            D = time
            width_range = self.time_mask_width
            
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D-mask_len.max()), (batch, 1), device=x.device).unsqueeze(1)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos<= arange) * (arange<(mask_pos+mask_len))
        mask = mask.any(dim=1)
        
        if dim==1:
            mask = mask.unsqueeze(2)
        elif dim==2:
            mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        mask = torch.zeros_like(x).masked_fill_(mask, 1)
        return x.view(*origin_size), mask
        
    
    def masked(self, x):
        x, mask1 = self.mask_along_axis(x, dim=2)
        x, mask2 = self.mask_along_axis(x, dim=1)
        return x, torch.max(mask1, mask2)

if __name__ == '__main__':
    # plot mask data
    data = torch.load("/path/to/00_01.pt")
    aug_data, mask = MaskAug().masked(data.unsqueeze(dim=0))
    data = aug_data[0].numpy()
    mask = mask[0].numpy()
    data_masked = np.ma.masked_where(mask==1, data)

    librosa.display.specshow(data_masked)
    plt.colorbar()
    plt.title("mask mfcc")
    plt.savefig("./masked.png", dpi=900)
