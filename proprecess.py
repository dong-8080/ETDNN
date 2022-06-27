import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa

import pandas as pd
import numpy as np

from tqdm import tqdm

import os


def mfcc_window_extract(wav_path):
    wav_form, sr = torchaudio.load(wav_path)

    mfcc_transform = T.MFCC(sample_rate = sr, n_mfcc=80, 
        melkwargs={
            'n_fft': 2048,
            'n_mels': 256,
            'hop_length': 512,
            'mel_scale': 'htk'
        })

    for index, start in enumerate(range(0, wav_form.shape[1], int(1.5*sr)), start=1):
        end = start + 3*sr

        if end > wav_form.shape[1]:
            break
        mfcc = mfcc_transform(wav_form[0,start:end])
        save_pt_name = os.path.basename(wav_path).split(".")[0]+f"_{index:>02}.pt"
        torch.save(mfcc, os.path.join(os.path.dirname(wav_path), save_pt_name))

if __name__ == '__main__':
    for root, dirs, files in tqdm(os.walk("/path/to/vad", topdown=False)):
        for name in files:
            wav_path = os.path.join(root, name)
            mfcc_window_extract(wav_path)
