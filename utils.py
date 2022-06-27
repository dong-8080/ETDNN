import torch
import torch.nn as nn
from resnet import resnet18, BasicBlock, Bottleneck
from ETDNN import ECAPA_TDNN
import numpy as np
import pandas as pd
import os
from loader import TrainSet, TestSet, DataLoader
from scheduler import CyclicCosineDecayLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def weight_init_tdnn(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weight_init_resnet(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
            
    # zero init the last bn in each residual branch
    # thie improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if isinstance(m, Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)
    elif isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)

def weight_init(model):
    model_name = model._get_name()
    if model_name=="ResNet":
        model.apply(weight_init_resnet)
    elif model_name=="ECAPA_TDNN":
        model.apply(weight_init_tdnn)

def plot_lr():
    model = resnet18()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CyclicCosineDecayLR(optimizer, init_decay_epochs=180, min_decay_lr=0.00001, \
        warmup_epochs=10, warmup_start_lr=0.00001)
    lr_plot = []
    for i in range(200):
        lr_plot.extend(scheduler.get_lr())
        optimizer.step()
        scheduler.step()
    
    plt.plot(lr_plot)
    plt.savefig("./learning_rate",dpi=900)

def utt_calc(ctime:str):
    filepath = "/path/to/result/2022/"
    kfold_acc = []
    for i in range(5):
        fname = os.path.join(filepath, f"{ctime}_{i}", "pred.txt")
        fname = f"/path/to/result/2022/{ctime}_{i}/pred.txt"
        id_ = pd.read_csv("/path/to/testset1.csv")["id"]
        with open(fname, "r") as f:
            data = f.readlines()
        data = [eval(i) for i in data]
        # transform to df
        data = pd.DataFrame(data)
        data = data.T
        data["id"] = id_
        result = data.groupby("id").agg(lambda x: x.value_counts().index[0]).reset_index()
        result["label"] = [int(i[1])-1 for i in result['id']]

        acc = []
        for i in range(result.shape[1]-2):
            cor = (result[i]==result['label']).sum()
            acc.append(cor/result.shape[0])
        kfold_acc.append(acc)
    pd.DataFrame(kfold_acc).T.to_csv(fname.replace(".txt",".csv"), index=False)
    print("save result at", fname.replace(".txt",".csv"))

def metrics_groups(ctime):
    # speaker-level
    data_all = pd.DataFrame(columns=["pred", "label"])
    for i in range(5):
        fpath = f"/path/to/result/2022/{ctime}_{i}/pred.txt"
        if not os.path.exists(fpath):
            utt_calc(ctime)
        label_path = f"/path/to/testset{i}.csv"
        with open(fpath, "r") as f:
            data = f.readlines()
        data = pd.DataFrame(eval(data[-1]), columns=["pred"])
        labels = pd.read_csv(label_path)
        data['label'] = labels['id'].apply(lambda x:int(x[1])-1)
        data['id'] = labels["id"]
        
        data_group = data.groupby("id").mean().round()
        data_all = data_all.append(data_group)
    speaker_labels, speaker_preds = data_all["label"], data_all["pred"]
    
    metrics = {
        "accuracy": accuracy_score(speaker_labels, speaker_preds),
        "precision": precision_score(speaker_labels, speaker_preds),
        "recall": recall_score(speaker_labels, speaker_preds),
        "f1_score": f1_score(speaker_labels, speaker_preds),
        "confusion_matrix": confusion_matrix(speaker_labels, speaker_preds)
    }
    return metrics

def metrics_uttrance(ctime):
    # uttrance-level
    metrics = []
    for i in range(5):
        fpath = f"/path/to/result/2022/{ctime}_{i}/pred.txt"
        label_path =  f"//path/to/testset{i}.csv"
        with open(fpath, "r") as f:
            result_end = f.readlines()[-1]
        
        utterance_preds = eval(result_end)
        
        label_data = pd.read_csv(label_path)
        # id的第二位减1即标签，快速计算
        utterance_labels = [int(i[1])-1 for i in label_data["id"]]
        
        metrics.append([
            accuracy_score(utterance_labels, utterance_preds),
            precision_score(utterance_labels, utterance_preds),
            recall_score(utterance_labels, utterance_preds),
            f1_score(utterance_labels, utterance_preds)
        ])
    mean_metrics = np.array(metrics).mean(axis=0)
    
    mean_metrics = dict(zip(["accuracy","precision","recall","f1 score"], mean_metrics))
    return mean_metrics


    
    
if __name__=="__main__":
    ctimes = ["0323_2212", "0323_2211", "0323_2209", "0322_1447", "0322_1009"]

    utt_metrics = pd.DataFrame([metrics_uttrance(i) for i in ctimes])
    print("The result in utterance level")
    print(utt_metrics)
    print(utt_metrics.describe())
    
    print("\n\n")
    groups_metrics = pd.DataFrame([metrics_groups(i) for i in ctimes])
    print("The result in speaker level")
    print(groups_metrics)
    print(groups_metrics.describe())