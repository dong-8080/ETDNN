import argparse
import torch
import torch.nn as nn
from ETDNN import ECAPA_TDNN
from loader import TrainSet, TestSet, DataLoader
from scheduler import CyclicCosineDecayLR
import utils

from train import train 
from utils import weight_init
from resnet import resnet18, resnet34, resnet50, resnet101
from SeResNet import se_resnet50, se_resnext50
from mask_aug import MaskAug
import time, os

def parse_args():
    parser = argparse.ArgumentParser(description="Model trainer")

    parser.add_argument("-m", "--model", type=str, default="ECAPA_TDNN")
    parser.add_argument("-gpu", "--gpu", type=str, default="cuda:1")
    parser.add_argument("-lr", "--lr" , type=float, default=0.0001)
    parser.add_argument("-s", "--save", type=bool, default=False, help="save model")
    parser.add_argument("-b", "--batchsize", type=int, default=512)
    parser.add_argument("-a", "--augment", type=bool, default=True)
    parser.add_argument("--max_epoch", default=200)
    parser.add_argument("--warmup_lr", default=0.00001)
    parser.add_argument("--warmup_epoch", default=20)
    parser.add_argument("--decay_lr", default=80)

    return parser.parse_args()

if __name__=="__main__":
    ## Training Settings
    args = parse_args()

    # choose the model
    model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192, class_num=2)

    # set learning rate
    print("learning rate: ", args.lr)

    # set batch size
    print("batch size: ", args.batchsize)

    # set gpu
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    if args.save:
        print("save model: ", "True")
    
    if args.augment:
        print("augment data: ", "True")


    ctime = time.strftime("%m%d_%H%M", time.localtime())
    augment = MaskAug()

    # 五折交叉验证
    for i in range(0,5):
        # initilization
        weight_init(model)

        save_path = os.path.join("/path/to/result/2022/", ctime+"_"+str(i))

        train_set = TrainSet(f"/path/to/trainset{i}.csv")
        test_set = TestSet(f"/path/to/testset{i}.csv")

        train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CyclicCosineDecayLR(optimizer, init_decay_epochs=180, min_decay_lr=0.00001, \
            warmup_epochs=20, warmup_start_lr=0.00001)

        # 损失函数    
        loss_fn = nn.CrossEntropyLoss()

        train(model, 200, train_loader, test_loader, optimizer, scheduler, loss_fn, save_path, device, ctime, args.augment)
    utils.utt_calc(ctime)

    print("train finished! The model parameters are as follows:")
    print(ctime)
    print("model: ", args.model)
    print("learning rate: ", args.lr)
    print("batch size: ", args.batchsize)
    print("gpu state: ", args.gpu)
    print("augment data: ", args.augment)
