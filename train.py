import torch
import torch.nn as nn

from ETDNN import ECAPA_TDNN
# from xvector import XVector

import matplotlib.pyplot as plt
import pandas as pd
from mask_aug import MaskAug
import time, os

def train_network(model, epoch, optimizer, scheduler, loader, loss_fn, device, is_augment):
    model.to(device)
    model.train()
    augment = MaskAug()

    loss_fn = loss_fn.to(device)
    loss, correct, total = 0, 0, len(loader.dataset)
    for i, (data, labels) in enumerate(loader, start=1):
        data, labels = data.to(device), torch.LongTensor(labels).to(device)
        if is_augment:
            data, _ = MaskAug().masked(data)
        optimizer.zero_grad()
        output = model(data)

        nloss = loss_fn.forward(output, labels)

        cor =  (output.argmax(dim=1) == labels).sum()
        correct += cor

        nloss.backward()
        optimizer.step()
        loss += nloss

    scheduler.step()
    return loss/len(loader), correct/total


def eval_network(model, loader, loss_fn, device, epoch, seg_save_path):
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    model.eval()
    
    correct, total, loss = 0, len(loader.dataset), 0
    perdicts = []
    for i, (data, labels) in enumerate(loader, start=1):
        data, labels = data.to(device), torch.LongTensor(labels).to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            nloss = loss_fn.forward(output, labels)
            loss += nloss

        correct += (pred==labels).sum()
        perdicts.extend(pred.cpu().numpy().tolist())
    save_segment_prediction(perdicts, epoch, seg_save_path)
    return  loss/len(loader), correct/total


def save_result(epoch, train_loss, train_acc, test_loss, test_acc, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fname = os.path.join(save_path, "loss.txt")
    with open(fname, "a") as f:
        data = [epoch, train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item()]
        data = f"{epoch}\t{train_loss.item()}\t{train_acc.item()}\t{test_loss.item()}\t{test_acc.item()}\n"
        f.write(data)

def save_segment_prediction(audio_pred, epoch, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fname = os.path.join(save_path, f"pred.txt")
    with open(fname, "a") as f:
        f.write(str(audio_pred)+"\n")

def save_embeddings(embeddings, epoch, type="train"):
    save_path = "/path/to/"
    if (epoch+1)%50==0 and epoch>0:
        if type == "train":
            save_path = os.path.join(save_path, f"train_{epoch}.csv")
        else:
            save_path = os.path.join(save_path, f"test_{epoch}.csv")

        embeddings = pd.DataFrame(embeddings)
        embeddings.to_csv(save_path, index=False)

def save_model(epoch, model, path="./default_checkpoint.pkl"):
    torch.save(model.state_dict(), path)
    print(f"save model at epoch {epoch+1} to path {path} success!")


def train(model, epoch, train_loader, test_loader, optimizer, scheduler, loss_fn, save_path, device, is_augment, ctime, save_model=False):    
    print("save result at", save_path)
    
    for e in range(epoch):
        train_loss, train_accuracy = train_network(model, e, optimizer, scheduler, train_loader, loss_fn, device, is_augment)
        test_loss, test_accuracy = eval_network(model, test_loader, loss_fn, device, e, save_path)

        # utt_acc = calc_uttarance()
        print(time.strftime("%m-%d %H:%M:%S")+ 
              " The epoch {:0>3d} train loss:{:.3f} train accuracy:{:.2f}, test loss:{:.2f} test accuracy:{:.3f}"
              .format(e, train_loss, train_accuracy, test_loss, test_accuracy))
        
        save_result(e, train_loss, train_accuracy, test_loss, test_accuracy, save_path)

        # 相隔十个epoch打印空格
        if e>0 and e%10==0:
            print("")
    final_loss_plot(save_path)


def final_loss_plot(save_path):
    loss_path = os.path.join(save_path, "loss.txt")
    data = pd.read_table(loss_path, header=None)
    epoch, train_loss, train_acc, test_loss, test_acc = \
        data[0], data[1], data[2], data[3], data[4]
    
    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    save_path = loss_path[:-8]
    plt.plot(epoch, train_acc, label="train")
    plt.plot(epoch, test_acc, label="test")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(222)
    plt.plot(epoch, train_loss, label="train")
    plt.plot(epoch, test_loss, label="test")
    plt.title("Loss")
    plt.legend()
    plt.savefig(save_path+"loss.png", dpi=300)
