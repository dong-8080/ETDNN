import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def kflod_plot(filename=None):
    file_path = "/path/to/result/2022/"

    if filename is None:
        files = [i for i in os.listdir(file_path) if i.endswith(".txt")]
        files.sort()
        files = files[-5:]
    else:
        files = [f"{filename}_{i}_loss.txt" for i in range(5)]
        print(files)
    # 创建一个全零的DF，与所有文件相加
    data_fold = pd.DataFrame(np.zeros((200, 5)))
    for i in files:
        data = pd.read_table(file_path+i, header=None)
        data_fold += data
    data_fold = data_fold/5
    
    epoch, train_loss, train_acc, test_loss, test_acc = \
        data_fold[0], data_fold[1], data_fold[2], data_fold[3], data_fold[4]
    
    # 插子图，两图
    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    save_path = file_path+files[0][:9]
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
    
    data_fold.to_csv(save_path+"loss.txt", index=False)

if __name__=="__main__":
    kflod_plot("0221_1941")