import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def data_loading(roots,datasets,batch_size):
       # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MNIST Dataset  下载训练集 MNIST 手写数字训练集
        train_dataset = getattr(dsets,datasets)(root=roots,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
                                    # 转化为取值范围是[0,1.0]的torch.FloadTensor
                                    download=True)  # 下载数据,download首次为True

        test_dataset = getattr(dsets,datasets)(root=roots,
                                   train=False,  # 测试集
                                   transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        # 数据的批处理，尺寸大小为batch_size,
        # 在训练集中，shuffle 必须设置为True, 表示次序是随机的
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        #  CIFAR10data数据集  下载训练集  CIFAR10data数据集训练集
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform1 = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return train_loader,test_loader