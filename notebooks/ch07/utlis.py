import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import pandas as pd

def load_data_fashion_mnist(batch_size, resize=None, pre_trans : list=None):
    
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    data_transform = transforms.Compose(transform)
    # Fashion MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='../data/',
                                               train=True, 
                                               transform=data_transform,
                                               download=True)
    
    test_dataset = torchvision.datasets.FashionMNIST(root='../data/',
                                              train=False, 
                                              transform=data_transform,
                                              download=True)
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader

def load_data_cifar10(batch_size, resize=None, pre_trans : list=None):
    
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    data_transform = transforms.Compose(transform)
    # Fashion MNIST dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                               train=True, 
                                               transform=data_transform,
                                               download=True);
    
    test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                              train=False, 
                                              transform=data_transform,
                                              download=True);
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader

def write2csv(filename : str, 
              data : list, 
              name : str):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame()
        
    df[name] = data
    df.to_csv(filename, index=False)
    # print('write '+ name + ' csv done!')
    


def draw_figures(pth:str, 
                 label_list:list, 
                 title:str = None):
    d = pd.read_csv(pth)
    legend_list = []
    for x, y in label_list:
        plt.plot(d[x], d[y])
        legend_list.append(y)
    plt.xlabel('epoch')
    
    plt.legend(legend_list)
    plt.grid()
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(arrowstyle = "->", connectionstyle = "angle,angleA=0,angleB=90,rad=10")
    bias = 0
    if title:
        plt.title(title)
        if 'acc' in title:
            plt.ylabel('accuracy')
            bias = 0
            for x, y in label_list:
                max_index = d[y].idxmax()
                plt.annotate(('max acc of {} = {:.4f}'.format(y, d[y][max_index])),
                             (d[x][max_index], d[y][max_index]), xytext=(0, -50 - bias), textcoords='offset points',
                             bbox=bbox, arrowprops=arrowprops)
                bias += 30
        elif 'loss' in title:
            plt.ylabel('loss')
            bias = 0
            for x, y in label_list:
                min_index = d[y].idxmin()
                plt.annotate( ('min loss of {} = {:.4f}'.format(y, d[y][min_index])),
                             (d[x][min_index], d[y][min_index]), xytext=(0, 50 + bias), textcoords='offset points',
                             bbox=bbox, arrowprops=arrowprops)
                bias += 30
    plt.show()