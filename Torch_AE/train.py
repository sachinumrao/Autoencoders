import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
from torchvision.transforms.transforms import Normalize, ToTensor
import wandb as wb
import matplotlib as mpl

import config
from cifar_autoencoder import AutoEncoder

# get dataloaders
def get_cifar_dataloaders():
    """
    Get train and test dataloaders for CIFAR-10 dataset.
    """
    transform = transforms.Compose(
        [ToTensor(),
         Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261)),])
    
    trainset = datasets.CIFAR10(root='~/Data/torchvision/', 
                                train=True, 
                                download=True, 
                                transform=transform)
    
    testset = datasets.CIFAR10(root='~/Data/torchvision/', 
                               train=True, 
                               download=True, 
                               transform=transform)
    
    train_dataloader = DataLoader(trainset,
                                  batch_size=config.BS,
                                  shuffle=False,
                                  num_workers=2)
    
    test_dataloader = DataLoader(testset,
                                 batch_size=config.BS,
                                 shuffle=False,
                                 num_workers=2)
    
    return train_dataloader, test_dataloader


def get_device():
    """
    Get device to train on.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    return device

    
# train for one epoch
def train_one_epoch():
    pass

# run validation after model epoch
def validate(model, dataloader):
    pass

# main train loop
def train():
    pass

# save model checkpoint
def save_model(model, epoch):
    pass

# load model
def load_model():
    pass

# run inference from model

# main function
def main():
    pass

if __name__ == "__main__":
    main()
