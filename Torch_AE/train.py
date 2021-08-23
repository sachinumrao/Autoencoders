import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
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
def train_one_epoch(model, train_dataloader, loss, optimizer, device):
    
    pass

# run validation after model epoch
def validate(model, dataloader):
    pass

# main train loop
def train(model, train_dataloader, test_dataloader, device):
    train_loss = []
    test_loss = []
    # define loss function
    loss = nn.MSELoss()
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    for i in range(config.EPOCHS):
        # train model for a epoch
        train_loss_epoch, model = train_one_epoch(
            model, train_dataloader, loss, optimizer, device)
        
        # validate model
        test_loss_epoch = validate(model, test_dataloader, loss, device)
        
        # save loss
        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)
        
        # print loss information
        print(f'Epoch: {i+1}/{config.EPOCHS} \t Train Loss: {train_loss_epoch}')
        print(f'Epoch: {i+1}/{config.EPOCHS} \t Test Loss: {test_loss_epoch}')
        
        # save model
        if (i+1)%5 == 0:
            save_model(model, i)        
    
    loss_dict = {'train_loss': train_loss, 'test_loss': test_loss}
    return loss_dict, model

# save model checkpoint
def save_model(model, epoch):
    output_dir = './model_checkpoint/'
    model_name = "autoencoder" + "_" + str(epoch) + ".pth"
    model.save(output_dir + model_name)
    

# load model
def load_model():
    model = torch.load('./model_checkpoint/autoencoder_100.pth')
    return model

# run inference from model
def run_inference(test_dataloader, device):
    model = load_model()
    
# save traning loss
def save_loss(loss_dict):
    ldf = pd.DataFrame.from_dict(loss_dict)
    mpl.plt.plot(ldf['train_loss'], label='train_loss')
    mpl.plt.plot(ldf['test_loss'], label='test_loss')
    mpl.plt.legend()
    mpl.plt.xlabel('Epoch')
    mpl.plt.ylabel('Loss')
    mpl.plt.title('Loss vs Epoch')
    mpl.plt.savefig('loss.png')
    

# main function
def main():
    # get dataloaders
    train_dataloader, test_dataloader = get_cifar_dataloaders()
    
    # get models
    model = AutoEncoder()
    
    # get device
    device = get_device()
    
    # train model
    loss_dict, model = train(model, train_dataloader, test_dataloader, device)
    
    # save model checkpoint
    save_model(model, 100)
    
    # save training loss
    save_loss(loss_dict)
    
    # generate some images
    run_inference(test_dataloader, device)
    

if __name__ == "__main__":
    main()
