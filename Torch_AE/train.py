import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from torchvision.transforms.transforms import Normalize, ToTensor
import matplotlib.pyplot as plt

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
                               train=False, 
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
    model.to(device)
    model.train()
    
    train_loss = 0.0
    for data in tqdm(train_dataloader):
        # get data
        inputs, _ = data
        inputs = inputs.to(device)
        
        # forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_ = loss(outputs, inputs)
        loss_.backward()
        optimizer.step()
        
        # calculate loss
        train_loss += loss_.item()*inputs.size(0)
        wandb.log({"batch_loss": loss_.item()})
    
    train_loss = train_loss / len(train_dataloader)
    wandb.log({"train_loss": train_loss})
    
    return train_loss, model


# run validation after model epoch
def validate(model, dataloader, loss, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    for data in tqdm(dataloader):
        # get data
        inputs, _ = data
        inputs = inputs.to(device)
        
        # forward
        outputs = model(inputs)
        loss_ = loss(outputs, inputs)
        
        # calculate loss
        test_loss += loss_.item()*inputs.size(0)
        
    test_loss = test_loss / len(dataloader)
    wandb.log({"test_loss": test_loss})
    
    return test_loss


# main train loop
def train(model, train_dataloader, test_dataloader, device):
    # set up tracking
    wandb.init(project='cifar10_autoencoder', entity='sumrao')
    wandb.watch(model)
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
        print(f'Epoch: {i+1}/{config.EPOCHS} \t Train Loss: {train_loss_epoch:.6f}')
        print(f'Epoch: {i+1}/{config.EPOCHS} \t Test Loss: {test_loss_epoch:.6f}')
        
        # save model
        if (i+1)%5 == 0:
            save_model(model, i)        
    
    loss_dict = {'train_loss': train_loss, 'test_loss': test_loss}
    return loss_dict, model


# save model checkpoint
def save_model(model, epoch):
    output_dir = './model_checkpoint/'
    model_name = "autoencoder" + "_" + str(epoch) + ".pth"
    torch.save(model, output_dir + model_name)
    

# load model
def load_model():
    model = torch.load('./model_checkpoint/autoencoder_100.pth')
    return model


# run inference from model
def run_inference(test_dataloader, device):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
    
    model = load_model()
    model.to(device)
    
    dataiter = iter(test_dataloader)
    images, labels = dataiter.next()
    images = images.to(device)
    
    output = model(images)
    
    output = output.cpu().detach().numpy()
    images = images.cpu().detach().numpy()
    
    
    
    # plot original images
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,
                                 figsize=(12,10))
    for idx in range(5):
        ax = fig.add_subplot(2,5, idx+1, xticks=[], yticks=[])
        img = images[idx,:,:,:].squeeze()
        img = np.transpose(img, (1, 2, 0))
        # img = (img * 255).astype(np.uint8)
        plt.imshow(img)
        ax.set_title(classes[labels[idx]])
        
        ax = fig.add_subplot(2,5, idx+5+1, xticks=[], yticks=[])
        rst = output[idx,:,:,:].squeeze()
        rst = np.transpose(rst, (1, 2, 0))
        # rst = (rst * 255).astype(np.uint8)
        plt.imshow(rst)
        
        ax.set_title(classes[labels[idx]])
    plt.savefig("Results.png")
  
    
# save traning loss
def save_loss(loss_dict):
    ldf = pd.DataFrame.from_dict(loss_dict)
    plt.plot(ldf['train_loss'], label='train_loss')
    plt.plot(ldf['test_loss'], label='test_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.savefig('loss.png')
    

# main function
def main():
    # get dataloaders
    train_dataloader, test_dataloader = get_cifar_dataloaders()
    
    # get models
    model = AutoEncoder()
    
    # get device
    device = get_device()
    
    # train model
    loss_dict, model = train(model, 
                             train_dataloader, 
                             test_dataloader, 
                             device)
    
    # save model checkpoint
    save_model(model, 100)
    
    # save training loss
    save_loss(loss_dict)
    
    # generate some images
    run_inference(test_dataloader, device)
    

if __name__ == "__main__":
    main()
