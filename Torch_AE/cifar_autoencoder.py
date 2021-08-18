import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for cifar10 dataset
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # define encoder blocks
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        
        # define decoder blocks
        pass
    
    def forward(self, x):
        pass

    
    