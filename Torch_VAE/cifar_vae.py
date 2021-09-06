import torch
import torch.nn as nn
from torchsummary import summary

class VariationalAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for cifar10 dataset
    """
    def __init__(self, latent_dim, batch_size):
        super(VariationalAutoEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # define encoder blocks
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(64*4*4, 128)
        
        self.fc_mu = nn.Linear(128, 64)
        self.fc_logvar = nn.Linear(128, 64)
        
        
        # define decoder blocks
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_relu1 = nn.ReLU()
        self.t_conv2 = nn.ConvTranspose2d(32, 8, 2, stride=2)
        self.t_relu2 = nn.ReLU()
        self.t_conv3 = nn.ConvTranspose2d(8, 3, 2, stride=2)
        self.t_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # encoder block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        
        # decoder block
        out = self.t_conv1(out)
        out = self.t_relu1(out)
        
        out = self.t_conv2(out)
        out = self.t_relu2(out)
        
        out = self.t_conv3(out)
        out = self.t_sigmoid(out)
        
        return out
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    
if __name__ == '__main__':
    model = VariationalAutoEncoder()
    print(summary(model, input_size=(3, 32, 32)))
    