import torch
import torch.nn as nn
from torchsummary import summary


# custom layer for reshape
class Reshape(nn.Module):
    """
    Class to reshape the tensor
    """
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    

class AutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for cifar10 dataset
    """
    def __init__(self, latent_dim=32):
        super(AutoEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # encoder block
        self.encoder = nn.Sequential(
            # conv blocks
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # project data to latent space
            nn.Flatten(),
            nn.Linear(64*4*4, self.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            # transform latent space data back to 3 channel
            nn.Linear(self.latent_dim, 64*4*4),
            Reshape(-1, 64, 4, 4),
            
            # transpose conv blocks
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32, 8, 2, stride=2),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        # encoder block
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    
if __name__ == '__main__':
    model = AutoEncoder()
    # print(model)
    print(summary(model, input_size=(3, 32, 32)))
    