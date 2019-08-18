# import depedencies
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# load the cifar-10 data
transform = transforms.Compose([transforms.ToTensor(),  
                               transforms.Normalize((0.4914, 0.4822, 0.4466), 
                                                    (0.247, 0.243, 0.261))])

trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), 
                                        tv.transforms.Normalize((0.4914, 0.4822, 0.4466), 
                                                                (0.247, 0.243, 0.261))])

trainset = tv.datasets.CIFAR10(root='./data',  train=True,download=True, 
                               transform=transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, 
                                         num_workers=4)

testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, 
                              transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
           'horse', 'ship', 'truck')

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, 
                                         num_workers=2)

# model architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        # design of encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=3),
            nn.ReLU(True))

        # design of decoder part
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=3),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# training params
n_epochs = 10

# loss function and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

# training loop
for epoch in range(n_epochs):
    for data in dataloader:
        img, _ = data

        # forward pass
        output = model(img)
        loss = criterion(output, img)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, loss.item()))