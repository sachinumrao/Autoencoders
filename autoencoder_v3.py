# import depedencies
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

# load the cifar-10 data
transform = transforms.Compose([transforms.ToTensor(),  
                               transforms.Normalize((0.4914, 0.4822, 0.4466), 
                                                    (0.247, 0.243, 0.261))])

trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), 
                                        tv.transforms.Normalize((0.4914, 0.4822, 0.4466), 
                                                                (0.247, 0.243, 0.261))])

print("Downloading CIFAR-10 Dataset...")

trainset = tv.datasets.CIFAR10(root='./data',  train=True,download=True, 
                               transform=transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, 
                                         num_workers=4)

testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, 
                              transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
           'horse', 'ship', 'truck')

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, 
                                         num_workers=2)

# model architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        # design of encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
            
        # design of decoder part
        self.decoder = nn.Sequential(   
            nn.Upsample(scale_factor=2, mode='nearest'),          
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.ConvTranspose2d(16, 3, 3, padding=1),
            nn.ReLU(True)
        )
            


    def forward(self,x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# Create Model 
model = Autoencoder()
model.cuda()

print("Model Summary")
print(summary(model, (3,32,32)))

# training params
n_epochs = 20

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

print("Setting Up Training Loop...")

# training loop
loss_vals = []
for epoch in range(n_epochs):
    loss = None
    for data in tqdm(dataloader, desc="Epoch: "+str(epoch+1)+"/"+str(n_epochs)+" :"):
        img, _ = data

        # forward pass
        img = img.cuda()
        output = model(img)
        loss = criterion(output, img)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, loss.cpu().item()))
    loss_vals.append(loss.cpu().item())

# Plot loss function values over epochs
plt.plot(loss_vals)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Convolutional Autoencoder on CIFAR-10 Data")
plt.show()