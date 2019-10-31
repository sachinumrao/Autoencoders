import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt

# convert data to Tensor
transform = transforms.ToTensor()

# load the MNIST data
print("Downloading MNIST Dataset...")
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
val_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

batch_size = 32
num_workers = 4
train_loader = torch.utils.data.DataLoader(train_data, 
                    batch_size=batch_size, 
                    num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(val_data, 
                batch_size=batch_size, 
                num_workers=num_workers)

# define the model architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
       
       # encoder block
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        #self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        
        # decoder block
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(4, 16, 3, padding=1),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 1, 3, padding=1),
            nn.ReLU(True)     
            
        )


    def forward(self, x):
        
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# Create Model 
model = Autoencoder()
model.cuda()

print("Model Summary")
print(summary(model, (1,28,28)))

# training params
n_epochs = 10

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

print("Setting Up Training Loop...")

# training loop
loss_vals = []
for epoch in range(n_epochs):
    loss = None
    for data in tqdm(train_loader, desc="Epoch: "+str(epoch+1)+"/"+str(n_epochs)+" :"):
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
plt.title("Convolutional Autoencoder on MNIST Data")
plt.show()

# Tests on validation data
# obtain one batch of test images
dataiter = iter(val_loader)
images, _ = dataiter.next()
img = images.cuda()


output = model(img)
images = images.numpy()

output = output.cpu().view(batch_size, 1, 28, 28)
output = output.detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)