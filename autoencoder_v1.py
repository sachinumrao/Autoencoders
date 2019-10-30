# import dependencies
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.backends import cudnn

# cuda optimizations
#cudnn.benchmark = True


# normalize data
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,))
                               ])

# load data
print("Downloading data...\n")

trainset = datasets.MNIST('./data', download=True, train=True, 
                          transform=transform)

valset = datasets.MNIST('./data', download=True, train=False, 
                          transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                          shuffle=True)

# model params
input_size = 784
hidden_size = [512, 256, 128, 300, 600]
op_size = 784

# model architecture
model = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[0], hidden_size[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[1], hidden_size[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[2], hidden_size[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[3], hidden_size[4]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[4], op_size))

# copy the model in gpu
model.cuda()

print("Model Summary \n")
print(model)

# use MSE loss 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

time_0 = time()
n_epochs = 20

print("Starting training loop...")
# train loop
loss_vals = []
for e in range(n_epochs):
    
    loss = None
    for images, labels in tqdm(trainloader, desc="Epoch: "+str(e+1)+"/20 :"):
        images = images.view(images.shape[0], -1).cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print("Loss : ", loss.cpu().item())
    loss_vals.append(loss.cpu().item())

print("Finished Training...")
print("\nTraining Time (mins) :",(time() - time_0)/60)

print("Plotting Training Curve...")
plt.plot(loss_vals)
plt.show()

# score images from validation set
images, labels = next(iter(valloader))
img = images[0].view(1,784)
with torch.no_grad():
    img = img.cuda()
    ops = model(img)

img_p = ops.cpu().view(28,28).numpy()

# plot original and oputput
plt.imshow(img_p, cmap='gray')
plt.show()

plt.imshow(img.cpu().view(28,28).numpy(), cmap='gray')
plt.show()