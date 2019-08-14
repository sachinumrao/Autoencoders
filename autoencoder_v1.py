# import dependencies
from time import time

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# normalize data
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5))
                               ])

# load data
trainset = datasets.MNIST('./data', download=True, train=True, 
                          transform=transform)

valset = datasets.MNIST('./data', download=True, train=False, 
                          transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                          shuffle=True)

# model params
input_size = 784
hidden_size = [512, 256, 128, 256, 512]
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

print(model)

# use MSE loss 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

time_0 = time()
n_epochs = 20

# train loop
for e in range(n_epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


print("\nTraining Time (mins) :",(time() - time_0)/60)

images, labels = next(iter(valloader))
img = images[0].view(1,784)
with torch.no_grad():
    ops = model(img)

img_p = ops.view(28,28).numpy()

# plot original and oputput
plt.imshow(img_p, cmap='gray')
plt.show()

plt.imshow(img.view(28,28).numpy(), cmap='gray')
plt.show()