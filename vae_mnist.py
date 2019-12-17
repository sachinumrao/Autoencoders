# import dependencies
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define parameters
batch = 32
num_epochs = 10
hidden_size = 256
latent_dims = 64
input_dims = 784
n_channels = 1
img_h, img_w = 28, 28
output_dims = 784
logging_interval = 100

# load training data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=batch,
    num_workers=8,
    shuffle=True
)

# load testing data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=batch,
    num_workers=8,
    shuffle=True
)

# define VAE architecture
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_dims)
        self.fc2_sigma = nn.Linear(hidden_size, latent_dims)
        self.fc3 = nn.Linear(latent_dims, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_dims)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_sigma(h1)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, x):
        h3 = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dims))
        z = self.reparameterization(mu, logvar)
        return self.decode(z), mu, logvar

# instantiate VAE model
model = VAE().to(device)

# instantiate optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# define loss function
def loss_fn(x_hat, x, mu, logvar):
    bce = F.binary_cross_entropy(x_hat, x.view(-1, input_dims),
        reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


# setup training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_id, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(data)
        loss = loss_fn(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_id%logging_interval == 0:
            print(f'Train Epoch : {epoch}    \
                {batch_id*len(data)}/{len(train_loader.dataset)} \
                {100.*batch_id/len(train_loader):.2f}% \
                Loss : {loss.item()/len(data):.5f}')

    print(f'Epoch : {epoch} \
            Average Loss : {train_loss/len(train_loader.dataset):.5f}')


# setup test function
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            x_hat, mu, logvar = model(data)
            test_loss += loss_fn(x_hat, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                result = torch.cat([data[:n],
                    x_hat.view(batch, n_channels, img_h, img_w)[:n]])
                save_image(result.cpu(),
                    'results/reconstructed_images_'+str(epoch)+'.png',
                    nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss : {test_loss:.5f}')

if __name__ == "__main__":
    for epoch in range(1, num_epochs+1):
        train(epoch)
        test(epoch)
        # generate some samples from random latent space points
        with torch.no_grad():
            n_samples = 32
            sample = torch.randn(n_samples, latent_dims).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(n_samples, n_channels, img_h, img_w),
                'results/sample_'+str(epoch)+'.png')



