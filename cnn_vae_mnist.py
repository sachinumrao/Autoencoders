import torch
from torch.utils import data
from torchvision.utils import save_image
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import CenterCrop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params
batch = 32
num_epochs = 10
hidden_size = 256
latent_dims = 64
n_latent = latent_dims
input_dims = 784
n_channels = 1
img_h, img_w = 28, 28
in_shape = (n_channels, img_h, img_w)
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

# define model architecture
class VAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        self.z_dim = h//2**2
        self.cc = CenterCrop((h,w))

        self.bc1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1)
        self.bc2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bc3 = nn.BatchNorm2d(64)

        self.z_mean = nn.Linear(64, self.z_dim**2, n_latent)
        self.z_var = nn.Linear(64, self.z_dim**2, n_latent)
        self.z_develop = nn.Linear(n_latent, 64*self.z_dim**2)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.bc4 = nn.BatchNorm2d(32)
        self.upconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)


    def encode(self, x):
        out = self.bc1(x)
        out = self.conv1(out)
        out = F.relu(self.bc2(out))
        out = self.conv2(out)
        out = F.relu(self.bc3(out))
        out_mean = self.z_mean(out)
        out_var = self.z_var(out)
        return out_mean, out_var
        

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.upconv1(out)
        out = F.relu(self.bc4(out))
        out = self.upconv2(out)
        out = self.cc(out)
        return torch.sigmoid(out)


    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mean + eps*std
        

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
        


# instantiate VAE model
model = VAE(in_shape, n_latent).to(device)

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
                    'cnn_results/reconstructed_images_'+str(epoch)+'.png',
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
                'cnn_results/sample_'+str(epoch)+'.png')