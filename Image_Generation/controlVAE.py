import torch
import torch.nn as nn
from torch.autograd import Variable


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, latent_dims, num_of_channels):
        super(Encoder, self).__init__()
        self.z_dims = latent_dims
        self.sequential = nn.Sequential(
            nn.Conv2d(num_of_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 4 * 4)),
            nn.Linear(256 * 4 * 4, latent_dims * 2),
        )

    def forward(self, x):
        sequential = self.sequential(x)
        mu, sigma = torch.split(sequential, self.z_dims, dim=-1)
        return sequential, mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims, num_of_channels):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(latent_dims, 256 * 4 * 4),
            View((-1, 256, 4, 4)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_of_channels, 4, 2, 1),
        )

    def forward(self, z):
        x_hat = self.sequential(z)
        return x_hat


def reparametrization_trick(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    std = torch.exp(log_var / 2)
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class controlVAE(nn.Module):
    def __init__(self, latent_dims=500, num_of_channels=3):
        super().__init__()
        self.encoder = Encoder(latent_dims, num_of_channels)
        self.decoder = Decoder(latent_dims, num_of_channels)

    def forward(self, x):
        sequential, mu, log_var = self.encoder(x)
        z = reparametrization_trick(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var
