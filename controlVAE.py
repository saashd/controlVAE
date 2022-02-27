import torch
import torch.nn as nn
import torch.utils
import torch.distributions


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, latent_dims, num_of_channels):
        super(Encoder, self).__init__()
        self.z_dims=latent_dims
        self.sequential = nn.Sequential(
            nn.Conv2d(num_of_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, 4096),  # B, 4096
            nn.Linear(4096, latent_dims * 2),  # B, z_dim*2
        )

    def forward(self, x):
        sequential = self.sequential(x)
        mu, sigma = torch.split(sequential, self.z_dims, dim=-1)
        return sequential, mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims, num_of_channels):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(latent_dims, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            #  Added to overcome different size of input and output
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(32, num_of_channels, 4, 2, 1),  # B, nc, 64, 64
        )

    def forward(self, z):
        x_hat = self.sequential(z)
        return x_hat


def reparametrization_trick(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    eps = torch.randn_like(log_var)
    return mu + torch.exp(log_var / 2) * eps


class controlVAE(nn.Module):
    def __init__(self, latent_dims=500, num_of_channels=3):
        super().__init__()
        self.encoder = Encoder(latent_dims, num_of_channels)
        self.decoder = Decoder(latent_dims, num_of_channels)

    def forward(self, x):
        sequential, mu, log_var = self.encoder(x)
        z = reparametrization_trick(mu, log_var)
        return self.decoder(z), mu, log_var
