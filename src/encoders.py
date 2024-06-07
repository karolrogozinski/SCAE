"""
Module containing the encoders.
"""
import numpy as np

from torch import nn
import torch.nn.functional as F

from utils.spectral_norm_fc import spectral_norm_fc


class EncoderControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            latent_dim_z=10,
            latent_dim_w=10,
            hid_channels=32,
            hidden_dim=512,
            device='cpu',
            ):
        """
        Encoder based on CorrVAE, adjusted to 44x44 ZDC images
        """
        super(EncoderControlVAE, self).__init__()

        self.hid_channels = hid_channels
        self.hidden_dim = hidden_dim
        self.latent_dim_z = latent_dim_z
        self.latent_dim_w = latent_dim_w
        self.img_size = img_size

        kernel_size = 4
        cnn_kwargs = dict(
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.reshape = (self.hid_channels, kernel_size, kernel_size)

        # Conv layers for 44x44
        self.conv1 = nn.Conv2d(
            self.img_size[0], self.hid_channels, **cnn_kwargs).to(device)
        self.conv2 = nn.Conv2d(
            self.hid_channels, self.hid_channels, **cnn_kwargs).to(device)
        self.conv3 = nn.Conv2d(
            hid_channels, hid_channels, kernel_size,
            stride=(3, 3), padding=(1, 1)).to(device)

        # Linear layers
        self.lin1 = nn.Linear(
            np.product(self.reshape),
            self.hidden_dim).to(device)
        self.lin2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim).to(device)
        self.lin3 = nn.Linear(
            self.hidden_dim,
            self.latent_dim_z+self.latent_dim_w).to(device)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)

        x_z = x.view((batch_size, -1))
        x_z = F.leaky_relu(self.lin1(x_z), negative_slope=0.1)
        x_z = F.leaky_relu(self.lin2(x_z), negative_slope=0.1)
        x_z = F.leaky_relu(self.lin3(x_z), negative_slope=0.1)

        return (
            x_z[:, :self.latent_dim_z],
            x_z[:, self.latent_dim_z:],
        )
