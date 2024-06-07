"""
Module containing the noise generator.
"""

from torch import nn
import torch.nn.functional as F


class NoiseGenerator(nn.Module):
    def __init__(
                self,
                hidden_dim: int = 512,
                input_size: int = 64,
                latent_dim: int = 10,
                device: str = 'cpu',
            ) -> None:
        """
        Noise Generator implemented as simple MLP
        """
        super(NoiseGenerator, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.device = device

        self.lin1 = nn.Linear(
            self.input_size, self.hidden_dim
            ).to(device)
        self.lin2 = nn.Linear(
            self.hidden_dim, self.hidden_dim
            ).to(device)
        self.lin3 = nn.Linear(
            self.hidden_dim, self.hidden_dim
            ).to(device)
        self.lin4 = nn.Linear(
            self.hidden_dim, self.latent_dim
            ).to(device)

    def forward(self, x):
        x = F.leaky_relu(self.lin1(x), negative_slope=0.1)
        x = F.leaky_relu(self.lin2(x), negative_slope=0.1)
        x = F.leaky_relu(self.lin3(x), negative_slope=0.1)
        x = F.leaky_relu(self.lin4(x), negative_slope=0.1)

        return x
