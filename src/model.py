# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:17:20 2021

@author: Shiyu
"""
import torch
from torch import nn
from torch.nn import functional as F

from utils.model_init import weights_init


class ControlVAE(nn.Module):
    def __init__(
            self,
            img_size,
            encoder,
            decoder,
            noise_generator_z,
            noise_generator_w,
            latent_dim,
            latent_dim_prop,
            num_prop,
            moving_alpha=0.1,
            hid_channels=32,
            device='cpu'):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        encoder: encoder
        decoder: decoder
        latent_dim: latent dimension
        num_prop: number of properties
        device: device
        """
        super(ControlVAE, self).__init__()

        self.num_prop = num_prop
        self.latent_dim_z = latent_dim
        self.latent_dim_w = latent_dim_prop

        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.device = device

        self.encoder = encoder(img_size, self.latent_dim_z,
                               self.latent_dim_w, hid_channels, device=device)
        self.decoder = decoder(img_size, self.latent_dim_z,
                               self.latent_dim_w, self.num_prop,
                               hid_channels, device=device)
        self.ng_z = noise_generator_z(latent_dim=latent_dim, device=device)
        self.ng_w = noise_generator_w(latent_dim=latent_dim, device=device)

        self.apply(weights_init)
        self.w_mask = torch.nn.Parameter(
            torch.randn(self.num_prop, self.latent_dim_w, 2))

        self.z_mean_avg = nn.Parameter(torch.zeros(1, 8), requires_grad=False)
        self.z_std_avg = nn.Parameter(torch.zeros(1, 8), requires_grad=False)
        self.w_mean_avg = nn.Parameter(torch.zeros(1, 8), requires_grad=False)
        self.w_std_avg = nn.Parameter(torch.zeros(1, 8), requires_grad=False)
        self.alpha = moving_alpha

    def forward(self, x, tau, mask=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        batch_size = x.shape[0]
        input_size = self.ng_z.input_size

        z, w = self.encoder(x)
        z_ = self.ng_z(torch.rand(batch_size, input_size).to(self.device))
        w_ = self.ng_w(torch.rand(batch_size, input_size).to(self.device))

        if mask is None:
            logit = torch.sigmoid(self.w_mask) / (
                1 - torch.sigmoid(self.w_mask))
            mask = F.gumbel_softmax(
                logit.to(self.device), tau, hard=True)[:, :, 1]

        reconstruct, y_reconstruct, _ = self.decoder(z, w, mask)

        z_mean = z.mean(axis=0)
        z_std = z.std(axis=0)
        w_mean = w.mean(axis=0)
        w_std = w.std(axis=0)

        with torch.no_grad():
            self.z_mean_avg.data = self.alpha * z_mean + (1 - self.alpha) * self.z_mean_avg.data
            self.z_std_avg.data = self.alpha * z_std + (1 - self.alpha) * self.z_std_avg.data
            self.w_mean_avg.data = self.alpha * w_mean + (1 - self.alpha) * self.w_mean_avg.data
            self.w_std_avg.data = self.alpha * w_std + (1 - self.alpha) * self.w_std_avg.data

        return (
            reconstruct, y_reconstruct, z, w, z_, w_, mask
        )

    def generate(self, mask, z=None, w=None):
        input_size = self.ng_z.input_size

        if z is None:
            z = self.ng_z(torch.rand(1, input_size).to(self.device))
        if w is None:
            w = self.ng_w(torch.rand(1, input_size).to(self.device))

        reconstruct, y_reconstruct, _ = self.decoder(z, w, mask)
        return reconstruct, y_reconstruct

    def iterate_get_w(self, label, w_latent_idx, maxIter=20):
        """
        Get the w for a kind of given property

        Note:
        It's not the w from laten space but the w' reversed from y.
        Dim is same as y!
        """
        w_n = label.view(-1, 1).to(self.device).float()  # [N]
        for _ in range(maxIter):
            summand = self.decoder.property_lin_list[w_latent_idx](w_n)
            w_n1 = label.view(-1, 1).to(self.device).float() - summand
            print('Iteration of difference:' +
                  str(torch.abs(w_n-w_n1).mean().item()))
            w_n = w_n1.clone()
        return w_n1.view(-1)
