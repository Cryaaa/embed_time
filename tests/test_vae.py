import pytest
import torch
from embed_time.model import VAE, Encoder, Decoder


def test_encoder():
    z_dim = 32
    encoder = Encoder((28, 28), 1, 64, 128, z_dim)
    x = torch.ones(1, 1, 28, 28)
    mu, log_var = encoder(x)
    assert mu.shape == (1, z_dim)
    assert log_var.shape == (1, z_dim)


def test_vae():
    z_dim = 32
    h_dim1 = 64
    h_dim2 = 128
    x_dim = 1
    spatial_shape = (64, 64)
    encoder = Encoder(spatial_shape, x_dim, h_dim1, h_dim2, z_dim=z_dim)
    decoder = Decoder(z_dim, h_dim2, h_dim1, x_dim, spatial_shape)
    vae = VAE(encoder, decoder)
    vae.check_shapes((1, 1, *spatial_shape), z_dim=z_dim)
