import pytest
import torch
from embed_time.model import (
    VAE,
    Encoder,
    Decoder,
    DownsamplingEncoder,
    UpsamplingDecoder,
)


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


def test_downsampling_encoder():
    input_shape = (64, 64)
    input_channels = 1
    encoder = DownsamplingEncoder(
        input_shape=input_shape,
        input_channels=input_channels,
        latent_size=32,
        downsample_factors=[(2, 2), (2, 2), (2, 2)],
        num_channels=[2, 4, 8],
    )
    x = torch.ones(1, input_channels, *input_shape)
    mu, log_var = encoder(x)
    assert mu.shape == (1, 32)
    assert log_var.shape == (1, 32)


def test_upsampling_decoder():
    output_shape = (64, 64)
    output_channels = 1
    latent_size = 32
    decoder = UpsamplingDecoder(
        output_shape=output_shape,
        output_channels=output_channels,
        latent_size=latent_size,
        upsample_factors=[(2, 2), (2, 2), (2, 2)],
        num_channels=[8, 4, 2],
    )
    z = torch.ones(1, latent_size)
    output = decoder(z)
    assert output.shape == (1, output_channels, *output_shape)


def test_up_down_vae():
    input_shape = (64, 64)
    input_channels = 1
    latent_size = 32
    downsample_factors = [(2, 2), (2, 2), (2, 2)]
    num_channels = [2, 4, 8]
    encoder = DownsamplingEncoder(
        input_shape=input_shape,
        input_channels=input_channels,
        latent_size=latent_size,
        downsample_factors=downsample_factors,
        num_channels=num_channels,
    )
    decoder = UpsamplingDecoder(
        output_shape=input_shape,
        output_channels=input_channels,
        latent_size=latent_size,
        upsample_factors=downsample_factors,
        num_channels=num_channels[::-1],
    )
    vae = VAE(encoder, decoder)
    input = torch.ones(1, input_channels, *input_shape)

    output, mu, log_var = vae(input)
    vae.check_shapes(input.shape, latent_size)
    assert output.shape == (1, input_channels, *input_shape)
    assert mu.shape == (1, latent_size)
    assert log_var.shape == (1, latent_size)
