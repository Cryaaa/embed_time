import math
import torch
from torch import nn
import torch.nn.functional as F


class DownsamplingEncoder(torch.nn.Module):
    """
    Encoder for the VAE that has convolution and downsampling layers, followed by a fully connected layer.
    """

    def __init__(
        self,
        input_shape,
        input_channels,
        latent_size,
        downsample_factors=[(1, 2, 2), (1, 2, 2), (2, 2, 2)],
        num_channels=[8, 16, 32],
        num_conv=3,
        padding="same",
    ):
        """
        Parameters
        ----------
        input_shape : tuple
            Spatial shape of the input tensor.
        input_channels : int
            Number of channels in the input tensor.
        latent_size : int
            Size of the latent space.
        downsample_factors : list of tuples
            Factors by which to downsample the input tensor at each downsampling step.
        num_channels : list of ints
            Number of channels in each convolutional block.
        num_conv : int
            Number of convolutional layers in each block.
        padding : str, int, optional
            Padding mode for the convolutional layers.
        """
        super(DownsamplingEncoder, self).__init__()

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.downsample_factors = downsample_factors

        # Check arguments
        assert len(downsample_factors) == len(num_channels)
        # Spatial dimensions of the input shape must be the same size as the downsample factors
        assert all(len(f) == len(input_shape) for f in downsample_factors)
        # TODO assert that the input shape is divisible by the downsample factors

        # Define 2 or 3D functions
        if len(input_shape) == 2:
            self.conv_func = torch.nn.Conv2d
            self.pool_func = torch.nn.MaxPool2d
        elif len(input_shape) == 3:
            self.conv_func = torch.nn.Conv3d
            self.pool_func = torch.nn.MaxPool3d
        else:
            raise ValueError("Input shape must be 2 or 3D.")

        # Create the convolutional layers
        self.conv = torch.nn.ModuleList()
        current_channels = input_channels
        for _, (factor, channels) in enumerate(zip(downsample_factors, num_channels)):
            for j in range(num_conv):
                # Define the number of input channels at this stage
                # Add the convolutional layer
                self.conv.append(
                    self.conv_func(
                        current_channels, channels, kernel_size=3, padding=padding
                    )
                )
                current_channels = channels
                # Add the activation function
                self.conv.append(torch.nn.ReLU())
            self.conv.append(self.pool_func(kernel_size=factor, stride=factor))

        # Calculate the size of the flattened tensor
        self.flat_size = self._calculate_flat_size(input_shape)

        # Create the fully connected layer
        self.mu_layer = nn.Linear(self.flat_size, latent_size)
        self.logvar_layer = nn.Linear(self.flat_size, latent_size)

    @torch.no_grad()
    def _calculate_flat_size(self, input_shape):
        """
        Calculate the size of the flattened tensor after the convolutional layers.
        """
        # Create a dummy tensor
        x = torch.zeros((1,) + input_shape)
        # Pass the tensor through the convolutional layers
        for layer in self.conv:
            x = layer(x)
        # Return the size of the flattened tensor
        return x.numel()

    def forward(self, x):
        """
        Forward pass of the encoder.
        """
        # Pass the input through the convolutional layers
        for layer in self.conv:
            x = layer(x)
        # Flatten the tensor
        x = x.view(-1, self.flat_size)
        # Calculate the mean and log variance of the latent space
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar


class UpsamplingDecoder(torch.nn.Module):
    """
    Decoder for the VAE that has a fully connected layer followed by convolution and upsampling layers.
    """

    def __init__(
        self,
        output_shape,
        output_channels,
        latent_size,
        upsample_factors=[(1, 2, 2), (1, 2, 2), (2, 2, 2)],
        num_channels=[32, 16, 8],
        num_conv=3,
        padding="same",
        final_activation=None,
    ):
        """
        Parameters
        ----------
        output_shape : tuple
            Shape of the output tensor.
        output_channels : int
            Number of channels in the output tensor.
        latent_size : int
            Size of the latent space.
        upsample_factors : list of tuples
            Factors by which to upsample the output tensor at each upsampling step.
        num_channels : list of ints
            Number of channels in each convolutional block.
        num_conv : int
            Number of convolutional layers in each block.
        padding : str, int, optional
            Padding mode for the convolutional layers.
        final_activation : torch.nn.Module, optional
            Activation function for the output layer.
        """
        super(UpsamplingDecoder, self).__init__()

        self.output_shape = output_shape
        self.latent_size = latent_size
        self.upsample_factors = upsample_factors
        self.final_activation = final_activation

        # Check arguments
        assert len(upsample_factors) == len(num_channels)
        assert all(len(f) == len(output_shape) for f in upsample_factors)

        # Define 2 or 3D functions
        if len(output_shape) == 2:
            self.conv_func = torch.nn.Conv2d
            self.upsample_func = torch.nn.Upsample
        elif len(output_shape) == 3:
            self.conv_func = torch.nn.Conv3d
            self.upsample_func = torch.nn.Upsample
        else:
            raise ValueError("output shape must be 2 or 3D.")

        # Determine how to reshape
        current_channels = num_channels[0]
        # TODO this only works for same padding!!
        hidden_shape = output_shape
        for factor in upsample_factors:
            assert all(s % f == 0 for s, f in zip(hidden_shape, factor))
            hidden_shape = [s // f for s, f in zip(hidden_shape, factor)]
        self.hidden_shape = (current_channels, *hidden_shape)
        # define the flat size, given this shape
        self.flat_size = math.prod(self.hidden_shape)

        # Create the fully connected layer
        self.fc = nn.Linear(latent_size, self.flat_size)

        # Create the convolutional layers
        self.conv = torch.nn.ModuleList()
        for _, (factor, channels) in enumerate(zip(upsample_factors, num_channels)):
            for j in range(num_conv):
                # Define the number of output channels at this stage
                # Add the convolutional layer
                self.conv.append(
                    self.conv_func(
                        current_channels, channels, kernel_size=3, padding=padding
                    )
                )
                # Add the activation function
                self.conv.append(torch.nn.ReLU())
                current_channels = channels
            self.conv.append(self.upsample_func(scale_factor=factor))

        # Create the output layer
        self.output_layer = self.conv_func(
            num_channels[-1], output_channels, kernel_size=1, padding=padding
        )

    def forward(self, z):
        """
        Forward pass of the decoder.
        """
        # Pass the output through the fully connected layer
        h = self.fc(z)
        h = h.view(-1, *self.hidden_shape)  # un-flatten the tensor
        # Pass the output through the convolutional layers and upsampling layers
        for layer in self.conv:
            h = layer(h)
        # Final convolution for getting the right shape
        x = self.output_layer(h)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, x_dim, h_dim1, h_dim2, z_dim):
        """
        Basic encoding model.

        Parameters
        ----------
        input_shape: tuple
            shape of the input data in spatial dimensions (not channels)
        x_dim: int
            input channels in the input data
        h_dim1: int
            number of features in the first hidden layer
        h_dim2: int
            number of features in the second hidden layer
        z_dim: int
            number of latent features
        """
        super().__init__()
        # encoder part
        self.conv1 = nn.Conv2d(x_dim, h_dim1, kernel_size=3, stride=1, padding=1)
        # o = [(i + 2*p - k) / s] + 1
        output_shape = [(s + 2 * 1 - 3) + 1 for s in input_shape]
        self.conv2 = nn.Conv2d(h_dim1, h_dim2, kernel_size=3, stride=1, padding=1)
        self.output_shape = [(s + 2 * 1 - 3) + 1 for s in output_shape]
        # Computing the shape of the data at this point
        linear_h_dim = h_dim2 * math.prod(output_shape)
        self.fc31 = nn.Linear(linear_h_dim, z_dim)
        self.fc32 = nn.Linear(linear_h_dim, z_dim)

    def forward(self, x):
        """
        x: torch.Tensor
            input tensor

        Returns
        -------
        mu: torch.Tensor
            mean tensor
        log_var: torch.Tensor
            log variance tensor
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        batch_size = h.size(0)
        h = h.view(batch_size, -1)
        return self.fc31(h), self.fc32(h)  # mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim1, h_dim2, x_dim, output_shape):
        """
        Basic decoding model

        Parameters
        ----------
        z_dim: int
            number of latent features
        h_dim1: int
            number of features in the first hidden layer
        h_dim2: int
            number of features in the second hidden layer
        x_dim: int
            number of output channels
        output_shape: tuple
            shape of the output data in the spatial dimensions
        """
        super().__init__()
        # decoder part
        self.z_spatial_shape = (h_dim1, *output_shape)
        spatial_shape = math.prod(self.z_spatial_shape)
        # "Upsample" the data back to the amount we need for the output shape
        self.fc = nn.Linear(z_dim, spatial_shape)
        # Here there will be a reshape
        self.conv1 = nn.Conv2d(h_dim1, h_dim2, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(h_dim2, x_dim, kernel_size=3, padding="same")

    def forward(self, z):
        z = F.relu(self.fc(z))
        h = z.view(-1, *self.z_spatial_shape)
        h = F.relu(self.conv1(h))
        return F.sigmoid(self.conv2(h))


class VAE(nn.Module):
    def __init__(self, encoder, decoder, clamp_value=10):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.clamp_value = clamp_value

    def check_shapes(self, data_shape, z_dim):
        with torch.no_grad():
            try:
                output, mu, var = self.forward(torch.zeros(data_shape))
                input_shape = data_shape
                assert (
                    output.shape == input_shape
                ), f"Output shape {output.shape} is not the same as input shape {input_shape}"
                assert (
                    mu.shape[-1] == z_dim
                ), f"Mu shape {mu.shape} is not the same as latent shape {z_dim}"
                assert (
                    var.shape[-1] == z_dim
                ), f"Var shape {var.shape} is not the same as latent shape {z_dim}"
                print("Model shapes are correct")
            except AssertionError as e:
                raise (e)
            except Exception as e:
                print("Error in checking shapes")
                raise (e)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        # clamping of the logvar
        log_var = torch.clamp(log_var, -self.clamp_value, self.clamp_value)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
