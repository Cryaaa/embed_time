import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        # SOLUTION 3.1: Initialize your modules and define layers.
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # SOLUTION 3.2: Apply the modules you defined to the input x
        return self.conv_pass(x)


class ConvolutionalEncoder(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            n_fmaps: int,
            depth: int, 
            in_spatial_shape:tuple, 
            z_dim: int,
            fmap_inc_factor=2,
            padding: str = "same",
            downsample_factor: int = 2,
            kernel_size: int = 3,
            n_convs: int = 2
        ):
        self.depth = depth
        self.num_fmaps = n_fmaps
        self.in_channels = in_channels
        self.in_spatial_shape = in_spatial_shape
        self.kernel_size = kernel_size
        self.downsample_factor = downsample_factor
        self.fmap_inc_factor = fmap_inc_factor
        self.padding = padding
        self.n_convs = n_convs
        super(ConvolutionalEncoder, self).__init__()
        self.downsample = nn.MaxPool2d(self.downsample_factor,self.downsample_factor)
        self.convs = nn.ModuleList()
        # SOLUTION 6.2A: Initialize list here
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.convs.append(
                ConvBlock(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            )
        self.fc_layer_len = self.compute_final_layers()
        self.fc1 = nn.Linear(in_features=self.fc_layer_len,out_features=z_dim)
        self.fc2 = nn.Linear(in_features=self.fc_layer_len,out_features=z_dim)

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # SOLUTION 6.1A: Implement this function
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out
    
    def compute_spatial_shape(self, level: int) -> tuple[int, int]:
        spatial_shape = np.array(self.in_spatial_shape)
        if level == 0:
            if self.padding == "same":
                return spatial_shape
            
            # 2 convolutions and 2 sizes 
            spatial_shape = spatial_shape - self.n_convs * (2 * (self.kernel_size //2))
            return spatial_shape

        if self.padding == "same":
            spatial_shape = (np.array(self.compute_spatial_shape(level-1))//(self.downsample_factor)) 
        else:
            spatial_shape = self.compute_spatial_shape(level-1)
            spatial_shape = spatial_shape // self.downsample_factor
            spatial_shape = spatial_shape - self.n_convs * (2 * (self.kernel_size //2))
        return spatial_shape
    
    def compute_final_layers(self):
        spatial_dims_final = self.compute_spatial_shape(self.depth-1)
        num_fmaps_final = self.compute_fmaps_encoder(self.depth-1)
        return num_fmaps_final[1] * np.prod(spatial_dims_final)

    
    def forward(self, x):
        for level in range(self.depth -1):
            x = self.convs[level](x)
            x = self.downsample(x)
        x = self.convs[-1](x)
        x = x.view(-1,self.fc_layer_len)
        return self.fc1(x),self.fc2(x)
    
if __name__ == "__main__":
    shape = (499,6173)
    depth = 6
    encoder = ConvolutionalEncoder(
        in_channels=1,
        in_spatial_shape=shape,
        kernel_size=3,
        n_fmaps=8,
        padding="valid",
        depth =depth,
        z_dim=5,
    )
    example_tensor = torch.zeros(2,1,shape[0],shape[1])
    out= encoder(example_tensor)
    print(out[0].shape,out[1].shape)
    #print(encoder.compute_final_layers())



