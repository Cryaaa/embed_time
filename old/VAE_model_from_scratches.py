import torch.nn as nn

# Incomplete

 #%% Encoder Block
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

        # TASK 3.1: Initialize your modules and define layers.
        # YOUR CODE HERE
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.ReLU()
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # TASK 3.2: Apply the modules you defined to the input x
        return self.block(x)  # YOUR CODE HERE
    

#%% Downsample

class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # TASK 2B1: Initialize the maxpool module
        self.down = torch.nn.MaxPool2d(kernel_size=downsample_factor)  # YOUR CODE HERE

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # TASK 2B2: Check that the image_size is valid to use with the downsample factor
        # YOUR CODE HERE
        image_x = image_size[0] % self.downsample_factor 
        image_y = image_size[1] % self.downsample_factor 
        return True if (image_x ==0) & (image_y==0) else False

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)

 #%% Output Block
 class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # TASK 5.1: Define the convolution submodule
        # YOUR CODE HERE
        self.convolution = torch.nn.Conv2d(in_channels, out_channels, kernel_size =1)
        
        self.activation = activation

    def forward(self, x):
        # TASK 5.2: Implement the forward function
        # YOUR CODE HERE
        if not self.activation:
            return self.convolution(x)
        else:
            return self.activation(self.convolution(x)) 
        

#%% Model



class AutoEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        num_hidden: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
       super().__init__()     

        """
        A U-Net for 2D input that expects tensors shaped like::
        ``(batch, channels, height, width)``.
         Args:
        depth:
            The number of levels in the U-Net. 2 is the smallest that really
            makes sense for the U-Net architecture, as a one layer U-Net is
            basically just 2 conv blocks.
        in_channels:
            The number of input channels in your dataset.
        out_channels (optional):
            How many output channels you want. Depends on your task. Defaults to 1.
        final_activation (optional):
            What activation to use in your final output block. Depends on your task.
            Defaults to None.
        num_fmaps (optional):
            The number of feature maps in the first layer. Defaults to 64.
        fmap_inc_factor (optional):
            By how much to multiply the number of feature maps between
            layers. Encoder level ``l`` will have ``num_fmaps*fmap_inc_factor**l``
            output feature maps. Defaults to 2.
        downsample_factor (optional):
            Factor to use for down- and up-sampling the feature maps between layers.
            Defaults to 2.
        kernel_size (optional):
            Kernel size to use in convolutions on both sides of the UNet.
            Defaults to 3.
        padding (optional):
            How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
        upsample_mode (optional):
            The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
            or "bilinear." Defaults to "nearest."
        """

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode
        # Set the number of hidden units
        self.num_hidden = num_hidden
        
        # Define the encoder part of the autoencoder
        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # TASK 6.2A: Initialize list here
        for l in range(0, self.depth):
            features_in, features_out = self.compute_fmaps_encoder(l)
            self.left_convs.append(ConvBlock(in_channels=features_in, out_channels=features_out, kernel_size=self.kernel_size, padding=self.padding)) 
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(784, 256),  # input size: 784, output size: 256
        #     nn.ReLU(),  # apply the ReLU activation function
        #     nn.Linear(256, self.num_hidden),  # input size: 256, output size: num_hidden
        #     nn.ReLU(),  # apply the ReLU activation function
        # )
        self.right_convs = torch.nn.ModuleList()
        # TASK 6.2A: Initialize list here
        for l in range(0, self.depth):
            features_out, features_in = self.compute_fmaps_encoder(l)
            self.left_convs.append(ConvBlock(in_channels=features_in, out_channels=features_out, kernel_size=self.kernel_size, padding=self.padding)) 
                
        # Define the decoder part of the autoencoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.num_hidden, 256),  # input size: num_hidden, output size: 256
        #     nn.ReLU(),  # apply the ReLU activation function
        #     nn.Linear(256, 784),  # input size: 256, output size: 784
        #     nn.Sigmoid(),  # apply the sigmoid activation function to compress the output to a range of (0, 1)
        # )

        # UP, DOWN-Sample
        self.down = Downsample(downsample_factor=self.downsample_factor)
        self.up = torch.nn.Upsample(scale_factor=self.downsample_factor, mode="nearest")  
        # Final
        self.final = OutputConv(in_channels=self.num_fmaps, out_channels= self.in_channels, activation=self.final_activation)
        
        # Hidden Layer
        _, last_conv_features = self.compute_fmaps_encoder(depth)
        self.hidden = nn.Sequential(
            nn.Linear(self., self.num_hidden),  # input size: num_hidden, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
        )
       
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
        # TASK 6.1A: Implement this function
        if level == 0:
            return (self.in_channels, self.num_fmaps)
        else:
            return (self.num_fmaps*self.fmap_inc_factor**(level-1), self.num_fmaps*self.fmap_inc_factor**level)
        
    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded