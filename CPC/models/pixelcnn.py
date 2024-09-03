import torch.nn as nn

from .nn import PixelCNNBlock


class PixelCNN(nn.Module):
    """Autoregressive decoder.

    Causal convolutions are one dimensional, with shape (1, kernel_size).
    To implement veritcal kernels, input is rotated 90 degrees on entry and
    rotated back on exit.
    Args:
        colour_channels (int): Number of colour channels in the target image.
        kernel_size (int): Size of the kernel in the convolutional layers.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers.
        gated (int): Whether to use gated activations (A. Oord 2016).

    """

    def __init__(
        self,
        n_channels,
        kernel_size,
        n_filters=64,
        n_layers=4,
        gated=False,
    ):
        super().__init__()
        self.n_channels = n_channels

        self.layers = nn.Sequential()
        for i in range(n_layers):
            c_in = n_channels if i == 0 else n_filters
            self.layers.append(
                PixelCNNBlock(
                    in_channels=c_in,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    gated=gated,
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=n_filters,
            out_channels=n_channels,
            kernel_size=1,
        )

    def forward(self, x):
        out = self.layers(x)

        return self.out_conv(out)
