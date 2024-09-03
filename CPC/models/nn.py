import torch
from torch import nn
import torch.nn.functional as F


class ShiftedConv(nn.Module):
    """
    Convolutional layer with receptive field shifted left on the last dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        shift = kernel_size - 1
        self.padding = (shift, 0)

        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
        )


    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return x


class PixelCNNBlock(nn.Module):
    """
    Residual block for autoregressive CNN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        gated=False,
    ):
        super().__init__()
        mid_channels = out_channels * 2 if gated else out_channels

        self.in_conv = ShiftedConv(
            in_channels,
            mid_channels,
            kernel_size,
        )
        if gated:
            self.act_fn = lambda x: torch.tanh(x[:, 0::2]) * torch.sigmoid(x[:, 1::2])
        else:
            self.act_fn = nn.ReLU()
        self.out_conv = nn.Conv1d(out_channels, out_channels, 1)

        self.do_skip = out_channels == in_channels

    def forward(self, x):
        """
        Forward pass of the PixelCNN block.

        Returns condiontal tensor to match other layers used in the PixelCNN.

        Args:
            x (torch.Tensor): Input tensor.
            s_code (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Condition tensor.
        """
        feat = self.in_conv(x)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out
