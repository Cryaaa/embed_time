from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def get_padded_size(size, n_downsc):
    """
    Calculates the necessary padded size of an image for a number of downscaling steps.

    Args:
        size (tuple): The desired size of the image as a tuple of (height, width).
        n_downsc (int): The number of downscaling steps.

    Returns:
        tuple: The padded size of the image as a tuple of (padded_height, padded_width).
    """
    dwnsc = 2 ** n_downsc
    padded_size = [((s - 1) // dwnsc + 1) * dwnsc for s in size]

    return padded_size


def spatial_pad_crop(x, target_size):
    """
    Pads or crops the input tensor `x` to match the target size.

    Args:
        x (torch.Tensor): The input tensor to be padded or cropped.
        target_size (tuple): The target size to match.

    Returns:
        torch.Tensor: The padded or cropped tensor.
    """
    x_size = x.size()[2:]
    delta = [ts - xs for ts, xs in zip(target_size, x_size)]
    crop_delta = [(abs(d) // 2, abs(d) // 2 + abs(d) % 2) if d < 0 else (0, 0) for d in delta]
    pad_delta = [(d // 2, d // 2 + d % 2) if d > 0 else (0, 0) for d in delta]
    
    pad = []
    for d in reversed(pad_delta):
        pad.append(d[0])
        pad.append(d[1])
    x = nn.functional.pad(x, pad)
    x_size = x.size()[2:]
    crop = [slice(0, x.size(0)), slice(0,  x.size(1))]
    crop += [slice(d[0], xs - d[1]) for d, xs in zip(crop_delta, x_size)]
    return x[crop]


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    canvas = figure.canvas
    width, height = canvas.get_width_height()
    canvas.draw()
    image = (
        np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        .reshape(height, width, 4)
        .transpose(2, 0, 1)
    )
    image = image / 255
    plt.close(figure)
    return image
