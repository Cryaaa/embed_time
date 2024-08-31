import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)

class MouseDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir=".", transform=None, img_transform=None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(self.root_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.], [0.5]),  #TODO: Compute for our data -> avg mean and std
            ]
        )

        self.loaded_imgs = [None] * len(self.samples)
        self.loaded_masks = [None] * len(self.samples)
        for sample_ind in range(len(self.samples)):
            img_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "image.tif"
            )
            image = Image.open(img_path)
            image.load()
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "mask.tif"
            )
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask


def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def pad_to_size(small_tensor, target_size):
    if small_tensor.size() > target_size:
        msg = f"Can't pad tensor of size {small_tensor.size()} to tensor of size {target_size}."
        raise ValueError(msg)
    if small_tensor.size() == target_size:
        return small_tensor
    pad_twoside = []
    for small_size, large_size in zip(small_tensor.shape, target_size):
        pad_twoside.append(math.floor((large_size - small_size) / 2))
        pad_twoside.append(math.ceil((large_size - small_size) / 2))
    return torch.nn.functional.pad(small_tensor, pad_twoside[::-1])


def apply_and_show_random_image(f, ds):
    # pick random raw image from dataset
    img_tensor = ds[np.random.randint(len(ds))][0]

    batch_tensor = torch.unsqueeze(
        img_tensor, 0
    )  # add batch dimension that some torch modules expect
    out_tensor = f(batch_tensor)  # apply torch module
    out_tensor = out_tensor.squeeze(0)  # remove batch dimension
    img_arr = img_tensor.numpy()[0]  # turn into numpy array, look at first channel
    out_arr = out_tensor.detach().numpy()[
        0
    ]  # turn into numpy array, look at first channel

    # intialilze figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))

    # Show input image, add info and colorbar
    img_min, img_max = (img_arr.min(), img_arr.max())  # get value range
    inim = axs[0].imshow(img_arr, vmin=img_min, vmax=img_max)
    axs[0].set_title("Input Image")
    axs[0].set_xlabel(f"min: {img_min:.2f}, max: {img_max:.2f}, shape: {img_arr.shape}")
    div = make_axes_locatable(axs[0])
    cb = fig.colorbar(inim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # Show ouput image, add info and colorbar
    out_min, out_max = (out_arr.min(), out_arr.max())  # get value range
    outim = axs[1].imshow(out_arr, vmin=out_min, vmax=out_max)
    axs[1].set_title("First Channel of Output")
    axs[1].set_xlabel(f"min: {out_min:.2f}, max: {out_max:.2f}, shape: {out_arr.shape}")
    div = make_axes_locatable(axs[1])
    cb = fig.colorbar(outim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # center images and remove ticks
    max_bounds = [
        max(ax.get_ybound()[1] for ax in axs),
        max(ax.get_xbound()[1] for ax in axs),
    ]
    for ax in axs:
        diffy = abs(ax.get_ybound()[1] - max_bounds[0])
        diffx = abs(ax.get_xbound()[1] - max_bounds[1])
        ax.set_ylim([ax.get_ybound()[0] - diffy / 2.0, max_bounds[0] - diffy / 2.0])
        ax.set_xlim([ax.get_xbound()[0] - diffx / 2.0, max_bounds[1] - diffx / 2.0])
        ax.set_xticks([])
        ax.set_yticks([])

        # for spine in ["bottom", "top", "left", "right"]: # get rid of box
        #     ax.spines[spine].set_visible(False)

