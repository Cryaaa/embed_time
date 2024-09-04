# %%
from pathlib import Path
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision.transforms as trans
from torchvision.transforms import v2

from embed_time.dataloader_ij import LiveGastruloidDataset
from embed_time.transforms_ij import CustomToTensor, ShiftIntensity, SelectRandomTPNumpy

from skimage import io


# %%
folder_imgs = r"/mnt/efs/dlmbl/G-et/data/live_gastruloid/240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis/Individual Raft Images Norm/"
loading_transforms = trans.Compose([
        SelectRandomTPNumpy(0),
        CustomToTensor(),
        v2.Resize((336,336)),
        ShiftIntensity(bf_factor=2),
        v2.RandomAffine(
            degrees=90,
            translate=[0.1,0.1],
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.GaussianBlur(kernel_size=15, sigma=(0.1,20.0)),
    ])
# loading_transforms = trans.Compose([
#         SelectRandomTPNumpy(0),
#         CustomToTensor(),
#         v2.Resize((336,336)),
#         v2.RandomAffine(
#             degrees=90,
#             translate=[0.1,0.1],
#         ),
#         v2.RandomHorizontalFlip(),
#         v2.RandomVerticalFlip(),
#         v2.GaussianBlur(kernel_size=15, sigma=(0.1,20.0)),
#     ])

dataset_w_t = LiveGastruloidDataset(
    img_dir = folder_imgs,
    transform = loading_transforms,
)


dataloader_train = DataLoader(dataset_w_t, batch_size=5, shuffle=True, pin_memory=True, num_workers=8)

# %%
for data in dataloader_train:
    example_tensor = data
    break

# %%
batch_idx = 3
io.imshow(np.array(example_tensor[batch_idx][1]))
np.max(np.array(example_tensor[batch_idx][1]))

# %%

# %%
