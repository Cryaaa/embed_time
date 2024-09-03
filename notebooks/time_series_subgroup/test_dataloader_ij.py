# %%
from pathlib import Path
import os
from skimage import io
from skimage.exposure import rescale_intensity
import tifffile as tiff
import numpy as np

# %%
img_dir = r'/mnt/efs/dlmbl/G-et/data/live_gastruloid/240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis/Individual Raft Images'
stacked_dir = Path(img_dir).parents[0] / 'Individual Raft Images Stacked'
stacked_dir.mkdir(parents=True, exist_ok=True)
print(stacked_dir)
arr = 1
num_tp = 9
channels = ['BF', 'Green']
for raft in range(1, (23**2 + 1)):
    imgs_tp = []
    for tp in range(num_tp):
        imgs_ch = []
        for ch in channels:
            img_path = Path(img_dir) / f'array{arr}_timepoint{tp:02}_{ch}' / f'array{arr}_timepoint{tp:02}_{ch}_raft{raft:03}.tif'
            try: 
                img = tiff.imread(img_path)
            except  FileNotFoundError:
                img = np.full((695, 695),-1)
            imgs_ch.append(np.array(img))
        stacked_ch = np.stack(imgs_ch, axis=0)
        imgs_tp.append(stacked_ch)
    stacked_img = np.stack(imgs_tp, axis=0)
    print(raft, stacked_img.shape)

    output_stacked = stacked_dir / f'array{arr}_raft{raft:03}.tif'
    if os.path.exists(str(output_stacked)):
        os.remove(str(output_stacked)) 
    else:
        tiff.imwrite(output_stacked, stacked_img)

# %%
img_dir = r'/mnt/efs/dlmbl/G-et/data/live_gastruloid/240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis/Individual Raft Images'
arr = 1
num_tp = 9

norm_dir = Path(img_dir).parents[0] / 'Individual Raft Images Norm'
norm_dir.mkdir(parents=True, exist_ok=True)

for raft in range(1, (23**2 + 1)):
    stacked_path = stacked_dir / f'array{arr}_raft{raft:03}.tif'
    img = tiff.imread(stacked_path)
    if img.min() < 0:
        continue

    scaled_bf = []
    for tp in range(num_tp):
        img_bf = np.array(img[tp, 0])
        quant_bf = np.quantile(img_bf, (0.001, 0.999))
        out_bf = rescale_intensity(img_bf, (quant_bf[0], quant_bf[1]), (0,1))
        out_bf = (out_bf - 1) * -1
        scaled_bf.append(np.clip(out_bf, 0, 1))
    scaled_bf = np.array(scaled_bf)
    scaled_bf = np.expand_dims(scaled_bf, axis=1)

    imgs_sox2 = img[:,1]
    imgs_sox2 = np.array(imgs_sox2)
    quant_sox2 = np.quantile(imgs_sox2, (0.001, 0.999))
    scaled_sox2 = rescale_intensity(imgs_sox2, (quant_sox2[0], quant_sox2[1]), (0,1))
    scaled_sox2 = np.clip(scaled_sox2, 0, 1)
    scaled_sox2 = np.expand_dims(scaled_sox2, axis=1)

    # print(scaled_bf.shape, scaled_sox2.shape)
    norm = np.concatenate((scaled_bf, scaled_sox2),axis=1)
    print(raft, norm.shape)
    # stacked_img_scaled = np.stack([scaled_bf, scaled_sox2], axis=0)
    # print(stacked_img_scaled.shape)

    output_norm = norm_dir / f'array{arr}_raft{raft:03}.tif'
    tiff.imwrite(output_norm, norm)

# %%
import matplotlib.pyplot as plt

print(scaled_bf.shape)
fig,axs = plt.subplots(1, 9, figsize=(15,5))
for i,ax in enumerate(axs):
    ax.imshow(scaled_bf[i])
plt.tight_layout()

# %%
import matplotlib.pyplot as plt

print(scaled_sox2.shape)
fig,axs = plt.subplots(1, 9, figsize=(15,5))
for i,ax in enumerate(axs):
    ax.imshow(scaled_sox2[i])
plt.tight_layout()