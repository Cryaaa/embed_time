# %%
import torch
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, CropAndReshapeTL
from embed_time.dataloader_rs import LiveTLSDataset
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from skimage.io import imshow
from tifffile import imwrite

data_location = Path(r"D:\Data\DLMBL")

folder_imgs = data_location / 'Control_Dataset_4TP_Normalized_wMask'
metadata = data_location / 'Control_Dataset_4TP_Ground_Truth'
# %%
loading_transforms = trans.Compose([
    CropAndReshapeTL(1,0,598,0),
    CustomToTensor(),
    #ColorJitterBrightfield(0.5,0.3,0,0),
    v2.Resize((576,576)),
    #v2.RandomAffine(
    #     degrees=90,
    #     translate=[0.1,0.1],
    # ),
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    # v2.GaussianNoise(0,0.2,clip=True)
])

dataset_w_t = LiveTLSDataset(
    metadata,
    folder_imgs,
    transform = loading_transforms,
    return_metadata = True,
    metadata_columns=["Run","Plate","ID","Axes","Image Name"]
)

image, label, met_info = dataset_w_t[0]
image.shape


# %%
import napari
viewer = napari.Viewer()
# %%
c_0 = torch.concat([image[0,i] for i in range(4)],1)
c_1 = torch.concat([image[1,i] for i in range(4)],1)
c_0.shape
# %%
viewer.add_image(c_0.numpy(),contrast_limits=[0,1.3])
viewer.add_image(c_1.numpy(),contrast_limits=[0,1.3],blending="additive",colormap="red")

# %%

screenshots = []
for j in tqdm(range(len(dataset_w_t))):
    viewer.layers.clear()
    image, label, met_info = dataset_w_t[j]
    c_0 = torch.concat([image[0,i] for i in range(4)],1)
    c_1 = torch.concat([image[1,i] for i in range(4)],1)
    c_0.shape
    viewer.add_image(c_0.numpy(),contrast_limits=[0,1.3])
    viewer.add_image(
        c_1.numpy(),
        contrast_limits=[0,1],
        blending="additive",
        colormap="red"
    )
    screenshots.append(viewer.screenshot())

# %%

imshow(screenshots[0][530:920,180:2300])

# %%

out = Path(data_location) / 'Control_Dataset_4TP_Composites_Masked'
out.mkdir(exist_ok=True)
for j,screen in enumerate(screenshots):
    image, label, met_info = dataset_w_t[j]
    name = met_info[-1]
    print(name)
    imwrite(out / name,screen[530:920,180:2300])

#%%

screenshots_tl = []
for j in tqdm(range(len(dataset_w_t))):
    
    image, label, met_info = dataset_w_t[j]
    tl_composites = []
    for i in range(4):
        c_0 = image[0,i]
        c_1 = image[1,i]
        viewer.layers.clear()
        viewer.add_image(
            c_0.numpy(),
            contrast_limits=[0,1.3]
        )
        viewer.add_image(
            c_1.numpy(),
            contrast_limits=[0,1],
            blending="additive",
            colormap="red"
        )
        tl_composites.append(viewer.screenshot())
    screenshots_tl.append(tl_composites)

screenshots_tl = np.array(screenshots_tl)
screenshots_tl.shape
#%%

imshow(screenshots_tl[0][0][:,135:734])
# %%
composite_tl = Path(data_location) / 'Control_Dataset_4TP_Composites_Masked_tl'
composite_tl.mkdir(exist_ok=True)
for j,screen in enumerate(screenshots_tl):
    image, label, met_info = dataset_w_t[j]
    name = met_info[-1]
    #print(name)
    imwrite(composite_tl / name,screen[:,:,135:734])

# %%
