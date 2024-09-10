# %%
import torch
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, CropAndReshapeTL
from embed_time.dataloader_rs import LiveTLSDataset
from datetime import datetime
from pathlib import Path


data_location = Path("/mnt/efs/dlmbl/G-et/data/live-TLS")

folder_imgs = data_location / 'Control_Dataset_4TP_Normalized_Across_Plates'
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
import os
os.environ["DISPLAY"] = ":1001"
import napari
viewer = napari.Viewer()
# %%
c_0 = torch.concat([image[0,i] for i in range(4)],1)
c_1 = torch.concat([image[1,i] for i in range(4)],1)
c_0.shape
# %%
viewer.add_image(c_0.numpy(),contrast_limits=[0,1])
viewer.add_image(c_1.numpy(),contrast_limits=[0,1],blending="additive",colormap="red")

# %%
from tqdm.auto import tqdm
screenshots = []
for j in tqdm(range(len(dataset_w_t))):
    viewer.layers.clear()
    image, label, met_info = dataset_w_t[j]
    c_0 = torch.concat([image[0,i] for i in range(4)],1)
    c_1 = torch.concat([image[1,i] for i in range(4)],1)
    c_0.shape
    viewer.add_image(c_0.numpy(),contrast_limits=[0,1])
    viewer.add_image(
        c_1.numpy(),
        contrast_limits=[0,1],
        blending="additive",
        colormap="red"
    )
    screenshots.append(viewer.screenshot())

# %%
from tifffile import imwrite
out = Path(data_location) / 'Control_Dataset_4TP_Composites'
out.mkdir(exist_ok=True)
for j,screen in enumerate(screenshots):
    image, label, met_info = dataset_w_t[j]
    name = met_info[-1]
    print(name)
    imwrite(out / name,screen[290:690,:])
# %%
from skimage.io import imshow
imshow(screenshots[0][290:690])
# %%
import numpy as np
screenshots_cropped = np.array(screenshots)[:,290:690,:]
imshow(screenshots_cropped[0])
# %%
from embed_time.interactive_plots import get_dash_app_2D_scatter_hover_images
import pandas as pd
table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
version = 2
model_name = "ben_model_03_pp_norm"
out_tabular_data = table_location / model_name
out_tabular_data.mkdir(exist_ok=True)

data = pd.read_csv(
    out_tabular_data / f"version_{2}" / "latent_dimensions_table.csv"
)
data
# %%
all_metdat = {
    key : [mtdt[i] for _,_,mtdt in dataset_w_t]
    for i,key in enumerate(["Run","Plate","ID","Axes","Image Name"])
}
all_metdat
# %%
id_to_file = pd.DataFrame(
    all_metdat
)
# %%
from sklearn.preprocessing import StandardScaler
from umap import UMAP
data_w_files = data.merge(id_to_file,on=["Run","Plate","ID"])
data_w_files

scaled_latents = StandardScaler().fit_transform(
    data.drop(
        ["Run","ID","Dev Outcome","Time","Axes","Unique Plate"],
        axis=1
    ))

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,data[["Run","ID","Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

# %%
from skimage.io import imread
plot_imgs = np.array([imread(out/file) for file in data_w_files["Image Name"]]).astype("uint8")
plot_imgs.shape
# %%
data

# %%
app = get_dash_app_2D_scatter_hover_images(
    dataframe=umap_latents,
    plot_keys=["UMAP_1","UMAP_2"], 
    hue = "Dev Outcome",
    images = plot_imgs,
    additional_info = "Unique Plate",
    image_size = 500,
)


app.run(debug=True)
# %%
print("here")
# %%
