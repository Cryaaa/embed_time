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


from tifffile import imwrite
out = Path(data_location) / 'Control_Dataset_4TP_Composites'

# %%
from skimage.io import imshow

# %%
import numpy as np

# %%
from embed_time.interactive_plots import get_dash_app_2D_scatter_hover_images
import pandas as pd
table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
version = 1
model_name = "ben_model_03_pp_norm"
out_tabular_data = table_location / model_name
out_tabular_data.mkdir(exist_ok=True)

data = pd.read_csv(
    out_tabular_data / f"version_{version}" / "context_dimensions_table.csv"
)
# data= data[data["Time"]>=2]
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
plot_imgs = np.array([imread(out/file) for file in data_w_files["Image Name"]]).astype("uint8")[:,:,:,:3]
print(plot_imgs.shape)
import sys
#sys.exit()
# %%
data

# %%
app = get_dash_app_2D_scatter_hover_images(
    dataframe=umap_latents,
    plot_keys=["UMAP_1","UMAP_2"], 
    hue = "Dev Outcome",
    images = plot_imgs,
    additional_info = "Unique Plate",
    image_size = 1000,
)


app.run(debug=True)
# %%
print("here")
# %%
