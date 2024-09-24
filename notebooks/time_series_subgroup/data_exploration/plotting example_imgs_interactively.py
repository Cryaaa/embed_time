# %%
import torch
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, CropAndReshapeTL
from embed_time.dataloader_rs import LiveTLSDataset
from datetime import datetime
from pathlib import Path


data_location = Path(r"D:\Data\DLMBL")

metadata = data_location / 'Control_Dataset_4TP_Ground_Truth'
out = data_location / 'Control_Dataset_4TP_Composites_Masked'
folder_imgs =  data_location / 'Control_Dataset_4TP_Normalized_wMask'

# %%
from embed_time.interactive_plots import get_dash_app_2D_scatter_hover_images
import pandas as pd
table_location = data_location / "tabular_data"
version = 3
model_name = "ben_model_04_masked"
out_tabular_data = table_location / model_name
out_tabular_data.mkdir(exist_ok=True)

data = pd.read_csv(
    out_tabular_data / f"version_{version}" / "context_dimensions_table.csv"
)
# %%
loading_transforms = trans.Compose([
    CropAndReshapeTL(1,0,598,0),
    CustomToTensor(),
    v2.Resize((576,576)),

])

dataset_w_t = LiveTLSDataset(
    metadata,
    folder_imgs,
    transform = loading_transforms,
    return_metadata = True,
    metadata_columns=["Run","Plate","ID","Axes","Image Name"]
)

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
import numpy as np

plot_imgs = np.array([imread(out/file) for file in data_w_files["Image Name"]]).astype("uint8")[:,:,:,:3]
print(plot_imgs.shape)

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
composite_tl = data_location / 'Control_Dataset_4TP_Composites_Masked_tl'

plot_imgs_single_tp = np.array(
    [
        imread(composite_tl/file)[tp] 
        for file,tp in data_w_files[
            ["Image Name","Time"]
        ].to_numpy()
    ]
).astype("uint8")[:,:,:,:3]
print(plot_imgs_single_tp.shape)
# %%

app = get_dash_app_2D_scatter_hover_images(
    dataframe=umap_latents,
    plot_keys=["UMAP_1","UMAP_2"], 
    hue = "Unique Plate",
    images = plot_imgs_single_tp,
    additional_info = "Dev Outcome",
    image_size = 400,
)

app.run(debug=True,jupyter_mode="external")
# %%
