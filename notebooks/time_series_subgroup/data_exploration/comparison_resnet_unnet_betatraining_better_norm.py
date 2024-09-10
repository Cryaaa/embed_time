# %%
import torch
import matplotlib.pyplot as plt
from embed_time.dataloader_rs import LiveTLSDataset
from embed_time.model import VAE
from embed_time.UNet_based_encoder_decoder import UNetDecoder, UNetEncoder
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import seaborn as sns
import os
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, SelectRandomTPNumpy, CustomCropCentroid
from embed_time.dataloader_rs import LiveTLSDataset
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
import pandas as pd

base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
checkpoint_dir = Path(base_dir) / "2024-09-01_Resnet18_VAE_04_norm_checkpoints"
print(checkpoint_dir)

checkpoint_dir.mkdir(exist_ok=True)
data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"
folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized'
metadata = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'

loading_transforms_test = trans.Compose([
    SelectRandomTPNumpy(0),
    CustomCropCentroid(0,0,598),
    CustomToTensor(),
    v2.Resize((576,576)),
    #v2.RandomAffine(
    #    degrees=90,
    #    translate=[0.1,0.1],
    #),
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    #v2.GaussianNoise(0,0.05)
])

dataset_w_t = LiveTLSDataset(
    metadata,
    folder_imgs,
    metadata_columns=["Run","Plate","ID"],
    return_metadata=False,
    transform = loading_transforms_test,
)

sample, label = dataset_w_t[0]
in_channels, y, x = sample.shape
print(in_channels)
print((y,x))
# %%
dict = torch.load(Path(checkpoint_dir)/'checkpoint_99.pth')
dict.keys()
# %%
model_params = dict["metadata"]
model_params
# %%
from embed_time.model_VAE_resnet18 import VAEResNet18

model = VAEResNet18(nc=model_params['in_channels'],z_dim=model_params['z_dim'])
dataloader = DataLoader(dataset_w_t, batch_size=4, shuffle=True, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
model_params = dict['model']
model.load_state_dict(model_params)
# %%
dataloader = DataLoader(dataset_w_t, batch_size=1, shuffle=False, pin_memory=True)

for i,first in enumerate(dataloader):
    if i == 50:
        test_image = first[0]
        break

test_image.squeeze(0).shape
# %%
plot_size = 5
plot_images = 2
fig, ax = plt.subplots(1,plot_images,figsize=(plot_images*plot_size,plot_size))

ax[0].imshow(test_image.squeeze(0).numpy()[0])
ax[1].imshow(test_image.squeeze(0).numpy()[1])
# %%
model.eval()
result = model(test_image.to(device))[0]
result = result.detach().cpu().squeeze().numpy()
result.shape
# %%
fig, ax = plt.subplots(2,2,figsize=(2*plot_size,2*plot_size))

ax[0,0].imshow(test_image.squeeze(0).numpy()[0])
ax[0,1].imshow(test_image.squeeze(0).numpy()[1])
ax[1,0].imshow(result[0])
ax[1,1].imshow(result[1])# %%
checkpoint_unet = Path(base_dir)/ '2024-09-01_unet_encdec_beta_norm_01_checkpoints'
dict_unet = torch.load(checkpoint_unet/'checkpoint_99.pth')
dict_unet.keys()
# %%
model_params_unet = dict_unet['metadata']
model_params_unet
# %%
encoder = UNetEncoder(
    in_channels = in_channels,
    n_fmaps = model_params_unet["n_fmaps"],
    depth = model_params_unet["depth"],
    in_spatial_shape = (y,x),
    z_dim = model_params_unet["z_dim"],
)

decoder = UNetDecoder(
    in_channels = in_channels,
    n_fmaps = model_params_unet["n_fmaps"],
    depth = model_params_unet["depth"],
    in_spatial_shape = (y,x),
    z_dim = model_params_unet["z_dim"],
    upsample_mode="bicubic"
)

model_unet = VAE(encoder, decoder)
model_unet.to(device)
# %%
model_unet.load_state_dict(dict_unet['model'])
# %%
model_unet.eval()
result = model_unet(test_image.to(device))[0]
result.shape
# %%
fig, ax = plt.subplots(2,2,figsize=(2*plot_size,2*plot_size))

ax[0,0].imshow(test_image.squeeze(0)[0])
ax[0,1].imshow(test_image.squeeze(0)[1])
ax[1,0].imshow(result.detach().cpu().squeeze(0)[0])
ax[1,1].imshow(result.detach().cpu().squeeze(0)[1])
# %%
from embed_time.transforms import SelectSpecificTPNumpy

time_specific_datasets = [
    LiveTLSDataset(
        metadata,
        folder_imgs,
        metadata_columns=["Run","Plate","ID"],
        return_metadata=False,
        transform = trans.Compose([
            SelectSpecificTPNumpy(0,i),
            CustomCropCentroid(0,0,598),
            CustomToTensor(),
            v2.Resize((576,576)),
        ]),
    ) for i in range(4)
]

time_specific_dataloader =[
    DataLoader(
        ds, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True
    ) for ds in time_specific_datasets
]
# %%
latents_unet = []
labels_unet = []
timepoints_unet = []
for tp in range(4):
    for j in range(5):
        for i,first in enumerate(time_specific_dataloader[tp]):
            image, label = first
            _, mean, var = model_unet(image.to(device))
            z = model_unet.sampling(mean,var)
            latents_unet.append(z.detach().cpu())
            labels_unet.append(label[0])
            timepoints_unet.append(tp)
latents_unet = np.array(latents_unet)
latents_unet
# %%
dataframe_unet = pd.DataFrame(np.array(latents_unet.squeeze()),columns = [f"LD_mu_{i+1}" for i in range(np.array(latents_unet.squeeze()).shape[1])])
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"
if not os.path.isdir(tabular_data):
    os.mkdir(tabular_data)
pd.DataFrame({
    "Label":labels_unet,
    "Time":timepoints_unet,
}).to_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_GoodNorm_Annotations.csv")
dataframe_unet.to_csv(Path(tabular_data)/"20240901_UNet_20z_Latent_Space_GoodNorm.csv")
dataframe_unet
# %%
components=5
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(dataframe_unet)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame["Label"] = labels_unet
pca_frame["Time"] = timepoints_unet
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Label",alpha = 0.5)
# %%
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Time",alpha = 0.5)
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pca_frame
# Assuming your DataFrame is named df
# Replace this with your actual DataFrame
# df = pd.DataFrame(data)

# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df["PC_1"]
y = df["PC_2"]
z = df["PC_3"]
labels = df["Label"] == "good"

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.3)

# Add a colorbar
#cbar = plt.colorbar(sc)
#cbar.set_label('Labels')

ax.view_init(elev=30, azim=20)
# %%

# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df["PC_1"]
y = df["PC_2"]
z = df["PC_3"]
labels = df["Time"]

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.3)

# Add a colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Time')

ax.view_init(elev=20, azim=100)
# %%
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(dataframe_unet)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
# %%
umap_df["Label"] = labels_unet
umap_df["Time"] = timepoints_unet
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
# %%
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Time",alpha=0.5,palette="viridis")
# %%
latents = []
labels = []
timepoints = []
for tp in range(4):
    for j in range(5):
        for i,first in enumerate(time_specific_dataloader[tp]):
            image, label = first
            _, z, mean, var = model(image.to(device))
            latents.append(z.detach().cpu())
            labels.append(label[0])
            timepoints.append(tp)
latents = np.array(latents)
latents
# %%
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"
flat_latt = np.array(
    [lat.flatten() for lat in latents]
).squeeze()
dataframe = pd.DataFrame(
    flat_latt,
    columns = [
        f"LD_mu_{i+1}" 
        for i in range(flat_latt.shape[1])
    ]
)
dataframe.to_csv(Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_GoodNorm.csv")
pd.DataFrame({
    "Label":labels,
    "Time":timepoints,
}).to_csv(Path(tabular_data)/"20240901_Resnet_26000z_Latent_Space_GoodNorm_Annotations.csv")
dataframe
# %%
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(dataframe)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame["Label"] = labels
pca_frame["Time"] = timepoints
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Label",alpha = 0.5)
# %%
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Time",alpha = 0.5)
# %%
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(dataframe)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
# %%
umap_df["Label"] = labels
umap_df["Time"] = timepoints
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
# %%
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Time",alpha=0.5,palette="viridis")
# %%

df = pca_frame
# Assuming your DataFrame is named df
# Replace this with your actual DataFrame
# df = pd.DataFrame(data)

# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df["PC_1"]
y = df["PC_2"]
z = df["PC_3"]
labels = df["Label"] == "good"

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.5)

# Add a colorbar
#cbar = plt.colorbar(sc)
#cbar.set_label('Labels')

ax.view_init(elev=30, azim=50)
# %%

# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df["PC_1"]
y = df["PC_2"]
z = df["PC_3"]
labels = df["Time"]

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.5)

# Add a colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Time')

ax.view_init(elev=60, azim=100)
# %%
