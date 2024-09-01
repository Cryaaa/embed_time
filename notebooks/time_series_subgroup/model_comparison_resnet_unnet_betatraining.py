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

base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
checkpoint_dir = Path(base_dir) / "2024-09-01_Resnet18_VAE_03_tinybeta_checkpoints"
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
ax[1,1].imshow(result[1])
# %%
means = []
variances = []
labels = []
for i,first in enumerate(dataloader):
    image, label = first
    _, z, mean, var = model(image.to(device))
    means.append(mean.detach().cpu().flatten())
    variances.append(var.detach().cpu().flatten())
    labels.append(label[0])
# %%
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(np.array(means),columns = [f"LD_mu_{i+1}" for i in range(np.array(means).shape[1])])
dataframe
# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
components = 5
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(dataframe)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame["Label"] = labels
pca_frame
# %%
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Label",alpha = 0.7)
# %%
import umap

umap_transformer = umap.UMAP()
umap_out = umap_transformer.fit_transform(dataframe)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])

# %%
umap_df["Label"] = labels

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.8)
# %%
latents = []
labels_more_samples = []
for j in range(10):
    for i,first in enumerate(dataloader):
        image, label = first
        _, z, mean, var = model(image.to(device))
        latents.append(z.detach().cpu().flatten())
        labels_more_samples.append(label[0])
# %%
latents = np.array(latents)
latents.shape
# %%
dataframe_sampled = pd.DataFrame(np.array(latents),columns = [f"LD_mu_{i+1}" for i in range(np.array(latents).shape[1])])
dataframe_sampled
# %%
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(dataframe_sampled)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame["Label"] = labels_more_samples
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Label",alpha = 0.5)
# %%
umap_transformer = umap.UMAP()
umap_out = umap_transformer.fit_transform(dataframe_sampled)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
# %%
umap_df["Label"] = labels_more_samples

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
# %%
checkpoint_unet = Path(base_dir)/ '2024-09-01_unet_encdec_beta_01_checkpoints'
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
latents_unet = []
labels_unet = []
for j in range(10):
    for i,first in enumerate(dataloader):
        image, label = first
        _, mean, var = model_unet(image.to(device))
        z = model_unet.sampling(mean,var)
        latents_unet.append(z.detach().cpu())
        labels_unet.append(label[0])
latents_unet = np.array(latents_unet)
latents_unet
# %%
dataframe_unet = pd.DataFrame(np.array(latents_unet.squeeze()),columns = [f"LD_mu_{i+1}" for i in range(np.array(latents_unet.squeeze()).shape[1])])
dataframe_unet
# %%
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(dataframe_unet)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame["Label"] = labels_unet
sns.scatterplot(pca_frame,x="PC_1",y="PC_2",hue="Label",alpha = 0.5)
# %%
umap_transformer = umap.UMAP()
umap_out = umap_transformer.fit_transform(dataframe_unet)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
# %%
umap_df["Label"] = labels_unet

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
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
