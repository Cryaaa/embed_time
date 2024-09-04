# %%
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


from embed_time.dataloader_ij import LiveGastruloidDataset
from embed_time.model import VAE
from embed_time.model_VAE_resnet18 import VAEResNet18
from embed_time.transforms import CustomToTensor, SelectRandomTPNumpy

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from torchvision.transforms import v2

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import umap

# %%
base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
# checkpoint_dir = Path(base_dir) / "2024-09-03_Resnet18_VAE_norm_+aug_sox2only_02_ij_checkpoints"
checkpoint_dir = Path(base_dir) / "2024-09-02_Resnet18_VAE_norm_01_ij_checkpoints"
print(checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)

folder_imgs = r"/mnt/efs/dlmbl/G-et/data/live_gastruloid/240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis/Individual Raft Images Norm/"

loading_transforms_test = trans.Compose([
    SelectRandomTPNumpy(0),
    CustomToTensor(),
    v2.Resize((336,336)),
])

dataset_w_t = LiveGastruloidDataset(
    folder_imgs,
    transform = loading_transforms_test,
)

sample = dataset_w_t[0]
in_channels, y, x = sample.shape
print(in_channels)
print((y,x))

# %%
dict = torch.load(Path(checkpoint_dir)/'checkpoint_99.pth')
dict.keys()

# %%
model_params = dict["metadata"]
model_params.keys()

# %%
model = VAEResNet18(nc=model_params['in_channels'], z_dim=model_params['z_dim'])
dataloader = DataLoader(dataset_w_t, batch_size=4, shuffle=True, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
model_params = dict['model']
model.load_state_dict(model_params)

# %%
dataloader = DataLoader(dataset_w_t, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

for i, first in enumerate(dataloader):
    if i == 50:
        test_image = first.float()
        break

test_image.squeeze(0).shape

# %%
plot_size = 5
plot_images = 2
fig, ax = plt.subplots(1,plot_images,figsize=(plot_images*plot_size,plot_size))

ax[0].imshow(test_image.squeeze(0).numpy()[0])
ax[1].imshow(test_image.squeeze(0).numpy()[1])
test_image.shape

# %%
model.eval()
result = model(test_image.to(device))
result = result[0].detach().cpu().squeeze().numpy()
result.shape

# %%
fig, ax = plt.subplots(2,2,figsize=(2*plot_size,2*plot_size))

ax[0,0].imshow(test_image.squeeze(0).numpy()[0])
ax[0,1].imshow(test_image.squeeze(0).numpy()[1])
ax[1,0].imshow(result[0])
ax[1,1].imshow(result[1])

# %% [markdown]
## Everything below for time-dependent plotting.

# %%
from embed_time.transforms import SelectSpecificTPNumpy

num_tp = 9
time_specific_datasets = [
    LiveGastruloidDataset(
        folder_imgs,
        transform = trans.Compose([
            SelectSpecificTPNumpy(0,i),
            CustomToTensor(),
            v2.Resize((336,336)),
        ]),
        return_raft = True
    ) for i in range(num_tp)
]

time_specific_dataloader =[
    DataLoader(
        dataset=ds, 
        batch_size=4, 
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    ) for ds in time_specific_datasets
]

# %%
num_samp = 5
latents = []
timepoints = []
rafts = []
for tp in range(num_tp):
    for i, first in enumerate(time_specific_dataloader[tp]):
        image, raft = first
        image = image.float()
        _, z, mean, var = model(image.to(device))
        for j in range(num_samp):
            z = model.reparameterize(mean, var)
            # print(z.shape)
            latents.append(torch.flatten(z.detach().cpu(), start_dim=1))
            timepoints.append([tp for _ in range(len(z))])
            rafts.append(raft)

latents = np.concatenate(latents, axis=0)
timepoints = np.concatenate(timepoints, axis=0)
rafts = np.concatenate(rafts, axis=0)
latents

# %%
print(latents.shape)
# x = 5 samples x 529 rafts x 9 tp
# y = (21 x 21 px) x 20 z_dim

# %%
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"
if not os.path.isdir(tabular_data):
    os.mkdir(tabular_data)
df_lat = pd.DataFrame(
    latents,
    columns = [f"LD_mu_{i+1}" for i in range(latents.shape[1])]
)
df_lat['Time'] = timepoints
df_lat['Raft'] = rafts

df_lat.to_csv(Path(tabular_data) / "20240903_Resnet_20z_LatentSpace_Norm_+aug_sox2only_ij.csv",
              index=False)

# %%

# Only run if loading df_lat from saved csv file.
tabular_data = "/mnt/efs/dlmbl/G-et/tabular_data"
df_lat = pd.read_csv(Path(tabular_data) / "20240902_Resnet_20z_LatentSpace_Norm_ij.csv")

# %%
timepoints = df_lat['Time'].tolist()
print(timepoints)
rafts = df_lat['Raft'].tolist()
print(rafts)

# %%
df_lat = StandardScaler().fit_transform(
    df_lat.drop(columns=['Time', 'Raft']))
# df_lat = StandardScaler().fit_transform(
#     df_lat.drop(columns=['Unnamed: 0', 'Time', 'Raft']))

# %%
components=5
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(df_lat)
pca_df = pd.DataFrame(pca_out,
                      columns=[f"PC_{i+1}" for i in range(components)])
pca_df['Time'] = timepoints
pca_df['Raft'] = rafts
sns.scatterplot(shuffle(pca_df),
                x="PC_1",
                y="PC_2",
                hue="Time",
                alpha = 0.5)

# %%
pca_df['Time'].unique()


# %%
df_shuffle = shuffle(pca_df)

# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df_shuffle["PC_1"]
y = df_shuffle["PC_2"]
z = df_shuffle["PC_3"]
labels = df_shuffle["Time"]

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o', alpha=0.3)

# Add a colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Time')

ax.view_init(elev=20, azim=100)

# %%

# Create UMAP
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(df_lat)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
umap_df["Time"] = timepoints
umap_df["Rafts"] = rafts

sns.scatterplot(shuffle(umap_df),
                x="UMAP_1",
                y="UMAP_2",
                hue="Time",
                alpha=0.5,
                palette="viridis")

# %%
print(len(rafts))

# %%
from skimage import io
from scipy import stats

arr = 1
raft_dict = {}

for img_name in os.listdir(folder_imgs):
    raft_num = str(img_name)[-7:-4]
    img = io.imread(Path(folder_imgs) / img_name)
    img = img[:, 1].flatten()

    per_25 = np.percentile(img, 25)
    stdev = np.std(img)
    mean = np.mean(img)

    histo = np.histogram(img, bins=256, range=(0,1), density=True)[0]
    ent = stats.entropy(histo, base=2)

    raft_dict[raft_num] = (per_25, stdev, mean, ent)
    

# %%
per_25_list = []
stdev_list = []
mean_list = []
ent_list = []

count = 0
for raft in rafts:
    raft_num = str(raft)[-7:-4]
    per_25_list.append(raft_dict[raft_num][0])
    stdev_list.append(raft_dict[raft_num][1])
    mean_list.append(raft_dict[raft_num][2])
    ent_list.append(raft_dict[raft_num][3])
    count += 1
print(len(ent_list))

# %%

# Create UMAP
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(df_lat)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
umap_df["Time"] = timepoints
umap_df["Rafts"] = rafts
umap_df["Per_25"] = per_25_list
umap_df["Stdev"] = stdev_list
umap_df["Mean"] = mean_list
umap_df["Ent"] = ent_list

umap_df.head()

# %% 

label = 'Per_25'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)

# %% 

label = 'Stdev'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)

# %% 

label = 'Mean'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)

# %% 

cov_list = np.divide(np.array(stdev_list), np.array(mean_list))
umap_df["COV"] = cov_list

label = 'COV'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)

# %% 

label = 'Ent'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)

# %% 

label = 'Time'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)


# %%
umap_df.to_csv(Path(tabular_data) / "20240903_Resnet_20z_LatentSpace_Norm_+aug_sox2only_ij_umap.csv",
              index=False)

# %%
