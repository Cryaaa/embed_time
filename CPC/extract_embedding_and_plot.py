# %%
import os
from embed_time.transforms import CustomToTensor, CropAndReshapeTL
from embed_time.dataloader_rs import LiveTLSDatasetPairedOutput
from torchvision.transforms import v2
import torchvision.transforms as trans
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.nn import ShiftedConv
from models.convnext import ConvNeXt
import cpc
from pathlib import Path

table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
version = 3
model_name = "ben_model_03_pp_norm"
out_tabular_data = table_location / model_name
out_tabular_data.mkdir(exist_ok=True)
out_tabular_data = out_tabular_data / f"version_{str(version)}"
out_tabular_data.mkdir(exist_ok=True)

base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"


GPU = 0
seed = 1
 # first training this will be zero
checkpoint_dir = Path(base_dir) / f"2024-09-03_{model_name}_checkpoints/version_{str(version)}"

# %%
batch_size =16
n_time = 4

convnext_dims = [16, 32, 64, 128]
in_channels = 2
# %%
parameters = torch.load(checkpoint_dir/"metadata_training.pt")
batch_size =parameters['batch_size']
n_time =parameters['n_time']
latent_dims =parameters['latent_dims']
convnext_dims =parameters['convnext_dims']
in_channels = parameters['in_channels']
latent_dims = 16

# %%
torch.manual_seed(seed)
device = torch.device(f"cuda:{str(GPU)}")



data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"

folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized_Across_Plates'
annotations = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'


loading_transforms = trans.Compose([
    CropAndReshapeTL(1,0,598,0),
    CustomToTensor(),
    v2.Resize((576,576)),
])

torch.manual_seed(seed)
device = torch.device(f"cuda:{str(GPU)}")
dataset_w_t = LiveTLSDatasetPairedOutput(
    annotations,
    folder_imgs,
    indices=range(312),
    transform = loading_transforms,
)


train_loader = torch.utils.data.DataLoader(
    LiveTLSDatasetPairedOutput(
        annotations,
        folder_imgs,
        indices=range(312),
        transform = loading_transforms,
    ),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)


encoder = ConvNeXt(in_chans=in_channels, num_classes=latent_dims, dims=convnext_dims)
ar_model = ShiftedConv(in_channels=latent_dims, out_channels=latent_dims, kernel_size=n_time)
query_weights = torch.nn.ModuleList()
for _ in range(n_time - 1):
    query_weights.append(torch.nn.Linear(latent_dims, latent_dims))

encoder.load_state_dict(
    torch.load(os.path.join(checkpoint_dir, "encoder.pt"), weights_only=True)
)
ar_model.load_state_dict(
    torch.load(
        os.path.join(checkpoint_dir, "ar_model.pt"), weights_only=True
    )
)
query_weights.load_state_dict(
    torch.load(
        os.path.join(checkpoint_dir, "query_weights.pt"), weights_only=True
    )
)

def test(test_loader,
         encoder,
         ar_model,
         query_weights,
):
        encoder = encoder.to(device)
        ar_model = ar_model.to(device)
        query_weights = query_weights.to(device)
        encoder.eval()
        ar_model.eval()
        query_weights.eval()
        latents_list = []
        context_list = []
        with torch.no_grad():
            for batch in tqdm(
                test_loader,
                bar_format=f"Predicting {{l_bar}}{{bar}}{{r_bar}}",
            ):
                batch_size = batch.shape[0]
                batch = batch.to(device)
                context, latents = cpc.forward(
                    batch, batch_size=batch_size, encoder=encoder, ar_model=ar_model
                )
                latents_list.append(latents)
                context_list.append(context)
        return latents_list, context_list

embeddings = test(
    test_loader=train_loader,
    encoder=encoder,
    ar_model=ar_model,
    query_weights=query_weights,
)
latents, context = embeddings
latents = torch.cat(latents, dim=0).cpu()
context = torch.cat(context, dim=0).cpu()
# %%
latents_reshape = latents.movedim(2,0).flatten(0,1)


import seaborn as sns
from embed_time.dataloader_rs import LiveTLSDataset
import pandas as pd

metadata_loader = LiveTLSDataset(
    annotations,
    folder_imgs,
    transform = loading_transforms,
    metadata_columns=["Axes","Run","Plate","ID"],
    return_metadata=True
)

labels = []
metadata = []

for sample in metadata_loader:
    _, label, metdat = sample
    labels.append(label)
    metadata.append(metdat)
time_labels = [i//312 for i in range(len(latents_reshape))]
axis_labels = [meta[0] for meta in metadata]
plate_labels = [f"{meta[1]}_{meta[2]}" for meta in metadata]

latents_df = pd.DataFrame(
     latents_reshape,
     columns = [f"LD_{i}" for i in range(latent_dims)]
)
for i,name in enumerate(["Axes","Run","Plate","ID"]):
    latents_df[name] = [met[i] for met in metadata]*4
latents_df["Dev Outcome"] = labels*4
latents_df["Time"] = time_labels
latents_df["Unique Plate"] = plate_labels*4
latents_df.to_csv(out_tabular_data / "latent_dimensions_table.csv")

# %%
context_reshape = context.movedim(2,0).flatten(0,1)
context_df = pd.DataFrame(
     context_reshape,
     columns = [f"CD_{i}" for i in range(latent_dims)]
)
for i,name in enumerate(["Axes","Run","Plate","ID"]):
    context_df[name] = [met[i] for met in metadata]*4
context_df["Dev Outcome"] = labels*4
context_df["Time"] = time_labels
context_df["Unique Plate"] = plate_labels*4
context_df.to_csv(out_tabular_data / "context_dimensions_table.csv")
context_df
# %%
scaled_latents = StandardScaler().fit_transform(latents_reshape)

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,latents_df[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_latents = PCA(n_components=3).fit_transform(scaled_latents)
pca_latents = pd.DataFrame(pca_latents,columns = [f"PC_{i}" for i in range(3)])
pca_latents = pd.concat([pca_latents,latents_df[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)


scaled_context = StandardScaler().fit_transform(context_reshape)

umap_context = UMAP(n_neighbors=10).fit_transform(scaled_context)
umap_context = pd.DataFrame(umap_context,columns=["UMAP_1","UMAP_2"])
umap_context = pd.concat([umap_context,context_df[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_context = PCA(n_components=3).fit_transform(scaled_context)
pca_context = pd.DataFrame(pca_context,columns = [f"PC_{i}" for i in range(3)])
pca_context = pd.concat([pca_context,context_df[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)
# %%
def plot_and_save_umap(df, label, output_location, prefix):
    ax = sns.scatterplot(
        data = df,
        x="UMAP_1",
        y="UMAP_2",
        hue=label,
        alpha=0.6,
    )
    sns.move_legend(ax,(1.01,0.5))
    plt.savefig(output_location / f"{prefix}_UMAP_{label}.pdf",format="pdf")
    plt.tight_layout()
    plt.close()
# %%
for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    plot_and_save_umap(umap_latents,label,out_tabular_data,"latents")
    plot_and_save_umap(umap_context,label,out_tabular_data,"context")
# %%
# Assuming your DataFrame is named df
def plot_3D_pca(
        df,
        output_location,
        labels,
        name,
        prefix,
        angle,
):
    # Replace this with your actual DataFrame
    # df = pd.DataFrame(data)

    # Create the 3D scatter plot
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    # Extract the relevant columns
    x = df["PC_0"]
    y = df["PC_1"]
    z = df["PC_2"]


    # Normalize labels to map to colormap
    norm = plt.Normalize(labels.min(), labels.max())
    colors = plt.cm.viridis(norm(labels))

    # Plot the points
    sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.5)

    # Add a colorbar
    #cbar = plt.colorbar(sc)
    #cbar.set_label('Labels')

    ax.view_init(elev=30, azim=angle)
    plt.savefig(Path(output_location) / f"{prefix}_3D_pca_{name}.pdf",format="pdf")
    plt.tight_layout()
# %%
for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    labels = np.max(np.array(pd.get_dummies(latents_df[label])) * np.arange(len(pca_latents[label].unique())),axis=1)
    plot_3D_pca(pca_latents,out_tabular_data,labels,label,"latents",angle= 70)
    plot_3D_pca(pca_context,out_tabular_data,labels,label,"context",angle = 20)
# %%

tp=0
print(tp)
latents_df_tp = latents_df[latents_df["Time"] == tp].reset_index()
scaled_latents = StandardScaler().fit_transform(latents[:,:,tp])

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_latents = PCA(n_components=3).fit_transform(scaled_latents)
pca_latents = pd.DataFrame(pca_latents,columns = [f"PC_{i}" for i in range(3)])
pca_latents = pd.concat([pca_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)


scaled_context = StandardScaler().fit_transform(context[:,:,tp])
context_df_tp = context_df[context_df["Time"] == tp].reset_index()
umap_context = UMAP(n_neighbors=10).fit_transform(scaled_context)
umap_context = pd.DataFrame(umap_context,columns=["UMAP_1","UMAP_2"])
umap_context = pd.concat([umap_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_context = PCA(n_components=3).fit_transform(scaled_context)
pca_context = pd.DataFrame(pca_context,columns = [f"PC_{i}" for i in range(3)])
pca_context = pd.concat([pca_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
    plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    labels = np.max(np.array(pd.get_dummies(pca_latents[label])) * np.arange(len(pca_latents[label].unique())),axis=1)
    plot_3D_pca(pca_latents,out_tabular_data,labels,label,f"latents_tp{tp}",angle= 70)
    plot_3D_pca(pca_context,out_tabular_data,labels,label,f"context_tp{tp}",angle=20)

# %%
plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")
# %%
tp=1
print(tp)
latents_df_tp = latents_df[latents_df["Time"] == tp].reset_index()
scaled_latents = StandardScaler().fit_transform(latents[:,:,tp])

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_latents = PCA(n_components=3).fit_transform(scaled_latents)
pca_latents = pd.DataFrame(pca_latents,columns = [f"PC_{i}" for i in range(3)])
pca_latents = pd.concat([pca_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)


scaled_context = StandardScaler().fit_transform(context[:,:,tp])
context_df_tp = context_df[context_df["Time"] == tp].reset_index()
umap_context = UMAP(n_neighbors=10).fit_transform(scaled_context)
umap_context = pd.DataFrame(umap_context,columns=["UMAP_1","UMAP_2"])
umap_context = pd.concat([umap_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_context = PCA(n_components=3).fit_transform(scaled_context)
pca_context = pd.DataFrame(pca_context,columns = [f"PC_{i}" for i in range(3)])
pca_context = pd.concat([pca_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
    plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    labels = np.max(np.array(pd.get_dummies(pca_latents[label])) * np.arange(len(pca_latents[label].unique())),axis=1)
    plot_3D_pca(pca_latents,out_tabular_data,labels,label,f"latents_tp{tp}",angle= 70)
    plot_3D_pca(pca_context,out_tabular_data,labels,label,f"context_tp{tp}",angle=20)

# %%
plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

# %%
tp=2
print(tp)
latents_df_tp = latents_df[latents_df["Time"] == tp].reset_index()
scaled_latents = StandardScaler().fit_transform(latents[:,:,tp])

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_latents = PCA(n_components=3).fit_transform(scaled_latents)
pca_latents = pd.DataFrame(pca_latents,columns = [f"PC_{i}" for i in range(3)])
pca_latents = pd.concat([pca_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)


scaled_context = StandardScaler().fit_transform(context[:,:,tp])
context_df_tp = context_df[context_df["Time"] == tp].reset_index()
umap_context = UMAP(n_neighbors=10).fit_transform(scaled_context)
umap_context = pd.DataFrame(umap_context,columns=["UMAP_1","UMAP_2"])
umap_context = pd.concat([umap_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_context = PCA(n_components=3).fit_transform(scaled_context)
pca_context = pd.DataFrame(pca_context,columns = [f"PC_{i}" for i in range(3)])
pca_context = pd.concat([pca_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
    plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    labels = np.max(np.array(pd.get_dummies(pca_latents[label])) * np.arange(len(pca_latents[label].unique())),axis=1)
    plot_3D_pca(pca_latents,out_tabular_data,labels,label,f"latents_tp{tp}",angle= 70)
    plot_3D_pca(pca_context,out_tabular_data,labels,label,f"context_tp{tp}",angle=20)

# %%
plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

# %%
tp=3
print(tp)
latents_df_tp = latents_df[latents_df["Time"] == tp].reset_index()
scaled_latents = StandardScaler().fit_transform(latents[:,:,tp])

umap_latents = UMAP(n_neighbors=10).fit_transform(scaled_latents)
umap_latents = pd.DataFrame(umap_latents,columns=["UMAP_1","UMAP_2"])
umap_latents = pd.concat([umap_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_latents = PCA(n_components=3).fit_transform(scaled_latents)
pca_latents = pd.DataFrame(pca_latents,columns = [f"PC_{i}" for i in range(3)])
pca_latents = pd.concat([pca_latents,latents_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)


scaled_context = StandardScaler().fit_transform(context[:,:,tp])
context_df_tp = context_df[context_df["Time"] == tp].reset_index()
umap_context = UMAP(n_neighbors=10).fit_transform(scaled_context)
umap_context = pd.DataFrame(umap_context,columns=["UMAP_1","UMAP_2"])
umap_context = pd.concat([umap_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

pca_context = PCA(n_components=3).fit_transform(scaled_context)
pca_context = pd.DataFrame(pca_context,columns = [f"PC_{i}" for i in range(3)])
pca_context = pd.concat([pca_context,context_df_tp[["Dev Outcome","Time","Axes","Unique Plate"]]],axis=1)

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
    plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")

for label in ["Dev Outcome","Time","Axes","Unique Plate"]:
    labels = np.max(np.array(pd.get_dummies(pca_latents[label])) * np.arange(len(pca_latents[label].unique())),axis=1)
    plot_3D_pca(pca_latents,out_tabular_data,labels,label,f"latents_tp{tp}",angle= 70)
    plot_3D_pca(pca_context,out_tabular_data,labels,label,f"context_tp{tp}",angle=20)

# %%
plot_and_save_umap(umap_latents,label,out_tabular_data,f"latents_tp{tp}")
plot_and_save_umap(umap_context,label,out_tabular_data,f"context_tp{tp}")
# %%
