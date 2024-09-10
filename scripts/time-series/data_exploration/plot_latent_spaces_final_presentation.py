# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import StandardScaler
table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
# UNet_VAE_01_old_normalisation
# UNet_VAE_02_new_normalisation
# "Resnet18_VAE_03_old_normalisation.csv", 
# "Resnet18_VAE_04_new_normalisation.csv", 
# LinearVAE_01_bicubic_latents_w_annot.csv, 
# LinearVAE_02_bicubic_latents_w_annot.csv

file = "LinearVAE_02_bicubic_latents_w_annot.csv"
plot_outs = table_location / "LinearVAE_02_bicubic_latents"
plot_outs.mkdir(exist_ok=True)
dataframe = pd.read_csv(table_location / file)
annotation_columns = ["Label","Time","Axes","Run","Plate","ID"]
dataframe

scaled_input = StandardScaler().fit_transform(dataframe.drop(annotation_columns,axis=1))
components = 5
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(scaled_input)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame = pd.concat([pca_frame,dataframe[annotation_columns]],axis=1)

# Assuming your DataFrame is named df
def plot_3D_pca(
        df,
        output_location,
        labels,
        name
):
    # Replace this with your actual DataFrame
    # df = pd.DataFrame(data)

    # Create the 3D scatter plot
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    # Extract the relevant columns
    x = df["PC_1"]
    y = df["PC_2"]
    z = df["PC_3"]


    # Normalize labels to map to colormap
    norm = plt.Normalize(labels.min(), labels.max())
    colors = plt.cm.viridis(norm(labels))

    # Plot the points
    sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.5)

    # Add a colorbar
    #cbar = plt.colorbar(sc)
    #cbar.set_label('Labels')

    ax.view_init(elev=30, azim=70)
    plt.savefig(Path(output_location) / f"3D_pca_{name}.pdf",format="pdf")
# %%
for name, label in zip(
    ["developmental_outcome","time","axis_number"],
    [pca_frame["Label"] == "good",pca_frame["Time"],pca_frame["Axes"] == "single"]
):
    plot_3D_pca(pca_frame,plot_outs,label,name)
# %%
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(scaled_input)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
umap_df = pd.concat([umap_df,dataframe[annotation_columns]],axis=1)

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
plt.savefig(plot_outs / "UMAP_label_annotated.pdf",format="pdf")
plt.close()

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Time",alpha=0.5,palette="viridis")
plt.savefig(plot_outs / "UMAP_time_annotated.pdf",format="pdf")
plt.close()

sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Axes",alpha=0.5)
plt.savefig(plot_outs / "UMAP_axes_annotated.pdf",format="pdf")
plt.close()
# %%
scaler = StandardScaler()
from tqdm.auto import tqdm
for tp in tqdm(range(4)):
    df_tp = dataframe[dataframe["Time"]==tp]
    scaled_tp = scaler.fit_transform(
        df_tp.drop(annotation_columns,axis=1)
    )
    pca_tp = (pca_transformer.fit_transform(scaled_tp))
    pca_tp = pd.DataFrame(pca_tp,columns=[f"PC_{i+1}" for i in range(components)])
    pca_tp = pd.concat([pca_tp,df_tp[annotation_columns].reset_index()],axis=1)

    for name, label in zip(
        [f"developmental_outcome_tp{tp}",f"axis_number_tp{tp}"],
        [pca_tp["Label"] == "good",pca_tp["Axes"] == "single"]
    ):
        plot_3D_pca(pca_tp,plot_outs,label,name)

# %%
scaler = StandardScaler()
from tqdm.auto import tqdm
for tp in tqdm(range(4)):
    df_tp = dataframe[dataframe["Time"]==tp]
    scaled_tp = scaler.fit_transform(
        df_tp.drop(annotation_columns,axis=1)
    )

    umap_tp = umap_transformer.fit_transform(scaled_tp)

    umap_tp = pd.DataFrame(umap_tp,columns=["UMAP_1","UMAP_2"])
    umap_tp = pd.concat([umap_tp,df_tp[annotation_columns].reset_index()],axis=1,)

    sns.scatterplot(umap_tp,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
    plt.savefig(plot_outs / f"UMAP_label_annotated_tp{tp}.pdf",format="pdf")
    plt.close()

    sns.scatterplot(umap_tp,x="UMAP_1",y="UMAP_2",hue="Axes",alpha=0.5)
    plt.savefig(plot_outs / f"UMAP_axes_annotated_tp{tp}.pdf",format="pdf")
    plt.close()

# %%
plot_outs
# %%
