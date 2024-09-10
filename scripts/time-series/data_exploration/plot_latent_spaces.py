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
file = "UNet_VAE_01_old_normalisation.csv"
plot_outs = table_location / "UNet_VAE_01_old_normalisation"
plot_outs.mkdir(exist_ok=True)
dataframe = pd.read_csv(table_location / file)
annotation_columns = ["Label","Time","Axes","Run","Plate","ID"]
dataframe

# %%
scaled_input = StandardScaler().fit_transform(dataframe.drop(annotation_columns,axis=1))
components = 5
pca_transformer = PCA(n_components = components)
pca_out = pca_transformer.fit_transform(scaled_input)
pca_frame = pd.DataFrame(pca_out,columns=[f"PC_{i+1}" for i in range(components)])
pca_frame = pd.concat([pca_frame,dataframe[annotation_columns]],axis=1)
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
plt.savefig(plot_outs / "3D_pca_label_annotated",format="pdf")
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
plt.savefig(plot_outs / "3D_pca_time_annotated.dpf",format="pdf")
# %%
# Create the 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Extract the relevant columns
x = df["PC_1"]
y = df["PC_2"]
z = df["PC_3"]
labels = df["Axes"] == "single"

# Normalize labels to map to colormap
norm = plt.Normalize(labels.min(), labels.max())
colors = plt.cm.viridis(norm(labels))

# Plot the points
sc = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.5)

# Add a colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Axes')

ax.view_init(elev=60, azim=100)
plt.savefig(plot_outs / "3D_pca_axes_annotated.dpf",format="pdf")
# %%
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(scaled_input)

umap_df = pd.DataFrame(umap_out,columns=["UMAP_1","UMAP_2"])
umap_df = pd.concat([umap_df,dataframe[annotation_columns]])
# %%
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Label",alpha=0.5)
plt.savefig(plot_outs / "UMAP_label_annotated.dpf",format="pdf")
# %%
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Time",alpha=0.5,palette="viridis")
plt.savefig(plot_outs / "UMAP_time_annotated.dpf",format="pdf")
# %%
sns.scatterplot(umap_df,x="UMAP_1",y="UMAP_2",hue="Axes",alpha=0.5)
plt.savefig(plot_outs / "UMAP_axes_annotated.dpf",format="pdf")
# %%
