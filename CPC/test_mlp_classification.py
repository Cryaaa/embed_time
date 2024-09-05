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
from .models.mlp import MLP
from models.nn import ShiftedConv
from models.convnext import ConvNeXt
import cpc
from pathlib import Path

table_location = Path("/mnt/efs/dlmbl/G-et/tabular_data")
version = 1

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
parameters = torch.load(checkpoint_dir/"metadata_training.pt")
latent_dims =parameters['latent_dims']


# %%
class DatasetLatents(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, ):
        self.df = dataframe


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image,# label


model = MLP(
    n_dims = latent_dims,
    n_classes = 2,
    n_hidden_layers= 3,
    n_hidden_dims=512
)

