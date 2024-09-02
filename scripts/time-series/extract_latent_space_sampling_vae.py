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
from embed_time.transforms import SelectSpecificTPNumpy
from embed_time.model_VAE_resnet18 import VAEResNet18
from embed_time.model_VAE_resnet18_linear_botlnek import VAEResNet18LinBotNek
from tqdm.auto import tqdm

base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
checkpoint_dir = Path(base_dir) / "2024-09-02_Resnet18_LinearVAE_01_bicubic_checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"
folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized'
anotations = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'
checkpoint_epoch = 'checkpoint_99.pth'
batch_size = 2
metadata_columns=["Axes","Run","Plate","ID"]
do_latent_flattening = False
tabular_data = Path("/mnt/efs/dlmbl/G-et/tabular_data")
table_name = "LinearVAE_01_bicubic_latents_w_annot.csv"


def load_vae_from_checkpoint(
        checkpoint_dir,
        checkpoint_name,
        linear=True
):
    dict = torch.load(
        Path(checkpoint_dir)/checkpoint_name
    )
    model_params = dict["metadata"]

    if linear:
        model = VAEResNet18LinBotNek(
            nc=model_params['in_channels'],
            z_dim=model_params['z_dim'],
            input_spatial_dim=(576,576)
        )
    else:
        model = VAEResNet18(
            nc=model_params['in_channels'],
            z_dim=model_params['z_dim']
        )
    model_params = dict['model']
    model.load_state_dict(model_params)
    return model

def sample_latent_space_per_timepoint(
        model,
        timepoint_loaders,
        metadata = False,
        n_samples_per_tp = 5,
        n_tp = 4,
        metadata_columns =["Run","Plate","ID"],
        flatten_latents = True,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    latents = []
    labels = []
    timepoints = []
    metadatas = []
    for tp in tqdm(range(n_tp),total = n_tp):
        
            for i,first in enumerate(timepoint_loaders[tp]):
                if metadata:
                    image, label, metdat = first
                    
                else: 
                    image, label = first
                _, z, mean, var = model(image.to(device))
                for j in range(n_samples_per_tp):
                    z = model.reparameterize(mean,var)
                    if flatten_latents:
                        latents.append(torch.flatten(z.detach().cpu(),start_dim=1))
                    else:
                        latents.append(z.detach().cpu())
                    labels.append(label)
                    timepoints.append([tp for i in range(len(z))])
                    metadatas.append(metdat)
    
    latents = np.concat(np.array(latents),axis=0)
    timepoints = np.concat(np.array(timepoints),axis=0)
    labels = np.concat(np.array(labels),axis=0)
    metadata_df = {}
    for i in range(len(metadata_columns)):
        metadata_df[metadata_columns[i]] = np.concatenate(
            np.array(metadatas)[:,i,:]
        )
    metadata_df=pd.DataFrame(
        metadata_df
    )
    out_df = pd.concat(
        [
            pd.DataFrame(
                latents,
                columns = [
                    f"LD_{i}" for i in range(latents.shape[-1])
                ]
            ),
            pd.DataFrame(
                {"Time":timepoints}
            ),
            pd.DataFrame(
                {"Label":labels}
            ),
            metadata_df
        ],axis=1
    )
    return out_df

if __name__ == "__main__":
    model = load_vae_from_checkpoint(
        checkpoint_dir,
        checkpoint_epoch,
        linear=not(do_latent_flattening)
    )

    time_specific_datasets = [
        LiveTLSDataset(
            anotations,
            folder_imgs,
            metadata_columns=metadata_columns,
            return_metadata=True,
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
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=batch_size
        ) for ds in time_specific_datasets
    ]

    out_df = sample_latent_space_per_timepoint(
        model=model,
        timepoint_loaders=time_specific_dataloader,
        n_samples_per_tp=5,
        n_tp=4,
        metadata=not(metadata_columns is None),
        metadata_columns=metadata_columns,
        flatten_latents = do_latent_flattening,
        )
    
    out_df.to_csv(tabular_data / table_name)

    print("Done!")
