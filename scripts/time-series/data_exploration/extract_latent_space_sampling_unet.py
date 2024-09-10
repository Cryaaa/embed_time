import torch
from embed_time.dataloader_rs import LiveTLSDataset
from embed_time.model import VAE
from embed_time.UNet_based_encoder_decoder import UNetDecoder, UNetEncoder
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, SelectSpecificTPNumpy, CustomCropCentroid
from embed_time.dataloader_rs import LiveTLSDataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
checkpoint_dir = Path(base_dir) / '2024-09-01_unet_encdec_beta_norm_01_checkpoints'
checkpoint_dir.mkdir(exist_ok=True)
data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"
folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized'
anotations = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'
checkpoint_epoch = 'checkpoint_99.pth'
batch_size = 2
metadata_columns=["Axes","Run","Plate","ID"]
tabular_data = Path("/mnt/efs/dlmbl/G-et/tabular_data")
table_name = "UNet_VAE_02_new_normalisation.csv"
input_spatial_dim = (576,576)
in_channels = 2

def load_unet_from_checkpoint(
        checkpoint_dir,
        checkpoint_name,
):
    dict = torch.load(
        Path(checkpoint_dir)/checkpoint_name
    )
    model_params = dict["metadata"]

    encoder = UNetEncoder(
        in_channels = in_channels,
        n_fmaps = model_params["n_fmaps"],
        depth = model_params["depth"],
        in_spatial_shape = input_spatial_dim,
        z_dim = model_params["z_dim"],
    )

    decoder = UNetDecoder(
        in_channels = in_channels,
        n_fmaps = model_params["n_fmaps"],
        depth = model_params["depth"],
        in_spatial_shape = input_spatial_dim,
        z_dim = model_params["z_dim"],
        upsample_mode="bicubic"
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = VAE(encoder, decoder)
    model.to(device)
    model.load_state_dict(dict['model'])
    model.eval()
    return model

def sample_latent_space_per_timepoint(
        model: VAE,
        timepoint_loaders,
        metadata = False,
        n_samples_per_tp = 5,
        n_tp = 4,
        metadata_columns =["Run","Plate","ID"],
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
            _, mean, var = model(image.to(device))
            for j in range(n_samples_per_tp):
                z = model.sampling(mean,var)
                latents.append(z.detach().cpu())
                labels.append(label)
                timepoints.append([tp for _ in range(len(z))])
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
    model = load_unet_from_checkpoint(
        checkpoint_dir,
        checkpoint_epoch,
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
        )
    
    out_df.to_csv(tabular_data / table_name)

    print("Done!")
