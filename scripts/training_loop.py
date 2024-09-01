#%%
import os
from embed_time.splitter_static import DatasetSplitter
from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from embed_time.model import Encoder, Decoder, VAE
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#%% Generate Dataset

# Usage example:
parent_dir = '/mnt/efs/dlmbl/S-md/'
output_path = '/mnt/efs/dlmbl/G-et/training_logs/'
output_file = csv_file = output_path + 'example_split.csv'
model_name = "static_vanilla_vae"
run_name= "initial_params"
train_ratio = 0.7
val_ratio = 0.15
num_workers = -1
#change to false if you already have tensorboard running
find_port = True
#%% Define the logger for tensorboard
# Function to find an available port
def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process

# Launch tensorboard and click on the link to view the logs.
if find_port:
    tensorboard_process = launch_tensorboard("embed_time_static_runs")

logger = SummaryWriter(f"embed_time_static_runs/{model_name}")

# Create the dataset split CSV file
DatasetSplitter(parent_dir, output_file, train_ratio, val_ratio, num_workers).generate_split()

split = 'train'
channels = [0, 1, 2, 3]
cell_cycle_stages = 'interphase'
transform = "masks"
crop_size = 100

# Create the dataset
dataset = ZarrCellDataset(parent_dir, csv_file, split, channels, transform, crop_size)

#%% Generate Dataloader

# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_wrapper(metadata_keys, images_keys)
)


#%% Create the model

encoder = Encoder(input_shape=(100, 100),
                  x_dim=4,
                  h_dim1=16,
                  h_dim2=8,
                  z_dim=4)
decoder = Decoder(z_dim=4,
                  h_dim1=8,
                  h_dim2=16,
                  x_dim=4,
                  output_shape=(100, 100))

# Initiate VAE
vae = VAE(encoder, decoder).to(device)

#%% Define Optimizar
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD   




#%% Define training function
training_log = []
epoch_log = []
loss_per_epoch = 0
def train(
    epoch,
    model = vae,
    loader = dataloader,
    optimizer = optimizer,
    loss_function = loss_function,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=device,
    early_stop=False,
    training_log = training_log,
    epoch_log = epoch_log,
    loss_per_epoch = loss_per_epoch
    ):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch['cell_image'].to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        BCE, KLD  = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD  
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        loss_per_epoch = train_loss / len(dataloader.dataset)
        
        # log to console
        if batch_idx % 5 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )
            
        if batch_idx % log_interval == 0:
            row = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'len_data': len(batch['cell_image']),
                'len_dataset': len(loader.dataset),
                'loss': loss.item() / len(batch['cell_image']),
                'BCE': BCE.item() / len(batch['cell_image']),
                'KLD': KLD.item() / len(batch['cell_image'])
            }
            training_log.append(row)



        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_idx
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                input_image = data.to("cpu").detach()       
                predicted_image = recon_batch.to("cpu").detach()

                tb_logger.add_images(
                    tag="input_channel_0", img_tensor=input_image[:,0:1,...], global_step=step
                )
                tb_logger.add_images(
                    tag= "reconstruction_0", img_tensor=predicted_image[:,0:1,...], global_step=step
                )
                
                tb_logger.add_images(
                    tag="input_1", img_tensor=input_image[:,1:2,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_1", img_tensor=predicted_image[:,1:2,...], global_step=step
                )

                tb_logger.add_images(
                    tag="input_2", img_tensor=input_image[:,2:3,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_2", img_tensor=predicted_image[:,2:3,...], global_step=step
                )

                tb_logger.add_images(
                    tag="input_3", img_tensor=input_image[:,3:4,...], global_step=step
                )
                tb_logger.add_images(
                    tag="reconstruction_3", img_tensor=predicted_image[:,3:4,...], global_step=step
                )

                
                metadata = list(zip(batch['gene'], batch['barcode'], batch['stage']))
                tb_logger.add_embedding(
                    torch.rand_like(mu), metadata=metadata, label_img = input_image[:,2:3,...],global_step=step
                )
               
                # TODO saving model
            
        # early stopping
        if early_stop and batch_idx > 5:
            print("Stopping test early!")
            break
    
    # save the DF

    epoch_raw = {
                'epoch': epoch,
                'Average Loss': train_loss / len(dataloader.dataset)}
    epoch_log.append(epoch_raw)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))

#%% Training loop

folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
checkpoint_path = output_path + "checkpoints/static/" + folder_suffix + "/"
log_path = output_path + "logs/static/"+ folder_suffix + "/"
for epoch in range(1, 10):
    train(epoch, log_interval=100, log_image_interval=20, tb_logger=logger)
    filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_") + "epoch_"+str(epoch) + "_"
    training_logDF = pd.DataFrame(training_log)
    training_logDF.to_csv(log_path + filename_suffix+"training_log.csv", index=False)

    epoch_logDF = pd.DataFrame(epoch_log)
    epoch_logDF.to_csv(log_path + filename_suffix+"epoch_log.csv", index=False)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_per_epoch 
    }
    torch.save(checkpoint, output_path + filename_suffix + str(epoch) + "checkpoint.pth")
