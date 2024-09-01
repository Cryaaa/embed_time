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


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#%% Generate Dataset

# Usage example:
parent_dir = '/mnt/efs/dlmbl/S-md/'
output_path = '/home/S-ab/embed_time/training_logs/'
output_file = csv_file = output_path + 'example_split.csv'
train_ratio = 0.7
val_ratio = 0.15
num_workers = -1

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
tensorboard_process = launch_tensorboard("embed_time_runs")
model_name = "First Try"
logger = SummaryWriter(f"embed_time_runs/{model_name}")

#%% Define training function
training_log = []
epoch_log = []

def train(
    epoch,
    model = vae,
    training_log = training_log,
    epoch_log = epoch_log,
    loader = dataloader,
    optimizer = optimizer,
    loss_function = loss_function,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=device,
    early_stop=False,
    ):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch['cell_image'].to(device)
        # print(data.shape)
        # print("min",data.min(), data.max(), data.mean())
        # label = 'gene'
        # metadata=list(batch[label])
        
        # print(metadata)
        # input('jhhhj')
        # zero the gradients for this iteration
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        BCE, KLD  = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD  
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
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
                # top = torch.hstack((input_image[:,0,...], input_image[:,1,...]))
                # bottom = torch.hstack((input_image[:,2,...], input_image[:,3,...]))  # Combine a and b horizontally
                # input_image = torch.vstack((top, bottom))                 
                tb_logger.add_images(
                    tag="input0", img_tensor=input_image[:,0:1,...], global_step=step
                )
                tb_logger.add_images(
                    tag="input1", img_tensor=input_image[:,1:2,...], global_step=step
                )
                tb_logger.add_images(
                    tag="input2", img_tensor=input_image[:,2:3,...], global_step=step
                )
                tb_logger.add_images(
                    tag="input3", img_tensor=input_image[:,3:4,...], global_step=step
                )
                label = 'gene'
                tb_logger.add_embedding(
                    torch.rand_like(mu), metadata=list(batch[label]), label_img = input_image[:,2:3,...],global_step=step
                )
                # torch.rand_like(mu)
                # TODO saving model
                ############################
            
                # train_loss = 0
                # writer.add_scalar("Loss/train", loss.item(), i)
                # 3D version
                # central_slice = recon_batch.shape[2] // 2
                # writer.add_images("Input", data[:, :, central_slice], i)
                # writer.add_images(
                #     "Reconstructions", recon_batch[:, :, central_slice], i
                # )
                # 2D version
                # writer.add_images("Input", data, i)
                # writer.add_images("Reconstructions", recon_batch, i)
                # Both versions
                # print(mu.shape)
                # print(batch[label].data.shape)
                # tb_logger.add_embedding(
                #     mu, metadata=batch[label].data.squeeze().tolist(), global_step=step
                # )

                # tb_logger.add_images(
                #     tag="reconstruction",
                #     img_tensor=torch.cat([recon_batch.to("cpu").detach()[:,i] for i in range(4)],dim=2),
                #     global_step=step,
                # )
                # combined_image = torch.cat(
                #     [data, recon_batch, data.size()], # pad_to_size() if we change pading of the model to valid
                #     dim=4,
                # )

                # tb_logger.add_images(
                #     tag="input_target_prediction",
                #     img_tensor=combined_image,
                #     global_step=step,
                # )

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

for epoch in range(1, 10):
    train(epoch, log_interval=100, log_image_interval=20, tb_logger=logger)

    training_logDF = pd.DataFrame(training_log)
    training_logDF.to_csv(output_path + "training_log.csv", index=False)

    epoch_logDF = pd.DataFrame(epoch_log)
    epoch_logDF.to_csv(output_path + "epoch_log.csv", index=False)
