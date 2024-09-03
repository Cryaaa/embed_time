#%%
import os
from embed_time.splitter_static import DatasetSplitter
from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from embed_time.model_VAE_resnet18 import VAEResNet18
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn import utils as U
from torch import optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#%% Generate Dataset

# Usage example:
parent_dir = '/mnt/efs/dlmbl/S-md/'
output_path = '/mnt/efs/dlmbl/G-et/training_logs/'
output_file = csv_file = output_path + 'example_split.csv'
beta = 1e-4
lr = 1e-3
z_dim = 20
model_name = "resnet18_vae_conv2D"
run_name= "z_dim-"+str(z_dim)+"_lr-"+str(lr)+"_beta-"+str(beta)
train_ratio = 0.7
val_ratio = 0.15
num_workers = 8
#change to false if you already have tensorboard running
find_port = True

#%%read config
def read_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract 'Dataset mean' and 'Dataset std' from the config
    mean = config['Dataset mean'][0]  # Access the first (and only) element of the list
    std = config['Dataset std'][0]

    # Split the strings and convert to floats
    mean = [float(i) for i in mean.split()]
    std = [float(i) for i in std.split()]
    
    # Convert to ndarrays
    mean = np.array(mean)
    std = np.array(std)

    return mean, std
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

logger = SummaryWriter(f"embed_time_static_runs/{run_name}")

# Create the dataset split CSV file
csv_file = '/mnt/efs/dlmbl/G-et/csv/dataset_split_2.csv'
split = 'train'
channels = [0, 1, 2, 3]
transform = "masks"
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
dataset_mean, dataset_std = read_config(yaml_file_path)

# Create the dataset
dataset = ZarrCellDataset(parent_dir, csv_file, split, channels, transform, normalizations, None, dataset_mean, dataset_std)

#%% Generate Dataloader

# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']

# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_wrapper(metadata_keys, images_keys),
    num_workers=num_workers
)


#%% Create the model

# Initiate VAE-ResNet18 model
vae = VAEResNet18(nc = 4, z_dim = z_dim ).to(device)

#%% Define Optimizar
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

#%% Define loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD   




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
    beta=beta,
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
        MSE, KLD  = loss_function(recon_batch, data, mu, logvar)
        loss = MSE + beta*KLD  
        
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                'mu': mu,
                'logvar': logvar,
                'loss': loss.item() / len(batch['cell_image']),
                'MSE': MSE.item() / len(batch['cell_image']),
                'KLD': KLD.item() / len(batch['cell_image'])
            }
            training_log.append(row)



        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_idx
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="MSE_loss", scalar_value=MSE.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="KLD_loss", scalar_value=KLD.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                input_image = data.to("cpu").detach()       
                predicted_image = recon_batch.to("cpu").detach()

                tb_logger.add_image(
                    tag="input_0", img_tensor=input_image[0,0,...], global_step=step
                )
                tb_logger.add_image(
                    tag= "reconstruction_0", img_tensor=predicted_image[0,0,...], global_step=step
                )
                
                tb_logger.add_image(
                    tag="input_1", img_tensor=input_image[0,1,...], global_step=step
                )
                tb_logger.add_image(
                    tag="reconstruction_1", img_tensor=predicted_image[0,1,...], global_step=step
                )

                tb_logger.add_image(
                    tag="input_2", img_tensor=input_image[0,2,...], global_step=step
                )
                tb_logger.add_image(
                    tag="reconstruction_2", img_tensor=predicted_image[0,2,...], global_step=step
                )

                tb_logger.add_image(
                    tag="input_3", img_tensor=input_image[0,3,...], global_step=step
                )
                tb_logger.add_image(
                    tag="reconstruction_3", img_tensor=predicted_image[0,3,...], global_step=step
                )

                
                metadata = [list(item) for item in zip(*[batch[key] for key in metadata_keys])]
                tb_logger.add_embedding(
                    torch.flatten(mu, start_dim=1), metadata=metadata, label_img = input_image[:,2:3,...], global_step=step, metadata_header = metadata_keys
                )
               
        
            
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
    return train_loss/len(dataloader.dataset)
#%% Training loop

#define the folder path for saving checkpoints and logs
folder_suffix = datetime.now().strftime("%Y%m%d") + run_name
checkpoint_path = '/mnt/efs/dlmbl/G-et/checkpoints/static/Akila/' + folder_suffix + "/"
os.makedirs(checkpoint_path, exist_ok=True)
log_path = '/mnt/efs/dlmbl/G-et/logs/static/Akila/'+ folder_suffix + "/"
os.makedirs(log_path, exist_ok=True)

#training loop
for epoch in range(0, 100):
    train_loss =train(epoch, beta = beta, log_interval=100, log_image_interval=20, tb_logger=logger)

    

    train_path = log_path + "_epoch_"+str(epoch)+"/"
    os.makedirs(train_path, exist_ok=True)

    training_logDF = pd.DataFrame(training_log)
    training_logDF.to_csv(train_path+"epoch_log.csv", index=False)

    
    
    epoch_logDF = pd.DataFrame(epoch_log)
    epoch_logDF.to_csv(train_path+"epoch_summary_log.csv", index=False)

  
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss 
    }
    torch.save(checkpoint, checkpoint_path+"epoch_"+str(epoch)+"_checkpoint.pth")
