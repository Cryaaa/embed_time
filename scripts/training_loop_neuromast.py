import os

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
from datasets.neuromast import NeuromastDatasetTrain
from torchview import draw_graph

beta = 1e-7
lr = 1e-4
z_dim = 22
model_name = "neuromast_resnet18_vae_conv2D"
run_name= "z_dim-"+str(z_dim)+"_lr-"+str(lr)+"_beta-"+str(beta)
metadata = pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_balanced_train.csv")
find_port = True

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#launch tensorboard

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

#%% Generate Dataset
Train_dataset = NeuromastDatasetTrain()

#dataloader
train_loader = DataLoader(Train_dataset, batch_size=2, shuffle=True, num_workers=8)

# Initiate VAE-ResNet18 model
vae = VAEResNet18(nc = 1, z_dim = z_dim ).to(device)

#%% Define Optimizar
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)

#%% Define loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD   
import torch
from torchviz import make_dot
import torch.nn.functional as F



#%% Define training function
training_log = []
epoch_log = []
loss_per_epoch = 0
def train(
    epoch,
    model = vae,
    loader =train_loader,
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
    for batch_idx, (batch,label) in enumerate(train_loader):
        data = batch.to(device)
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
                'len_data': len(batch),
                'len_dataset': len(loader.dataset),
                'loss': loss.item() / len(batch),
                'MSE': MSE.item() / len(batch),
                'KLD': KLD.item() / len(batch)
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
                    tag="input_0", img_tensor=input_image[0:1,0,...], global_step=step
                )
                tb_logger.add_image(
                    tag= "reconstruction_0", img_tensor=predicted_image[0:1,0,...], global_step=step
                )
                
                # tb_logger.add_embedding(
                #     torch.flatten(mu, start_dim=1), metadata=label[0:1], label_img = input_image[0:1,...], global_step=step
                # )
               
        


    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset)
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

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss 
    }
    save_path = checkpoint_path + "_epoch_"+str(epoch)+"/"
    os.makedirs(save_path, exist_ok=True)

    torch.save(checkpoint, save_path+"checkpoint.pth")

