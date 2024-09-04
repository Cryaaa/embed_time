"""
This script was used to train the pre-trained model weights that were given as an option during the exercise.
"""
from pathlib import Path
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torchvision.transforms as trans
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter

from embed_time.dataloader_ij import LiveGastruloidDataset
from embed_time.model_VAE_resnet18 import VAEResNet18
from embed_time.transforms_ij import CustomToTensor, ShiftIntensity, SelectRandomTPNumpy

from tqdm import tqdm
import subprocess
from datetime import datetime

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var, beta_vae):
    MSE = F.mse_loss(recon_x, x[:, 1], reduction='mean')
    KLD = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) * beta_vae
    return MSE + KLD, MSE, KLD

def train(
        epoch, 
        model, 
        loss_fn,
        beta_vae,
        optimizer, 
        train_loader,
        checkpoint_dir, 
        metadata=None,
        tb_logger=None,
        log_image_interval = 50,
    ):
    model.train()
    train_loss = 0
    losses = {
        "SUM":[],
        "MSE":[],
        "KLD":[],
    }
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda().float()
        optimizer.zero_grad()
        
        recon_batch, z, mu, log_var = model(data)
        loss, mse_loss, KLD_loss = loss_fn(recon_batch[:, 1], # only calculate loss from SOX2
                                           data,
                                           mu,
                                           log_var,
                                           beta_vae)
        loss.backward()

        for (key, value), loss_funcs in zip(losses.items(),[loss,mse_loss,KLD_loss]):
            value.append(loss_funcs.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
        
         # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(train_loader) + batch_idx
            tb_logger.add_scalar(
                tag="sum_loss/Training", scalar_value=loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="mse_loss/Training", scalar_value=mse_loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="kld_loss/Training", scalar_value=KLD_loss.item(), global_step=step
            )

            # check if we log images in this iteration
            if step % log_image_interval == 0:
                ##print(data.to("cpu")[:,0].shape)
                #print(recon_batch.to("cpu")[:,0].shape)

                # tb_logger.add_image(
                #     tag="input_ch1/Training", 
                #     img_tensor=data.to("cpu")[0,0], 
                #     global_step=step,
                #     dataformats="HW"
                # )
                # tb_logger.add_image(
                #     tag="reconstruction_ch1/Training", 
                #     img_tensor=recon_batch.to("cpu")[0,0], 
                #     global_step=step,
                #     dataformats="HW"
                # )

                tb_logger.add_image(
                    tag="input_ch2/Training", 
                    img_tensor=data.to("cpu")[0,1], 
                    global_step=step,
                    dataformats="HW"
                )
                tb_logger.add_image(
                    tag="reconstruction_ch2/Training", 
                    img_tensor=recon_batch.to("cpu")[0,1], 
                    global_step=step,
                    dataformats="HW"
                )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return losses

def validate(
        epoch, 
        model, 
        loss_fn,
        beta_vae,
        optimizer, 
        val_loader,
        checkpoint_dir, 
        metadata=None,
        tb_logger=None,
        log_image_interval = 10,
    ):
    model.eval()
    validation_loss = 0
    losses = {
        "SUM":[],
        "MSE":[],
        "KLD":[],
    }
    for batch_idx, data in enumerate(val_loader):
        data = data.cuda().float()
        recon_batch, z, mu, log_var = model(data)
        loss, mse_loss, KLD_loss = loss_fn(recon_batch[:, 1],
                                           data,
                                           mu,
                                           log_var,
                                           beta_vae)

        for (key, value), loss_funcs in zip(losses.items(),[loss,mse_loss,KLD_loss]):
            value.append(loss_funcs.item())
        validation_loss += loss.item()
        
         # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(val_loader) + batch_idx
            tb_logger.add_scalar(
                tag="sum_loss/Validation", scalar_value=loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="mse_loss/Validation", scalar_value=mse_loss.item(), global_step=step
            )
            tb_logger.add_scalar(
                tag="kld_loss/Validation", scalar_value=KLD_loss.item(), global_step=step
            )

            # check if we log images in this iteration
            if step % log_image_interval == 0:
                ##print(data.to("cpu")[:,0].shape)
                #print(recon_batch.to("cpu")[:,0].shape)

                # tb_logger.add_image(
                #     tag="input_ch1/Validation", 
                #     img_tensor=data.to("cpu")[0,0], 
                #     global_step=step,
                #     dataformats="HW"
                # )
                # tb_logger.add_image(
                #     tag="reconstruction_ch1/Validation", 
                #     img_tensor=recon_batch.to("cpu")[0,0], 
                #     global_step=step,
                #     dataformats="HW"
                # )

                tb_logger.add_image(
                    tag="input_ch2/Validation", 
                    img_tensor=data.to("cpu")[0,1], 
                    global_step=step,
                    dataformats="HW"
                )
                tb_logger.add_image(
                    tag="reconstruction_ch2/Validation", 
                    img_tensor=recon_batch.to("cpu")[0,1], 
                    global_step=step,
                    dataformats="HW"
                )

    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(epoch, validation_loss / len(val_loader.dataset)))

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata
        },
        checkpoint_dir / f"checkpoint_{epoch}.pth",
    )

    return losses

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


if __name__ == "__main__":
    base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
    model_name = "Resnet18_VAE_norm_+aug_sox2only_03_ij"
    checkpoint_dir = Path(base_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{model_name}_checkpoints"
    print(checkpoint_dir)

    checkpoint_dir.mkdir(exist_ok=True)
    folder_imgs = r"/mnt/efs/dlmbl/G-et/data/live_gastruloid/240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis/Individual Raft Images Norm/"

    tensorboard_process = launch_tensorboard("resnet_runs")
    logger = SummaryWriter(f"{base_dir}/{datetime.today().strftime('%Y-%m-%d')}_{model_name}")

    loading_transforms = trans.Compose([
        SelectRandomTPNumpy(0),
        CustomToTensor(),
        v2.Resize((336,336)),
        ShiftIntensity(bf_factor=2),
        v2.RandomAffine(
            degrees=90,
            translate=[0.1,0.1],
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.GaussianBlur(kernel_size=15, sigma=(0.1,20.0)),
    ])

    dataset_w_t = LiveGastruloidDataset(
        img_dir = folder_imgs,
        transform = loading_transforms,
    )
    train_set, validation_set = torch.utils.data.random_split(dataset_w_t,[0.8,0.2])

    sample = dataset_w_t[0]
    in_channels, y, x = sample.shape
    print(in_channels)
    print((y,x))

    NUM_EPOCHS = 100
    z_dim = 20
    batch_size = 5
    beta_vae = 1e-7
    model_dict = {
        'num_epochs': NUM_EPOCHS,
        'in_channels': in_channels,
        'z_dim': z_dim,
        'batch_size': batch_size,
        'beta_vae': beta_vae,
    }
    
    model = VAEResNet18(nc=in_channels, z_dim=z_dim)
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # print(device)
    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS): # train for one epoch, validate
        loss_epoch = train(
            epoch,
            model,
            loss_function,
            beta_vae,
            optimizer,
            dataloader_train,
            checkpoint_dir=checkpoint_dir,
            metadata=model_dict,
            tb_logger=logger)
        train_losses.append(loss_epoch)
        
        loss_epoch = validate(
            epoch,
            model,
            loss_function,
            beta_vae,
            optimizer,
            dataloader_val,
            checkpoint_dir=checkpoint_dir,
            metadata=model_dict,
            tb_logger=logger)
        val_losses.append(loss_epoch)

    # TODO SAVE LOSSES SOMEWHERE
    train_loss_df = []
    for epoch,loss in enumerate(train_losses):
        loss_df = pd.DataFrame(loss)
        loss_df["Epoch"] = [epoch for _ in range(len(loss_df))]
        train_loss_df.append(loss_df)
        
    # TODO SAVE LOSSES SOMEWHERE
    val_loss_df = []
    for epoch,loss in enumerate(val_losses):
        loss_df = pd.DataFrame(loss)
        loss_df["Epoch"] = [epoch for _ in range(len(loss_df))]
        val_loss_df.append(loss_df)

    total_train_loss = pd.concat(train_loss_df,axis=0,ignore_index=True)
    total_train_loss.to_csv(checkpoint_dir / "train losses.csv")

    total_val_loss = pd.concat(val_loss_df,axis=0,ignore_index=True)
    total_val_loss.to_csv(checkpoint_dir / "validation losses.csv")
