"""
This script was used to train the pre-trained model weights that were given as an option during the exercise.
"""
import pandas as pd
from embed_time.dataloader_rs import LiveTLSDataset
from embed_time.model import VAE
from embed_time.UNet_based_encoder_decoder import UNetDecoder, UNetEncoder
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from tqdm import tqdm
from pathlib import Path
import os
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, SelectRandomTPNumpy, CustomCropCentroid
from embed_time.dataloader_rs import LiveTLSDataset
import subprocess
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x,x,reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD, MSE, KLD

def train(
        epoch, 
        model, 
        loss_fn, 
        optimizer, 
        train_loader,
        checkpoint_dir, 
        metadata=None,
        tb_logger=None,
        log_image_interval = 300,
    ):
    model.train()
    train_loss = 0
    losses = {
        "SUM":[],
        "MSE":[],
        "KLD":[],
    }
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss, mse_loss, KLD_loss = loss_fn(recon_batch, data, mu, log_var)
        loss.backward()

        for (key, value),loss_funcs in zip(losses.items(),[loss,mse_loss,KLD_loss]):
            value.append(loss_funcs.item())

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
                tb_logger.add_image(
                    tag="input/Training", 
                    img_tensor=data.to("cpu")[0,0], 
                    global_step=step,
                    dataformats="HW"
                )
                tb_logger.add_image(
                    tag="reconstruction/Training", 
                    img_tensor=recon_batch.to("cpu")[0,0], 
                    global_step=step,
                    dataformats="HW"
                )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return losses

def validate(
        epoch, 
        model, 
        loss_fn, 
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
    for batch_idx, (data, _) in enumerate(val_loader):
        data = data.cuda()
        recon_batch, mu, log_var = model(data)
        loss, mse_loss, KLD_loss = loss_fn(recon_batch, data, mu, log_var)

        for (key, value),loss_funcs in zip(losses.items(),[loss,mse_loss,KLD_loss]):
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
                tb_logger.add_image(
                    tag="input/Validation", 
                    img_tensor=data.to("cpu")[0,0], 
                    global_step=step,
                    dataformats="HW"
                )
                tb_logger.add_image(
                    tag="reconstruction/Validation", 
                    img_tensor=recon_batch.to("cpu")[0,0], 
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
    model_name = "UNEt_encdec_06_validation"
    checkpoint_dir = Path(base_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{model_name}_checkpoints"
    print(checkpoint_dir)

    checkpoint_dir.mkdir(exist_ok=True)
    data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"
    folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized'
    metadata = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'

    tensorboard_process = launch_tensorboard("unet_runs")
    logger = SummaryWriter(f"{base_dir}/{datetime.today().strftime('%Y-%m-%d')}_{model_name}")

    loading_transforms_wcrop = trans.Compose([
        SelectRandomTPNumpy(0),
        CustomCropCentroid(0,0,598),
        CustomToTensor(),
        v2.Resize((576,576)),
        v2.RandomAffine(
            degrees=90,
            translate=[0.1,0.1],
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),
    ])

    dataset_w_t = LiveTLSDataset(
        metadata,
        folder_imgs,
        metadata_columns=["Run","Plate","ID"],
        return_metadata=False,
        transform = loading_transforms_wcrop,
    )
    train_set, validation_set = torch.utils.data.random_split(dataset_w_t,[0.8,0.2])

    sample, label = dataset_w_t[0]
    in_channels, y, x = sample.shape
    print(in_channels)
    print((y,x))

    NUM_EPOCHS = 50
    n_fmaps = 32
    depth = 4
    z_dim = 20
    upsample_mode = "bicubic"
    model_dict = {
        'num_epochs': NUM_EPOCHS,
        'n_fmaps': n_fmaps,
        'depth': depth,
        'z_dim': z_dim,
        'upsample_mode':upsample_mode
    }
    encoder = UNetEncoder(
        in_channels = in_channels,
        n_fmaps = n_fmaps,
        depth = depth,
        in_spatial_shape = (y,x),
        z_dim = z_dim,
    )

    decoder = UNetDecoder(
        in_channels = in_channels,
        n_fmaps = n_fmaps,
        depth = depth,
        in_spatial_shape = (y,x),
        z_dim = z_dim,
        upsample_mode=upsample_mode
    )

    model = VAE(encoder, decoder)
    dataloader_train = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(validation_set, batch_size=8, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS): # train for one epoch, validate
        loss_epoch = train(
            epoch,
            model,
            loss_function,
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
