"""
This script was used to train the pre-trained model weights that were given as an option during the exercise.
"""

from embed_time.dataloader_rs import LiveTLSDataset
from embed_time.model import Encoder, Decoder, VAE
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
import os
import skimage.io as io
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, SelectRandomTimepoint
from embed_time.dataloader_rs import LiveTLSDataset


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x,x,reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD

def train(epoch, model, loss_fn, optimizer, train_loader,checkpoint_dir):
    model.train()
    train_loss = 0
    losses = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_fn(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
        losses.append(loss.item())
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    PATH = os.path.join(checkpoint_dir, f'chkpnt_e{epoch}.pth')

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_dir / f"checkpoint_{epoch}.pth",
    )
from datetime import datetime

if __name__ == "__main__":
    base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
    checkpoint_dir = Path(base_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_checkpoints"
    print(checkpoint_dir)

    checkpoint_dir.mkdir(exist_ok=True)
    data_location = "/mnt/efs/dlmbl/G-et/data/live-TLS"
    folder_imgs = data_location +"/"+'Control_Dataset_4TP_Normalized'
    metadata = data_location + "/" +'Control_Dataset_4TP_Ground_Truth'

    loading_transforms = trans.Compose([
        CustomToTensor(),
        SelectRandomTimepoint(0),
        v2.RandomAffine(
            degrees=90,
            translate=[0.1,0.1],
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.GaussianNoise(0,0.05)
    ])

    dataset_w_t = LiveTLSDataset(
        metadata,
        folder_imgs,
        metadata_columns=["Run","Plate","ID"],
        return_metadata=False,
        transform = loading_transforms,
    )

    sample, label = dataset_w_t[0]
    in_channels, y, x = sample.shape
    print(in_channels)
    print((y,x))

    NUM_EPOCHS = 50
    encoder = Encoder(input_shape=(y,x),
                      x_dim=in_channels,
                      h_dim1=8,
                      h_dim2=16,
                      z_dim=10)
    decoder = Decoder(z_dim=10,
                      h_dim1=16,
                      h_dim2=8,
                      x_dim=2,
                      output_shape=(y,x))
    model = VAE(encoder, decoder)
    dataloader = DataLoader(dataset_w_t, batch_size=4, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    for epoch in range(NUM_EPOCHS):
        train(
            epoch,
            model,
            loss_function,
            optimizer,
            dataloader,
            checkpoint_dir=checkpoint_dir)
        # test()
    

   