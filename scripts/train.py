"""
This script was used to train the pre-trained model weights that were given as an option during the exercise.
"""

from embed_time.dataloader import LiveTLSDataset
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
from embed_time.dataloader import LiveTLSDataset


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch, model, loss_fn, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_fn(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def train_classifier(base_dir, loss_fn, epochs=25):
    checkpoint_dir = Path(base_dir) / "../checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    data_dir = Path(base_dir) / "../data"
    data_dir.mkdir(exist_ok=True)
    
    encoder = Encoder(input_shape=...,
                      x_dim=...,
                      h_dim1=...,
                      h_dim2=...,
                      z_dim=...)
    decoder = Decoder(z_dim=...,
                      h_dim1=...,
                      h_dim2=...,
                      x_dim=...,
                      output_shape=...)
    model = VAE(encoder, decoder)
    data = LiveTLSDataset(data_dir, download=True, train=True)
    dataloader = DataLoader(data, batch_size=32, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    losses = []
    for epoch in range(epochs):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = loss_fn(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        losses.append(loss.item())
        # TODO save every epoch instead of overwriting?
        torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    with open(checkpoint_dir / "losses.txt", "w") as f:
        f.write("\n".join(str(l) for l in losses))


if __name__ == "__main__":
    # checkpoint_dir = Path(base_dir) / "../checkpoints"
    # checkpoint_dir.mkdir(exist_ok=True)
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
        return_metadata=True,
        transform = loading_transforms,
    )
    
    

    NUM_EPOCHS = 50
    encoder = Encoder(input_shape=...,
                      x_dim=...,
                      h_dim1=...,
                      h_dim2=...,
                      z_dim=...)
    decoder = Decoder(z_dim=...,
                      h_dim1=...,
                      h_dim2=...,
                      x_dim=...,
                      output_shape=...)
    model = VAE(encoder, decoder)
    data = LiveTLSDataset(data_dir, download=True, train=True)
    dataloader = DataLoader(data, batch_size=32, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        # test()