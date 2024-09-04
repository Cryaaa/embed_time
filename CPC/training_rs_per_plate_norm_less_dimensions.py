# %%
import torch
from sklearn.model_selection import train_test_split
from models.nn import ShiftedConv
from models.convnext import ConvNeXt
import train
import torchvision.transforms as trans
from torchvision.transforms import v2
from embed_time.transforms import CustomToTensor, CropAndReshapeTL
from embed_time.dataloader_rs import LiveTLSDatasetPairedOutput
from datetime import datetime
from pathlib import Path

GPU = 0
patience = 20
seed = 1
batch_size = 16
dataset_length = 312 
max_epochs = 300
n_time = 4
latent_dims = 16
convnext_dims = [16, 32, 64, 128]
in_channels =2
lr = 4e-4

model_training_settings = dict(
    GPU = 0,
    patience = 10,
    seed = 1,
    batch_size = 16,
    dataset_length = 312,
    max_epochs = 300,
    n_time = 4,
    latent_dims = 64,
    convnext_dims = [16, 32, 64, 128],
    in_channels =2,
    lr = 4e-3, 
)

model_name = "ben_model_03_pp_norm"
base_dir = "/mnt/efs/dlmbl/G-et/checkpoints/time-series"
checkpoint_dir = Path(base_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{model_name}_checkpoints"
print(checkpoint_dir)

checkpoint_dir.mkdir(exist_ok=True)


data_location = Path("/mnt/efs/dlmbl/G-et/data/live-TLS")

folder_imgs = data_location / 'Control_Dataset_4TP_Normalized_Across_Plates'
metadata = data_location / 'Control_Dataset_4TP_Ground_Truth'
# %%
# if __name__ == "__main__":
loading_transforms = trans.Compose([
    CropAndReshapeTL(1,0,598,0),
    CustomToTensor(),
    #ColorJitterBrightfield(0.5,0.3,0,0),
    v2.Resize((576,576)),
    v2.RandomAffine(
        degrees=90,
        translate=[0.1,0.1],
    ),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.GaussianNoise(0,0.2,clip=True)
])

torch.manual_seed(seed)
device = torch.device(f"cuda:{str(GPU)}")
dataset_w_t = LiveTLSDatasetPairedOutput(
    metadata,
    folder_imgs,
    indices=range(312),
    transform = loading_transforms,
)
# %%
dataset_w_t[0][0]
# %%
train_indices, test_indices = train_test_split(
    range(dataset_length), test_size=0.2, shuffle=True
)

train_loader = torch.utils.data.DataLoader(
    LiveTLSDatasetPairedOutput(
        metadata,
        folder_imgs,
        indices=train_indices,
        transform = loading_transforms,
    ),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
        LiveTLSDatasetPairedOutput(
        metadata,
        folder_imgs,
        indices=test_indices,
        transform = loading_transforms,
    ),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)


encoder = ConvNeXt(in_chans=in_channels, num_classes=latent_dims, dims=convnext_dims)
ar_model = ShiftedConv(in_channels=latent_dims, out_channels=latent_dims, kernel_size=n_time)
query_weights = torch.nn.ModuleList()
for _ in range(n_time - 1):
    query_weights.append(torch.nn.Linear(latent_dims, latent_dims))

parameters = (
    list(encoder.parameters())
    + list(ar_model.parameters())
    + list(query_weights.parameters())
)
optimiser = torch.optim.AdamW(parameters, lr=lr)

train.train(
    train_loader,
    val_loader,
    encoder=encoder,
    ar_model=ar_model,
    query_weights=query_weights,
    n_time=n_time,
    optimiser=optimiser,
    max_epochs=max_epochs,
    device=device,
    checkpoint_dir=checkpoint_dir,
    patience=patience,
    metadata_training=model_training_settings,
)

# Version 1 decreased lr, more patience
