import os
import re

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from models.nn import ShiftedConv
from models.convnext import ConvNeXt
import cpc


GPU = 0
max_epochs = 100
patience = 10
seed = 1
crop_size = 280
batch_size = 32
train_split = 0.9
checkpoint_dir = "checkpoints/"
latent_dims = 2
encoder_dims = [8, 16, 32, 64]
n_time = 5
n_channels = 1

torch.manual_seed(seed)
device = torch.device(f"cuda:{str(GPU)}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.n_images = len(images)
        self.transform = transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

#### MNIST DATA EXAMPLE
mnist = torchvision.datasets.MNIST("./data", download=True)

classes = []
for i in range(10):
    current_class = (mnist.targets == i).nonzero()
    classes.append(current_class[:, 0])

n_images = 1000

sorted_mnist = []
classes_ok = True
while classes_ok and len(sorted_mnist) < n_images:
    time_series = []
    for i in range(10):
        img = mnist[classes[i][-1].item()][0]
        img = torchvision.transforms.functional.pil_to_tensor(img)[None]
        img = torch.nn.functional.interpolate(img, scale_factor=10)
        time_series.append(img)
        classes[i] = classes[i][:-1]
        if len(classes[i]) == 0:
            classes_ok = False
    time_series = torch.stack(time_series, dim=2)
    sorted_mnist.append(time_series)
data = torch.cat(sorted_mnist, dim=0).to(torch.float)

data = data.unfold(2, 5, 5)
data = data.movedim(2, 1)
data = data.flatten(0, 1)
data = data.movedim(4, 2)

data -= data.mean()
data /= data.std()
#### MNIST DATA EXAMPLE


def get_next_version(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    files = os.listdir(checkpoint_dir)
    old_versions = [
        int(x.split("_")[1])
        for x in files
        if re.fullmatch(r"version_\d+", x) is not None
    ]
    old_versions.sort()
    if len(old_versions) > 0:
        version = f"version_{old_versions[-1] + 1}"
    else:
        version = "version_0"
    checkpoint_dir = os.path.join(checkpoint_dir, version)
    return checkpoint_dir


def train(
    train_loader,
    val_loader,
    encoder,
    ar_model,
    query_weights,
    n_time,
    optimiser,
    max_epochs,
    device,
    checkpoint_dir,
    patience,
    metadata_training = None,
):
    patience_counter = 0
    best_loss = torch.inf
    checkpoint_dir = get_next_version(checkpoint_dir)
    writer = SummaryWriter(checkpoint_dir)
    if metadata_training is not None:
        torch.save(
            metadata_training,
            os.path.join(checkpoint_dir, "metadata_training.pt"),
        )
    encoder = encoder.to(device)
    ar_model = ar_model.to(device)
    query_weights = query_weights.to(device)
    try:
        for epoch in range(max_epochs):
            running_train_loss = 0.0
            encoder.train()
            ar_model.train()
            query_weights.train()
            for batch in tqdm(
                train_loader,
                bar_format=f"Epoch {epoch} {{l_bar}}{{bar}}{{r_bar}}",
                leave=False,
            ):
                batch = batch.to(device)
                batch_size = batch.shape[0]
                optimiser.zero_grad()
                loss = cpc.cal_loss(
                    x=batch,
                    batch_size=batch_size,
                    n_time=n_time,
                    encoder=encoder,
                    ar_model=ar_model,
                    query_weights=query_weights,
                )
                loss.backward()
                optimiser.step()
                running_train_loss += loss / len(train_loader)
            writer.add_scalar("loss/train", running_train_loss.item(), epoch)

            with torch.no_grad():
                encoder.eval()
                ar_model.eval()
                query_weights.eval()
                running_val_loss = 0.0
                for batch in tqdm(
                    val_loader,
                    bar_format=f"Validation {{l_bar}}{{bar}}{{r_bar}}",
                    leave=False,
                ):
                    batch = batch.to(device)
                    batch_size = batch.shape[0]
                    loss = cpc.cal_loss(
                        x=batch,
                        batch_size=batch_size,
                        n_time=n_time,
                        encoder=encoder,
                        ar_model=ar_model,
                        query_weights=query_weights,
                    )
                    running_val_loss += loss / len(val_loader)
                writer.add_scalar("loss/val", running_val_loss.item(), epoch)
                writer.flush()
                if running_val_loss < best_loss:
                    best_loss = running_val_loss
                    patience_counter = 0
                    torch.save(
                        encoder.state_dict(), os.path.join(checkpoint_dir, "encoder.pt")
                    )
                    torch.save(
                        ar_model.state_dict(),
                        os.path.join(checkpoint_dir, "ar_model.pt"),
                    )
                    torch.save(
                        query_weights.state_dict(),
                        os.path.join(checkpoint_dir, "query_weights.pt"),
                    )
                else:
                    patience_counter += 1
                    if patience_counter == patience:
                        break
    except KeyboardInterrupt:
        print(f"Keyboard interrupt")
    print(f"{epoch} epochs. Version {checkpoint_dir.split('_')[-1]}")

if __name__ == "__main__":
    transform = torchvision.transforms.RandomCrop(crop_size)
    train_set = data[: int(len(data) * train_split)]
    val_set = data[int(len(data) * train_split) :]
    train_set = Dataset(train_set, transform=transform)
    val_set = Dataset(val_set, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    encoder = ConvNeXt(in_chans=n_channels, num_classes=latent_dims, dims=encoder_dims)
    ar_model = ShiftedConv(in_channels=latent_dims, out_channels=latent_dims, kernel_size=n_time)
    query_weights = torch.nn.ModuleList()
    for _ in range(n_time - 1):
        query_weights.append(torch.nn.Linear(latent_dims, latent_dims))

    parameters = (
        list(encoder.parameters())
        + list(ar_model.parameters())
        + list(query_weights.parameters())
    )
    optimiser = torch.optim.AdamW(parameters, lr=4e-3)

    train(
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
    )
