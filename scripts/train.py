from constants import *
from embed_time.dataloader import PumpingDataset
from embed_time.model import ResNet2D
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.nn.functional as F


ckpt_path = "/home/alicia/store1/alicia/pumping/ckpts"
device = "cuda:2"
cycle_size = 3
chunk_size = 10
batch_size = 4
n_epochs = 10_000_000
learning_rate = 1e-4
save_ckpt_freq = 1

input_channels = chunk_size
output_classes = chunk_size * 2

train_datasets, valid_datasets = train_test_split(
        PAIRED_DATASETS[:2],
        test_size=0.5,
        train_size=0.5,
        random_state=42
)

print(len(train_datasets), len(valid_datasets))

train_dataset = PumpingDataset(
        dataset_names=train_datasets,
        chunk_size=chunk_size,
        cycle_size=cycle_size,
        device=device
)

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
)

valid_dataset = PumpingDataset(
        dataset_names=valid_datasets,
        chunk_size=chunk_size,
        cycle_size=cycle_size,
        device=device
)

valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
)

model = ResNet2D(
        output_classes=output_classes,
        input_channels=input_channels,
        batch_size=batch_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate)

model.train()

for n_epoch in tqdm(range(n_epochs)):

    running_loss = 0.0

    for i, (x, label, item_idx) in enumerate(train_dataloader):

        # x shape: (n_batch, chunk_size, width, height)
        # y shape: (n_batch, chunk_size, 2)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = 1 - F.cosine_similarity(pred, label, dim=2).mean() + \
                F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    if n_epoch % save_ckpt_freq == 0:
        torch.save(
            {
                'epoch': n_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            },
            filename=f'{ckpt_path}/ckpt_epoch_{n_epoch}.pth.tar'
        )

