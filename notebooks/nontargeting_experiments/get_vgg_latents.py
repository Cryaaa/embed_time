# %%
# The purpose of this is to use the VGG model to get latents (taking the last convolutional layer as the latent space).
# I want to see whether the mis-classified cells are recognizable in the latent space.
# 
# Created 2024-09-04 by @adjavon

# %% 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from funlib.learn.torch.models import Vgg2D
from embed_time.static_utils import read_config
from torchvision import transforms as v2
import seaborn as sns

def load_best_checkpoint(directory, metrics):
    # get epoch in metric with highest val_accuracy
    best_index = metrics['val_accuracy'].idxmax()
    best_epoch = metrics['epoch'][best_index]
    checkpoint = directory / f"{best_epoch}.pth"
    return checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloader(dataset, label_type, batch_size=16, num_workers=8, balance_dataset=None):
    csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv"
    subsub = f"{label_type}_{balance_dataset}" if balance_dataset else label_type
    subdir = Path(f"/mnt/efs/dlmbl/G-et/da_testing/vgg2d_{dataset}/{subsub}")
    df = pd.read_csv(csv_file)
    class_names = df[label_type].sort_values().unique().tolist()
    num_classes = len(class_names)

    metadata_keys = ['gene', 'barcode', 'stage']
    images_keys = ['cell_image']
    crop_size = 96
    normalizations = v2.Compose([v2.CenterCrop(crop_size)])
    yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
    dataset = "benchmark"
    dataset_mean, dataset_std = read_config(yaml_file_path)

    val_dataset = ZarrCellDataset(
        parent_dir = '/mnt/efs/dlmbl/S-md/',
        csv_file = csv_file, 
        split='val',
        channels=[0, 1, 2, 3], 
        mask='min', 
        normalizations=normalizations,
        interpolations=None, 
        mean=dataset_mean, 
        std=dataset_std
    )

    # Create a DataLoader for the validation dataset
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_wrapper(metadata_keys, images_keys),
        drop_last=False
    )
    return subdir, val_dataloader, class_names, num_classes


# %%
dataset = "benchmark_nontargeting_barcode_with_cct2"
label_type =  "gene"  # 'gene'
batch_size = 16
num_workers = 8
balance_dataset = True  # False

subdir, val_dataloader, class_names, num_classes = create_dataloader(dataset, label_type, batch_size, num_workers, balance_dataset=balance_dataset)

metrics = pd.read_csv(subdir / "metrics.csv")

# %% Get the model to load the best checkpoint, create a confusion matrix
checkpoint = load_best_checkpoint(subdir, metrics)
model = Vgg2D(
    input_size=(96, 96),
    input_fmaps=4,
    output_classes=num_classes,
)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
model.eval()

# %% [markdown]
# The funlib model has two parts which are called features and classifier; 
# We want the output from the second to last Linear layer in the classifier part.
# Hook to capture the activations
activations = []

def hook(module, input, output):
    activations.append(output.cpu().numpy())

# Register the hook on the second-to-last Linear layer
second_to_last_layer = model.classifier[-4]
hook_handle = second_to_last_layer.register_forward_hook(hook)

# Get the activations
latents = []
for batch in tqdm(val_dataloader):
    images = batch['cell_image'].to(device)
    with torch.inference_mode():
        model(images)  # Forward pass to trigger the hook
    latents.append(np.concatenate(activations, axis=0))
    activations.clear()  # Clear activations for the next batch

# latents = []
# for batch in tqdm(val_dataloader):
#     images = batch['cell_image'].to(device)
#     with torch.inference_mode():
#         output = model.features(images)
#     latents.append(output.flatten(1).cpu().numpy())
# %%
latents = np.concatenate(latents, axis=0)
# Store the latents
np.save(subdir / "vgg_latents.npy", latents)
# %%
# TODO should we also get the predicted labels
# TODO should we also get the correct vs predicted labels?