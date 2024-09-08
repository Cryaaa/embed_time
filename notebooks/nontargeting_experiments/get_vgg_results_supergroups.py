# %% 
# Loading the results of vgg experiments and showing their losses, accuracies, and confusion matrices.
# 
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

# %% Utilities
def plot_metrics(metrics): 
    metrics.plot(subplots=True, figsize=(10, 10))
    plt.show()

def load_best_checkpoint(directory, metrics, metric="val_accuracy", mode="max"):
    # get epoch in metric with highest val_accuracy
    if mode == "max":
        best_index = metrics[metric].idxmax()
    else:
        best_index = metrics[metric].idxmax()
    best_epoch = metrics['epoch'][best_index]
    checkpoint = directory / f"{best_epoch}.pth"
    return checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_confusion_matrix(model, val_dataloader, class_names, label_type, normalize='true'):
    model.eval()
    predictions = []
    labels = []

    for batch in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
        images, batch_labels = batch['cell_image'], batch[label_type]
        batch_labels = torch.tensor(
            [class_names.index(label) for label in batch_labels]
        )
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        output = model(images)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    cm = confusion_matrix(np.concatenate(labels), np.concatenate(predictions), normalize=normalize)
    return cm


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

# %% Setup happens here
dataset = "benchmark_nontargeting_barcode_with_cct2"
label_type =  "barcode"  # 'gene'
batch_size = 16
num_workers = 8
balance_dataset = True  # False

subdir, val_dataloader, class_names, num_classes = create_dataloader(dataset, label_type, batch_size, num_workers, balance_dataset=balance_dataset)

metrics = pd.read_csv(subdir / "metrics.csv")
plot_metrics(metrics)
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

cm = get_confusion_matrix(model, val_dataloader, class_names, label_type, normalize=None)

# %% Validation loop for confusion matrix
normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(normalized_cm, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
# Set tick labels
# plt.xticks(np.arange(num_classes) + 0.5, class_names)
# plt.yticks(np.arange(num_classes) + 0.5, class_names)
plt.show()

# %% 
# Group the rows of the confusion matrix by the gene
df = pd.read_csv(f"/mnt/efs/dlmbl/G-et/csv/dataset_split_{dataset}.csv")
df = df[df.split == 'val']
supergroups = df.gene.unique().tolist()

class_to_supergroup = df[["barcode", "gene"]].sort_values(by=["barcode", "gene"]).drop_duplicates().set_index("barcode").to_dict()["gene"]

# %%
# Map class names to their supergroups
class_to_supergroup_idx = {cls: supergroups.index(class_to_supergroup[cls]) for cls in class_names}

# Sort class names based on their supergroup
sorted_class_names = sorted(class_names, key=lambda x: class_to_supergroup_idx[x])

# Re-order the confusion matrix
sorted_indices = [class_names.index(cls) for cls in sorted_class_names]
reordered_cm = cm[sorted_indices, :][:, sorted_indices]

# %% Plot the reordered confusion matrix
normalized_reordered_cm = reordered_cm / reordered_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(normalized_reordered_cm, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')

xticks = [
    list(class_to_supergroup.values()).count(supergroup) for supergroup in supergroups
]
xticks = np.cumsum([0] + xticks)
xticks = xticks[:-1] + (xticks[1:] - xticks[:-1]) / 2
plt.xticks(xticks, supergroups)
plt.yticks(xticks, supergroups)

# Add lines to separate supergroups
for i in range(len(supergroups) - 1):
    plt.axhline(xticks[i] * 2, color='red', linewidth=2)
    plt.axvline(xticks[i]* 2, color='red', linewidth=2)

# %% Get the confusion matrix given supergroups -- if two samples are from the same supergroup, they are considered the same class
super_cm = np.zeros((len(supergroups), len(supergroups)))
for i in range(num_classes):
    for j in range(num_classes):
        super_cm[class_to_supergroup_idx[class_names[i]], class_to_supergroup_idx[class_names[j]]] += cm[i, j]

# %% Plot the confusion matrix given supergroups
normalized_super_cm = super_cm / super_cm.sum(axis=1)[:, np.newaxis]
# sns.heatmap(normalized_super_cm, annot=True, fmt='.2f', cmap='Blues')
sns.heatmap(normalized_super_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=supergroups, yticklabels=supergroups)
plt.xlabel('Predicted')
plt.ylabel('True')

# %% Balanced super-group accuracy
balanced_super_acc = np.mean(np.diag(normalized_super_cm))
balanced_super_acc
# %%
