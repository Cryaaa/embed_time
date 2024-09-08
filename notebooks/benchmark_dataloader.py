# %% Notebook to find out the time take to load a single image in the dataset
from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from torchvision.transforms import v2
from embed_time.static_utils import read_config
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
csv_file = '/mnt/efs/dlmbl/G-et/csv/dataset_split_17_sampled.csv'
split = 'train'
channels = [0, 1, 2, 3]
transform = "masks"
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"

parent_dir = '/mnt/efs/dlmbl/S-md/'

metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']

dataset_mean, dataset_std = read_config(yaml_file_path)
dataset = ZarrCellDataset(parent_dir, csv_file, split, channels, transform, normalizations, None, dataset_mean, dataset_std)


# Create a DataLoader for the dataset
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=8,
    collate_fn=collate_wrapper(metadata_keys, images_keys)
)

# %% Timing the dataloader
for data in tqdm(dataloader):
    pass

# %%
