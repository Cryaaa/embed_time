import os
import pandas as pd
# from torchvision.io import read_image
from torch.utils.data import Dataset
import tifffile as tiff

class LiveGastruloidDataset(Dataset):
    def __init__(
        self, 
        img_dir,
        transform=None, 
        target_transform=None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_folders = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_folders)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_names[idx]
        )

        image = tiff.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)

        return image