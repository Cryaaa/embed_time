import os
from pathlib import Path
import numpy as np
# from torchvision.io import read_image
from torch.utils.data import Dataset
import tifffile as tiff

class LiveGastruloidDataset(Dataset):
    def __init__(
        self, 
        img_dir,
        transform=None,
        return_raft=False
    ):
        self.img_dir = img_dir
        self.file_names = [file for file in os.listdir(img_dir) if file.endswith(".tif")]
        self.transform = transform
        self.return_raft = return_raft

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = Path(self.img_dir) / self.file_names[idx]
        img = tiff.imread(img_path)
        
        if self.transform:
            img = self.transform(img)

        if self.return_raft:
            return img.float(), self.file_names[idx]

        return img.float()