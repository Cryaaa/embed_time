import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np

class LiveTLSDataset(Dataset):
    def __init__(
        self, 
        annotations_file, 
        img_dir, 
        file_name_column = "Image Name", 
        label_column ="Morph",
        metadata_columns = ["Plate","ID",], 
        transform=None, 
        target_transform=None,
        return_metadata =False,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.metadata_columns = metadata_columns
        self.label_column = label_column
        self.file_name_column = file_name_column
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.annotations.iloc[idx][self.file_name_column]
        )

        image = tiff.imread(img_path)
        label = self.annotations.iloc[idx][self.label_column]
        

        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)

        if self.return_metadata:
            metadata = self.annotations[self.metadata_columns].iloc[idx].to_numpy()
            return image, label, metadata
        return image, label
    
class LiveTLSDatasetPairedOutput(Dataset):
    def __init__(
        self, 
        annotations_file, 
        img_dir,
        indices,
        label_column ="Morph",
        file_name_column = "Image Name", 
        transform=None,
    ):
        self.annotations = pd.read_csv(annotations_file).iloc[indices]
        self.img_dir = img_dir
        self.transform = transform
        self.file_name_column = file_name_column
        self.label_column = label_column

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.annotations.iloc[idx][self.file_name_column]
        )
        image = tiff.imread(img_path)
        if self.transform:
            image = self.transform(image)

        return image