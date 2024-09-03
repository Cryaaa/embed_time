#%%
import zarr
from typing import Union, Optional, Callable, Dict
from torch.utils.data import get_worker_info, Dataset
from pathlib import Path
import numpy as np
from iohub import open_ome_zarr
import scipy
import matplotlib.pyplot as plt

class ZarrDataset(Dataset):
    """Dataset to extract patches from a zarr storage."""
    def __init__(
        self,
        data_path: Union[str, Path],
        image_transform: Optional[Callable] = None,
        image_transform_params: Optional[Dict] = None,
        ) -> None:
        self.data_path = Path(data_path)
        self.image_transform = image_transform
        self.patch_transform_params = image_transform_params

        self.data = open_ome_zarr(data_path)
        self.indices = list(self.data.positions())[:4]
        self.mean = self.calculate_mean()
        self.std = self.calculate_std()

    def calculate_mean(self):  
        total_sum = np.zeros(2)  #(1, 2, 32, 2048, 2048)
        total_count = 0
        for name, pos in self.indices:
            image = pos[0].numpy()
            total_sum += image.sum(axis=(0, 2, 3, 4))
            total_count += image.shape[2] * image.shape[3] * image.shape[3]
        mean = total_sum / total_count
        return mean
    
    def calculate_std(self):
        sum_squared_diff = np.zeros(2)
        total_count = 0
        for name, pos in self.indices:
            image = pos[0].numpy()
            sum_squared_diff += ((image - self.mean[None, :, None, None, None]) ** 2).sum(
                axis=(0, 2, 3, 4)
            )
            total_count += image.shape[2] * image.shape[3] * image.shape[3]
        variance = sum_squared_diff / total_count
        std = np.sqrt(variance)
        return std
    
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        name, pos = self.indices[idx]
        array = pos[0].numpy() # (t,c,z,y,x)
        print(array.shape)
        patient = name.split("/")[0]

        # transformation
        transform_array = np.max(array, axis=2).squeeze(0)
        print(transform_array.shape)
        transform_array = scipy.ndimage.zoom(transform_array, zoom=(1, 0.5, 0.5))
        print(transform_array.shape)
        flip_prob = np.random.rand(1)
        
        if flip_prob >0.5:
            transform_array = np.flip(transform_array, axis=(1,2)) #(2,2024,2024) 
            print(transform_array.shape)
        transform_array = np.rot90(transform_array, k=np.random.randint(4), axes=(1,2))
        print(transform_array.shape)
        return transform_array
    
dataset = ZarrDataset("/home/S-ac/embed_time/zarrdata/mitochondria.zarr")

for batch in dataset:
    _, ax = plt.subplots(2)
    ax[0].imshow(batch[0])
    ax[1].imshow(batch[1])


# %%
