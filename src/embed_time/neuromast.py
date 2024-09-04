
from iohub.ngff import open_ome_zarr
from natsort import natsorted
from glob import glob
from pathlib import Path 
import torch
from torch.utils.data import Dataset
from scipy.ndimage import measurements
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NeuromastDatasetTrain(Dataset):
    def __init__(self):
        file_path = "/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/"
        zarr_file = 'structured_celltype_classifier_data.zarr/*/*/*'
        position_paths = natsorted(glob(file_path + zarr_file))
        self.position_paths = position_paths[:500]
        

        self.metadata = pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_balanced_train.csv")


        # Find the maximum range across all dimensions
        max_x_range = 256
        max_y_range = 256
        max_z_range = 48 # not used for cropping

        self.crop_size = [max_z_range, max_y_range, max_x_range]

        self.shape = (open_ome_zarr(self.position_paths[0], mode="r")).data.shape  

        
    def crop_image(self, idx):
        
        row = self.metadata.iloc[idx]
        # Get centroid coordinates
        centroid_z = int(row['Centroid_Z'])
        centroid_y = int(row['Centroid_Y'])
        centroid_x = int(row['Centroid_X'])
        
        #get the label number
        label = int(row['Label'])

        timepoint = int(row['T_value'])

        # Compute the cropping box boundaries
        z_min = int(row['Z_min'])
        z_max = int(row['Z_max'])
        y_min = int(max((int(centroid_y - self.crop_size[1] // 2)),0))
        y_max = int(min((int(centroid_y + self.crop_size[1] // 2)), self.shape[3]-1))
        x_min = int(max((int(centroid_x - self.crop_size[2] // 2)), 0))
        x_max = int(min((int(centroid_x + self.crop_size[2] // 2)), self.shape[4]-1))

        mid_z = (z_min + z_max) // 2


        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        dataset = open_ome_zarr(self.position_paths[timepoint], mode="r")
        image = dataset.data[0,0:1,mid_z,y_min:y_max, x_min:x_max]
        segmented_data = dataset.data[0,2:3,mid_z,y_min:y_max, x_min:x_max] #segmention masks
        # celltypes = dataset.data[0,3:,:,:,:]
        # Get a binary mask of the current segment
        segment_mask = segmented_data == label
        

        # Find the unique label numbers in the celltypes image for this segment
        cell_type = int(row['Cell_Type'])
        cropped_image=np.where(segment_mask, image, 0)
        
        # if z_max - z_min != 64 & z_max == self.shape[2]-1:
        #     z_min = z_max - 64

        # if z_max - z_min != 64 & z_min == 0:
        #     z_max = z_min + 64
        # Crop the image
        
        
        return cropped_image, cell_type


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        cell, cell_type = self.crop_image(idx)
        return cell, cell_type
    
class NeuromastDatasetTrain_T10(Dataset):
    def __init__(self):
        file_path = "/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/"
        zarr_file = 'structured_celltype_classifier_data.zarr/*/*/*'
        position_paths = natsorted(glob(file_path + zarr_file))
        self.position_paths = position_paths[:500]
        self.cell_count = 40 # number of cells to sample from each timepoint
       

        self.metadata = pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_train_T10.csv")

        # Find the maximum range across all dimensions
        max_x_range = 256
        max_y_range = 256
        max_z_range = 48 # not used for cropping

        self.crop_size = [max_z_range, max_y_range, max_x_range]

        self.shape = (open_ome_zarr(self.position_paths[0], mode="r")).data.shape  

        
    def crop_image(self, idx):
        
        row = self.metadata.iloc[idx]
        # Get centroid coordinates
        centroid_z = int(row['Centroid_Z'])
        centroid_y = int(row['Centroid_Y'])
        centroid_x = int(row['Centroid_X'])
        
        #get the label number
        label = int(row['Label'])

        timepoint = int(row['T_value'])

        # Compute the cropping box boundaries
        z_min = int(row['Z_min'])
        z_max = int(row['Z_max'])
        y_min = int(max((int(centroid_y - self.crop_size[1] // 2)),0))
        y_max = int(min((int(centroid_y + self.crop_size[1] // 2)), self.shape[3]-1))
        x_min = int(max((int(centroid_x - self.crop_size[2] // 2)), 0))
        x_max = int(min((int(centroid_x + self.crop_size[2] // 2)), self.shape[4]-1))

        mid_z = (z_min + z_max) // 2
        
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        dataset = open_ome_zarr(self.position_paths[timepoint], mode="r")
        image = dataset.data[0,0:1,mid_z,y_min:y_max, x_min:x_max]
        segmented_data = dataset.data[0,2:3,mid_z,y_min:y_max, x_min:x_max] #segmention masks
        # celltypes = dataset.data[0,3:,:,:,:]
        # Get a binary mask of the current segment
        segment_mask = segmented_data == label
        

        # Find the unique label numbers in the celltypes image for this segment
        cell_type = int(row['Cell_Type'])
        cropped_image=np.where(segment_mask, image, 0)
        
        # if z_max - z_min != 64 & z_max == self.shape[2]-1:
        #     z_min = z_max - 64

        # if z_max - z_min != 64 & z_min == 0:
        #     z_max = z_min + 64
        # Crop the image
        
        
        return cropped_image, cell_type

    def __len__(self):
        
        return len(self.metadata)

    def __getitem__(self, idx):
        cell, cell_type = self.crop_image(idx)
        return cell, cell_type
    

class NeuromastDatasetTest(Dataset):
    def __init__(self):
        file_path = "/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/"
        zarr_file = 'structured_celltype_classifier_data.zarr/*/*/*'
        position_paths = natsorted(glob(file_path + zarr_file))
        self.position_paths = position_paths[500:]
        self.cell_count = 40 # number of cells to sample from each timepoint
        

        self.metadata = pd.read_csv("/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast_test_T10.csv")

        # Find the maximum range across all dimensions
        max_x_range = 256
        max_y_range = 256
        max_z_range = 48 # not used for cropping

        self.crop_size = [max_z_range, max_y_range, max_x_range]

        self.shape = (open_ome_zarr(self.position_paths[0], mode="r")).data.shape  

        
    def crop_image(self, idx):
        
        row = self.metadata.iloc[idx]
        # Get centroid coordinates
        centroid_z = int(row['Centroid_Z'])
        centroid_y = int(row['Centroid_Y'])
        centroid_x = int(row['Centroid_X'])
        
        #get the label number
        label = int(row['Label'])

        timepoint = int(row['T_value'])

        # Compute the cropping box boundaries
        z_min = int(row['Z_min'])
        z_max = int(row['Z_max'])
        y_min = int(max((int(centroid_y - self.crop_size[1] // 2)),0))
        y_max = int(min((int(centroid_y + self.crop_size[1] // 2)), self.shape[3]-1))
        x_min = int(max((int(centroid_x - self.crop_size[2] // 2)), 0))
        x_max = int(min((int(centroid_x + self.crop_size[2] // 2)), self.shape[4]-1))

        mid_z = (z_min + z_max) // 2
        
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        # Load the corresponding image from the dataset (assuming 5D dataset [T, C, Z, Y, X])
        dataset = open_ome_zarr(self.position_paths[timepoint], mode="r")
        image = dataset.data[0,0:1,mid_z,y_min:y_max, x_min:x_max]
        segmented_data = dataset.data[0,2:3,mid_z,y_min:y_max, x_min:x_max] #segmention masks
        # celltypes = dataset.data[0,3:,:,:,:]
        # Get a binary mask of the current segment
        segment_mask = segmented_data == label
        

        # Find the unique label numbers in the celltypes image for this segment
        cell_type = int(row['Cell_Type'])
        cropped_image=np.where(segment_mask, image, 0)
        
        
        return cropped_image, cell_type


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        cell, cell_type = self.crop_image(idx)
        return cell, cell_type