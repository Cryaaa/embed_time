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

zarr_dir = "/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/"
# defines input zarr file name with the zarr file structure
zarr_file = 'structured_celltype_classifier_data.zarr/*/*/*'
position_paths = natsorted(glob(zarr_dir + zarr_file))
# print(position_paths)



centroids = {}
bounding_boxes = {}
data = []
for i, paths in enumerate(position_paths):
    dataset = open_ome_zarr(paths, mode="r")
    image = dataset.data[:,0:2,:,:,:]
    celltype = dataset.data[0,3:4,:,:,:]
    segmented_data = dataset.data[0,2:3,:,:,:]
    
    segment_labels = np.unique(segmented_data)
    segment_labels = segment_labels[segment_labels != 0]  # Exclude background


    # Calculate the centroid for each segment
    for label in segment_labels:
        # Get a binary mask of the current segment
        segment_mask = segmented_data == label
        
        # Find the indices where the segment is present
        t, z_indices, y_indices, x_indices = np.where(segment_mask)
        # Mask the nuclei image with the segment
        masked_image_green=np.where(segment_mask, image, 0)

        # Calculate the bounding box (min and max in each dimension)
        z_min, z_max = z_indices.min(), z_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        
        # # Crop the segment using the bounding box
        # cropped_image_green = masked_image_green[0,0,z_min-2:z_max+2, y_min-2:y_max+2, x_min-2:x_max+2]
        # # cropped_image_red = masked_image_red[0,1,z_min-2:z_max+2, y_min-2:y_max+2, x_min-2:x_max+2]
        
        # Compute the centroid
        coords = np.array(np.nonzero(segment_mask))
        centroid = np.mean(coords, axis=1)
        string = Path(paths).parts[-3:]
        # Extract neuromast ID and t from the paths
       
        neuromast_id = int(string[-3]) # Assuming neuromast ID is in this position
        timepoint = int(string[-2])      # Assuming t value is in this position
        celltypes_segment = celltype[segment_mask]
        cell_type = int(np.unique(celltypes_segment))


        # Append the data to the list
        data.append({
            "Neuromast_ID": neuromast_id,
            "Label": label,
            "Cell_Type": cell_type,
            "Z_min": z_min,
            "Z_max": z_max,
            "Y_min": y_min,
            "Y_max": y_max,
            "X_min": x_min,
            "X_max": x_max,
            "Centroid_Z": centroid[-3],
            "Centroid_Y": centroid[-2],
            "Centroid_X": centroid[-1],
            "T_value": timepoint
        })
        print(f'collected info from celltype {cell_type},timepoint {timepoint} and neuromast {neuromast_id}')
    
# Convert the list of data into a pandas DataFrame
df = pd.DataFrame(data)

# Calculate the ranges for X, Y, and Z
df['X_range'] = df['X_max'] - df['X_min']
df['Y_range'] = df['Y_max'] - df['Y_min']
df['Z_range'] = df['Z_max'] - df['Z_min']

# Find the maximum range across all dimensions
max_x_range = df['X_range'].max()
max_y_range = df['Y_range'].max()
max_z_range = df['Z_range'].max()

# Print the maximum ranges
print(f"Maximum X range: {max_x_range}")
print(f"Maximum Y range: {max_y_range}")
print(f"Maximum Z range: {max_z_range}")

filepath = '/mnt/efs/dlmbl/G-et/data/neuromast/Dataset/metadata_neuromast.csv'
df.to_csv(filepath, index=False)

print("Data saved to segment_data.csv")