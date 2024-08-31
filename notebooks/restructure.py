
from iohub.ngff import open_ome_zarr
from iohub.ngff_meta import TransformationMeta
import numpy as np
from natsort import natsorted
from glob import glob
import click
from pathlib import Path
from tqdm import tqdm



sample_dir = '/hpc/projects/jacobo_group/iSim_processed_files/hair_cell_classification/training_data_DL_MBL/'
# defines input zarr file name with the zarr file structure
sample_zarr_file = 'celltype_classifier_data_pyramid.zarr/*/*/*'
# generates a list of paths to the zarr files that match the specified zarr file structure
position_paths = natsorted(glob(sample_dir + sample_zarr_file))
output_zarr_file = 'structured_celltype_classifier_data_pyramid.zarr'
# constructs the full path for the output zarr file
output_path = sample_dir + output_zarr_file
output_path = Path(output_path)

"""Create an empty zarr store mirroring another store"""
DTYPE = np.float32
MAX_CHUNK_SIZE = 500e6  # in bytes
bytes_per_pixel = np.dtype(DTYPE).itemsize

# Load the first position to infer dataset information
input_dataset = open_ome_zarr(position_paths[0], mode="r")
T, C, Z, Y, X = input_dataset.data.shape
output_zyx_shape = (Z, Y, X)
voxel_size = tuple(input_dataset.scale[-3:])
click.echo("Creating empty array...")

"""Create an empty zarr store mirroring another store"""


# Handle transforms and metadata
transform = TransformationMeta(
    type="scale",
    scale=2 * (1,) + voxel_size,
)

channel_names = input_dataset.channel_names

# Output shape needed for the datloader
output_shape = (1, len(channel_names)) + output_zyx_shape
click.echo(f"Number of positions: {len(position_paths)}")
click.echo(f"Output shape: {output_shape}")

# Create output dataset
output_dataset = open_ome_zarr(
    output_path, layout="hcs", mode="w", channel_names=channel_names
)

chunk_zyx_shape = list(output_zyx_shape)
    # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
    # XY image is larger than MAX_CHUNK_SIZE
while (
    chunk_zyx_shape[-3] > 1
    and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
):
    chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
chunk_zyx_shape = tuple(chunk_zyx_shape)

chunk_size = 2 * (1,) + chunk_zyx_shape
click.echo(f"Chunk size: {chunk_size}")
# This takes care of the logic for single position or multiple position by wildcards

for path in position_paths:
    path_strings = Path(path).parts[-3:]
    dataset = open_ome_zarr(path)
    for t_idx in range(T):
        pos = output_dataset.create_position(str(path_strings[1]), str(t_idx), "0")
        output_array = pos.create_zeros(
            name="0",
            shape=output_shape,
            chunks=chunk_size,
            dtype=DTYPE,
            transform=[transform],
        )
input_dataset.close()
output_dataset.close()


total_iterations = len(position_paths) * T
progress_bar = tqdm(total=total_iterations, desc="Processing")

# Copy data from input to output in the dataloader format
for path in position_paths:

    path_strings = Path(path).parts[-3:]
    dataset = open_ome_zarr(path)
    
    for t_idx in range(T):
        output_dataset = open_ome_zarr(
            output_path / str(path_strings[1]) / str(t_idx) / "0", mode="r+"
        )
        output_dataset.data[0,:,:,:,:] = dataset.data[t_idx,:,:,:,:]
        progress_bar.update(1)  # Update the progress bar
        output_dataset.close()
    dataset.close()


