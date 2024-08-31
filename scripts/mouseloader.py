import dask.array
import zarr
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
import dask #like np but on big data



raw = gp.ArrayKey('RAW') # this the raw image
mask = gp.ArrayKey('MASK') # this is the mask

datapath = Path("/mnt/efs/dlmbl/G-et/data/mousepanc")
sources = []
for test_zarr in (datapath / "zarrtransposed").iterdir():
    test_image = Array(zarr.open(datapath / "zarrtransposed"/ test_zarr.name), axis_names=["color^", "x", "y"])
    test_mask = Array(zarr.open(datapath / "masks" / test_zarr.name), axis_names=["x", "y"], voxel_size=(16, 16))

    image_source = gp.ArraySource(raw, test_image, interpolatable=True) # put image into gunpowder pipeline
    mask_source = gp.ArraySource(mask, test_mask, interpolatable=False) # put mask into gunpowder pipeline

    source = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline
    source += gp.RandomLocation(mask = mask, min_masked = 0.9) # random location in image
    sources.append(source)
    break

source = tuple(sources) + gp.RandomProvider()

# Augment image
source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(10.0, 10.0), rotate=True, spatial_dims = 2) # augment image
#TODO: Add more augmentations


# write request
request = gp.BatchRequest()
size = 320
request.add(raw, (size, size))
request.add(mask, (size, size))


with gp.build(source):
    for i in range(10): # number of patches 
        batch = source.request_batch(request)
        x = batch.arrays[raw].data
        y = batch.arrays[mask].data
        plt.imshow(x.transpose(1, 2, 0))
        plt.show()
        plt.imshow(y)
        plt.show()


# treat this as dataloader. Same pipeline for sick and treated data 
# put x through the model 
# 