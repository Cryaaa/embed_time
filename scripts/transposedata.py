
from pathlib import Path
import zarr
import dask.array
import numpy as np
import matplotlib.pyplot as plt

# write out images and masks in format readable by gunpowder

datapath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/zarr/')
newpath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/zarrtransposed/')
maskpath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/masks/')

maskth = 230 # threshold for mask


for zarrfile in datapath.iterdir():
    #TODO: run for all. Some files are empty!

    print(zarrfile)
    inarray = dask.array.from_zarr(zarrfile) # load the image
    inarray = inarray[:].transpose(2, 0, 1)
    inarray.to_zarr(newpath/zarrfile.name, overwrite=True) # save the image

    masked = dask.array.coarsen(np.min, inarray, axes = {0:1, 1:16, 2:16}, trim_excess=True)

    masked = masked.sum(axis=0)
    masked = masked < maskth*3
    plt.imshow(masked)
    plt.show()
    masked.to_zarr(maskpath/zarrfile.name, overwrite=True) # save the image
    
    zarrmask = zarr.open(maskpath/zarrfile.name)
    zarrmask.attrs['voxel_size'] = (16,16)
    #break