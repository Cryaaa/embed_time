
from pathlib import Path
import zarr
import dask.array

datapath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/zarr/')
newpath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/zarrtransposed/')

for zarrfile in datapath.iterdir():
    print(zarrfile)
    inarray = dask.array.from_zarr(zarrfile) # load the image
    inarray = inarray[:].transpose(2, 0, 1)
    inarray.to_zarr(newpath/zarrfile.name, overwrite=True) # save the image
    