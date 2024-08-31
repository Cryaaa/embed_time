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


datapath = Path('/mnt/efs/dlmbl/G-et/data/mousepanc/zarrtransposed/')

testimage = datapath.glob( '25868*3.2*.zarr') # glob made for unix wildcard

# select the first image
testimage = list(testimage)[0] 

testimage = zarr.open(testimage)

print(testimage.shape)

testarray = Array(testimage, axis_names = ["color^", "x", "y"])

raw = gp.ArrayKey('RAW') # this the raw image
source = gp.ArraySource(raw, testarray, interpolatable=True) # put image into gunpowder pipeline
source += gp.RandomLocation() # random location in image
source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(30.0, 30.0), rotate=True, spatial_dims = True) # augment image
# write request
request = gp.BatchRequest()
request.add(raw, (5000, 5000))



with gp.build(source):
    batch = source.request_batch(request)
    x = batch.arrays[raw].data
    plt.imshow(x.transpose(1, 2, 0))
    plt.show()


