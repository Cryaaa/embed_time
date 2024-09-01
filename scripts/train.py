import zarr
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
import dask #like np but on big data
from embed_time.model import ResNet2D
import torch
import re
import numpy as np


raw = gp.ArrayKey('RAW') # this the raw image
mask = gp.ArrayKey('MASK') # this is the mask
label = gp.ArrayKey('LABEL') # this is the label

datapath = Path("/mnt/efs/dlmbl/G-et/data/mousepanc") # path to the data
sources = []

class AddLabel(gp.BatchFilter):

    def __init__(self, array_key, label):
        self.array_key = array_key
        self.label = label

    def setup(self):

        self.provides(self.array_key, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):

        array = gp.Array(np.array(self.label), spec=gp.ArraySpec(nonspatial=True))
        batch = gp.Batch()
        batch[self.array_key] = array
        return batch

# def getlabel(name, labels):
#     if '25868' in name:
#         if bool(re.search('1.', name)):
#             labels.append(1) #healthy
#         elif re.search('2.', name):
#             labels.append(2) #sick
#         else:
#             labels.append(3)
#     elif '14550' in name:
#         if re.search("A*PBS", name):
#             labels.append(1)
#         elif 'CaerPBS' in name:
#             labels.append(2)
#         elif bool(re.search("F*PBS*", name)):
#             labels.append(3)
#     else: 
        
#     return labels



labels = []

for test_zarr in (datapath / "zarrtransposed").iterdir():
    test_image = Array(zarr.open(datapath / "zarrtransposed"/ test_zarr.name), axis_names=["color^", "x", "y"])
    test_mask = Array(zarr.open(datapath / "masks" / test_zarr.name), axis_names=["x", "y"], voxel_size=(16, 16))
    
    #generate random labels 
    gtlabel = np.random.randint(0, 3)
    
    #label = np.ones(test_image.shape) * gtlabel

    image_source = gp.ArraySource(raw, test_image, interpolatable=True) # put image into gunpowder pipeline
    mask_source = gp.ArraySource(mask, test_mask, interpolatable=False) # put mask into gunpowder pipeline
    #label_source = gp.ArraySource(label, np.array(gtlabel), interpolatable=False)



    source = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline
    source += gp.RandomLocation(mask = mask, min_masked = 0.9) # random location in image. at least 90% of the patch is foreground
    source += AddLabel(label, gtlabel) # "for this testzar my label is x"
    sources.append(source)
    #break

source = tuple(sources) + gp.RandomProvider()

# Augment image
source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(10.0, 10.0), rotate=True, spatial_dims = 2) # augment image
#TODO: Add more augmentations
batch_size = 32
source += gp.Stack(batch_size) # stack the batch

# write request
request = gp.BatchRequest()
size = 320
request.add(raw, (size, size))
request.add(mask, (size, size))
request[label] = gp.ArraySpec(nonspatial=True)

output_classes = 256 # dimensions of the latent space
input_channels = 3
n_iter = 10


model = ResNet2D(output_classes, input_channels, batch_size)
with gp.build(source):
    for n in range(n_iter): # number of iterations
        batch = source.request_batch(request)
        x = batch.arrays[raw].data
        x_torch = torch.tensor(x, dtype=torch.float32)
        print(x.shape, type(x))
        #y = batch.arrays[mask].data
        print(x.shape)
        pred = model(x_torch)
        # TODO: add loss function 
        #print(pred.shape)
        # plt.imshow(x[0].transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(y[0])
        # plt.show()


# treat this as dataloader. Same pipeline for sick and treated data 
# put x through the model 
# 

# output space: dimensions of the latent space 


