import zarr
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
import dask #like np but on big data
#from embed_time.model import ResNet2D
from embed_time.vae import VAEResNet18
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



labels = []

def getlabel(test_zarr):
    if '25868' in test_zarr.name:
        if '25868$1.' in test_zarr.name:
            return 0
        elif '25868$2.' in test_zarr.name:
            return 1
        else: 
            return 2



for test_zarr in (datapath / "zarrtransposed").iterdir():

    if not "25868" in test_zarr.name: # subset to only 25868 for now 
        print("skipping", test_zarr.name)
        continue

    test_image = Array(zarr.open(datapath / "zarrtransposed"/ test_zarr.name), axis_names=["color^", "x", "y"])
    test_mask = Array(zarr.open(datapath / "masks" / test_zarr.name), axis_names=["x", "y"], voxel_size=(16, 16))
    
    #print(test_zarr.name)
    gtlabel = getlabel(test_zarr) #np.random.randint(0, 3) #generate random labels 
    #print(gtlabel)
    
    image_source = gp.ArraySource(raw, test_image, interpolatable=True) # put image into gunpowder pipeline
    mask_source = gp.ArraySource(mask, test_mask, interpolatable=False) # put mask into gunpowder pipeline

    source = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline
    source += gp.RandomLocation(mask = mask, min_masked = 0.9) # random location in image. at least 90% of the patch is foreground
    source += AddLabel(label, gtlabel) # "for this testzar my label is x"
    sources.append(source)
    break

source = tuple(sources) + gp.RandomProvider()

# Augment image
source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(10.0, 10.0), rotate=True, spatial_dims = 2) # augment image
#TODO: Add more augmentations
batch_size = 16
source += gp.Stack(batch_size) # stack the batch

# write request
request = gp.BatchRequest()
size = 64
request.add(raw, (size, size))
request.add(mask, (size, size))
request[label] = gp.ArraySpec(nonspatial=True)

output_classes = 256 # dimensions of the latent space
input_channels = 3
n_iter = 500

vae = VAEResNet18(nc = input_channels, z_dim = 16)
loss_function: torch.nn.Module = torch.nn.MSELoss()
kl_divergence = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

device = torch.device("cuda")

vae.train()
vae = vae.to(device)

losses = []
with gp.build(source):
    for n in range(n_iter): # number of iterations
        batch = source.request_batch(request)
        
        x = batch.arrays[raw].data
        x_torch = torch.tensor(x, dtype=torch.float32)

        x_torch = x_torch.to(device)
        optimizer.zero_grad()


        x, z = vae(x_torch) # data goes to the foreward function
        loss = loss_function(x, x_torch) + kl_divergence(z, torch.zeros_like(z))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # add a loss-function
x_out = x.cpu().detach().numpy()
plt.imshow(x.cpu().detach().numpy()[0][0])


plt.plot(losses)
plt.show()

        # TODO: add loss function 



# treat this as dataloader. Same pipeline for sick and treated data 
# put x through the model 
# 

# output space: dimensions of the latent space 


