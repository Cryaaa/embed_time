import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
from embed_time.resnet import ResNet18
import torch
import numpy as np
from tqdm import tqdm


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
    
    gtlabel = getlabel(test_zarr) #np.random.randint(0, 3) #generate random labels 
    
    image_source = gp.ArraySource(raw, test_image, interpolatable=True) # put image into gunpowder pipeline
    mask_source = gp.ArraySource(mask, test_mask, interpolatable=False) # put mask into gunpowder pipeline

    source = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline
    source += gp.RandomLocation(mask = mask, min_masked = 0.9) # random location in image. at least 90% of the patch is foreground
    source += AddLabel(label, gtlabel) # "for this testzar my label is x"
    sources.append(source)
    break
    

def gunpowderthings(sources, batch_size = 16): 
    source = tuple(sources) + gp.RandomProvider()
    # Augment image
    source += gp.Normalize(raw) # normalize image: devides by max (255) # TODO: normalize using mean and std of data 
    source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(10.0, 10.0), rotate=True, spatial_dims = 2) # augment image
    #TODO: Add more augmentations

    source += gp.Stack(batch_size) # stack the batch
    return source


def writerequest(patch_size = 64):
    request = gp.BatchRequest()
    request.add(raw, (patch_size, patch_size))
    request.add(mask, (patch_size, patch_size))
    request[label] = gp.ArraySpec(nonspatial=True)
    return request


source = gunpowderthings(sources)
request = writerequest()
# write request

out_channels = 3 # dimensions of the latent space
input_channels = 3
n_iter = 100

resnet = ResNet18(nc = input_channels, oc = out_channels)

loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
device = torch.device("cuda")


losses = []
total_correct = 0
total_wrong = 0

resnet.train()
resnet = resnet.to(device)

with gp.build(source):
    for n in tqdm(range(n_iter)): # number of iterations
        batch = source.request_batch(request)
        
        x_in = batch.arrays[raw].data 

        x_in_t = torch.tensor(x_in, dtype=torch.float32)

        x_in_t = x_in_t.to(device)
        optimizer.zero_grad()


        y_pred = resnet(x_in_t) # data goes to the foreward function

        y = torch.tensor(batch.arrays[label].data).to(device)#.unsqueeze(1).to(device)
        # print(y)
        # print(x)
        loss = loss_function(y_pred, y)
        losses.append(loss.item())

        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_wrong += (y_pred.argmax(dim=1) != y).sum().item()
        loss.backward()
        optimizer.step()

        # add a loss-function





fig, ax = plt.subplots(1)
ax.plot(losses)


# confusion matrix


# treat this as dataloader. Same pipeline for sick and treated data 
# put x through the model 
# 

# output space: dimensions of the latent space 


# How to tmux

# tmux new -s train # where train is the name of the session
# tmux ls # list all sessions
# tmux a -t train # attach to session train
# conda activate embed_time
# python scripts/train.py
# control + b, d # detach from session