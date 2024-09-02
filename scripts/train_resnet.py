import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
from embed_time.resnet import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm


raw = gp.ArrayKey('RAW') # this the raw image
mask = gp.ArrayKey('MASK') # this is the mask
label = gp.ArrayKey('LABEL') # this is the label

datapath = Path("/mnt/efs/dlmbl/G-et/data/mousepanc") # path to the data

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


def getlabel(inzarr):
    if '25868' in inzarr.name:
        if '25868$1.' in inzarr.name:
            return 0
        elif '25868$2.' in inzarr.name:
            return 1
        else: 
            return 2




trainsources = []

i = 0

train = [
    item for item in (datapath / 'zarrtransposed').iterdir() if ("25868" in item.name) and (int(item.name[8]) % 2 == 0)  # A list of image names
]


for inzarr in train:
    i += 1
    train_image = Array(zarr.open(datapath / "zarrtransposed"/ inzarr.name), axis_names=["color^", "x", "y"])
    train_mask = Array(zarr.open(datapath / "masks" / inzarr.name), axis_names=["x", "y"], voxel_size=(16, 16))
    gtlabel = getlabel(inzarr)
    
    image_source = gp.ArraySource(raw, train_image, interpolatable=True) # put image into gunpowder pipeline
    mask_source = gp.ArraySource(mask, train_mask, interpolatable=False) # put mask into gunpowder pipeline

    trainsource = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline
    trainsource += gp.RandomLocation(mask = mask, min_masked = 0.9) # random location in image. at least 90% of the patch is foreground
    trainsource += AddLabel(label, gtlabel) # "for this testzar my label is x"

    trainsources.append(trainsource)
    if i > 3:
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


trainsource = gunpowderthings(trainsources)
trainrequest = writerequest()

out_channels = 3 # dimensions of the latent space
input_channels = 3
n_iter = 100

resnet = ResNet18(nc = input_channels, oc = out_channels)

loss_function = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
device = torch.device("cuda")


losses = []
total_correct = 0
total_wrong = 0
allpred = []
resnet.train()
resnet = resnet.to(device)

with gp.build(trainsource):
    for n in tqdm(range(n_iter)): # number of iterations
        batch = trainsource.request_batch(trainrequest)
        
        x_in = batch.arrays[raw].data 

        x_in_t = torch.tensor(x_in, dtype=torch.float32)

        x_in_t = x_in_t.to(device)
        optimizer.zero_grad()


        y_pred = resnet(x_in_t) # data goes to the foreward function

        y = torch.tensor(batch.arrays[label].data).to(device)#.unsqueeze(1).to(device)

        loss = loss_function(y_pred, y)
        losses.append(loss.item())

        # total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        # total_wrong += (y_pred.argmax(dim=1) != y).sum().item()
        loss.backward()
        optimizer.step()



fig, ax = plt.subplots(1)
ax.plot(losses)

prediction =  y_pred.argmax(dim=1).cpu().numpy()
real = y.cpu().numpy()
# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
# Compute the confusion matrix
cm = confusion_matrix(real, prediction)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


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