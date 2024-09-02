import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
from embed_time.vae import VAEResNet18
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
source += gp.Normalize(raw) # normalize image: devides by max (255)
# TODO: normalize using mean and std of data 

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

z_dim = 128 # dimensions of the latent space
input_channels = 3
n_iter = 10000

vae = VAEResNet18(nc = input_channels, z_dim = z_dim)
#loss_function: torch.nn.Module = torch.nn.MSELoss() #TODO: try L1 loss instead
# use L1 loss
loss_function: torch.nn.Module = torch.nn.L1Loss()
# add perceptual loss (structural similarity loss)

kl_divergence = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

device = torch.device("cuda")

vae.train()
vae = vae.to(device)

reclosses = []
kllosses = []
losses = []


def kld_loss( mu, logvar):

    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

# k
beta = 1e-4 # zdim / total n of pixels (patchsize^2 * nchannels)

with gp.build(source):
    for n in tqdm(range(n_iter)): # number of iterations
        batch = source.request_batch(request)
        
        x_in = batch.arrays[raw].data

        x_in_t = torch.tensor(x_in, dtype=torch.float32)

        x_in_t = x_in_t.to(device)
        optimizer.zero_grad()


        x, z, mean, logvar = vae(x_in_t) # data goes to the foreward function
        rec_loss = loss_function(x, x_in_t)

        kl_loss = kld_loss(mean, logvar)


        loss = beta * kl_loss + rec_loss
        losses.append(loss.item())
        reclosses.append(rec_loss.item())
        kllosses.append(kl_loss.item())
        loss.backward()
        optimizer.step()

        # add a loss-function



def plotoutput(x_in, x_out):
    np.random.seed(0)
    shuffled = np.arange(len(x_in))

    np.random.shuffle(shuffled)
    fig, ax = plt.subplots(2,4)

    for i in range(4):
        ax[0, i].imshow(x_in_t[shuffled[i]].cpu().detach().numpy().transpose(1,2,0))
        ax[1, i].imshow(x_out[shuffled[i]].cpu().detach().numpy().transpose(1,2,0))

    plt.show()


plotoutput(x_in_t, x)


fig, ax = plt.subplots(1,2)
ax[0].plot(reclosses)
ax[0].set_title("Reconstruction loss")
ax[1].plot(kllosses)
ax[1].set_title("KL loss")

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