import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
from embed_time.resnet import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import datetime

raw = gp.ArrayKey('RAW') # this the raw image
mask = gp.ArrayKey('MASK') # this is the mask
label = gp.ArrayKey('LABEL') # this is the label
prediction = gp.ArrayKey('PREDICTION') # this is the prediction

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
    # if i > 3:
    #     break

    

# def load_checkpoint(checkpoint_path, model, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model, checkpoint['epoch']


out_channels = 3 # dimensions of the latent space
input_channels = 3
n_iter = 10000


resnet = ResNet18(nc = input_channels, oc = out_channels)
batch_size = 16
patch_size = 64


loss_function = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
device = torch.device("cuda")


# losses = []
# total_correct = 0
# total_wrong = 0
allpred = []
resnet.train()
resnet = resnet.to(device)


source = tuple(trainsources) + gp.RandomProvider()

# Augment image
source += gp.Normalize(raw) # normalize image: devides by max (255) # TODO: normalize using mean and std of data 
source += gp.DeformAugment(control_point_spacing=gp.Coordinate(100, 100), jitter_sigma=gp.Coordinate(10.0, 10.0), rotate=True, spatial_dims = 2) # augment image
#TODO: Add more augmentations

source += gp.Stack(batch_size) # stack the batch
source += gp.torch.Train(resnet, loss_function, optimizer, inputs = {0:raw}, loss_inputs = {0: prediction, 1: label}, outputs = {0: prediction},
                            checkpoint_basename = "/home/S-mt/embed_time/checkpoints/resnetcheck.pt", save_every = 1000, 
                            log_dir = "/mnt/efs/dlmbl/G-et/embed_time/logs/resnetcheck/",
                            log_every = 1)



trainrequest = gp.BatchRequest()
trainrequest.add(raw, (patch_size, patch_size))
trainrequest.add(mask, (patch_size, patch_size))
trainrequest[label] = gp.ArraySpec(nonspatial=True)
trainrequest[prediction] = gp.ArraySpec(nonspatial=True)





# Additional information
# ITERATION = n_iter
# PATH = "/home/S-mt/embed_time/checkpoints/resnetcheck.pt"
# LOSS = 0.4

# torch.save({
#             'iteration': ITERATION,
#             'model_state_dict': resnet.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)


# checkpoint_dir = "/mnt/efs/dlmbl/"
# checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
# checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
# resnet, epoch = load_checkpoint(checkpoint_path, resnet, device)



# output_dir = '/mnt/efs/dlmbl/G-et/embed_time/'
# run_name= "resnet"

# folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
#log_path = output_dir + "logs"+ folder_suffix + "/"

# checkpoint_path = "/home/S-mt/embed_time/checkpoints/"



# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)




with gp.build(source):
    for n in tqdm(range(n_iter)): # number of iterations
        batch = source.request_batch(trainrequest)
        y_pred = batch[prediction].data
        y = batch[label].data


        
        # x_in = batch.arrays[raw].data 

        # x_in_t = torch.tensor(x_in, dtype=torch.float32)

        # x_in_t = x_in_t.to(device)
        # optimizer.zero_grad()


        # y_pred = resnet(x_in_t) # data goes to the foreward function

        # y = torch.tensor(batch.arrays[label].data).to(device)#.unsqueeze(1).to(device)

        # loss = loss_function(y_pred, y)
        # losses.append(loss.item())

        # # total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        # # total_wrong += (y_pred.argmax(dim=1) != y).sum().item()
        # loss.backward()
        # optimizer.step()






# fig, ax = plt.subplots(1)
# ax.plot(losses)

# pred =  y_pred.argmax(axis=1)
# real = y
# # confusion matrix

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# # Compute the confusion matrix
# cm = confusion_matrix(real, pred)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()


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


# output_dir = '/mnt/efs/dlmbl/G-et/'
# run_name= "resnet_linear_1168sub_64dim"

# folder_suffix = datetime.now().strftime("%Y%m%d_%H%M_") + run_name
# log_path = output_dir + "logs/static/Matteo/"+ folder_suffix + "/"

# checkpoint_path = output_dir + "checkpoints/static/Matteo/" + folder_suffix + "/"

# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)

# for epoch in range(1, 100):
#     train(epoch, log_interval=100, log_image_interval=20, tb_logger=logger)
#     filename_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_") + "epoch_"+str(epoch) + "_"
#     training_logDF = pd.DataFrame(training_log)
#     training_logDF.to_csv(log_path + filename_suffix+"training_log.csv", index=False)
# epoch_logDF = pd.DataFrame(epoch_log)
# epoch_logDF.to_csv(log_path + filename_suffix+"epoch_log.csv", index=False)

# checkpoint = {
#     'epoch': epoch,
#     'model_state_dict': vae.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss_per_epoch 
# }
# torch.save(checkpoint, checkpoint_path + filename_suffix + str(epoch) + "checkpoint.pth")