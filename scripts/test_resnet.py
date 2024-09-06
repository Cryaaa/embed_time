import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
from funlib.persistence import Array
from embed_time.resnet import ResNet18
import torch
import numpy as np
from tqdm import tqdm

# TODO: Copied from train_resnet. modify! 

# create resnet

input_channels = 3
out_channels = 3
resnet = ResNet18(nc = input_channels, oc = out_channels)
patch_size = 64

# load the model from the checkpoint
checkpoint10k = torch.load("/home/S-mt/embed_time/checkpoints/resnetcheck.pt_checkpoint_10000")
resnet.load_state_dict(checkpoint10k['model_state_dict']) # put into resnet model 

class DummyModel(torch.nn.Module):
    def forward(self, x, mask):
        return torch.ones(1, 3, 1, 1)


class WrappedModel(torch.nn.Module):
    """
    Turns the ResNet model output into a "spatial" output
    Return some "null" value when the mask is empty
    """
    def __init__(self, model, null_value=np.nan, num_classes=3, fraction_masked=0.9):
        super().__init__()
        self.null_output = torch.full((num_classes, 1, 1), null_value)
        self.model = model
        self.fraction_masked = fraction_masked

    def forward(self, x, mask):
       
        if mask.sum() < mask.numel() * self.fraction_masked:
            return self.null_output
        else:
            # softmax
            output = self.model(x.unsqueeze(0))
            #return torch.nn.functional.softmax(output, dim = 1).squeeze[0] # return the output of the model
            return torch.nn.functional.softmax(output, dim = 1)[..., None, None].squeeze(0) # return the output of the model


wrapped_model = WrappedModel(resnet)
wrapped_model.eval()

# wrapped_model = DummyModel()
# wrapped_model.eval()


datapath = Path("/mnt/efs/dlmbl/G-et/data/mousepanc") # path to the data

test = [
    item for item in (datapath / 'zarrtransposed').iterdir() if ("25868" in item.name) and (int(item.name[8]) % 2 == 1 and (int(item.name[8]) <= 4))  # A list of image names
]

testname = test[0].name

raw = gp.ArrayKey('RAW') # this the raw image
mask = gp.ArrayKey('MASK') # this is the mask
prediction = gp.ArrayKey('PREDICTION') # this is the prediction

#for inzarr in test:
test_image = Array(zarr.open(datapath / "zarrtransposed"/ testname), axis_names=["color^", "x", "y"])
test_mask = Array(zarr.open(datapath / "masks" / testname), axis_names=["x", "y"], voxel_size=(16, 16))

image_source = gp.ArraySource(raw, test_image, interpolatable=True) # put image into gunpowder pipeline
mask_source = gp.ArraySource(mask, test_mask, interpolatable=False) # put mask into gunpowder pipeline

testsource = (image_source, mask_source) + gp.MergeProvider() # put both into pipeline

#testsource += gp.Pad(raw, None)
testsource += gp.Normalize(raw)

patch_roi = gp.Roi((0, 0), (patch_size, patch_size))


# total_roi = test_image.roi
# print(test_image.shape)
# imgshape = test_image.shape

# centerx = test_image.shape[1] // 2
# centery = test_image.shape[2] // 2

multiplesx = test_image.shape[1]//patch_size
multiplesy = test_image.shape[2]//patch_size



# xmax = multiplesx * patch_size
# ymax = multiplesy * patch_size

newx0 = int(np.floor(0.5 * multiplesx) * patch_size)
newy0 = int(np.floor(0.5 * multiplesy) * patch_size)

# newx1 = int(xmax - newx0)
# newy1 = int(ymax - newy0)


total_roi = gp.Roi((300*patch_size, 240*patch_size), (200*patch_size, 200*patch_size))


testsource += gp.torch.Predict(
    wrapped_model,
    inputs = {
        'x': raw,
        'mask': mask
    },
    outputs = {
        0: prediction
    },
    array_specs = {
        prediction: gp.ArraySpec(roi=total_roi, voxel_size=(patch_size, patch_size), dtype=np.float32)
    },
    device = "cpu",
    #spawn_subprocess = True
)

testsource += gp.ZarrWrite(
    dataset_names = {
        prediction: 'prediction'
    },
    store = f'/mnt/efs/dlmbl/G-et/data/mousepanc/predictions/{testname}'
)

scan_request = gp.BatchRequest()
scan_request[raw] = gp.ArraySpec(roi=patch_roi, voxel_size=(1, 1))
scan_request[prediction] = gp.ArraySpec(roi=patch_roi, voxel_size=(patch_size, patch_size))
scan_request[mask] = gp.ArraySpec(roi=patch_roi, voxel_size=(16, 16))

testsource += gp.Scan(scan_request, num_workers=1)


request = gp.BatchRequest()

with gp.build(testsource):
    testsource.request_batch(request)





img = zarr.open(f'/mnt/efs/dlmbl/G-et/data/mousepanc/predictions/{testname}')


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im = ax[0].imshow(img['prediction'][0], cmap='coolwarm', vmin = 0, vmax = 1)
ax[0].set_title('Healthy')
ax[1].imshow(img['prediction'][1], cmap='coolwarm', vmin = 0, vmax = 1)
ax[1].set_title('Sick')
ax[2].imshow(img['prediction'][2], cmap='coolwarm', vmin = 0, vmax = 1)
ax[2].set_title('Treated')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
