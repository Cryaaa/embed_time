import torch
import torch.nn as nn
import torchview as tv
import matplotlib.pyplot as plt
from embed_time.model_VAE_resnet18 import VAEResNet18
from embed_time.model_VAE_resnet18_linear_ac import VAEResNet18_linear

output_path = '/mnt/efs/dlmbl/G-et/logs/'
filename = "VAEResNet18_zdim10"

# Example model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = x.view(-1, 320)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    


# Instantiate the model and create a dummy input
model = VAEResNet18(nc=4, z_dim=10)
dummy_input = torch.randn(1, 4, 128, 128)

# Draw the model graph
graph = tv.draw_graph(model, input_data=dummy_input, 
                      save_graph=True, filename=filename,
                      directory=output_path)


