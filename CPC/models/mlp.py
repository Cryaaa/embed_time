import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_dims, n_classes, n_hidden_layers, n_hidden_dims):
        super.__init__()

        self.layers = nn.ModuleList()
        first = nn.Linear(n_dims, n_hidden_dims)
        self.layers.append(first)
        for _ in n_hidden_layers:
            self.layers.append(nn.Linear(n_hidden_dims, n_hidden_dims))
        last = nn.Linear(n_hidden_dims, n_classes)
        self.layers.append(last)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)
