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

        self.activation = nn.ReLU()
        self.final_activation = nn.Softmax(1)
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return self.final_activation(x)
