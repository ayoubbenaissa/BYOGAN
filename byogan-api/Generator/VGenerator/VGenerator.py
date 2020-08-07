import torch.nn as nn 

    #ANN-based Generator
class VGenerator(nn.Module):
    def __init__(self, layers, in_features=100, out_features=28*28, negative_slope=0.2, drop_out=0.3, n_layers=2):
        super(VGenerator, self).__init__()
        def network_layer(in_neurons, out_neurons, ns=negative_slope, do=drop_out):
            if (do == 0):
                return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns)]
            return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns), nn.Dropout(do)]

        # layers are created using ModuleList since we are iterating through a dynamic number of layers given as input:
        self.main = nn.ModuleList(network_layer(in_features, layers[0], do=0))
        for i in range(n_layers):
            self.main.extend(network_layer(layers[i], layers[i+1], do=float(drop_out[i])))
        self.main.extend([nn.Linear(layers[n_layers], out_features), nn.Tanh()])

    def forward(self, input):
        for f in self.main:
            input = f(input)
        return input
