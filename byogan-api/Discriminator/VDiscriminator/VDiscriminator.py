import torch
import torch.nn as nn

    #ANN-based Discriminator
class VDiscriminator(nn.Module):
    def __init__(self, layers, in_features=28*28, negative_slope=0.2, drop_out=0.3, n_layers=2, tanh_out_layer=False):
        super(VDiscriminator, self).__init__()
        def network_layer(in_neurons, out_neurons, ns=negative_slope, do=drop_out):
            if (do == 0):
                return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns)]
            return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns), nn.Dropout(do)]

        def out_activation(tanh_out_layer):
            if (tanh_out_layer):
                return nn.Tanh()
            return nn.Sigmoid()

        self.main = nn.ModuleList(network_layer(in_features, layers[0], do=0))
        for i in range(n_layers):
            self.main.extend(network_layer(layers[i], layers[i+1], do=drop_out[i]))
        self.out = nn.ModuleList([nn.Linear(layers[n_layers], 1), out_activation(tanh_out_layer)])

    def forward(self, input, Feature_Matching=False):
        for f in self.main:
            input = f(input)
        #feature matching will applied on the last hidden layer:
        if(Feature_Matching):
            return input
        #otherwise, we use the output of the last unit:
        for out_lay in self.out:
            input = out_lay(input)
        return input
