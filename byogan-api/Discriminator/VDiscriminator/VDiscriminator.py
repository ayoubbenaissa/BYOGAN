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

        if (n_layers==2):    
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #2 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                )
            self.out = nn.Sequential(
                #output layer:
                nn.Linear(layers[2], 1),
                out_activation(tanh_out_layer)
            )

        if (n_layers==3):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #3 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                *network_layer(layers[2], layers[3], do=drop_out[2]),
            )
            self.out = nn.Sequential(
                #output layer:
                nn.Linear(layers[3], 1),
                out_activation(tanh_out_layer)
            )

        if (n_layers==4):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #4 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                *network_layer(layers[2], layers[3], do=drop_out[2]),
                *network_layer(layers[3], layers[4], do=drop_out[3]),
            )
            self.out = nn.Sequential(
                #output layer:
                nn.Linear(layers[4], 1),
                out_activation(tanh_out_layer)
            )

        if (n_layers==5):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #5 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                *network_layer(layers[2], layers[3], do=drop_out[2]),
                *network_layer(layers[3], layers[4], do=drop_out[3]),
                *network_layer(layers[4], layers[5], do=drop_out[3]),
            )
            self.out = nn.Sequential(
                #output layer:
                nn.Linear(layers[5], 1),
                out_activation(tanh_out_layer)
            )

    def forward(self, input, Feature_Matching=False):
        input = self.main(input)
        #feature matching will applied on the last hidden layer
        if(Feature_Matching):
            #otherwise, we use the output of the last unit
            return input
        return self.out(input)
