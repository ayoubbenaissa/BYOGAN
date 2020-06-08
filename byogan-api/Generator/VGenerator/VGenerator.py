import torch.nn as nn 

    #ANN-based Generator
class VGenerator(nn.Module):
    def __init__(self, layers, in_features=100, out_features=28*28, negative_slope=0.2, drop_out=0.3, n_layers=2):
        super(VGenerator, self).__init__()
        def network_layer(in_neurons, out_neurons, ns=negative_slope, do=drop_out):
            if (do == 0):
                return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns)]
            return [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(ns), nn.Dropout(do)]
        if (n_layers==2):    
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #2 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                #output layer:
                nn.Linear(layers[2], out_features),
                nn.Tanh())

        if (n_layers==3):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #3 hidden layers
                *network_layer(layers[0], layers[1], do=drop_out[0]),
                *network_layer(layers[1], layers[2], do=drop_out[1]),
                *network_layer(layers[2], layers[3], do=drop_out[2]),
                #output layer:
                nn.Linear(layers[3], out_features),
                nn.Tanh()
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
                #output layer:
                nn.Linear(layers[4], out_features),
                nn.Tanh()
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
                *network_layer(layers[4], layers[5], do=drop_out[4]),
                #output layer:
                nn.Linear(layers[5], out_features),
                nn.Tanh()
                )

    def forward(self, input):
        return self.main(input)
