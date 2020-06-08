import torch.nn as nn 

class WGenerator(nn.Module):
    def __init__(self, layers, in_features=100, out_features=28*28,
                negative_slope=0.2, drop_out=0.3, n_layers=2,
                batchNorm=False, eps=0.00001, momentum=0.1):
        super(WGenerator, self).__init__()
        def network_layer(in_neurons, out_neurons, do=drop_out, batchN=False, epsilon=0.00001, mmt=0.1, ns=negative_slope):
            block = [nn.Linear(in_neurons, out_neurons)]
            if (batchN): block.append(nn.BatchNorm1d(out_neurons, epsilon, mmt))
            block.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            if (do): block.append(nn.Dropout(do))
            return block
        if (n_layers==2):    
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #2 hidden layers
                *network_layer(layers[0], layers[1], do=float(drop_out[0]), batchN= float(batchNorm[0]), epsilon= float(eps[0]), mmt= float(momentum[0])),
                *network_layer(layers[1], layers[2], do=float(drop_out[1]), batchN= float(batchNorm[1]), epsilon= float(eps[1]), mmt= float(momentum[1])),
                #output layer:
                nn.Linear(layers[2], out_features),
                nn.Tanh())

        if (n_layers==3):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #3 hidden layers
                *network_layer(layers[0], layers[1], do=float(drop_out[0]), batchN= float(batchNorm[0]), epsilon= float(eps[0]), mmt= float(momentum[0])),
                *network_layer(layers[1], layers[2], do=float(drop_out[1]), batchN= float(batchNorm[1]), epsilon= float(eps[1]), mmt= float(momentum[1])),
                *network_layer(layers[2], layers[3], do=float(drop_out[2]), batchN= float(batchNorm[2]), epsilon= float(eps[2]), mmt= float(momentum[2])),
                #output layer:
                nn.Linear(layers[3], out_features),
                nn.Tanh()
            )

        if (n_layers==4):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #4 hidden layers
                *network_layer(layers[0], layers[1], do=float(drop_out[0]), batchN= float(batchNorm[0]), epsilon= float(eps[0]), mmt= float(momentum[0])),
                *network_layer(layers[1], layers[2], do=float(drop_out[1]), batchN= float(batchNorm[1]), epsilon= float(eps[1]), mmt= float(momentum[1])),
                *network_layer(layers[2], layers[3], do=float(drop_out[2]), batchN= float(batchNorm[2]), epsilon= float(eps[2]), mmt= float(momentum[2])),
                *network_layer(layers[3], layers[4], do=float(drop_out[3]), batchN= float(batchNorm[3]), epsilon= float(eps[3]), mmt= float(momentum[3])),
                #output layer:
                nn.Linear(layers[4], out_features),
                nn.Tanh()
                )

        if (n_layers==5):
            self.main = nn.Sequential(
                #input layer
                *network_layer(in_features, layers[0], do=0),
                #5 hidden layers
                *network_layer(layers[0], layers[1], do=float(drop_out[0]), batchN= float(batchNorm[0]), epsilon= float(eps[0]), mmt= float(momentum[0])),
                *network_layer(layers[1], layers[2], do=float(drop_out[1]), batchN= float(batchNorm[1]), epsilon= float(eps[1]), mmt= float(momentum[1])),
                *network_layer(layers[2], layers[3], do=float(drop_out[2]), batchN= float(batchNorm[2]), epsilon= float(eps[2]), mmt= float(momentum[2])),
                *network_layer(layers[3], layers[4], do=float(drop_out[3]), batchN= float(batchNorm[3]), epsilon= float(eps[3]), mmt= float(momentum[3])),
                *network_layer(layers[4], layers[5], do=float(drop_out[4]), batchN= float(batchNorm[4]), epsilon= float(eps[4]), mmt= float(momentum[4])),
                #output layer:
                nn.Linear(layers[5], out_features),
                nn.Tanh()
                )

    def forward(self, input):
        return self.main(input)
