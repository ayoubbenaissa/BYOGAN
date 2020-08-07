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
        
        # layers are created using ModuleList since we are iterating through a dynamic number of layers given as input:
        self.main = nn.ModuleList(network_layer(in_features, layers[0], do=0))
        for i in range(n_layers):
            self.main.extend(network_layer(layers[i], layers[i+1], do=float(drop_out[i]),
            batchN=float(batchNorm[i]), epsilon=float(eps[i]), mmt=float(momentum[i])))
        self.main.extend([nn.Linear(layers[n_layers], out_features), nn.Tanh()])

    def forward(self, input):
        for f in self.main:
            input = f(input)
        return input
