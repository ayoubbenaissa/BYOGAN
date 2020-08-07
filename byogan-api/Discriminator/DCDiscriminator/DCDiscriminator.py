import torch
import torch.nn as nn

        #CNN-based Discriminator Network:
class DCDiscriminator(nn.Module):
    def __init__(self, in_size=3, leaky_relu=0, drop_out=0, n_layers=2, batchNorm=True, eps=0.00001, momentum=0.1, tanh_out_layer=False, img_size=64):
        super(DCDiscriminator, self).__init__()
        #define the basic cnn-layer which will compose the network
        def conv_layer(in_size, out_size, kernel, stride, padding, do, bn=batchNorm, epsilon= eps, mmt= momentum):
            block = [
                nn.Conv2d(in_size, out_size, kernel_size=kernel, stride=stride, padding=padding, bias=False)
                ]
            if(bn): block.append(nn.BatchNorm2d(out_size, epsilon, mmt))
            block.append(nn.LeakyReLU(negative_slope=leaky_relu, inplace=True))
            if (do): block.append(nn.Dropout2d(do))
            return block

        #output layer
        def out_layer(n_size, out_size, k=4):
            if (tanh_out_layer):
                return [
                nn.Conv2d(n_size, out_size, kernel_size=k, stride=1, padding=0, bias=False),
                nn.Tanh()
            ]
            return [
                nn.Conv2d(n_size, out_size, kernel_size=k, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            ]

        #since the cnn kernels depend heavily on the image size, our tool covers only 64*64 & 28*28 images 
        if (img_size == 64):
            if(n_layers==2):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 4, 2, 1, do=0, bn=False))
                self.main.extend(conv_layer(64, 256, 4, 2, 1, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(256, 512, 8, 2, 1, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                
                self.out = nn.ModuleList(out_layer(512, 1, 6))

            if(n_layers==3):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 4, 2, 1, do=0, bn=False))
                self.main.extend(conv_layer(64, 128, 4, 2, 1, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(128, 256, 4, 2, 1, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                self.main.extend(conv_layer(256, 512, 4, 2, 1, do=drop_out[2], bn=batchNorm[2], epsilon=eps[2], mmt=momentum[2]))
                
                self.out = nn.ModuleList(out_layer(512, 1))

            if(n_layers==4):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 4, 2, 1, do=0, bn=False))
                self.main.extend(conv_layer(64, 128, 4, 2, 1, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(128, 256, 4, 2, 1, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                self.main.extend(conv_layer(256, 512, 3, 2, 1, do=drop_out[2], bn=batchNorm[2], epsilon=eps[2], mmt=momentum[2]))
                self.main.extend(conv_layer(512, 1028, 3, 2, 1, do=drop_out[3], bn=batchNorm[3], epsilon=eps[3], mmt=momentum[3]))
                
                self.out = nn.ModuleList(out_layer(1028, 1, 2))

        if (img_size == 28):
            if(n_layers==2):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 3, 1, 0, do=0, bn=False))
                self.main.extend(conv_layer(64, 256, 4, 2, 1, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(256, 512, 5, 2, 1, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                
                self.out = nn.ModuleList(out_layer(512, 1, 6))

            if(n_layers==3):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 3, 1, 0, do=0, bn=False))
                self.main.extend(conv_layer(64, 128, 3, 1, 0, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(128, 256, 5, 2, 1, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                self.main.extend(conv_layer(256, 512, 5, 2, 1, do=drop_out[2], bn=batchNorm[2], epsilon=eps[2], mmt=momentum[2]))
                
                self.out = nn.ModuleList(out_layer(512, 1, 5))

            if(n_layers==4):
                self.main = nn.ModuleList(conv_layer(in_size, 64, 3, 1, 0, do=0, bn=False))
                self.main.extend(conv_layer(64, 128, 3, 1, 0, do=drop_out[0], bn=batchNorm[0], epsilon=eps[0], mmt=momentum[0]))
                self.main.extend(conv_layer(128, 256, 3, 1, 0, do=drop_out[1], bn=batchNorm[1], epsilon=eps[1], mmt=momentum[1]))
                self.main.extend(conv_layer(256, 512, 5, 2, 1, do=drop_out[2], bn=batchNorm[2], epsilon=eps[2], mmt=momentum[2]))
                self.main.extend(conv_layer(512, 1028, 5, 2, 1, do=drop_out[3], bn=batchNorm[3], epsilon=eps[3], mmt=momentum[3]))
                
                self.out = nn.ModuleList(out_layer(1028, 1, 4))


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
