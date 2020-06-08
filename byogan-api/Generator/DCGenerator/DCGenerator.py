import torch
import torch.nn as nn

    #CNN-based Generator
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, leaky_relu=0, drop_out=0, n_layers=2, batchNorm=True, eps=0.00001, momentum=0.1, out_channels=3, img_size=64):
        super(DCGenerator, self).__init__()

        #conv-transpose will be used to upsample the input
        def convT_layer(in_size, out_size, kernel, stride, padding, do, batchN, epsilon, mmt, LR= leaky_relu):
            block = [
                    nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel, stride=stride, padding=padding, bias=False)
                ]
            if (batchN): block.append(nn.BatchNorm2d(out_size, epsilon, mmt))
            block.append(nn.LeakyReLU(negative_slope=LR, inplace=True))
            if (do): block.append(nn.Dropout2d(do))
            return block

        def out_layer(k=4, out= out_channels):
            return [
                nn.ConvTranspose2d(64, out, kernel_size=k, stride=2, padding=1, bias=False),
                nn.Tanh(),
            ]
        
        #since the cnn kernels depend on the image size, our tool will cover 64*64 & 28*28 images
        if (img_size == 64):
            if(n_layers==2):
                self.network = nn.Sequential(
                    #input layer:
                    *convT_layer(z_dim, 512, 4, 1, 0, do=0, batchN= False, epsilon=0, mmt= 0), #512 (4*4)
                    #2 hidden layers:
                    *convT_layer(512, 128, 8, 2, 1, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]), #128 (12*12) 
                    *convT_layer(128, 64, 12, 2, 1, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]),  #64 (32*32)
                    #output layer:
                    *out_layer() #3 (64*64)
                    )

            if(n_layers==3):
                self.network = nn.Sequential(
                    #input
                    *convT_layer(z_dim, 512,4, 1, 0, do=0, batchN= False, epsilon=0, mmt= 0),
                    #3 hidden layers
                    *convT_layer(512, 256, 4, 2, 1, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]),
                    *convT_layer(256, 128, 4, 2, 1, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]),
                    *convT_layer(128, 64, 4, 2, 1, do=drop_out[2], batchN= batchNorm[2], epsilon= eps[2], mmt= momentum[2]),
                    #output
                    *out_layer(),
                    )

            if(n_layers==4):
                self.network = nn.Sequential(
                    nn.ConvTranspose2d(z_dim, 1024, 3, 1, 0), #1024 (3*3)

                    *convT_layer(1024, 512, 3, 2, 1, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]), #512 (5*5)
                    *convT_layer(512, 256, 2, 2, 1, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]), #256 (8*8)
                    *convT_layer(256, 128, 2, 2, 1, do=drop_out[2], batchN= batchNorm[2], epsilon= eps[2], mmt= momentum[2]), #128 (14*14)
                    *convT_layer(128, 64, 9, 2, 1, do=drop_out[3], batchN= batchNorm[3], epsilon= eps[3], mmt= momentum[3]), #64 (33*33)

                    *out_layer(k=2), #3 (64*64)
                )
            
        if (img_size == 28):
                if(n_layers==2):
                    self.network = nn.Sequential(
                        #input layer:
                        *convT_layer(z_dim, 256, 3, 1, 0, do=0, batchN= False, epsilon=0, mmt= 0), #512 (4*4)
                        #2 hidden layers:
                        *convT_layer(256, 128, 4, 2, 1, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]), #128 (12*12) 
                        *convT_layer(128, 64, 5, 2, 1, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]),  #64 (32*32)
                        #output layer:
                        *out_layer(6)
                        )

                if(n_layers==3):
                    self.network = nn.Sequential(
                        #input
                        *convT_layer(z_dim, 512, 3, 1, 0, do=0, batchN= False, epsilon=0, mmt= 0),
                        #3 hidden layers
                        *convT_layer(512, 256, 3, 1, 0, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]),
                        *convT_layer(256, 128, 3, 1, 0, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]),
                        *convT_layer(128, 64, 4, 2, 1, do=drop_out[2], batchN= batchNorm[2], epsilon= eps[2], mmt= momentum[2]),
                        #output
                        *out_layer(4),
                        )

                if(n_layers==4):
                    self.network = nn.Sequential(
                        #input
                        *convT_layer(z_dim, 1024, 3, 1, 0, do=0, batchN= False, epsilon=0, mmt= 0),
                        #4 hidden
                        *convT_layer(1024, 512, 3, 1, 0, do=drop_out[0], batchN= batchNorm[0], epsilon= eps[0], mmt= momentum[0]), #512 (5*5)
                        *convT_layer(512, 256, 3, 1, 0, do=drop_out[1], batchN= batchNorm[1], epsilon= eps[1], mmt= momentum[1]), #256 (8*8)
                        *convT_layer(256, 128, 3, 1, 1, do=drop_out[2], batchN= batchNorm[2], epsilon= eps[2], mmt= momentum[2]), #128 (14*14)
                        *convT_layer(128, 64, 3, 1, 0, do=drop_out[3], batchN= batchNorm[3], epsilon= eps[3], mmt= momentum[3]), #64 (33*33)
                        #out layer
                        *out_layer(k=4),
                    )

    def forward(self, x):
        return self.network(x)
