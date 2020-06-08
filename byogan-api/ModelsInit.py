import torch
import torch.nn as nn

#normal initialization:
def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

#uniform (not recommended):
def weight_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.uniform_(m.weight.data, 0.0, 0.02)
    if classname.find('ConvTranspose2d') != -1:
        nn.init.uniform_(m.weight.data, 0.0, 0.02)
    if classname.find('Conv2d') != -1:
        nn.init.uniform_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm1d') != -1:
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('BatchNorm2d') != -1:
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('Linear') != -1:
        nn.init.uniform_(m.weight.data, -1.0, 1.0)
        nn.init.constant_(m.bias.data, 0)

#Xavier Uniform (efficient)
def weight_init_Xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('Conv1d') != -1:
        nn.init.xavier_uniform(m.weight)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight)
    if classname.find('ConvTranspose2d') != -1:
        nn.init.xavier_uniform(m.weight)

#Xavier Normal (recommended)
def weight_init_Xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
        nn.init.constant_(m.bias.data, 0)
    if classname.find("Conv") != -1:
        nn.init.xavier_normal(m.weight)
    if classname.find("ConvTranspose") != -1:
        nn.init.xavier_normal(m.weight)

#Kaiming uniform:
def weight_init_Kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    if classname.find("ConvTranspose") != -1:
        nn.init.kaiming_uniform(m.weight, mode='fan_in', nonlinearity='leaky_relu')

#Kaiming normal:
def weight_init_Kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias.data, 0)
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    if classname.find("ConvTranspose") != -1:
        nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='leaky_relu')

def init_model(model, initMethod):
    if (initMethod == 'normal'): model.apply(weight_init_normal)
    if (initMethod == 'uniform'): model.apply(weight_init_uniform)
    if (initMethod == 'Xavier normal'): model.apply(weight_init_Xavier_normal)
    if (initMethod == 'Xavier uniform'): model.apply(weight_init_Xavier_uniform)
    if(initMethod == 'Kaiming uniform'): model.apply(weight_init_Kaiming_uniform)
    if(initMethod == 'Kaiming normal'): model.apply(weight_init_Kaiming_normal)

def get_init_method_description(initMethod):
    if (initMethod == 'default'): return 'used inisialization method is default one provided by Pytorch'
    if (initMethod == 'normal'): return 'The normal distribution should have a mean of 0 and a pre-defined standard deviation.'
    if (initMethod == 'uniform'): return 'A uniform distribution has the equal probability of picking any number from a set of numbers' 
    if (initMethod == 'Xavier normal'): return 'Glorot normal initialization, The resulting tensor will have values sampled from N(0, std**) where std in a pre-calculated value'
    if (initMethod == 'Xavier uniform'): return 'Glorot uniform initialization, The resulting tensor will have values sampled from U[-A, A] where is A is a pre-calculated value'
    if (initMethod == 'Kaiming uniform'): return 'Initialize the layer parameters according to the paper:Delving deep into rectifiers, with uniform distribution in a pre-defined range'
    if (initMethod == 'Kaiming normal'): return 'Initialize the layer parameters according to the paper:Delving deep into rectifiers, with a Normal distribution with pre-defined standard deviation'
