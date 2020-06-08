import torch 
from torchvision import transforms, datasets
import torchvision
import torchvision.datasets as dsets
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

def Dataloader(name, path, batch_size, img_size, channels):

    #necessary transformations to apply on input Data
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if(name == 'MNIST'):
        dataset = datasets.MNIST(root=path, train=True, transform=trans, download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    if(name == 'FashionMNIST'):
        dataset = datasets.FashionMNIST(root=path, train=True, transform=trans, download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    elif (name == 'ImageFolder'):
        dataset = datasets.ImageFolder(root=path, transform=trans)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    elif (name == 'CSV'):
        dtype = torch.FloatTensor
        file = open(path)
        data_train = pd.read_csv(file)
        img_digits = (data_train.loc[:, :].values/255 - 0.5)*2
        # change DataFrame to numpy
        digits_Tensor = torch.Tensor(img_digits).type(dtype)
        # build Dataset
        digits_DataSet = torch.utils.data.TensorDataset(digits_Tensor)
        # build DataLoader
        digits_DataLoader = torch.utils.data.DataLoader(digits_DataSet, batch_size = batch_size)
        return digits_DataLoader
    else:
        # for the CSV create dataloader manually, make it torch(batch_size, channels, img_size, img_size)
        # then with Vanilla => make vector
        return 1