import torch.nn as nn #used to build NN models

#two Loss options: Binary Cross Entropy & Mean Squarred Error
def loss_function(loss_name):
    if (loss_name == 'BCE'):
        return nn.BCELoss()
    elif (loss_name == 'MSE'):
        return nn.MSELoss()
