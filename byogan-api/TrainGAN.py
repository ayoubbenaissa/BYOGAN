#import libraries:
import torch #top-level package
import torch.nn as nn #used to build NN models
import torch.optim as optim #optimization algorithms (SGD, Adam)
from torch.autograd.variable import Variable #differentiation operations on tensors
import copy
import numpy as np
import torch.autograd as autograd

# vector referring to real data (in general it is a vector filled with 1)
def real_data(size, device, flip, smooth, symmetric_labels):
    if (symmetric_labels):
        data = Variable(-torch.ones(size, 1)) if (flip) else Variable(torch.ones(size, 1))
        if (smooth):
            data = torch.FloatTensor(size).uniform_()/10 + 0.9 #smoothing values between 0.9 and 1
            if (flip):
                data = torch.FloatTensor(size).uniform_()/10 - 1 #smoothing values between -1 and -0.9
    else:
        data = Variable(torch.zeros(size, 1)) if (flip) else Variable(torch.ones(size, 1))
        if (smooth):
            data = torch.FloatTensor(size).uniform_()/10 + 0.9 #smoothing values between 0.9 and 1
            if (flip):
                data = torch.FloatTensor(size).uniform_()/10 #user choosed flipping & smoothing
    return data.to(device)

# vector referring to fake data (in general it is a vector filled with 0)
def fake_data(size, device, flip, smooth, symmetric_labels):
    if (symmetric_labels):
      data = Variable(torch.ones(size, 1)) if (flip) else Variable(-torch.ones(size, 1))
      if (smooth):
          data = torch.FloatTensor(size).uniform_()/10 - 1 #smoothing values between -1 and -0.9
          if (flip):
            data = torch.FloatTensor(size).uniform_()/10 + 0.9 #flipping & smoothing 
    else:
        data = Variable(torch.ones(size, 1)) if (flip) else Variable(torch.zeros(size, 1))
        if (smooth):
            data = torch.FloatTensor(size).uniform_()/10 #smoothing values between 0 and 0.1
            if (flip):
                data = torch.FloatTensor(size).uniform_()/10 + 0.9 #flipping & smoothing
    return data.to(device)

def compute_gradient_penalty(modelD, deviceD, real, fake):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor if (deviceD == 'cuda: 0') else torch.FloatTensor
    alpha = Tensor(np.random.random((real.size(0), 1)))
    alpha = alpha.expand_as(real)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real.data + ((1 - alpha) * fake.data)).requires_grad_(True)
    interpolates = Variable(interpolates, requires_grad=True)
    interpolates = interpolates.to(deviceD)
    d_interpolates = modelD(interpolates)
    fake = Variable(Tensor(real.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_discriminator(optimizer, real, fake, modelD, lossD,
                        device, flip, smooth, symmetric_labels,
                        apply_gradient_penalty, lambda_gp,
                        type_modelG, clip_d, apply_clip_d, apply_divide_d_cost):
    gradient_penalty = torch.Tensor()
    #reset gardients:
    optimizer.zero_grad()
    
    if (apply_gradient_penalty):
        #train on real data:
        prediction_real = modelD(real.to(device), False)
        #train on fake data:
        prediction_fake = modelD(fake.to(device), False)
        #gradient penalty:
        gradient_penalty = compute_gradient_penalty(modelD, device, real, fake)
        error_real = lossD(prediction_real, real_data(real.size(0), device, flip, smooth, symmetric_labels))
        error_fake = lossD(prediction_fake, fake_data(fake.size(0), device, flip, smooth, symmetric_labels))
        total_loss = -error_real + error_fake + lambda_gp * gradient_penalty
        if (apply_divide_d_cost):
            total_loss = total_loss / 2
        total_loss.backward()
    
    else:
        if (apply_divide_d_cost):
            #train on real data:
            prediction_real = modelD(real.to(device), False)
            #calculate error and backpropagation:
            error_real = lossD(prediction_real, real_data(real.size(0), device, flip, smooth, symmetric_labels))
            
            #train on fake data:
            prediction_fake = modelD(fake.to(device), False)
            #calculate error and backpropagation:
            error_fake = lossD(prediction_fake, fake_data(fake.size(0), device, flip, smooth, symmetric_labels))
            total_loss = (error_real + error_fake) / 2
            total_loss.backward()
        else:
            #train on real data:
            prediction_real = modelD(real.to(device), False)
            #calculate error and backpropagation:
            error_real = lossD(prediction_real, real_data(real.size(0), device, flip, smooth, symmetric_labels))
            error_real.backward()
            
            #train on fake data:
            prediction_fake = modelD(fake.to(device), False)
            #calculate error and backpropagation:
            error_fake = lossD(prediction_fake, fake_data(fake.size(0), device, flip, smooth, symmetric_labels))
            error_fake.backward()
            total_loss = error_real + error_fake
    
    #update weights with gradients
    optimizer.step()
    if (apply_gradient_penalty == False) and (type_modelG == 'WGAN') and (apply_clip_d == True):
        # Clip weights of discriminator
        for p in modelD.parameters():
            p.data.clamp_(-clip_d, clip_d)
    
    #return total error, and both prediction:
    return total_loss, prediction_real, prediction_fake, gradient_penalty

def unrolled_train_discriminator_iteration(optimizer, real, fake, modelD, lossD, device, flip, smooth, symmetric_labels):
    #reset gradients:
    optimizer.zero_grad()
    
    #train on real data:
    prediction_real = modelD(real.to(device), False)
    #calculate error and backpropagation:
    error_real = lossD(prediction_real, real_data(real.size(0), device, flip, smooth, symmetric_labels))
    
    #train on fake data:
    prediction_fake = modelD(fake.to(device), False)
    #calculate error and backpropagation:
    error_fake = lossD(prediction_fake, fake_data(fake.size(0), device, flip, smooth, symmetric_labels))
    #total loss:
    d_loss = error_real + error_fake
    d_loss.backward(create_graph=True)

    #update weights with gradients
    optimizer.step()
    
    #return total error, and both prediction:
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake, modelD, lossG, deviceG, deviceD, flip, smooth, symmetric_labels,
                    unrolled_step, optimizerD, real_unrolled, lossD, real_batch, feature_matching, batch_size):
    #reset gradients:
    optimizer.zero_grad()

    #unrolled training:
    if unrolled_step > 0:
        init_state_modelD = copy.deepcopy(modelD.state_dict())
        for i in range(unrolled_step):
            unrolled_train_discriminator_iteration(optimizerD, real_unrolled[i], fake, modelD, lossD, deviceD, flip, smooth, symmetric_labels)
    
    #train:
     #if we use feature matching
    if(feature_matching):
        feature_real = modelD(real_batch.to(deviceD), feature_matching)
        feature_generated = modelD(fake.to(deviceD), feature_matching)
        error = lossG(feature_real, feature_generated)
        error.backward()

        #if we do not use feature matching
    else:
        generated_data = modelD(fake.to(deviceD), False)
        #calculate error and backpropagation:
        error = lossG(generated_data, real_data(batch_size, deviceG, flip, smooth, symmetric_labels))
        error.backward()
    
    #update weights:
    optimizer.step()
    
    if unrolled_step > 0:
        #retrieve the state before the unrolling procedure
        modelD.load_state_dict(init_state_modelD)    
        del init_state_modelD
    return error
