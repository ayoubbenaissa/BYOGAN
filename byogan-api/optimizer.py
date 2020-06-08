import torch.optim as optim #optimization algorithms (SGD, Adam)

#three Optimizers:
def model_optimizer(name, model, learning_rate=0.0002, in_betas=(0.5, 0.999),
                    epsilon=0.00000001, in_weight_decay=0, in_ams_grad=False, in_momentum=0,
                    alpha=0.99, centered_rms=False, in_nostevor=False):
    if (name == "SGD"):
            return optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=in_momentum, weight_decay=in_weight_decay, nesterov=in_nostevor)
    if (name == "Adam"):
        return optim.Adam(model.parameters(), lr=learning_rate, betas=in_betas,
                          eps=epsilon, weight_decay=in_weight_decay, amsgrad=in_ams_grad)
    if(name=="RMS"):
        return optim.RMSprop(model.parameters(), lr=learning_rate,
                            alpha=alpha, eps=epsilon, weight_decay=in_weight_decay, momentum=in_momentum, centered=centered_rms)