import torch #top-level package
from torch.autograd.variable import Variable #differentiation operations on tensors
from torch.distributions.normal import Normal #also called Gaussian
from torch.distributions.multivariate_normal import MultivariateNormal

#the Latent Vector shape depends on the network type og the Generator
# (we have two types: ANN & CNN based network types)

def DCLatentVector(in_size, out_size, noise_type, device):
    if (noise_type == 'uniform'):
        n = Variable(torch.randn(in_size, out_size, 1, 1, device=device)) 
        #expected input of DCmodel is 4D tensor 
    if (noise_type == 'gaussian'):
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        n = m.sample([in_size, out_size, 1]) #will give 4D tensor
        n = n.to(torch.device(device))
    if (noise_type == 'multimodal gaussian'):
        m = MultivariateNormal(torch.zeros(out_size), torch.eye(out_size))
        # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        n = m.sample([in_size])
        n = n.view(n.shape[0], n.shape[1], 1, 1)
        n = n.to(torch.device(device))
    return n

def VLatentVector(in_size, out_size, noise_type, device):
    if (noise_type == 'uniform'):
        n = Variable(torch.randn(in_size, out_size, device=device)) 
        #expected input of Vmodel is 2D tensor 
    if (noise_type == 'gaussian'):
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        n = m.sample([in_size, out_size]) #will give 3D tensor
        n = torch.squeeze(n) #to squeeze the 3D tensor into 2D => basically drops an unnecessary column
        n = n.to(torch.device(device))
    if (noise_type == 'multimodal gaussian'):
        m = MultivariateNormal(torch.zeros(out_size), torch.eye(out_size))
        # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        n = m.sample([in_size])
        n = n.to(torch.device(device))
    return n
