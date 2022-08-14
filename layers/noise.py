import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NoiseNormalize(nn.Module):
    def forward(self, noises):
        for noise in noises:
            mean = noise.mean()
            std = noise.std()

            noise.data.add_(-mean).div_(std)
        return noises

def noise_normalize(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

# ------------------------------------------------------------------------------------------------------------------
# Normal Distribution with Learnable Scale
# ------------------------------------------------------------------------------------------------------------------
class IntermediateNoise(nn.Module):
    def __init__(self, inp_c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, inp_c, 1, 1), requires_grad=True)
        self.noise = None
    
    def forward(self, x, noise=None):
        if self.training:
            if noise is None and self.noise is None:
                noise = torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(x.device)
            elif noise is None:
                noise = self.noise
            return x + (noise * self.weight)
        else:
            return x

# ------------------------------------------------------------------------------------------------------------------
# Unit Hypersphere Projection
# ------------------------------------------------------------------------------------------------------------------
class BallProjection(nn.Module):
    """
    Constraint norm of an input noise vector to be sqrt(latent_code_size)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div((torch.mean(x.pow(2), dim=1, keepdim=True).add(1e-8)).pow(0.5))            