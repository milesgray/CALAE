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
