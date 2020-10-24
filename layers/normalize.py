
import functools
from math import log2, ceil
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import numpy as np

from .switch_norm import SwitchNorm1d, SwitchNorm2d, SwitchNorm3d

####################################################################################################################
###### N O R M A L I Z A T I O N #########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Activation Norm - Normalized
# ------------------------------------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """ ActNorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]
# ------------------------------------------------------------------------------------------------------------------
# Layer Norm
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L812
# ------------------------------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
# ------------------------------------------------------------------------------------------------------------------
# Grouped Channel Norm
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1391
# ------------------------------------------------------------------------------------------------------------------
class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, style, alpha=1e-8):
        """
        x - (N x C x H x W)
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)  # [N1HW]
        y = x / y  # normalize the input x volume 
        return y
# ------------------------------------------------------------------------------------------------------------------
# Switchable Norm
# Switchable Normalization is a normalization technique that is able to learn different normalization operations for different normalization layers in a deep neural network in an end-to-end manner.
# https://github.com/switchablenorms/Switchable-Normalization
# ---------------------------------------------------------------------------------------------------
class SwitchNorm:
    def __init__(self, dims, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        if dims == 1:
            self.norm = SwitchNorm1d(num_features, eps=eps, momentum=momentum, using_moving_average=using_moving_average)
        if dims == 2:
            self.norm = SwitchNorm2d(num_features, eps=eps, momentum=momentum, using_moving_average=using_moving_average)
        if dims == 3:
            self.norm = SwitchNorm3d(num_features, eps=eps, momentum=momentum, using_moving_average=using_moving_average)
        
    def reset_parameters(self):
        self.norm.reset_parameters()

    def forward(self, x):
        return self.norm.forward(x)


# ------------------------------------------------------------------------------------------------------------------
# SPADE
# https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py#L66
# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
# ------------------------------------------------------------------------------------------------------------------
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
