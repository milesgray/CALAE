import torch
import torch.nn as nn

from CALAE.layers import lreq
from CALAE.layers.scaled import ScaledLinear

# ------------------------------------------------------------------------------------------------------------------
# Learnable Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class LearnableGaussianTransform0d(nn.Module):
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(scale), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(scale), requires_grad=True)

    def forward(self, x):
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + (x - 1)
        return z
class LearnableGaussianTransform1d(nn.Module):
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        self.A = lreq.Linear(scale[0], scale[1], bias=True)

    def forward(self, x):
        z = self.A.weight * x
        x = self.A.bias.exp() * x

        z = z + (x - 1)
        return z
class LearnableGaussianTransform2d(nn.Module):
    def __init__(self, scale=(4,4)):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)

    def forward(self, x):
        x = self.weight * x
        z = self.bias.exp() * x
        return x + (z - 1)
# ------------------------------------------------------------------------------------------------------------------
# Learnable Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class LearnableAffineTransform0d(nn.Module):
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        #self.A = ScaledLinear(scale[0], scale[1], bias=True)
        self.weight = nn.Parameter(torch.zeros(1, scale), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(scale), requires_grad=True)

    def forward(self, x):
        #z = self.A.linear.__dict__['_parameters']['weight_orig'] * x
        #x = self.A.linear.bias.exp() * x
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + x
        return z
class LearnableAffineTransform1d(nn.Module):
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        self.A = ScaledLinear(scale[0], scale[1], bias=True)
        #self.weight = nn.Parameter(torch.zeros(1, 1, scale[1]), requires_grad=True)
        #self.bias = nn.Parameter(torch.zeros(scale[1]), requires_grad=True)

    def forward(self, x):
        z = self.A.linear.__dict__['_parameters']['weight_orig'] * x
        x = self.A.linear.bias.exp() * x

        z = z + x
        return z
class LearnableAffineTransform2d(nn.Module):
    def __init__(self, scale=(4,4)):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)

    def forward(self, x):
        x = self.weight.exp() * x
        z = self.bias + x
        return z
