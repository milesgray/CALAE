'''
Custom Activation Functions
'''
from torch import nn
import torch
import torch.nn.functional as F

@torch.jit.script
def mish(x):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)

@torch.jit.script
def logcosh(x):
    return torch.cosh(x + 1e-12).log()

class LogCosh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return logcosh(x)

@torch.jit.script
def xtanh(x):
    return torch.tanh(x) * x

class XTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xtanh(x)

@torch.jit.script
def xsigmoid(x):
    y = 1 + torch.exp(-x)
    y = torch.abs(y - x)
    z = 2 * y / x
    return z

class XSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xsigmoid(x)