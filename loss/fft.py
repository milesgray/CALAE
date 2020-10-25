import torch
from torch import nn

class FTTLoss(nn.Module):
    def __init__(self, dim, diff_fn=lambda x,y: torch.abs(x-y)):
        """" Calculates a real-valued FFT and then applies the `diff_fn` to get the loss."

        Args:
            dim (number, collection): The axis or axes of the RFFT result to use in `diff_fn`,
                a list of axes calculates each diff sequentially and then sums the total. 
                The mean is returned as the loss.            
            diff_fn (func, optional): A function that takes in two tensors (a single dimensional slice of the
                output of `rfft`) and calulcates some difference between them as the basis of the 
                loss function's calculation.
                 Defaults to ```python 
                lambda x,y: torch.abs(x-y)
                ```.
        """
        self.dim = dim
        self.diff_fn = diff_fn
    
    def forward(self, x, y):
        xf = torch.rfft(x, 3)
        yf = torch.rfft(y, 3)
        if isinstance(self.dim, (list, tuple, set)):
            diff = 0
            for d in self.dim:
                diff += self.diff_fn(xf[d], yf[d])
        elif isinstance(self.dim, (int, float)):
            diff += self.diff_fn(xf[self.dim], yf[self.dim])
        loss = diff.mean()
        return loss


def fft_loss(x, y, axes=2, ndims=3, diff_fn=lambda x,y: torch.abs(x-y)):
    """ Calculates a real-valued FFT and then applies the `diff_fn` to get the loss.

    Args:
        x (Tensor): Network Output Data
        y (Tensor): Label Data
        dim (number, collection, optional): The axis or axes of the RFFT result to use in `diff_fn`,
            a list of axes calculates each diff sequentially and then sums the total. 
            The mean is returned as the loss.
        ndims (int, optional): Number of dimensions the `torch.rfft` method should expect in the input.
        diff_fn (func, optional): A function that takes in two tensors (a single dimensional slice of the
            output of `rfft`) and calulcates some difference between them as the basis of the 
            loss function's calculation.
             Defaults to ```python 
             lambda x,y: torch.abs(x-y)
             ```.
    
    Returns:
        loss (Tensor): The mean of the sum of all `diff_fn` outputs
    """        
    xf = torch.rfft(x, ndims)
    yf = torch.rfft(y, ndims)
    if isinstance(dim, (list, tuple, set)):
        assert max(axes) == ndims
        diff = 0
        for a in axes:
            diff += diff_fn(xf[a], yf[a])
    elif isinstance(axes, (int, float)):
        diff += diff_fn(xf[axes], yf[axes])
    loss = diff.mean()
    return loss
