import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NoiseRegularizeLoss(nn.Module):
    def __init__(self, shifts=1, roll_dims=[2,3], pow=2, mean_axes=[3,5], exit_size=8):
        """Creates a loss based on the noise being shifted around and applied to itself.
        No way to actually minimize it, thus it is a regularization type loss.

        Args:
            shifts (int, optional): Used in the roll method. Defaults to 1.
            roll_dims (list, optional): Applies the roll self multiply for each
                of these dimensions, at each size. Defaults to [2,3].
            pow (int, optional): Exponential to apply to each self multiply. Defaults to 2.
            mean_axes (list, optional): Reshape and put equal portion of the dimensions
                into each of these axes, then reduce using mean on these axes. 
                Defaults to [3,5].
            exit_size (int, optional): Exits the algorithm when the size of the noise
                3rd dimension gets to this size. Defaults to 8.
        """
        super().__init__()
        self.shifts
        self.roll_dims = roll_dims
        self.pow = pow
        self.mean_axes = mean_axes
        self.exit_size = exit_size

    def forward(self, noises):
        loss = 0

        for noise in noises:
            size = noise.shape[2]

            while True:
                for dims in self.roll_dims:
                    loss += (noise * torch.roll(noise, 
                                                shifts=self.shifts, 
                                                dims=dims)).mean().pow(self.pow)

                if size <= self.exit_size:
                    break
                
                reshape_shape = [-1]
                first_axis = min(self.mean_axes)
                last_axis = max(self.mean_axes)
                total_axes = len(self.mean_axes)
                for _ in range(1, first_axis-1):
                    reshape_shape.append(1)
                for i in range(first_axis-1, int(total_axes*2)+1):
                    if i in self.mean_axes:
                        reshape_shape.append(total_axes)
                    else:
                        reshape_shape.append(size // total_axes)
                
                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean(self.mean_axes)
                size //= total_axes

        return loss

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss
