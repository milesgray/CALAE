import torch
from torch import Tensor

from CALAE.loss.IQA import *
from loss_provider import LossProvider

def euclidean_distance(tensor_1: Tensor, tensor_2: Tensor, mask: Tensor = None) -> Tensor:
    difference = tensor_1 - tensor_2
    if mask is not None:
        difference = mask * difference

    distance = difference.square().sum().sqrt() / tensor_1.shape.numel()
    return distance
