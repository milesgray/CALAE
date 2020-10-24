import torch
from torch import nn
from torch import Tensor

def gram_matrix(features: Tensor, mask: Tensor) -> Tensor:
    batch_size, num_channels, height, width = features.shape

    if mask is not None:
        normalize_denominator = mask.square().sum(dim=(2, 3)).sqrt()
        normalize_denominator = normalize_denominator.expand(1, 1, -1, -1)
        normalize_denominator = normalize_denominator.permute((2, 3, 0, 1))
        normalize_denominator = normalize_denominator.repeat((1,) + mask.shape[1:])
        normalized_mask = mask / normalize_denominator
        features = normalized_mask * features

    features = features.view(batch_size * num_channels, height * width)
    features = features.permute((1, 0))
    return torch.mm(features.T, features)

class StyleLoss(nn.Module):

    def __init__(self, target_feature: torch.Tensor, mask=None):
        super().__init__()
        self.target_gram_matrix = gram_matrix(target_feature, mask).detach()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = x.shape[:2]

        G = gram_matrix(x, self.mask)
        loss = (G - self.target_gram_matrix).square().sum() / (4 * (batch_size * num_channels)**2)

        return loss
