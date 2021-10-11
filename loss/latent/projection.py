import typing as tp

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .lpips import PerceptualLoss
from .perceptual_style_loss import FixedPerceptualAndStyleLoss

class WplusLoss(nn.Module):
    def __init__(self, lambdas: tp.Dict[str, float], device: str):
        super().__init__()
        self.lambdas = lambdas
        self.device = device
        self.perceptual_loss_function = PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
        )

    def forward(self, input_img: Tensor, target_img: Tensor) -> Tensor:
        perceptual_loss = lambdas['l_percept'] * perceptual_loss_function(input_img, target_img).sum()
        mse_loss = lambdas['l_mse'] * F.mse_loss(input_img, target_img, reduction='none')
        mse_loss = mse_loss.mean(dim=(1, 2, 3)).sum()
        loss = perceptual_loss + mse_loss
        return loss

class WplusStyleLoss(nn.Module):
    def __init__(self, lambdas: tp.Dict[str, float], content_image: Tensor, style_image: Tensor, mask_image: Tensor, device: str):
        super().__init__()
        self.lambdas = lambdas
        self.device = device
        self.perceptual_and_style_loss = FixedPerceptualAndStyleLoss(content_image,
                                                                     style_image,
                                                                     mask_image.detach(),
                                                                     (1 - mask_image).detach())
        self.perceptual_and_style_loss.to(device)

    def forward(self, input_img: Tensor, target_img: Tensor) -> Tensor:
        style_loss, perceptual_loss = perceptual_and_style_loss(input_img)
        style_loss = lambdas['l_style'] * style_loss
        perceptual_loss = lambdas['l_percept'] * perceptual_loss

        mse_loss = torch.square(mask_image * (input_img - target_img)).mean()
        mse_loss = lambdas['l_mse'] * mse_loss
        loss = mse_loss + style_loss + perceptual_loss
        return loss

class NaiveNoiseLoss(nn.Module):
    def __init__(self, lambdas: tp.Dict[str, float]):
        super().__init__()
        self.lambdas = lambdas

    def forward(self, input_img: Tensor, target_img: Tensor) -> Tensor:
        mse_loss = lambdas['l_mse'] * F.mse_loss(input_img, target_img, reduction='none')
        mse_loss = mse_loss.mean(dim=(1, 2, 3)).sum()
        loss_dict = {'mse_loss': mse_loss.item()}
        return mse_loss

class NoiseLoss(nn.Module):
    def __init__(self, lambdas: tp.Dict[str, float], content_image: Tensor, style_image: Tensor, mask_image: Tensor, device: str):
        super().__init__()
        self.lambdas = lambdas
        self.content_image = content_image
        self.style_image = style_image
        self.mask_image = mask_image
        self.device = device

    def forward(self, input_img: Tensor, target_img: Tensor) -> Tensor:
        mse_loss_1 = lambdas['l_mse_1'] * torch.square(self.mask_image * (input_img - self.content_image.detach())).mean()
        mse_loss_2 = lambdas['l_mse_2'] * torch.square((1 - self.mask_image) * (input_img - self.style_image.detach())).mean()
        loss = mse_loss_1 + mse_loss_2
        return loss

def w_plus_loss(lambdas: tp.Dict[str, float], device: str) -> tp.Callable:
    perceptual_loss_function = PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    def loss_impl(generated_image: Tensor, original_image: Tensor) -> tp.Tuple[torch.Tensor, dict]:
        perceptual_loss = lambdas['l_percept'] * perceptual_loss_function(generated_image, original_image).sum()
        mse_loss = lambdas['l_mse'] * F.mse_loss(generated_image, original_image, reduction='none')
        mse_loss = mse_loss.mean(dim=(1, 2, 3)).sum()
        loss = perceptual_loss + mse_loss
        loss_dict = {
            'perceptual_loss': perceptual_loss.item(),
            'mse_loss': mse_loss.item(),
        }
        return loss, loss_dict

    return loss_impl


def naive_noise_loss(lambdas: tp.Dict[str, float]) -> tp.Callable:
    def loss_impl(generated_image: Tensor, original_image: Tensor) -> tp.Tuple[torch.Tensor, dict]:
        mse_loss = lambdas['l_mse'] * F.mse_loss(generated_image, original_image, reduction='none')
        mse_loss = mse_loss.mean(dim=(1, 2, 3)).sum()
        loss_dict = {'mse_loss': mse_loss.item()}
        return mse_loss, loss_dict

    return loss_impl


def w_plus_style_loss(lambdas: tp.Dict[str, float], content_image: Tensor, style_image: Tensor, mask_image: Tensor, device: str) -> tp.Callable:
    perceptual_and_style_loss = FixedPerceptualAndStyleLoss(content_image, style_image, mask_image.detach(), (1 - mask_image).detach())
    perceptual_and_style_loss.to(device)

    def loss_impl(generated_image: Tensor, original_image: Tensor) -> tp.Tuple[Tensor, dict]:
        style_loss, perceptual_loss = perceptual_and_style_loss(generated_image)
        style_loss = lambdas['l_style'] * style_loss
        perceptual_loss = lambdas['l_percept'] * perceptual_loss

        mse_loss = torch.square(mask_image * (generated_image - content_image)).mean()
        mse_loss = lambdas['l_mse'] * mse_loss
        loss_dict = {
            'mse_loss': mse_loss.item(),
            'style_loss': style_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
        }
        loss = mse_loss + style_loss + perceptual_loss

        return loss, loss_dict

    return loss_impl


def noise_loss(lambdas: tp.Dict[str, float], content_image: Tensor, style_image: Tensor, mask_image: Tensor) -> tp.Callable:

    def loss_impl(generated_image: Tensor, original_image: Tensor) -> tp.Tuple[Tensor, dict]:
        mse_loss_1 = lambdas['l_mse_1'] * torch.square(mask_image * (generated_image - content_image.detach())).mean()
        mse_loss_2 = lambdas['l_mse_2'] * torch.square((1 - mask_image) * (generated_image - style_image.detach())).mean()

        loss_dict = {
            'mse_1': mse_loss_1.item(),
            'mse_2': mse_loss_2.item(),
        }
        loss = mse_loss_1 + mse_loss_2

        return loss, loss_dict

    return loss_impl
