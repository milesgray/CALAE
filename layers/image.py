import torch
import torch.nn as nn

from .scaled import ScaledConv2d 

__all__ = ["FromRGB", "ToRGB", "ToRGB_StyleGAN2"]

# ------------------------------------------------------------------------------------------------------------------
# Project from RGB space to Feature space
# Entry point for sending an image into the network
# ------------------------------------------------------------------------------------------------------------------
class FromRGB(nn.Module):
    def __init__(self, inp_c, oup_c):
        super().__init__()
        self.from_rgb = nn.Sequential(ScaledConv2d(inp_c, oup_c, 1, 1, 0), nn.LeakyReLU(0.2))
        self.downsample = nn.AvgPool2d(2)
        
    def forward(self, x, downsample=False):
        if downsample:
            return self.from_rgb(self.downsample(x.contiguous()))
        else:
            return self.from_rgb(x.contiguous())  
# ------------------------------------------------------------------------------------------------------------------
# Projection from Feature Space to RGB Space
# Outputs an image with values scaled to [-1, 1]
# ------------------------------------------------------------------------------------------------------------------
class ToRGB(nn.Module):
    def __init__(self, inp_c, oup_c, use_bias=False):
        super().__init__()
        self.to_rgb = ScaledConv2d(inp_c, oup_c, 1, 1, 0)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
            
    def forward(self, x, upsample=False):
        x = x.contiguous()
        if upsample:
            x = self.upsample(x)
        x = self.to_rgb(x)
        if self.use_bias:
            x = x + self.bias
        return torch.tanh(x)        
# ------------------------------------------------------------------------------------------------------------------
# Projection from Feature Space to RGB Space from StyleGAN2, also considers a style context
# Outputs an image with values scaled to [-1, 1]
# ------------------------------------------------------------------------------------------------------------------
class ToRGB_StyleGAN2(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.upsampler = stylegan2.Upsample(blur_kernel)

        self.conv = stylegan2.ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None, upsample=False):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:            
            skip = self.upsampler(skip)

            out = out + skip

        return out
