import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm
import torch.distributions as D
 
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

import random
from math import log2, ceil

from torch.autograd import Function
from torch.nn import functional as F

from CALAE.layers import Factory
from CALAE.layers import ToRGB, FromRGB, AdaIN, AdaPN
from CALAE.layers import IntermediateNoise, Blur
from CALAE.layers import LearnableGaussianTransform2d, LearnableAffineTransform2d
from CALAE.layers import ExplicitCoordConv
from CALAE.layers import TripletAttention
from CALAE.layers.scaled import ScaledConv2d, ScaledLinear, set_scale
import CALAE.layers.lreq as lreq

####################################################################################################################
################################################## Level 2 blocks ##################################################
####################################################################################################################

####################################################################################################################
############## E N C O D E R #############--------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, scale, act='leaky', final=False, blur_downsample=False, fused_scale=True, learn_blur=False):
        super().__init__()
        
        self.final = final
        self.blur_downsample = blur_downsample
        self.learn_blur = learn_blur

        if self.learn_blur:
            self.learned_affine = set_scale(LearnableAffineTransform2d(scale=(oup_c if final else inp_c, scale)))
        
        self.in1 = nn.InstanceNorm2d(inp_c, affine=False)
        self.in2 = nn.InstanceNorm2d(oup_c, affine=False)
        
        self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, stride=1, padding=1)
        self.style_mapping1 = ScaledLinear(2 * inp_c, code)
        
        if final:
            self.fc = ScaledLinear(inp_c * 4 * 4, oup_c)
            self.style_mapping2 = ScaledLinear(code, oup_c)
        else:
            self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, stride=1, padding=1)    
            self.style_mapping2 = ScaledLinear(2 * oup_c, code)
            
        if act:
            self.act = Factory.get_activation(act)
        self.downsample = nn.AvgPool2d(2, 2)
        
        self.blur = Blur(inp_c)
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        statistics1 = torch.cat([x.mean(dim=[2,3]), x.std(dim=[2,3])], dim=1)
        style1 = self.style_mapping1(statistics1)
        x = self.in1(x)
        norm = x
        if self.final:
            x = x.view(x.shape[0], -1)
            statistics2 = self.act(self.fc(x))
        else:    
            if self.blur_downsample:
                x = self.blur(x)
            if self.learn_blur:
                x = self.learned_affine(x)
            x = self.downsample(self.act(self.conv2(x)))
            statistics2 = torch.cat([x.mean(dim=[2,3]), x.std(dim=[2,3])], dim=1)
            
        style2 = self.style_mapping2(statistics2)
        
        if not self.final:
            x = self.in2(x)
        
        return x, style1, style2, norm

####################################################################################################################
############ G E N E R A T O R ###########--------------------------------------------------------------------------
class GeneratorBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, initial=False, blur_upsample=True, 
                 fused_scale=True, learn_blur=True, learn_residual=False, 
                 learn_style=False, learn_noise=False, use_attn=True, use_coord=True,
                 scale=4, act="leaky", norm="pixel"):
        super().__init__()
                
        self.initial = initial
        self.blur_upsample = blur_upsample
        self.learn_blur = learn_blur
        self.learn_residual = learn_residual
        self.learn_style = learn_style
        self.learn_noise = learn_noise
        self.use_coord = use_coord

        # learnable affine transform to correct blur
        if self.learn_blur:
            if self.initial:
                self.blur_affine = set_scale(LearnableAffineTransform2d(scale=inp_c))
            else:
                self.blur_affine = set_scale(LearnableAffineTransform2d(scale=(inp_c, scale)))
        if self.learn_residual:
            self.residual_gain = nn.Parameter(torch.from_numpy(np.array([1, -1], dtype=np.float32)))
            self.res_conv = lreq.Conv2d(inp_c, oup_c, kernel_size=3, stride=1, padding=1) # 
            self.res_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            self.res_ups_affine = LearnableGaussianTransform2d(scale=(oup_c, scale))

        # Learnable noise coefficients
        self.B1 = set_scale(IntermediateNoise(inp_c))
        self.B2 = set_scale(IntermediateNoise(oup_c))
        # Learnable affine transform
        if self.learn_noise:
            self.B1_affine = LearnableGaussianTransform2d(scale=(inp_c, scale))
            self.B2_affine = LearnableGaussianTransform2d(scale=(oup_c, scale))
        if self.learn_style:
            self.w1_affine = LearnableGaussianTransform0d(scale=code)
            self.w2_affine = LearnableGaussianTransform0d(scale=code)
        
        
        # Each Ada PN contains learnable parameters A
        if norm == "pixel":
            self.ada_norm1 = AdaPN(inp_c, code)
            self.ada_norm2 = AdaPN(oup_c, code)
        elif norm == "instance":
            self.ada_norm1 = AdaIN(inp_c, code)
            self.ada_norm2 = AdaIN(oup_c, code)
        
        # In case if it is the initial block, learnable constant is created
        if self.initial:            
            self.constant = nn.Parameter(torch.randn(1, inp_c, 4, 4), requires_grad=True)
            if self.use_coord:                
                self.conv1 = ExplicitCoordConv(inp_c, inp_c, kernel_size=3, padding=1)
        else:
            if self.use_coord:                
                self.conv1 = ExplicitCoordConv(inp_c, inp_c, kernel_size=3, padding=1)
            else:
                self.conv1 = lreq.Conv2d(inp_c, inp_c, kernel_size=3, padding=1)
            
        self.conv2 = lreq.Conv2d(inp_c, oup_c, kernel_size=3, padding=1)
        
        self.upsample = Upsample(inp_c) # nn.UpsamplingBilinear2d(scale_factor=2)
        self.blur = Blur(inp_c)

        if use_attn:
            self.attn = TripletAttention()
        
        if act:
            self.activation = Factory.get_activation(act)        
        
    def forward(self, x, w, bbox=None, e=None):
        """
        x - (N x C x H x W) -> N x C x H*2 x W*2
        w - (N x C), where A: (N x C) -> (N x (C x 2))
        e - (N x C x H x W)
        """
        i = x
        if self.initial:
            x = x.repeat(w.shape[0], 1, 1, 1)
            if self.use_coord and bbox is not None:
                x = self.conv1(x, bbox)
        else:
            x = self.upsample(x)            
            
            if self.blur_upsample:
                x = self.blur(x)
            if self.learn_blur:
                x = self.blur_affine(x)

            
            if self.use_coord:
                if bbox is not None:
                    x = self.conv1(x, bbox)
                else:
                    raise ValueError("Must pass 'bbox' when using coord conv ('use_coord' = True)")
            else:
                x = self.conv1(x)
            x = self.attn(x)
            x = self.activation(x)    
            
        x = self.B1(x)
        if self.learn_noise:
            x = self.B1_affine(x)   
        if self.learn_style:
            w = self.w1_affine(w)
        x = self.ada_norm1(x, w)        
        x = self.activation(self.conv2(x))
        
        x = self.B2(x)
        if self.learn_noise:
            x = self.B2_affine(x)
        if self.learn_style:
            w = self.w2_affine(w)
        x = self.ada_norm2(x, w)

        if self.learn_residual and not self.initial:
            i = self.res_conv(i)
            i = self.res_upsample(i)
            i = self.res_ups_affine(i)
            ratio = F.softmax(self.residual_gain, dim=0)
            x = x * ratio[0] + i * ratio[1]

        return x

   
class GeneratorSkipBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, blur_upsample=True, 
                 fused_scale=True, learn_blur=True, learn_residual=False, 
                 learn_style=False, learn_noise=False,
                 scale=4, act="leaky", norm="pixel"):
        super().__init__()

        print(f"[GeneratorBlock]\t Input Channel: {inp_c}, Output Channel: {oup_c}, Code Size: {code}, Scale: {scale}")
                
        self.initial = initial
        self.blur_upsample = blur_upsample
        self.learn_blur = learn_blur
        self.learn_residual = learn_residual
        self.learn_style = learn_style
        self.learn_noise = learn_noise

        # learnable affine transform to correct blur
        if self.learn_blur:
            self.blur_affine = set_scale(LearnableAffineTransform2d(scale=(inp_c, scale)))
        
        self.residual_gain = nn.Parameter(torch.from_numpy(np.array([1, -1], dtype=np.float32)))
        self.res_upsample = Upsample(inp_c) # nn.UpsamplingBilinear2d(scale_factor=2)
        self.res_blur = Blur(inp_c)
        if self.learn_blur:
            self.res_blur_affine = set_scale(LearnableAffineTransform2d(scale=inp_c))    

        # Learnable noise coefficients
        self.B = set_scale(IntermediateNoise(inp_c))
        # Learnable affine transform
        if self.learn_noise:
            self.B_affine = set_scale(LearnableAffineTransform2d(scale=(inp_c, scale)))
        
        # Each Ada PN contains learnable parameters A
        if norm == "pixel":
            self.norm = PixelNorm()
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d()
        
        # In case if it is the initial block, learnable constant is created
        if self.initial:
            self.constant = nn.Parameter(torch.randn(1, inp_c, 4, 4), requires_grad=True)
        else:
            self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, padding=1)
            
        self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, padding=1)
        
        self.upsample = Upsample(inp_c) # nn.UpsamplingBilinear2d(scale_factor=2)
        self.blur = Blur(inp_c)
        if act:
            self.activation = Factory.get_activation(act) 
        
    def forward(self, x):
        """
        x - (N x C x H x W)
        w - (N x C), where A: (N x C) -> (N x (C x 2))
        """
        x = self.upsample(x)            
            
        if self.blur_upsample:
            x = self.blur(x)
        if self.learn_blur:
            x = self.blur_affine(x)

        x = self.conv1(x)
        x = self.activation(x)    
        
        x = self.B(x)
        if self.learn_noise:
            x = self.B_affine(x)   
        x = self.norm(x)      

        return x
    
####################################################################################################################
################################################## Level 3 blocks ##################################################
####################################################################################################################

####################################################################################################################
###### L A T E N T - M A P P I N G #######--------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    def __init__(self, code=512, depth=4):
        super().__init__()
        self.code = code
        self.act = nn.LeakyReLU(0.2)
        
        self.f = [BallProjection()]
        for _ in range(depth-1):
            self.f.extend([lreq.Linear(code, code), nn.LeakyReLU(0.2)])
        self.f = self.f + [lreq.Linear(code, code)]
        self.f = nn.Sequential(*self.f)
    
    def forward(self, z1, scale, z2=None, p_mix=0.9):
        """
        Outputs latent code of size (bs x n_blocks x latent_code_size), performing style mixing
        """
        n_blocks = int(log2(scale) - 1)
        
        # Make latent code of style (bs x n_blocks x latent_code_size)
        style1 = self.f(z1)[:, None, :].repeat(1, n_blocks, 1)
        
        # Randomly decide if style mixing should be performed or not
        if (random.random() < p_mix) & (z2 is not None) & (n_blocks!=1):
            style2 = self.f(z2)[:, None, :].repeat(1, n_blocks, 1)
            layer_idx = torch.arange(n_blocks)[None, :, None].to(z1.device)
            mixing_cutoff = random.randint(1, n_blocks-1) #Insert style2 in 8x8 ... 1024x1024 blocks
            return torch.where(layer_idx < mixing_cutoff, style1, style2)
        else:
            return style1 # If style2 is not used

####################################################################################################################
####### D I S C R I M I N A T O R ########--------------------------------------------------------------------------
                       
class Discriminator(nn.Module):
    def __init__(self, code=512, depth=3):
        super().__init__()
        
        self.disc = []
        for index in range(depth - 1):            
            self.disc.extend([lreq.Linear(code, code), nn.LeakyReLU(0.2)])
        self.disc = self.disc + [lreq.Linear(code, 1)]
        self.disc = nn.Sequential(*self.disc)
                
    def forward(self, x):
        return self.disc(x)
    
####################################################################################################################
############## E N C O D E R #############--------------------------------------------------------------------------
    
class Encoder(nn.Module):
    def __init__(self, max_fm, code, 
                 blocks={1024:{"enc":[8,8],"rgb":8},
                         512:{"enc":[8,4],"rgb":8},
                         256:{"enc":[4,4],"rgb":4},
                         128:{"enc":[4,2],"rgb":4},
                         64:{"enc":[2,2],"rgb":2},
                         32:{"enc":[2,1],"rgb":2},
                         16:{"enc":[1,1],"rgb":1},
                         8:{"enc":[1,1],"rgb":1},
                         4:{"enc":[1,1],"rgb":1}}, 
                 fc_intital=True, blur_downsample=False, learn_blur=False, verbose=False):
        super().__init__()
        
        self.code = code  
        encoder_blocks = []
        from_rgb_blocks = []
        self.max_scale = 0
        for i, (scale, settings) in enumerate(blocks.items()):
            if verbose: print(f"[Encoder]\t Block {i} for scale {scale} with settings: {settings}")      
            encoder_blocks.append(EncoderBlock(max_fm//settings["enc"][0], max_fm//settings["enc"][1], code, 
                                               final=scale==4, 
                                               blur_downsample=blur_downsample, 
                                               learn_blur=learn_blur))
            from_rgb_blocks.append(FromRGB(3, max_fm//settings["rgb"]))
            self.max_scale = max(scale, self.max_scale)
        print(f"[Encoder]\t Max scale achievable: {self.max_scale}")

        self.encoder = nn.ModuleList(encoder_blocks)        
        self.fromRGB =  nn.ModuleList(from_rgb_blocks)
        
    def forward(self, x, alpha=1., return_norm=False):
        if return_norm:
            # return norm is not 0, it should be set to the layer index (negative)
            # of the layer to use for the return value
            norm_layer_num = return_norm
            return_norm = True
        n_blocks = int(log2(x.shape[-1]) - 1) # Compute the number of required blocks

        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            _, w1, w2, n = self.encoder[-1](self.fromRGB[-1](x, downsample=False), return_norm=return_norm)
            if return_norm: return n
            else: return w1 + w2
            
        # Store w
        w = torch.zeros(x.shape[0], self.code).to(x.device)
        
        # Convert input from RGB and blend across 2 scales
        if alpha < 1:
            inp_top, w1, w2, n = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))
            inp_left = self.fromRGB[-n_blocks+1](x, downsample=True)
            x = inp_left.mul(1 - alpha) + inp_top.mul(alpha)
        else: # Use top shortcut
            x, w1, w2, n = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))

        w += (w1 + w2)

        for index in range(-n_blocks + 1, 0):
            x, w1, w2, n = self.encoder[index](x, return_norm=return_norm)
            w += (w1 + w2)
            if return_norm and index == norm_layer_num:
                return n

        return w

####################################################################################################################
############ G E N E R A T O R ###########--------------------------------------------------------------------------

class StyleGenerator(nn.Module):
    def __init__(self, max_fm, code, 
                 blocks={4:{"gen":[1,1],"rgb":1},
                         8:{"gen":[1,1],"rgb":1},
                         16:{"gen":[1,1],"rgb":1},
                         32:{"gen":[1,2],"rgb":2},
                         64:{"gen":[2,2],"rgb":2},
                         128:{"gen":[2,4],"rgb":4},
                         256:{"gen":[4,4],"rgb":4},
                         512:{"gen":[4,8],"rgb":8},
                         1024:{"gen":[8,8],"rgb":8}}, 
                 blur_upsample=False, learn_blur=False, verbose=False):
        super().__init__()

        generator_blocks = []
        to_rgb_blocks = []
        self.max_scale = 0
        for i, (scale, settings) in enumerate(blocks.items()):
            if verbose: print(f"[StyleGenerator]\t Block {i} for scale {scale} with settings: {settings}")      
            generator_blocks.append(GeneratorBlock(max_fm//settings["gen"][0], max_fm//settings["gen"][1], code, 
                                    initial=i==0, 
                                    blur_upsample=blur_upsample, 
                                    learn_blur=learn_blur,
                                    scale=scale))
            to_rgb_blocks.append(ToRGB(max_fm//settings["rgb"], 3))
            self.max_scale = max(scale, self.max_scale)
        print(f"[StyleGenerator]\t Max scale achievable: {self.max_scale}")
        self.generator = nn.ModuleList(generator_blocks)          
        self.toRGB =  nn.ModuleList(to_rgb_blocks)
        
    def get_blocks_parameters(self):
        pars = []
        for block in self.generator:
            named_block = list(block.named_parameters())
            for index in range(len(named_block)):
                if 'ada_in' not in named_block[index][0]:
                    pars.append(named_block[index][1])
        return pars
    
    def get_styles_parameters(self):
        # Get modules, corresponding to mapping from latent codes to Feature map's channel-wise coefficients
        return nn.ModuleList([module.ada_in1.A for module in self.generator] + \
                             [module.ada_in2.A for module in self.generator]).parameters()
    
    def ema(self, model, beta=0.999):
        """
        If Generator is used in running average regime, takes optimized model during training and
        adds it's weights into a linear combination
        """        
        runing_parameters = dict(self.named_parameters())
        for key in runing_parameters.keys():
            runing_parameters[key].data.mul_(beta).add_(1 - beta, dict(model.named_parameters())[key].data)
        
    def forward(self, w, scale, alpha=1, return_norm=False):
        if return_norm:
            # return norm is not 0, it should be set to the layer index (positive)
            # of the layer to use for the return value
            norm_layer_num = return_norm
            return_norm = True
        n_blocks = int(log2(scale) - 1) # Compute the number of required blocks        
                
        # Take learnable constant as an input
        inp = self.generator[0].constant

        if return_norm:
            norm = None
        
        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            norm = self.generator[0](inp, w[:, 0])
            if return_norm: return norm
            else: return self.toRGB[0](norm, upsample=False)

        # If scale >= 8
        activations_2x = []
        for index in range(n_blocks):
            inp = self.generator[index](inp, w[:, index])
            
            # if returning norm, cut out early
            if return_norm and norm_layer_num == index:
                return inp

            # Save last 2 scales
            if index in [n_blocks-2, n_blocks-1]:
                activations_2x.append(inp)
        
        inp = self.toRGB[n_blocks-1](activations_2x[1], upsample=False)
        if alpha < 1: # In case if blending is applied            
            inp = (1 - alpha) * self.toRGB[n_blocks-2](activations_2x[0], upsample=True) + alpha * inp
        return inp
