import random
import functools
import math
from math import log2, ceil

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm, remove_spectral_norm
import torch.distributions as D
from torch.autograd import Function
from torch.nn import functional as F

import numpy as np

from CALAE.layers.scaled import set_scale, ScaledLinear, ScaledConv2d 
from CALAE.layers import lreq
import CALAE.losses
from CALAE.layers.activations import Mish
from CALAE.layers.attention import UNetAttention, SelfAttention, TripletAttention
from CALAE.layers.spectralnorm import SN, SNConv2d, SNLinear
from CALAE.layers.coordconv import ExplicitCoordConv
from CALAE.layers.image import ToRGB, FromRGB

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
                 fc_intital=True, 
                 blur_downsample=False, 
                 learn_blend=True,
                 learn_blur=True, 
                 use_coord=True,
                 use_attn=False,
                 verbose=False):
        """ALAE based encoder that takes an image of a certain scale and outputs a style vector that exists in a latent space being learned.

        Depending on the parameters passed to `forward`, can return a variety of information calculated by this network. By default,
        only the final style embedding is returned. The style embeddings from each layer can be returned as well, or the first instance 
        norm of each layer, or a combination of all 3 as a tuple.

        Args:
            max_fm (int): Max resolution
            code (int): Length of style vector to output, length of the dimension of the random code sent into the generator
            blocks (dict, optional): A description of the shape of each layer to be created, defaults to a `max_fm` of 1024, so reduce it to save 
                memory. The keys of the dict are the resolution of the input, then the keys of each inner dict are the settings of the "encoder block"
                and of the "FromRGB" layers - the input and output dimensions for the block and the input dimension of the layer (the value is a divisor
                of the `max_fm`). Defaults to {1024:{"enc":[8,8],"rgb":8}, 512:{"enc":[8,4],"rgb":8}, 256:{"enc":[4,4],"rgb":4}, 128:{"enc":[4,2],"rgb":4}, 64:{"enc":[2,2],"rgb":2}, 32:{"enc":[2,1],"rgb":2}, 16:{"enc":[1,1],"rgb":1}, 8:{"enc":[1,1],"rgb":1}, 4:{"enc":[1,1],"rgb":1}}.
            fc_intital (bool, optional): Whether or not the first layer is a fully connected dense layer instead of a convolutional. Defaults to True.
            blur_downsample (bool, optional): Apply blur after downsampling. Defaults to False.
            learn_blend (bool, optional): Apply a learnable ratio when combining each block's style vector outputs, which are otherwise added together. 
                Defaults to True.
            learn_blur (bool, optional): Apply a learnable affine transformation after the downsample blur. Defaults to True.
            use_coord (bool, optional): Use a CoordConv instead of a normal convolution, requires that a bbox param with the coordinates that the
                input originate from. Only useful if being trained on images much larger than the max size. Defaults to True.
            use_attn (bool, optional): Use a self attention layer in the encoder block. Defaults to False.
            verbose (bool, optional): Output additional console info during building of model. Defaults to False.
        """
        super().__init__()
        self.code = code

        # layer toggles
        self.learn_blend = learn_blend
        self.use_coord = use_coord
        self.use_attn = use_attn

        # temporary module lists
        encoder_blocks = []
        from_rgb_blocks = []
        if self.use_attn:
            attn_blocks = []
        if self.use_coord:
            coord_blocks = []
        if self.learn_blend: 
            blend_gains = []
        self.max_scale = 0
        for i, (scale, settings) in enumerate(blocks.items()):
            if verbose: print(f"[Encoder]\t Block {i} for scale {scale} with settings: {settings}")      
            encoder_blocks.append(EncoderBlock(max_fm//settings["enc"][0], max_fm//settings["enc"][1], code,  
                                               scale=scale,
                                               final=scale==4, 
                                               blur_downsample=blur_downsample, 
                                               learn_blur=learn_blur))
            from_rgb_blocks.append(FromRGB(3, max_fm//settings["rgb"]))
            if self.use_attn:
                attn_blocks.append(SelfAttention(max_fm//settings["enc"][0], 'leaky'))
            if self.use_coord:
                coord_blocks.append(ExplicitCoordConv(max_fm//settings["enc"][1], max_fm//settings["enc"][1], kernel_size=1, padding=0))
            if self.learn_blend:
                blend_gains.append(nn.Parameter(torch.from_numpy(np.array([1, -1], dtype=np.float32)), requires_grad=True))
            self.max_scale = max(scale, self.max_scale)
        print(f"[Encoder]\t Max scale achievable: {self.max_scale}")

        self.encoder = nn.ModuleList(encoder_blocks)        
        self.fromRGB =  nn.ModuleList(from_rgb_blocks)
        if self.use_attn:
            self.attn = nn.ModuleList(attn_blocks)
        if self.use_coord:
            self.coord = nn.ModuleList(coord_blocks)
        if self.learn_blend:
            self.blend = nn.ModuleList(blend_gains)
        
    def forward(self, x, alpha=1., return_norm=False, return_blocks=False, bbox=None):
        if return_blocks:
            blocks = []
        if return_norm:
            # return norm is not 0, it should be set to the layer index (negative)
            # of the layer to use for the return value
            norm_layer_num = return_norm
            return_norm = True
            norms = []
        n_blocks = int(log2(x.shape[-1]) - 1) # Compute the number of required blocks

        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            x = self.fromRGB[-1](x, downsample=False)
            if self.use_attn:
                x = self.attn[-1](x)
            if self.use_coord and bbox is not None:
                x = self.coord[-1](x, bbox)
            x, w1, w2, n = self.encoder[-1](x)
            if return_norm and not return_blocks and -1 in norm_layer_num: 
                return n
            if self.learn_blend:
                ratio = F.softmax(self.blend[-1], dim=0)
                w = (w1 * ratio[0] + w2 * ratio[1])
            else:
                w = (w1 + w2)  
            if return_blocks:
                if return_norm:
                    return n, [(x, w, n)]
                else:
                    return [(x, w, n)]
            else:
                return w
            
        # Store w
        w = torch.zeros(x.shape[0], self.code).to(x.device)
        
        # Convert input from RGB and blend across 2 scales
        if alpha < 1:
            x_top = self.fromRGB[-n_blocks](x, downsample=False)
            if self.use_attn:
                x_top = self.attn[-n_blocks](x_top)
            if self.use_coord and bbox is not None:
                x_top = self.coord[-n_blocks](x_top, bbox)
            inp_top, w1, w2, n = self.encoder[-n_blocks](x_top)

            inp_left = self.fromRGB[-n_blocks+1](x, downsample=True)
            x = inp_left.mul(1 - alpha) + inp_top.mul(alpha)
        else: # Use top shortcut
            x = self.fromRGB[-n_blocks](x, downsample=False)
            if self.use_attn:
                x = self.attn[-n_blocks](x)
            if self.use_coord and bbox is not None:
                x = self.coord[-n_blocks](x, bbox)
            x, w1, w2, n = self.encoder[-n_blocks](x)

        if self.learn_blend:
            ratio = F.softmax(self.blend[-n_blocks], dim=0)
            w += (w1 * ratio[0] + w2 * ratio[1])
        else:
            w += (w1 + w2) 

        if return_blocks:
            blocks.append((x, w, n))

        for index in range(-n_blocks + 1, 0):
            if self.use_attn:
                x = self.attn[index](x)
            if self.use_coord:
                x = self.coord[index](x, bbox)
            x, w1, w2, n = self.encoder[index](x)
            if return_norm and index in norm_layer_num:
                norms.append(n)
            if self.learn_blend:
                ratio = F.softmax(self.blend[index], dim=0)
                w += (w1 * ratio[0] + w2 * ratio[1])
            else:
                w += (w1 + w2)
            
            if return_blocks:
                blocks.append((x, w, n))
                
        if return_norm and return_blocks:
            return norms, w, blocks
        elif return_norm:
            return norms
        elif return_blocks:
            return w, blocks
        else:
            return w


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
