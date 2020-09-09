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

from scaled_layers import set_scale, ScaledLinear, ScaledConv2d 
from downsample import Downsample
from losses import logcosh, xtanh, xsigmoid

####################################################################################################################
################################################## Level 0 blocks ##################################################
####################################################################################################################

####################################################################################################################
############### L O S S ##################--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        return logcosh(y_t, y_prime_t)

class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):        
        return xtanh(y_t, y_prime_t)

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        return xsigmoid(y_t, y_prime_t)

####################################################################################################################
###### N O R M A L I Z A T I O N #########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ------------------------------------------------------------------------------------------------------------------
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

# ------------------------------------------------------------------------------------------------------------------
# Activation Norm - Normalized
# ------------------------------------------------------------------------------------------------------------------
class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]
        
# ------------------------------------------------------------------------------------------------------------------
# Spectral Norm - Normalized
# ------------------------------------------------------------------------------------------------------------------
class SpectralNorm(nn.Module):
    """ Spectral Normalization - Normalized Flow layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.norm = nn.utils.spectral_norm()
        self.denorm = nn.utils.remove_spectral_norm()

    def forward(self, x):
        z = self.norm(x)
        return z

    def inverse(self, z):
        x = self.denorm(z)
        return x

####################################################################################################################
###### T R A N S F O R M A T I O N #######--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Model Layer Weights as Learnable Gaussian
# per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
# ------------------------------------------------------------------------------------------------------------------
class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet

# ------------------------------------------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------------------------------------------
class PlanarTransform(nn.Module):
    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return f_z, sum_log_abs_det_jacobians

# ------------------------------------------------------------------------------------------------------------------
# High Pass Filter Transformation
# ------------------------------------------------------------------------------------------------------------------
class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# ------------------------------------------------------------------------------------------------------------------
# Relative X,Y Coordinate Values Channel
# adds coords for each filter location, resulting in size [B,H,W,C+1] - should be the first layer
# ------------------------------------------------------------------------------------------------------------------
class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper StarGANv2."""
    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel

####################################################################################################################
################################################## Level 1 blocks ##################################################
####################################################################################################################

####################################################################################################################
###### N O R M A L I Z A T I O N #########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Adaptive Instance normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ------------------------------------------------------------------------------------------------------------------
class AdaIN(nn.Module):
    def __init__(self, n_channels, code):
        super().__init__()
        
        self.insance_norm = nn.InstanceNorm2d(n_channels, affine=False, eps=1e-8)
        self.A = ScaledLinear(code, n_channels * 2)
        
        # StyleGAN
        # self.A.linear.bias.data = torch.cat([torch.ones(n_channels), torch.zeros(n_channels)])
        
    def forward(self, x, style):
        """
        x - (N x C x H x W)
        style - (N x (Cx2))
        """        
        # Project project style vector(w) to  mu, sigma and reshape it 2D->4D to allow channel-wise operations        
        style = self.A(style)
        y = style.view(style.shape[0], 2, style.shape[1]//2).unsqueeze(3).unsqueeze(4)
        
        return torch.addcmul(y[:, 1], value=1., tensor1=y[:, 0] + 1, tensor2 = x)        

####################################################################################################################
###### T R A N S F O R M A T I O N #######--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Unit Hypersphere Projection
# ------------------------------------------------------------------------------------------------------------------
class BallProjection(nn.Module):
    """
    Constraint norm of an input noise vector to be sqrt(latent_code_size)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div((torch.mean(x.pow(2), dim=1, keepdim=True).add(1e-8)).pow(0.5))

# ------------------------------------------------------------------------------------------------------------------
# Normal Distribution with Learnable Scale
# ------------------------------------------------------------------------------------------------------------------
class IntermediateNoise(nn.Module):
    def __init__(self, inp_c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, inp_c, 1, 1), requires_grad=True)
    
    def forward(self, x):
        if self.training:
            noise = torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(x.device)
            return x + (noise * self.weight)
        else:
            return x
    
# ------------------------------------------------------------------------------------------------------------------
# Differentiable Blur 
# based on Scaled Conv2D kernel
# ------------------------------------------------------------------------------------------------------------------
class BlurFunctionBackward(Function):
    """
    Official Blur implementation
    https://github.com/adambielski/perturbed-seg/blob/master/stylegan.py
    """
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

# ------------------------------------------------------------------------------------------------------------------
# Learnable Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class AffineTransform(nn.Module):
    def __init__(self, learnable=True, scale=4):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, scale, scale, 1)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(1, scale, scale, 1)).requires_grad_(learnable)

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x        
        return z

# ------------------------------------------------------------------------------------------------------------------
# Project from RGB space to Feature space
# Entry point for sending an image into the network
# ------------------------------------------------------------------------------------------------------------------
class FromRGB(nn.Module):
    def __init__(self, inp_c, oup_c):
        super(FromRGB, self).__init__()
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
    def __init__(self, inp_c, oup_c):
        super(ToRGB, self).__init__()
        self.to_rgb = ScaledConv2d(inp_c, oup_c, 1, 1, 0)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            
    def forward(self, x, upsample=False):
        if upsample:
            return self.to_rgb(self.upsample(x.contiguous()))
        else:
            return self.to_rgb(x.contiguous())

####################################################################################################################
################################################## Level 2 blocks ##################################################
####################################################################################################################

####################################################################################################################
############## E N C O D E R #############--------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, final=False, blur_downsample=False, fused_scale=True, learn_blur=False):
        super().__init__()
        
        self.final = final
        self.blur_downsample = blur_downsample
        self.learn_blur = learn_blur

        self.learned_affine = AffineTransform(learnable=True, scale=oup_c if final else inp_c)
        
        self.in1 = nn.InstanceNorm2d(inp_c, affine=False)
        self.in2 = nn.InstanceNorm2d(oup_c, affine=False)
        
        self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, stride=1, padding=1)
        self.style_mapping1 = ScaledLinear(2 * inp_c, code)
        
        if final:
            self.fc = ScaledLinear(inp_c * 4 * 4, oup_c)
            self.style_mapping2 = ScaledLinear(oup_c, code)
        else:
            self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, stride=1, padding=1)    
            self.style_mapping2 = ScaledLinear(2 * oup_c, code)
            
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2, 2)
        
        self.blur = Blur(inp_c)
        
    def forward(self, x, return_norm=False):
        
        x = self.act(self.conv1(x))
        statistics1 = torch.cat([x.mean(dim=[2,3]), x.std(dim=[2,3])], dim=1)
        style1 = self.style_mapping1(statistics1)
        x = self.in1(x)
        if return_norm: norm = x
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
        
        if return_norm: return x, style1, style2, norm
        else: return x, style1, style2, None

####################################################################################################################
############ G E N E R A T O R ###########--------------------------------------------------------------------------
    
class GeneratorBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, initial=False, blur_upsample=False, fused_scale=True, learn_blur=False, scale=4):
        super().__init__()
                
        self.initial = initial
        self.blur_upsample = blur_upsample
        self.learn_blur = learn_blur

        # learnable affine transform to correct blur
        if self.learn_blur:
            self.learned_affine = AffineTransform(learnable=True, scale=inp_c)

        # Learnable noise coefficients
        self.B1 = set_scale(IntermediateNoise(inp_c))
        self.B2 = set_scale(IntermediateNoise(oup_c))
        
        # Each Ada IN contains learnable parameters A
        self.ada_in1 = AdaIN(inp_c, code)
        self.ada_in2 = AdaIN(oup_c, code)
        
        # In case if it is the initial block, learnable constant is created
        if self.initial:
            self.constant = nn.Parameter(torch.randn(1, inp_c, 4, 4), requires_grad=True)
        else:
            self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, padding=1)
            
        self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, padding=1)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.blur = Blur(inp_c)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        """
        x - (N x C x H x W)
        w - (N x C), where A: (N x C) -> (N x (C x 2))
        """
        if self.initial:
            x = x.repeat(w.shape[0], 1, 1, 1)            
        else:
            x = self.upsample(x)
            x = self.conv1(x)
            
            if self.blur_upsample:
                x = self.blur(x)
            if self.learn_blur:
                x = self.learned_affine(x)
            
            x = self.activation(x)    
            
        x = self.B1(x)
        if self.initial and self.learn_blur:
            x = self.learned_affine(x)
        
        x = self.ada_in1(x, w)        
        x = self.activation(self.conv2(x))
        
        x = self.B2(x)
            
        return self.ada_in2(x, w)
    
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
            self.f.extend([ScaledLinear(code, code), nn.LeakyReLU(0.2)])
        self.f = self.f + [ScaledLinear(code, code)]
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
            self.disc.extend([ScaledLinear(code, code), nn.LeakyReLU(0.2)])
        self.disc = self.disc + [ScaledLinear(code, 1)]
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