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
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

#from models.unet_generator import *
from CALAE.models import stylegan2
from CALAE.metrics.perceptual import PerceptualLoss
from CALAE.layers.scaled import set_scale, ScaledLinear, ScaledConv2d 
from CALAE.layers import lreq
import CALAE.losses
from CALAE.layers.activations import Mish
from CALAE.layers.attention import UNetAttention, SelfAttention, TripletAttention
from CALAE.layers.spectralnorm import SN, SNConv2d, SNLinear
from CALAE.layers.coordconv import ExplicitAddCoords

import lpips
import piq

# ------------------------------------------------------------------------------------------------------------------

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat

def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)

# ------------------------------------------------------------------------------------------------------------------

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))

class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)

class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super().__init__()
        self.use_bias = True
        # initialize padding
        self.pad = Factory.get_pad_layer(padding)       

        # initialize normalization
        self.norm = Factory.get_normalization(norm, norm_dim=output_dim)        

        # initialize activation
        self.activation = Factory.get_activation(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        self.norm = Factory.get_normalization(norm, norm_dim=output_dim)        

        # initialize activation
        self.activation = Factory.get_activation(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class PixelAttention1D(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):
        super(PixelAttention1D, self).__init__()
        self.conv = lreq.Linear(nf, nf)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

####################################################################################################################
############# C O M P O N E N T ######################################----------------------------------------------
############# F A C T O R Y ####################--------------------------------------------------------------------
######## -----------------------------------------------------------------------------------------------------------
class Factory:
    @staticmethod
    def get_filter(filt_size=3):
        if(filt_size == 1):
            a = np.array([1., ])
        elif(filt_size == 2):
            a = np.array([1., 1.])
        elif(filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)

        return filt

    @staticmethod
    def get_pad_layer(pad_type):
        if(pad_type in ['refl', 'reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl', 'replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type == 'zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized' % pad_type)
        return PadLayer

    @staticmethod
    def get_activation(activation, **kwargs):
        # initialize activation
        if isinstance(activation, str):
            activation = activation.lower()

            if activation == 'relu':
                return nn.ReLU(inplace=True, **kwargs)
            elif activation in ['lrelu', 'leaky', 'leakyrelu']:
                return nn.LeakyReLU(0.2, inplace=True, **kwargs)
            elif activation == 'prelu':
                return nn.PReLU(**kwargs)
            elif activation == 'selu':
                return nn.SELU(inplace=True, **kwargs)
            elif activation == 'tanh':                
                return nn.Tanh(**kwargs)
            elif activation == 'mish':
                return Mish(**kwargs)
            elif activation == 'logcosh':
                return LogCoshLoss(**kwargs)
            elif activation == 'xtanh':
                return XTanhLoss(**kwargs)
            elif activation == 'xsigmoid':
                return XSigmoidLoss(**kwargs)
            elif activation == 'gelu':
                return nn.GELU(**kwargs)
            elif activation == 'celu':
                return nn.CELU(**kwargs)
            elif activation == "rrelu":
                return nn.RReLU(**kwargs)
            elif activation == 'sigmoid':
                return nn.Sigmoid(**kwargs)
            elif activation == 'elu':
                return nn.ELU(**kwargs)
            elif activation == 'hardshrink':
                return nn.Hardshrink(**kwargs)
            elif activation == 'hardtanh':
                return nn.Hardtanh(**kwargs)
            elif activation == 'hardswish':
                return nn.Hardswish(**kwargs)
            elif activation == 'logsigmoid':
                return nn.LogSigmoid(**kwargs)
            elif activation == 'relu6':
                return nn.Relu6(**kwargs)
            elif activation == 'softplus':
                return nn.Softplus(**kwargs)
            elif activation == 'softshrink':
                return nn.Softshrink(**kwargs)
            elif activation == 'softsign':
                return nn.Softsign(**kwargs)
            elif activation == 'softmin':
                return nn.Softmin(**kwargs)
            elif activation == 'softmax':
                return nn.Softmax(**kwargs)
            elif activation == 'softmax2d':
                return nn.Softmax2d(**kwargs)
            elif activation == 'logsoftmax':
                return nn.LogSoftmax(**kwargs)
            elif activation == 'tanhshrink':
                return nn.Tanhshrink(**kwargs)
            elif activation == 'threshold':
                return nn.Threshold(**kwargs)
            elif activation == 'none':
                return nn.Identity()
            else:
                assert 0, f"Unsupported activation: {activation}"
        return nn.Identity()

    @staticmethod
    def get_normalization(norm, norm_dim=1, **kwargs):
        if norm in ['bn2d', 'batch', 'bn', 'batchnorm2d', 'batchnorm']:
            return nn.BatchNorm2d(norm_dim, **kwargs)
        elif norm in ['bn1d', 'batchnorm1d']:
            return nn.BatchNorm1d(norm_dim, **kwargs)
        elif norm in ['bn3d', 'batchnorm3d']:
            return nn.BatchNorm3d(norm_dim, **kwargs)
        elif norm in ['group', 'groupnorm', 'gn']:
            return nn.GroupNorm(norm_dim, **kwargs)
        elif norm in ['localresponse', 'local', 'localresponsenorm', 'localnorm', 'lr', 'lrn']:
            return nn.LocalResponseNorm(norm_dim, **kwargs)
        elif norm in ['in1d', 'instancenorm1d']:
            return nn.InstanceNorm1d(norm_dim, **kwargs)
        elif norm in ['in2d', 'inst', 'in', 'instancenorm', 'instancenorm2d']:
            return nn.InstanceNorm2d(norm_dim, **kwargs)
        elif norm in ['in3d', 'instancenorm3d']:
            return nn.InstanceNorm3d(norm_dim, **kwargs)
        elif norm in ['pixel', 'pn', 'pixelnorm']:
            return PixelNorm(**kwargs)
        elif norm in ['act', 'an', 'actnorm']:
            return ActNorm(**kwargs)
        elif norm in ['ln', 'layer', 'layernorm']:
            return LayerNorm(norm_dim, **kwargs)
        elif norm == 'none':
            return nn.Identity()
        else:
            assert 0, f"Unsupported normalization: {norm}"
        return None

    @staticmethod
    def make_norm_1d(norm):
        if isinstance(norm, str):
            norm = 'batchnorm1d' if norm in ['batchnorm', 'bn', 'bn2d', 'bn3d', 'batch', 'batchnorm2d', 'batchnorm3d'] else norm
            norm = 'instancenorm1d' if norm in ['instancenorm', 'in', 'in2d', 'in3d', 'instance', 'instancenorm2d', 'instancenorm3d'] else norm
        return norm
    @staticmethod
    def make_norm_2d(norm):
        if isinstance(norm, str):
            norm = 'batchnorm2d' if norm in ['batchnorm', 'bn', 'bn1d', 'bn3d', 'batch', 'batchnorm1d', 'batchnorm3d'] else norm
            norm = 'instancenorm2d' if norm in ['instancenorm', 'in', 'in1d', 'in3d', 'instance', 'instancenorm1d', 'instancenorm3d'] else norm
        return norm
    @staticmethod
    def make_norm_3d(norm):
        if isinstance(norm, str):
            norm = 'batchnorm3d' if norm in ['batchnorm', 'bn', 'bn2d', 'bn1d', 'batch', 'batchnorm2d', 'batchnorm1d'] else norm
            norm = 'instancenorm3d' if norm in ['instancenorm', 'in', 'in2d', 'in1d', 'instance', 'instancenorm2d', 'instancenorm1d'] else norm
        return norm

# ------------------------------------------------------------------------------------------------------------------
####################################################################################################################
################################################## Level 0 blocks ##################################################
####################################################################################################################

####################################################################################################################
###### N O R M A L I Z A T I O N #########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Activation Norm - Normalized
# ------------------------------------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """ ActNorm layer; cf Glow section 3.1 """
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
# Layer Norm
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L812
# ------------------------------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * max((x.dim() - 2), 1)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
class LayerNorm1D(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
# ------------------------------------------------------------------------------------------------------------------
# Grouped Channel Norm
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1391
# ------------------------------------------------------------------------------------------------------------------
class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        x - (N x C x H x W)
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)  # [N1HW]
        y = x / y  # normalize the input x volume 
        return y
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
    Here f(x1) is a conv layer initialized to identity. (this has optionally been modified to use scaled init)
    """
    def __init__(self, n_channels, scaled=True):
        super().__init__()
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        if scaled:
            self.net = ScaledConv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)
        else:
            self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian        
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
# Planar Transformation
# Used in normalized flows, computes and returns the jacobian as well as a learnable complex affine transform
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
# Downsample with Efficient Filters
# ------------------------------------------------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = Factory.get_filter(self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = Factory.get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
# ------------------------------------------------------------------------------------------------------------------
# Upsample with Efficient Filters
# ------------------------------------------------------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = Factory.get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = Factory.get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

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
        
        self.norm = nn.InstanceNorm2d(n_channels, affine=False, eps=1e-8)
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

        x = self.norm(x)
        
        return torch.addcmul(y[:, 1], value=1., tensor1=y[:, 0] + 1, tensor2 = x)        
# ------------------------------------------------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ------------------------------------------------------------------------------------------------------------------
class AdaPN(nn.Module):
    def __init__(self, n_channels, code):
        super().__init__()
        self.A = ScaledLinear(code, n_channels * 2)

    def forward(self, x, style, alpha=1e-8):
        """
        x - (N x C x H x W)
        style - (N x (Cx2))
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # Project project style vector(w) to  mu, sigma and reshape it 2D->4D to allow channel-wise operations  
        style = self.A(style)
        z = style.view(style.shape[0], 2, style.shape[1]//2).unsqueeze(3).unsqueeze(4)
        # original PixelNorm
        y = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)  # [N1HW]
        y = x / y  # normalize the input x volume        
        # addcmul like in AdaIN
        return torch.addcmul(z[:, 1], value=1., tensor1=z[:, 0] + 1, tensor2=y)
# ------------------------------------------------------------------------------------------------------------------
# Minibatch standard deviation layer 
# reference: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py#L300
# ------------------------------------------------------------------------------------------------------------------
class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
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
        self.noise = None
    
    def forward(self, x, noise=None):
        if self.training:
            if noise is None and self.noise is None:
                noise = torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(x.device)
            elif noise is None:
                noise = self.noise
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
# Blur from original ALAE
# https://github.com/podgorskiy/ALAE/blob/master/net.py#L49
# ------------------------------------------------------------------------------------------------------------------
class BlurSimple(nn.Module):
    def __init__(self, channels):
        super().__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)
# ------------------------------------------------------------------------------------------------------------------
# Learnable Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class LearnableGaussianTransform0d(nn.Module):        
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(scale), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(scale), requires_grad=True)

    def forward(self, x):
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + (x - 1)
        return z
class LearnableGaussianTransform1d(nn.Module):        
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        self.A = lreq.Linear(scale[0], scale[1], bias=True)

    def forward(self, x):
        z = self.A.weight * x
        x = self.A.bias.exp() * x
      
        z = z + (x - 1)
        return z
class LearnableGaussianTransform2d(nn.Module):
    def __init__(self, scale=(4,4)):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)

    def forward(self, x):
        x = self.weight * x
        z = self.bias.exp() * x       
        return x + (z - 1)
# ------------------------------------------------------------------------------------------------------------------
# Learnable Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class LearnableAffineTransform0d(nn.Module):        
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        #self.A = ScaledLinear(scale[0], scale[1], bias=True)
        self.weight = nn.Parameter(torch.zeros(1, scale), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(scale), requires_grad=True)

    def forward(self, x):
        #z = self.A.linear.__dict__['_parameters']['weight_orig'] * x
        #x = self.A.linear.bias.exp() * x
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + x
        return z
class LearnableAffineTransform1d(nn.Module):        
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        self.A = ScaledLinear(scale[0], scale[1], bias=True)
        #self.weight = nn.Parameter(torch.zeros(1, 1, scale[1]), requires_grad=True)
        #self.bias = nn.Parameter(torch.zeros(scale[1]), requires_grad=True)

    def forward(self, x):
        z = self.A.linear.__dict__['_parameters']['weight_orig'] * x
        x = self.A.linear.bias.exp() * x
      
        z = z + x
        return z
class LearnableAffineTransform2d(nn.Module):
    def __init__(self, scale=(4,4)):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=True)

    def forward(self, x):
        x = self.weight.exp() * x
        z = self.bias + x       
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
    def __init__(self, inp_c, oup_c, use_bias=False):
        super(ToRGB, self).__init__()
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
####################################################################################################################
################################################## Level 2 blocks ##################################################
####################################################################################################################

####################################################################################################################
############## D I S C R I M I N A T O R #############--------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# Discriminator Block from original ALAE 
# https://github.com/podgorskiy/ALAE/blob/master/net.py#L129
# ------------------------------------------------------------------------------------------------------------------
class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=False, dense=False):
        super().__init__()
        self.conv_1 = ScaledConv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = BlurSimple(inputs)
        self.last = last
        self.dense_ = dense
        self.fused_scale = fused_scale
        if self.dense_:
            self.dense = ScaledLinear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ScaledConv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ScaledConv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.dense_:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)

        return x

# ------------------------------------------------------------------------------------------------------------------
# N-Layer Discriminator Base for Patch-GAN Style
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1285
# ------------------------------------------------------------------------------------------------------------------
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, scale, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        size = scale
        kw = 3
        padw = 1
        stride = 2 if (size % 2) == 0 else 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
            if stride > 1:
                sequence += [Downsample(ndf)]
        size = max(size // stride, 1)
        stride = 2 if (size % 2) == 0 else 1
        nf_mult = 1
        nf_mult_prev = 1
        if n_layers > 1:
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                if(no_antialias):
                    sequence += [
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)]
                    if stride > 1:
                        sequence += [Downsample(ndf * nf_mult)]
                
                size = max(size // stride, 1)
                stride = 2 if (size % 2) == 0 else 1

                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n_layers, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)

# ------------------------------------------------------------------------------------------------------------------
# Pixel Segmentation Discriminator
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1343
# ------------------------------------------------------------------------------------------------------------------
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

# ------------------------------------------------------------------------------------------------------------------
# Patch Discriminator for GAN Contrastive Learning using Patches from same image
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1375
# ------------------------------------------------------------------------------------------------------------------
class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, patch_size, scale, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc=input_nc, scale=scale, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, no_antialias=no_antialias)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        size = self.patch_size
        Y = H // size
        X = W // size
        x = x.view(B, C, Y, size, X, size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(x)



# ------------------------------------------------------------------------------------------------------------------
# Spectral Norm Discriminator
# 
# ------------------------------------------------------------------------------------------------------------------
# 2D Conv layer with spectral norm
class SNConv2d(lreq.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        lreq.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(lreq.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        lreq.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)

class SpectralDiscriminator(nn.Module):
    def __init__(self, code=512, depth=3, norm='layer', act='mish', verbose=False):
        super().__init__()

        self.norm = norm
        self.act = act
        
        self.disc = []
        for i in range(depth - 1):
            if verbose: print(f"[Discriminator]\t Block {i} for {code}d code using norm {norm}, act {act} and a residual skip delay of {skip_delay} (only applies if non zero)")   
            self.disc.extend(self.build_layer(code))
        self.disc = self.disc + [SNLinear(code, 1)]
        self.disc = nn.Sequential(*self.disc)

    def build_layer(self, code):
        layer = []
        layer.append(SNLinear(code, code))

        if self.norm:
            layer.append(Factory.get_normalization(Factory.make_norm_2d(self.norm)))
        if self.act:
            layer.append(Factory.get_activation(self.act))  
            
        return layer
        
    def forward(self, x):
        return self.disc(x)

       # ------------------------------------------------------------------------------------------------------------------
# Spectral Norm Feature Projection Network
# 
# ------------------------------------------------------------------------------------------------------------------ 
class SpectralFeatureProjectionNetwork(nn.Module):
    def __init__(self, code=512, depth=4, norm='none', act='leaky', use_attn=True, skip_delay=0, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.code = code
        self.act = act
        self.norm = norm
        self.use_attn = use_attn
        
        self.f = [BallProjection()]
        for i in range(depth-1):
            if verbose: print(f"[FeatureProjectionNetwork]\t Block {i} for {code}d code using norm {norm}, act {act} and a residual skip delay of {skip_delay} (only applies if non zero)")   
                        
            layer = self.build_layer(code)
            if skip_delay > 0 and i >= skip_delay:
                layer += self.f[i-skip_delay]
            
            self.f.extend(layer)
        self.f = self.f + [SNLinear(code, code)]
        self.f = nn.Sequential(*self.f)

    def build_layer(self, code, skip=None):
        layer = []
        layer.append(SNLinear(code, code))
        
        if self.act:
            layer.append(Factory.get_activation(self.act)) 
        if self.norm:
            layer.append(Normalize(power=2))
        if self.act == "selu":
            layer.append(nn.AlphaDropout(0.05))
        if self.use_attn:   
            layer.append(PixelAttention1D(code))
            
        return layer
    
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

"""
# ------------------------------------------------------------------------------------------------------------------
# Discriminator used in StyleGAN2
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1375
# ------------------------------------------------------------------------------------------------------------------
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, size, 
                 channel_multiplier=2, 
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        channels = {
                    4: 512,
                    8: 512,
                    16: 512,
                    32: 512,
                    64: 256 * channel_multiplier,
                    128: 128 * channel_multiplier,
                    256: 64 * channel_multiplier,
                    512: 32 * channel_multiplier,
                    1024: 16 * channel_multiplier,
                }
        convs = [stylegan2.ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(stylegan2.ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = stylegan2.ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            stylegan2.EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            stylegan2.EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
"""

# ------------------------------------------------------------------------------------------------------------------
# Feature Projection for GAN Contrastive Learning using Patches from same image
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L535
# ------------------------------------------------------------------------------------------------------------------
class PatchSampleFeatureProjection(nn.Module):
    def __init__(self, scale, patch_size, use_perceptual=False, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp        
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.use_percept = use_perceptual
        percep_layer_lookup = {
            4: 3,
            8: 8,
            16: 15,
            32: 22
        }
        if self.use_percept:
            self.percept = PerceptualLoss(ilayer=percep_layer_lookup[scale])

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            if self.use_percept:
                C, H, W = feat.size(0), feat.size(1), feat.size(2)
                size = self.patch_size
                Y = H // size
                X = W // size
                feat = feat.view(C, Y, size, X, size)
                feat_reshape = feat.permute(1, 3, 0, 2, 4).contiguous().view(Y * X, C, size, size)
                if num_patches > 0:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[0], device=feats[0].device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    x_sample = self.percept(feat_reshape[patch_id, ...])
                else:
                    x_sample = feat_reshape
                    patch_id = [range(feat_reshape.shape[0])]
            else:
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                if num_patches > 0:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                else:
                    x_sample = feat_reshape
                    patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


# ------------------------------------------------------------------------------------------------------------------
# Encoder for GAN Contrastive Learning using Patches from same image
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L535
# ------------------------------------------------------------------------------------------------------------------
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super().__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None

        for layer_id, layer in enumerate(self.model):
            print(layer_id, layer)

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
                 learn_style=False, learn_noise=False, use_attn=True,
                 scale=4, act="leaky", norm="pixel"):
        super().__init__()
                
        self.initial = initial
        self.blur_upsample = blur_upsample
        self.learn_blur = learn_blur
        self.learn_residual = learn_residual
        self.learn_style = learn_style
        self.learn_noise = learn_noise

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
        else:
            self.conv1 = lreq.Conv2d(inp_c, inp_c, kernel_size=3, padding=1)
            
        self.conv2 = lreq.Conv2d(inp_c, oup_c, kernel_size=3, padding=1)
        
        self.upsample = Upsample(inp_c) # nn.UpsamplingBilinear2d(scale_factor=2)
        self.blur = Blur(inp_c)

        if use_attn:
            self.attn = TripletAttention()
        
        if act:
            self.activation = Factory.get_activation(act)        
        
    def forward(self, x, w, e=None):
        """
        x - (N x C x H x W) -> N x C x H*2 x W*2
        w - (N x C), where A: (N x C) -> (N x (C x 2))
        e - (N x C x H x W)
        """
        i = x
        if self.initial:
            x = x.repeat(w.shape[0], 1, 1, 1)            
        else:
            x = self.upsample(x)            
            
            if self.blur_upsample:
                x = self.blur(x)
            if self.learn_blur:
                x = self.blur_affine(x)

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
###### F E A T U R E -> P R O J E C T I O N -> L A T E N T #######--------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Feature Projection Network from ALAE, customized
# ------------------------------------------------------------------------------------------------------------------
class FeatureProjectionNetwork(nn.Module):
    def __init__(self, code=512, depth=4, norm='none', act='leaky', use_attn=True, skip_delay=0, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.code = code
        self.act = act
        self.norm = norm
        self.use_attn = use_attn
        
        self.f = [BallProjection()]
        for i in range(depth-1):
            if verbose: print(f"[FeatureProjectionNetwork]\t Block {i} for {code}d code using norm {norm}, act {act} and a residual skip delay of {skip_delay} (only applies if non zero)")   
                        
            layer = self.build_layer(code)
            if skip_delay > 0 and i >= skip_delay:
                layer += self.f[i-skip_delay]
            
            self.f.extend(layer)
        self.f = self.f + [lreq.Linear(code, code)]
        self.f = nn.Sequential(*self.f)

    def build_layer(self, code, skip=None):
        layer = []
        layer.append(lreq.Linear(code, code))
        
        if self.act:
            layer.append(Factory.get_activation(self.act)) 
        if self.norm:
            layer.append(Normalize(power=2))
        if self.act == "selu":
            layer.append(nn.AlphaDropout(0.05))
        if self.use_attn:   
            layer.append(PixelAttention1D(code))
            
        return layer
    
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
# ------------------------------------------------------------------------------------------------------------------
# Progressive Discriminator Network from ALAE, customized
# ------------------------------------------------------------------------------------------------------------------          
class Discriminator(nn.Module):
    def __init__(self, code=512, depth=3, norm='layer', act='mish', verbose=False):
        super().__init__()

        self.norm = norm
        self.act = act
        
        self.disc = []
        for i in range(depth - 1):
            if verbose: print(f"[Discriminator]\t Block {i} for {code}d code using norm {norm}, act {act} and a residual skip delay of {skip_delay} (only applies if non zero)")   
            self.disc.extend(self.build_layer(code))
        self.disc = self.disc + [ScaledLinear(code, 1)]
        self.disc = nn.Sequential(*self.disc)

    def build_layer(self, code):
        layer = []
        layer.append(ScaledLinear(code, code))

        if self.norm:
            layer.append(Factory.get_normalization(Factory.make_norm_2d(self.norm)))
        if self.act:
            layer.append(Factory.get_activation(self.act))  
            
        return layer
        
    def forward(self, x):
        return self.disc(x)

####################################################################################################################
############## E N C O D E R #############--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Progressive Encoder Network from ALAE, customized
# ------------------------------------------------------------------------------------------------------------------
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
                 verbose=False):
        super().__init__()
        
        self.learn_blend = learn_blend
        
        
        self.code = code  
        encoder_blocks = []
        from_rgb_blocks = []
        attn_blocks = []
        if learn_blend:
            self.blend_gains = []
        self.max_scale = 0
        for i, (scale, settings) in enumerate(blocks.items()):
            if verbose: print(f"[Encoder]\t Block {i} for scale {scale} with settings: {settings}")      
            encoder_blocks.append(EncoderBlock(max_fm//settings["enc"][0], max_fm//settings["enc"][1], code,  
                                               scale=scale,
                                               final=scale==4, 
                                               blur_downsample=blur_downsample, 
                                               learn_blur=learn_blur))
            from_rgb_blocks.append(FromRGB(3, max_fm//settings["rgb"]))
            attn_blocks.append(SelfAttention(max_fm//settings["enc"][1], 'leaky'))
            if learn_blend:
                self.blend_gains.append(nn.Parameter(torch.from_numpy(np.array([1, -1], dtype=np.float32)), requires_grad=True))
            self.max_scale = max(scale, self.max_scale)
        print(f"[Encoder]\t Max scale achievable: {self.max_scale}")

        self.encoder = nn.ModuleList(encoder_blocks)        
        self.fromRGB =  nn.ModuleList(from_rgb_blocks)
        self.attn = nn.ModuleList(attn_blocks)
        
    def forward(self, x, alpha=1., return_norm=False, return_blocks=False):
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
            x, w1, w2, n = self.encoder[-1](self.fromRGB[-1](x, downsample=False))
            if return_norm and not return_blocks and -1 in norm_layer_num: 
                return n
            if self.learn_blend:
                ratio = F.softmax(self.blend_gains[-1], dim=0)
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
            inp_top, w1, w2, n = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))
            inp_left = self.fromRGB[-n_blocks+1](x, downsample=True)
            x = inp_left.mul(1 - alpha) + inp_top.mul(alpha)
        else: # Use top shortcut
            x, w1, w2, n = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))

        #w += (w1 + w2)
        if self.learn_blend:
            ratio = F.softmax(self.blend_gains[-n_blocks], dim=0)
            w += (w1 * ratio[0] + w2 * ratio[1])
        else:
            w += (w1 + w2) 

        if return_blocks:
            blocks.append((x, w, n))

        for index in range(-n_blocks + 1, 0):
            x, w1, w2, n = self.encoder[index](x)
            if return_norm and index in norm_layer_num:
                norms.append(n)
            if self.learn_blend:
                ratio = F.softmax(self.blend_gains[index], dim=0)
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

####################################################################################################################
############ G E N E R A T O R ###########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Progressive 2d RGB Image Generator Network with Style-normalization Inputs from ALAE, customized
# ------------------------------------------------------------------------------------------------------------------
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
                 blur_upsample=False, 
                 learn_blur=False, 
                 learn_residual=False, 
                 learn_style=False, 
                 learn_noise=False, 
                 use_attn=True,
                 act="leaky",
                 norm="instance",
                 verbose=False):
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
                                    learn_residual=learn_residual,
                                    learn_style=learn_style,
                                    learn_noise=learn_noise,
                                    use_attn=use_attn,
                                    act=act,
                                    norm=norm,
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
                if 'ada_norm' not in named_block[index][0]:
                    pars.append(named_block[index][1])
        return pars
    
    def get_styles_parameters(self):
        # Get modules, corresponding to mapping from latent codes to Feature map's channel-wise coefficients
        return nn.ModuleList([module.ada_norm1.A for module in self.generator] + \
                             [module.ada_norm2.A for module in self.generator]).parameters()
    
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
            norm_layer_nums = return_norm
            return_norm = True
            norms = []
        n_blocks = int(log2(scale) - 1) # Compute the number of required blocks        
                
        # Take learnable constant as an input
        inp = self.generator[0].constant
        
        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            norm = self.generator[0](inp, w[:, 0]) 
            if return_norm: return [norm]
            else: return self.toRGB[0](norm, upsample=False)

        # If scale >= 8
        activations_2x = []
        for index in range(n_blocks):
            inp = self.generator[index](inp, w[:, index])
            
            # if returning norm, cut out early
            if (return_norm and index in norm_layer_nums):
                norms.append(inp)

            # Save last 2 scales
            if index in [n_blocks-2, n_blocks-1]:
                activations_2x.append(inp)
        
        inp = self.toRGB[n_blocks-1](activations_2x[1], upsample=False)
        
        if alpha < 1: # In case if blending is applied            
            inp = (1 - alpha) * self.toRGB[n_blocks-2](activations_2x[0], upsample=True) + alpha * inp
            if return_norm and n_blocks-1 in norm_layer_nums:
                norms.append(inp)
        else:
            if return_norm and n_blocks-1 in norm_layer_nums:
                norms.append(inp)
        if return_norm:
            return norms
        else:
            return inp


class UNetStyle(nn.Module):
    def __init__(self, g_params, e_params, f_params, verbose=False):
        super().__init__()
        self.encoder = Encoder(**e_params)
        self.generator = StyleGenerator(**g_params)
        self.projector = FeatureProjectionNetwork(**f_params)
        self.verbose = verbose

    def forward(self, x, w, scale, alpha, return_norm=False, mode='enc-dec'):
        if mode == 'enc-dec':
            if return_norm:
                E_n, w, E_x = self.encoder.forward(x, scale, alpha, return_norm=True, return_blocks=True)
                return E_n, self.generator.forward(w, scale, E_x, alpha, return_norm=True)
            else:
                w, E_x = self.encoder.forward(x, scale, alpha, return_norm=False, return_blocks=True)
                return self.generator.forward(w, scale, E_x, alpha, return_norm=False)
            
    def generate(self, w, scale, e_x, alpha=1, return_norm=False):
        if return_norm:
            # return norm is not 0, it should be set to the layer index (positive)
            # of the layer to use for the return value
            norm_layer_nums = return_norm
            return_norm = True
            norms = []
        n_blocks = int(log2(scale) - 1) # Compute the number of required blocks        
                
        # Take learnable constant as an input
        inp = self.generator.generator[0].constant
        
        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            norm = self.generator.generator[0](inp, w[:, 0])             
            if return_norm: 
                return [norm]
            else: 
                E_out = E_x[-1]
                norm = torch.cat(E_out, norm, dim=1)
                return self.generator.toRGB[0](norm, upsample=True)

        # If scale >= 8
        activations_2x = []
        for index in range(n_blocks):
            inp = self.generator.generator[index](inp, w[:, index])
            
            # if returning norm, cut out early
            if (return_norm and index in norm_layer_nums):
                norms.append(inp)

            E_out = E_x[-(index+1)]
            inp = torch.cat(E_out, inp, dim=1)

            # Save last 2 scales
            if index in [n_blocks-2, n_blocks-1]:
                activations_2x.append(inp)
        
        inp = self.generator.toRGB[n_blocks-1](activations_2x[1], upsample=True)
        
        if alpha < 1: # In case if blending is applied            
            inp = (1 - alpha) * self.generator.toRGB[n_blocks-2](activations_2x[0], upsample=True) + alpha * inp
            if return_norm and n_blocks-1 in norm_layer_nums:
                norms.append(inp)
        else:
            if return_norm and n_blocks-1 in norm_layer_nums:
                norms.append(inp)
        if return_norm:
            return norms
        else:
            return inp
# ------------------------------------------------------------------------------------------------------------------
# Resnet 2D RGB Image Generator Network with Contrastive Feature outputs, from Contrastive Unpaired Style Transfer
# Used with NCELoss
# ------------------------------------------------------------------------------------------------------------------
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(x)
            return fake            
