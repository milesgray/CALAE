import torch
import torch.nn as nn
import numpy as np

from .activations import Mish, LogCosh, XTanh, XSigmoid, Hat
from .normalize import ActNorm, PixelNorm, GroupedChannelNorm, LayerNorm, SwitchNorm

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
    def get_activation(activation, *args, **kwargs):
        # initialize activation
        if isinstance(activation, str):
            activation = activation.lower()

            if activation == 'relu':
                return nn.ReLU(*args, **kwargs)
            elif activation in ['lrelu', 'leaky', 'leakyrelu']:
                return nn.LeakyReLU(*args, **kwargs)
            elif activation == 'prelu':
                return nn.PReLU(*args, **kwargs)
            elif activation == 'selu':
                return nn.SELU(*args, **kwargs)
            elif activation == 'tanh':                
                return nn.Tanh(*args, **kwargs)
            elif activation == 'mish':
                return Mish(*args, **kwargs)
            elif activation == 'logcosh':
                return LogCosh(*args, **kwargs)
            elif activation == 'xtanh':
                return XTanh(*args, **kwargs)
            elif activation == 'xsigmoid':
                return XSigmoid(*args, **kwargs)
            elif activation == 'gelu':
                return nn.GELU(*args, **kwargs)
            elif activation == 'celu':
                return nn.CELU(*args, **kwargs)
            elif activation == "rrelu":
                return nn.RReLU(*args, **kwargs)
            elif activation == 'sigmoid':
                return nn.Sigmoid(*args, **kwargs)
            elif activation == 'elu':
                return nn.ELU(*args, **kwargs)
            elif activation == 'hardshrink':
                return nn.Hardshrink(*args, **kwargs)
            elif activation == 'hardtanh':
                return nn.Hardtanh(*args, **kwargs)
            elif activation == 'hardswish':
                return nn.Hardswish(*args, **kwargs)
            elif activation == 'logsigmoid':
                return nn.LogSigmoid(*args, **kwargs)
            elif activation == 'relu6':
                return nn.Relu6(*args, **kwargs)
            elif activation == 'softplus':
                return nn.Softplus(*args, **kwargs)
            elif activation == 'softshrink':
                return nn.Softshrink(*args, **kwargs)
            elif activation == 'softsign':
                return nn.Softsign(*args, **kwargs)
            elif activation == 'softmin':
                return nn.Softmin(*args, **kwargs)
            elif activation == 'softmax':
                return nn.Softmax(*args, **kwargs)
            elif activation == 'softmax2d':
                return nn.Softmax2d(*args, **kwargs)
            elif activation == 'logsoftmax':
                return nn.LogSoftmax(*args, **kwargs)
            elif activation == 'tanhshrink':
                return nn.Tanhshrink(*args, **kwargs)
            elif activation == 'threshold':
                return nn.Threshold(*args, **kwargs)
            elif activation == 'hat':
                return Hat(*args, **kwargs)
            elif activation == 'none':
                return nn.Identity()
            else:
                assert 0, f"Unsupported activation: {activation}"
        return nn.Identity()

    @staticmethod
    def get_normalization(norm, norm_dim=1, *args, **kwargs):
        if norm in ['bn2d', 'batch', 'bn', 'batchnorm2d', 'batchnorm']:
            return nn.BatchNorm2d(norm_dim, *args, **kwargs)
        elif norm in ['bn1d', 'batchnorm1d']:
            return nn.BatchNorm1d(norm_dim, *args, **kwargs)
        elif norm in ['bn3d', 'batchnorm3d']:
            return nn.BatchNorm3d(norm_dim, *args, **kwargs)
        elif norm in ['group', 'groupnorm', 'gn']:
            return nn.GroupNorm(norm_dim, *args, **kwargs)
        elif norm in ['localresponse', 'local', 'localresponsenorm', 'localnorm', 'lr', 'lrn']:
            return nn.LocalResponseNorm(norm_dim, *args, **kwargs)
        elif norm in ['in1d', 'instancenorm1d']:
            return nn.InstanceNorm1d(norm_dim, *args, **kwargs)
        elif norm in ['in2d', 'inst', 'in', 'instancenorm', 'instancenorm2d']:
            return nn.InstanceNorm2d(norm_dim, *args, **kwargs)
        elif norm in ['in3d', 'instancenorm3d']:
            return nn.InstanceNorm3d(norm_dim, *args, **kwargs)
        elif norm in ['pixel', 'pn', 'pixelnorm']:
            return PixelNorm(*args, **kwargs)
        elif norm in ['act', 'an', 'actnorm']:
            return ActNorm(*args, **kwargs)
        elif norm == 'ln':
            return nn.LayerNorm(norm_dim, *args, **kwargs)
        elif norm in ['switch', 'sn', 'switchnorm']:
            return SwitchNorm(norm_dim, *args, **kwargs)
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
