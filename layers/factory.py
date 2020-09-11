import torch
import torch.nn as nn
import numpy as np

from .activations import Mish, LogCosh, XTanh, XSigmoid

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
                return LogCosh(**kwargs)
            elif activation == 'xtanh':
                return XTanh(**kwargs)
            elif activation == 'xsigmoid':
                return XSigmoid(**kwargs)
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
        elif norm == 'ln':
            return nn.LayerNorm(norm_dim, **kwargs)
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
