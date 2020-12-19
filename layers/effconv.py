import torch
from torch import nn
import torch.nn.functional as F
import math

from .factory import Factory

__all__ = ["Shuffle", "CBR", "CBA", "BR", "CB", "C", "CDilated", "CDilatedB", \
           "EffDWSepConv", "StridedEffDWise", "EfficientPyrPool"]

def activation_fn(features, name='prelu', inplace=True):
    '''
    :param features: # of features (only for PReLU)
    :param name: activation name (prelu, relu, selu)
    :param inplace: Inplace operation or not
    :return:
    '''
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        NotImplementedError('Not implemented yet')
        exit()

class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class CBA(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and activation
    '''

    def __init__(self, nIn, nOut, kSize, act="relu", stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = Factory.get_activation(act)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilatedB(nn.Module):
    '''
    This class defines the dilated convolution with batch normalization.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.bn(self.conv(input))

class EffDWSepConv(nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super().__init__()
        self.conv_channel = CBR(channel_in, channel_in, kSize=kernel_size, stride=1, groups=channel_in)

        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, stride=1, groups=groups_proj)

        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.ksize=kernel_size

    def forward(self, x):
        '''
        :param x: input of dimension C x H x W
        :return: output of dimension C1 x H x W
        '''
        bsz, channels, height, width = x.size()
        x = self.conv_channel(x)
        proj_out  =self.proj_layer(x)
        linear_comb_out = self.linear_comb_layer(x)
        return proj_out * linear_comb_out

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class StridedEffDWise(nn.Module):
    '''
    This class implements the strided volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, kernel_size=3):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        :param weight_avg: Waighted average for fusing the feature maps in volume-wise separable convolutions
        :param res_conn: Residual connection in the volume-wise separabel convolutions
        :param proj: Want to project the feature maps from channel_in to channel_out or not
        '''
        super().__init__()

        self.pool_layer = CBR(channel_in, channel_in, 3, stride=2, groups=channel_in)
        self.dw_layer =  EffDWSepConv(channel_in, channel_in, kernel_size=kernel_size)
        self.channel_in = channel_in
        self.channel_out = 2*channel_in
        self.ksize = kernel_size

    def forward(self, x):
        x = self.pool_layer(x)
        return torch.cat([x, self.dw_layer(x)], 1)

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, ' \
            'width={width}, height={height})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class EfficientPWConv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.wt_layer = nn.Sequential(
                            nn.AdaptiveAvgPool2d(output_size=1),
                            nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                            nn.Sigmoid()
                        )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)

        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super().__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            # perform one big batch normalization instead of p small ones
            BR(proj_planes * len(scales)),
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br),
        )
        if last_layer_br:
            self.br = BR(out_planes)
        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)

        out = torch.cat(hs, dim=1)
        out = self.merge_layer(out)
        if self.last_layer_br:
            return self.br(out)
        return out        
