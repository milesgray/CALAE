import torch.nn as nn
import math
from box_convolution.box_convolution_module import BoxConv2d

# Runtime scaling, analog of https://github.com/podgorskiy/ALAE/blob/5d8362f3ce468ece4d59982ff531d1b8a19e792d/lreq.py
# Motivated by https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/model.py

class ScaleWeights:
    def __init__(self, name):
        self.name = name
    
    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        
        return weight * math.sqrt(2 / fan_in)
    
    @staticmethod
    def apply(module, name):
        hook = ScaleWeights(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)
    
    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)

def set_scale(module, name='weight'):
    ScaleWeights.apply(module, name)
    return module

class ScaledLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out, bias=bias)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        
        self.linear = set_scale(linear)

    def forward(self, x):
        return self.linear(x)

class ScaledConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        use_box_conv = kwargs.get('use_box_conv', False)        
        if use_box_conv:
            if "kernel_size" in kwargs:
                kwargs.pop("kernel_size")
            self.padding = 0
            if "padding" in kwargs:
                padding = kwargs.pop("padding")
            if "max_input_h" not in kwargs:
                kwargs["max_input_h"] = 512
            if "max_input_w" not in kwargs:
                kwargs["max_input_w"] = 512
            conv = BoxConv2d(*args, **kwargs)
            if padding > 0:
                self.pad = nn.ZeroPad2d(padding)
        else:
            conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        
        self.conv = set_scale(conv)

    def forward(self, x):
        x = self.conv(x)
        #if self.padding > 0:
        #    x = self.pad(x)
        return x