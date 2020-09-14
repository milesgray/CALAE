""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 128)
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet_Sigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Sigmoid, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.sigmoid(self.outc(x))

        return logits

class UNet_Sigmoid_3Tail(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Sigmoid_3Tail, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc1 = OutConv(64, n_classes)
        self.outc2 = OutConv(64, n_classes)
        self.outc3 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out1 = self.sigmoid(self.outc1(x))
        x_out2 = self.sigmoid(self.outc2(x))
        x_out3 = self.sigmoid(self.outc3(x))

        return torch.cat((x_out1, x_out2, x_out3), dim=1)

class UNet_NTail_128(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_NTail_128, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_Sigmoid_NTail_128(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_Sigmoid_NTail_128, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = self.sigmoid(layer(x))
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_NTail_32_Mod(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_NTail_32_Mod, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 512)
        self.down1 = Down(512, 512)
        self.down2 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.outc_modules = nn.ModuleList()
        for _ in range(n_tails):
            self.outc_modules.append(OutConv2(512, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_NTail_128_Mod(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_NTail_128_Mod, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 512)
        self.down1 = Down(512, 512)
        self.down2 = Down(512, 512)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(1024, 512, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv2(512, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

# Bigger capacity
class UNet_NTail_128_Mod1(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_NTail_128_Mod1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 1024)
        self.down1 = Down(1024, 1024)
        self.down2 = Down(1024, 1024)
        factor = 2 if bilinear else 1
        self.down3 = Down(1024, 1024)
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(2048, 1024, bilinear)
        self.up3 = Up(2048, 1024, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv2(1024, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out


class UNet_Sigmoid_NTail_128_Mod(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_Sigmoid_NTail_128_Mod, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 512)
        self.down1 = Down(512, 512)
        self.down2 = Down(512, 512)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(1024, 512, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv2(512, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = self.sigmoid(layer(x))
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_Sigmoid_NTail_256(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_Sigmoid_NTail_256, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 16, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out = torch.empty(0)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_Small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
#         x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits