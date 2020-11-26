import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *


class UNetDiscriminator(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
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
        self.outc = OutConv(64, 1)

        self.first_run = True

    def forward(self, x):
        x1 = self.inc(x)
        if self.first_run: print(f"x1: {x1.shape}")
        x2 = self.down1(x1)
        if self.first_run: print(f"x2: {x2.shape}")
        x3 = self.down2(x2)
        if self.first_run: print(f"x3: {x3.shape}")
        x4 = self.down3(x3)
        if self.first_run: print(f"x4: {x4.shape}")
        x5 = self.down4(x4)
        if self.first_run: print(f"x5: {x5.shape}")
        y5 = self.up1(x5, x4)
        if self.first_run: print(f"y5: {y5.shape}")
        y4 = self.up2(y5, x3)
        if self.first_run: print(f"y4: {y4.shape}")
        y3 = self.up3(y4, x2)
        if self.first_run: print(f"y3: {y3.shape}")
        y2 = self.up4(y3, x1)
        if self.first_run: print(f"y2: {y2.shape}")
        logits = self.outc(y2)
        if self.first_run: 
            print(f"logits: {logits.shape}")
            self.first_run = False

        return logits
