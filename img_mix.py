"""
From FAMOS - https://github.com/zalandoresearch/famos/blob/master/utils.py
"""
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn

#large alpha emphasizes new -- conv. generation , less effect on old, the mix template output
#@param I_G is parametric generation
#@param I_M is mixed template image
def blend(I_G, I_M, alpha, beta):
    if opt.blendMode==0:
        out= I_M*(1 - beta) + alpha * I_G[:, :3]
    if opt.blendMode==1:
        out = I_G[:, :3] * alpha * 2 + I_M
    if opt.blendMode==2:##this is the mode described in paper, convex combination
        out= I_G[:, :3] * alpha + (1 - alpha) * I_M
    return torch.clamp(out,-1,1)

##show the different btw final image and mixed image -- this shows the parametric output of our network
def invblend(I,I_M,alpha,beta):
    return torch.clamp(I-I_M,-1,1)

##visualization routine to show mix arrayA as many colourful channels
def rgb_channels(x):
    N=x.shape[1]
    if N ==1:
        return torch.cat([x,x,x],1)##just white dummy

    cu= int(N**(1/3.0))+1
    a=x[:,:3]*0##RGB image
    for i in range(N):
        c1=int(i%cu)
        j=i//cu
        c2=int(j%cu)
        j=j//cu
        c3=int(j%cu)
        a[:,:1]+= c1/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,1:2]+=c2/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,2:3]+=c3/float(cu+1)*x[:,i].unsqueeze(1)
    return a#*2-1##so 0 1