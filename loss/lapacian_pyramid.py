import sys
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import PIL.Image

def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = torch.reshape(torch.nn.Parameter(gauss_kernel(size=k_size, sigma=sigma), 
                                                                                            requires_grad=False, 
                                                                                            trainable=False, 
                                                                                            dtype=torch.float32),
                                                [k_size, k_size, 1, 1])
    t_kernel3 = torch.cat([t_kernel]*t_input.get_shape()[3], axis=2)
    t_result = t_input
    for r in range(repeats):
        t_result = nn.depthwise_conv2d(t_result, t_kernel3,
                strides=[1, stride, stride, 1], padding='SAME')
    return t_result

def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = torch.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
    t_pyr.append(current)
    return t_pyr

def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = [torch.norm(a-b,ord=1)/torch.size(a, out_type=torch.float32) for a,b in zip(t_pyr1, t_pyr2)]
    t_loss = torch.sum(t_losses)*torch.shape(t_img1, out_type=torch.float32)[0]
    return t_loss
