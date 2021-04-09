import time

import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage, signal
from numpy.fft import fft2, ifft2

class SR_SIMLoss(nn.Module):
    """Creates a criterion that measures the SR_SIM Index for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value `1 - clip(SR_SIM, min=0, max=1)` is returned. If you need SR_SIM value,
    use function `sr_sim` instead.
    SR_SIM Index with automatic downsampling, Version 1.0

    This is an implementation of the algorithm for calculating the
    Spectral Residual based Similarity (SR-SIM) index between two images. For
    more details, please refer to our paper:
    Lin Zhang and Hongyu Li, "SR-SIM: A fast and high performance IQA index based on spectral residual", in: Proc. ICIP 2012.
    ----------------------------------------------------------------------
    Input : (1) x: the first image being compared
        (2) y: the second image being compared
    Output: srsim: the similarity score between two images, a real number
    """
    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.metric_fn = sr_sim

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            score = self.metric_fn(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = self.metric_fn(x, y)
            return score


def rgb2yiq(rgbarray):
    rbgarray = rgbarray.to(torch.float32)
    Y = 0.299 * rgbarray[:,:,0] + 0.587 * rgbarray[:,:,1] + 0.114 * rgbarray[:,:,2]
    I = 0.5959 * rgbarray[:,:,0] - 0.2746 * rgbarray[:,:,1] - 0.3213 * rgbarray[:,:,2]
    Q = 0.2115 * rgbarray[:,:,0] - 0.5227 * rgbarray[:,:,1] + 0.3112 * rgbarray[:,:,2]
    return torch.stack((Y,I,Q))

def conv2d_same(data, kernel):
    data = F.conv2d(data, (1,kernel[0],kernel[1]))
    p = (kernel.shape[1]-1)//2
    data = F.pad(data, (p,p,p,p),'reflect')
    return data

def downsample(image, F, aveKernel):
    """ Downsample image like in matlab
    Args:
        image (torch.arrait uint8): image to downsample
        F (float): downsampling factor
        aveKernel (torch.array): averaging kernel
    Returns:
        torch.array: downsampled image
    """
    assert aveKernel.shape == (F,F)
    aveI = conv2d_same(image, (1,aveKernel[0],aveKernel[1]))
    return aveI[::F, ::F]


def sr_sim(image1, image2):
    """
    SR_SIM Index with automatic downsampling, Version 1.0
    Copyright(c) 2011 Lin ZHANG
    All Rights Reserved.
    ----------------------------------------------------------------------
    Permission to use, copy, or modify this software and its documentation
    for educational and research purposes only and without fee is hereQ
    granted, provided that this copyright notice and the original authors'
    names appear on all copies and supporting documentation. This program
    shall not be used, rewritten, or adapted as the basis of a commercial
    software or hardware product without first obtaining permission of the
    authors. The authors make no representations about the suitability of
    this software for any purpose. It is provided "as is" without express
    or implied warranty.
    ----------------------------------------------------------------------
    This is an implementation of the algorithm for calculating the
    Spectral Residual based Similarity (SR-SIM) index between two images. For
    more details, please refer to our paper:
    Lin Zhang and Hongyu Li, "SR-SIM: A fast and high performance IQA index based on spectral residual", in: Proc. ICIP 2012.
    ----------------------------------------------------------------------
    Input : (1) image1: the first image being compared
        (2) image2: the second image being compared
    Output: srsim: the similarity score between two images, a real number
    """
    if image1.shape != image2.shape:
        raise ValueError("Image 1 and 2 must be of same shape")

    if image1.ndim == 3:
        Y1, I1, Q1 = cv2.split(rgb2yiq(image1))
        Y2, I2, Q2 = cv2.split(rgb2yiq(image2))
    else:
        Y1 = image1.to(torch.float32)
        Y2 = image2.to(torch.float32)

    # Downsample (first apply averaging filter to prevent sampling bias)
    minDimension = min(image1.shape[0], image1.shape[1])
    F = max(1,round(minDimension / 256))
    aveKernel = torch.ones((F,F))/F**2

    Y1 = downsample(Y1, F, aveKernel)
    Y2 = downsample(Y2, F, aveKernel)

    if image1.ndim == 3:
        I1 = downsample(I1, F, aveKernel)
        I2 = downsample(I2, F, aveKernel)
        Q1 = downsample(Q1, F, aveKernel)
        Q2 = downsample(Q2, F, aveKernel)

    # Calculate the visual saliency maps
    saliencyMap1 = spectralResidueSaliency(Y1)
    saliencyMap2 = spectralResidueSaliency(Y2)

    # Calculate the gradient magnitude map using Scharr filters
    dx = torch.Tensor([[3., 0., -3.],
                    [10., 0., -10.],
                    [3., 0., -3.]]) / 16
    dy = torch.Tensor([[3., 10., 3.],
                    [0., 0., 0.],
                    [-3., -10., -3.]]) / 16

    IxY1 = signal.convolve(Y1, dx, "same")
    IyY1 = signal.convolve(Y1, dy, "same")
    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve(Y2, dx, "same")
    IyY2 = signal.convolve(Y2, dy, "same")
    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2)

    # Calculate the SR-SIM
    C1 = 0.40 # constant
    C2 = 225 # constant
    alpha = 0.50 # constant

    GBVSSimMatrix = (2 * saliencyMap1 * saliencyMap2 + C1) / (saliencyMap1**2 + saliencyMap2**2 + C1)
    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + C2) / (gradientMap1**2 + gradientMap2**2 + C2)

    Rm = torch.max(saliencyMap1, saliencyMap2) # weight factor
    SimMatrix = GBVSSimMatrix * (gradientSimMatrix ** alpha)

    if image1.ndim == 3:
        # Formulas (8-11) from FSIMc paper page 4
        # https://sse.tongji.edu.cn/linzhang/IQA/FSIM/Files/Fsim%20a%20feature%20similarity%20index%20for%20image%20quality%20assessment.pdf
        T3 = 200 # constant, 130 in VSI
        T4 = 200 # constant, 130 in VSI
        # get matrices from I, Q color channels
        ISimMatrix = (2 * I1 * I2 + T3) / (I1**2 + I2**2 + T3)
        QSimMatrix = (2 * Q1 * Q2 + T4) / (Q1**2 + Q2**2 + T4)

        lambd = 0.03 # constant, use same as FSIM, =0,02 in VSI

        # New formula for SR-SIM based on Y,I,Q color channels
        prod = ISimMatrix * QSimMatrix
        SimMatrixC = SimMatrix * (torch.sign(prod) * torch.abs(prod) ** lambd).real
        SRSIMc = torch.sum(SimMatrixC * Rm) / torch.sum(Rm)
        return SRSIMc

    else:
        SRSIM = torch.sum(SimMatrix * Rm) / torch.sum(Rm)
        return SRSIM


def spectralResidueSaliency(image):
    """
    this function is used to calculate the visual saliency map for the given
    image using the spectral residue method proposed by Xiaodi Hou and Liqing
    Zhang. For more details about this method, you can refer to the paper:
    Saliency detection: a spectral residual approach.
    there are some parameters needed to be adjusted
    """
    scale = 0.25 # constant
    aveKernelSize = 3 # constant
    gauSigma = 3.8 # constant
    gauSize = 9 # constant

    # correction of built-in round function which 
    # "for values exactly halfway between rounded decimal values, rounds to the nearest even value"
    # as opposite to matlab which always round it up
    def _round(a):
        return int(torch.rint(torch.nextafter(a, a+1)))

    inImg = cv2.resize(image, (_round(scale*image.shape[1]), _round(scale*image.shape[0])), interpolation=cv2.INTER_CUBIC)

    myFFT = fft2(inImg)
    myLogAmplitude = torch.log(torch.abs(myFFT))
    myPhase = torch.angle(myFFT)
    mySpectralResidual = myLogAmplitude - cv2.boxFilter(myLogAmplitude, -1, (aveKernelSize, aveKernelSize), cv2.BORDER_REPLICATE)
    saliencyMap = torch.abs(ifft2(torch.exp(mySpectralResidual + 1j*myPhase)))**2

    blurred = cv2.GaussianBlur(saliencyMap,(gauSize, gauSize), gauSigma, gauSigma )
    saliencyMap = torch.nn.functional.normalize(blurred)
    return cv2.resize(saliencyMap, (image.shape[1], image.shape[0]))
