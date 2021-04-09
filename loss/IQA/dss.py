from math import floor, ceil
import numpy as np
from scipy import ndimage
from scipy.fftpack import dct
import cv2
import time

def dss(img1, img2, sigma=1.55, C=(1000, 300)):
    """
    DSS DCT Subband Similarity index for measuring image quality
    DSS = DSS_INDEX(IMG1, IMG2) calculates the DCT Subband Similarity (DSS)
    score for image IMG1, with the image IMG2 as the reference. IMG1 and
    IMG2 should be 2D grayscale images, and must be of the same size.
    DSS = DSS_INDEX(IMG1, IMG2, SIGMA, C) calculates the DSS score
    with control over parameters of the computation. Parameters include:
    sigma - Specifies the standard deviation of the Gaussian that
            determines the proportion between the weight given to low
            spatial frequencies and to high spatial frequencies.
            Default value: sigma = 1.55
    C     - Specifies constants in the DSS subband similarity equations
            (see the reference paper below). C should be a vector of
            size [1 2] where the first element in C is for the DC
            equation and the second element in C is for the AC equation.
            Default value: C = [1000 300]
    This function is an implementation of the algorithm described in the
    following paper:
        Amnon Balanov, Arik Schwartz, Yair Moshe, and Nimrod Peleg,
        "Image Quality Assessment based on DCT Subband Similarity," 22nd
        IEEE International Conference on Image PRocessing (ICIP 2015).
    Example
    ---------
    This example shows how to compute DSS score for a noisy image given
    the original reference image.
    img2 = imread('cameraman.tif');
    img1 = imnoise(img2, 'gaussian', 0, 0.001);
    subplot(1,2,1); imshow(img2); title('Reference Image');
    subplot(1,2,2); imshow(img1); title('Noisy Image');
    dss = dss_index(img1,img2);
    fprintf('The DSS score is 0.4f\n',dss);
    Version 1.00
    Copyright(c) 2015, Yair Moshe
    Signal and Image Processing Laboratory (SIPL)
    Department of Electrical Engineering
    Technion - Israel Institute of Technology
    Technion City, Haifa 32000, Israel
    Tel: +(972)4-8294746
    e-mail: yair@ee.technion.ac.il
    WWW: sipl.technion.ac.il
    """

    # Parse inputs
    if img1.ndim != 2:
        raise TypeError("Images must be greyscale")
    if img2.shape != img2.shape:
        raise ValueError("Image 1 and 2 must be of same size")
    try:
        float(sigma)
    except Exception:
        raise TypeError("Sigma must be a scalar")
    if not isinstance(C, tuple):
        raise TypeError("C must be a tuple")

    # Compute DSS
    # Crop images size to the closest multiplication of 8
    nRows, nCols = img1.shape
    nRows = 8*floor(nRows/8)
    nCols = 8*floor(nCols/8)

    img1 = img1[:nRows, :nCols]
    img2 = img2[:nRows, :nCols]

    # Channel decomposition for both images by 8x8 2D DCT
    img1Decomp = dct_decomp(img1)
    img2Decomp = dct_decomp(img2)
    # Create a Gaussian window that will be used to weight subbands scores
    r = np.arange(1, 9, 1)
    X,Y = np.meshgrid(r, r)
    distance = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    w = np.exp(-((distance**2)/(2*sigma**2))).real

    # Compute similarity between each subband in img1 and img2
    subbandSimilarity = np.zeros((8, 8))
    smallWeightThresh = 1e-2
    for m in range(8):
        for n in range(8):
            # Skip subbands with very small weight
            if(w[m,n] < smallWeightThresh):
                w[m,n] = 0
                continue

            subbandSimilarity[m,n] = subband_similarity(
                img1Decomp[m::8, n::8],
                img2Decomp[m::8, n::8],
                m, n, C)

    # Weight subbands similarity scores
    dss = np.sum(np.sum(subbandSimilarity * (w/np.sum(w))))
    #np.savetxt('img_python.txt', sortedVarLeftTerm, fmt="%.5f", delimiter=",", newline="\n")
    return dss

# Channel decomposition for an image by 8x8 2D DCT
def dct_decomp(img):
    nRows, nCols = img.shape
    D = dctmtx(8)
    doubleImg = img.astype(np.float64)
    decomp = np.zeros(img.shape)
    for blockNumInRow in range(nRows//8):
        for blockNumInColumn in range(nCols//8):
            curBlock = doubleImg[0 + 8*(blockNumInRow) : 8 + 8*(blockNumInRow), 0 + 8*(blockNumInColumn):8 + 8*(blockNumInColumn)]
            decomp[0 + 8*(blockNumInRow) : 8 + 8*(blockNumInRow), 0 + 8*(blockNumInColumn):8 + 8*(blockNumInColumn)] = np.matmul(np.matmul(D,curBlock), D.transpose())
    return decomp

# Discrete cosine transform matrix by matlab
# https://fr.mathworks.com/help/images/discrete-cosine-transform.html#f21-16137
def dctmtx(N):
    p, q = np.ogrid[1:N, 1:2*N:2]
    return np.concatenate((np.sqrt(1/N)*np.ones((1,N)) , np.sqrt(2/N) * np.cos(np.pi/(2*N) * p * q)), axis=0)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# Compute similarity between a subband in img1Subband and img2Subband
def subband_similarity(img1Subband, img2Subband, m, n, Cs):
    size = 3 # constant
    sigma = 1.5 # constant
    percentile = 0.05 # constant

    if((m == 0) and (n == 0)):
        C = Cs[0] # DC
    else:
        C = Cs[1] # AC
    # Compute local variance
    window = matlab_style_gauss2D(shape=(size,size), sigma=sigma)
    mu1 = ndimage.convolve(img1Subband, window, mode="constant")
    mu2 = ndimage.convolve(img2Subband, window, mode="constant")
    sigma1_sq = ndimage.convolve(img1Subband*img1Subband, window, mode="constant") - mu1*mu1
    sigma2_sq = ndimage.convolve(img2Subband*img2Subband, window, mode="constant") - mu2*mu2
    sigma12 = ndimage.convolve(img1Subband*img2Subband, window, mode="constant") - mu1*mu2
    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq[sigma2_sq < 0] = 0
    varLeftTerm = (2*np.sqrt(sigma1_sq * sigma2_sq) + C) / (sigma1_sq + sigma2_sq + C)
    # Spatial pooling of worst scores
    percentileIndex = round(percentile * (varLeftTerm.size))
    sortedVarLeftTerm = np.sort(varLeftTerm.flatten())
    similarity = np.mean(sortedVarLeftTerm[:percentileIndex])
    # For DC, multiply by a right term
    if ((m == 0) and (n == 0)):
        varRightTerm = ((sigma12 + C) / (np.sqrt(sigma1_sq * sigma2_sq) + C))
        sortedVarRightTerm = np.sort(varRightTerm.flatten())
        simRightTerm = np.mean(sortedVarRightTerm[:percentileIndex])
        similarity = similarity * simRightTerm
    return similarity
