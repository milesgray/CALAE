""" Utility methods for handling complex numbers in networks, converted from TF """
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_psnr(predictions, ground_truths, maxpsnr=100):
    """Compute PSNR."""
    ndims = len(predictions.get_shape().as_list())
    mse = torch.mean(
        torch.square(torch.abs(predictions - ground_truths)), axis=list(range(1, ndims))
    )
    maxvals = torch.max(torch.abs(ground_truths), axis=list(range(1, ndims)))
    psnrs = (
        20 * torch.log(maxvals / torch.sqrt(mse)) /
        torch.log(nn.Constant(10, dtype=mse.dtype))
    )
    # Handle case where mse = 0.
    psnrs = torch.min(psnrs, maxpsnr)
    return psnrs


def hartley(im):
    ft = fft2c(im)
    hart = torch.real(ft) - torch.imag(ft)
    return hart


def getReal(pt_output, data_format):
    if data_format == "channels_last":
        real = pt_output[:, :, :, ::2]
    else:
        real = pt_output[:, ::2, :, :]
    return real


def getImag(pt_output, data_format):
    if data_format == "channels_last":
        imag = pt_output[:, :, :, 1::2]
    else:
        imag = pt_output[:, 1::2, :, :]
    return imag


def interleave(pt_output, data_format):
    if data_format == "channels_last":
        output_shape = torch.shape(pt_output)
        s = output_shape[3]
        realOut = pt_output[:, :, :, 0: s // 2]
        imagOut = pt_output[:, :, :, s // 2: s]
        pt_output = torch.cat([realOut, imagOut], 2)
        pt_output = torch.reshape(pt_output, output_shape)
    else:
        output_shape = torch.shape(pt_output)
        s = output_shape[1]
        realOut = pt_output[:, 0: s // 2, :, :]
        imagOut = pt_output[:, s // 2: s, :, :]
        pt_output = torch.cat([realOut, imagOut], 0)
        pt_output = torch.reshape(pt_output, output_shape)
    return pt_output


def complex_to_channels(image, requires_grad=False):
    """Convert data from complex to channels."""
    image_out = torch.stack([torch.real(image), torch.imag(image)], axis=-1)
    shape_out = torch.cat([torch.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
    image_out = torch.reshape(image_out, shape_out)
    return image_out


def channels_to_complex(image, requires_grad=False):
    """Convert data from channels to complex."""    
    image_out = torch.reshape(image, [-1, 2])
    image_out = torch.tensor(image_out[:, 0], image_out[:, 1], dtype=torch.cfloat)
    shape_out = torch.cat([torch.shape(image)[:-1], [image.shape[-1] // 2]], axis=0)
    image_out = torch.reshape(image_out, shape_out)
    return image_out


def fftshift(im, axis=0, name="fftshift"):
    """Perform fft shift.
    This function assumes that the axis to perform fftshift is divisible by 2.
    """
    split0, split1 = torch.split(im, 2, axis=axis)
    output = torch.concat((split1, split0), axis=axis)

    return output


def ifftc(im, name="ifftc", do_orthonorm=True):
    """Centered iFFT on second to last dimension."""
    im_out = im
    if do_orthonorm:
        fftscale = torch.sqrt(1.0 * im_out.get_shape().as_list()[-2])
    else:
        fftscale = 1.0
    fftscale = torch.cast(fftscale, dtype=torch.complex64)
    if len(im.get_shape()) == 4:
        im_out = torch.transpose(im_out, [0, 3, 1, 2])
        im_out = fftshift(im_out, axis=3)
    else:
        im_out = torch.transpose(im_out, [2, 0, 1])
        im_out = fftshift(im_out, axis=2)
    with torch.device("/gpu:0"):
        # FFT is only supported on the GPU
        im_out = torch.ifft(im_out) * fftscale
    if len(im.get_shape()) == 4:
        im_out = fftshift(im_out, axis=3)
        im_out = torch.transpose(im_out, [0, 2, 3, 1])
    else:
        im_out = fftshift(im_out, axis=2)
        im_out = torch.transpose(im_out, [1, 2, 0])

    return im_out


def fftc(im, name="fftc", do_orthonorm=True):
    """Centered FFT on second to last dimension."""
    im_out = im
    if do_orthonorm:
        fftscale = torch.sqrt(1.0 * im_out.get_shape().as_list()[-2])
    else:
        fftscale = 1.0
    fftscale = torch.cast(fftscale, dtype=torch.complex64)
    if len(im.get_shape()) == 4:
        im_out = torch.transpose(im_out, [0, 3, 1, 2])
        im_out = fftshift(im_out, axis=3)
    else:
        im_out = torch.transpose(im_out, [2, 0, 1])
        im_out = fftshift(im_out, axis=2)
    with torch.device("/gpu:0"):
        im_out = torch.fft(im_out) / fftscale
    if len(im.get_shape()) == 4:
        im_out = fftshift(im_out, axis=3)
        im_out = torch.transpose(im_out, [0, 2, 3, 1])
    else:
        im_out = fftshift(im_out, axis=2)
        im_out = torch.transpose(im_out, [1, 2, 0])

    return im_out


def ifft2c(im, name="ifft2c", do_orthonorm=True):
    """Centered inverse FFT2 on second and third dimensions."""
    im_out = im
    dims = torch.shape(im_out)
    if do_orthonorm:
        fftscale = torch.sqrt(torch.cast(dims[1] * dims[2], dtype=torch.float32))
    else:
        fftscale = 1.0
    fftscale = torch.cast(fftscale, dtype=torch.complex64)

    # permute FFT dimensions to be the last (faster!)
    tpdims = list(range(len(im_out.get_shape().as_list())))
    tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
    tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

    im_out = torch.transpose(im_out, tpdims)
    im_out = fftshift(im_out, axis=-1)
    im_out = fftshift(im_out, axis=-2)

    # with torch.device('/gpu:0'):
    im_out = torch.ifft2d(im_out) * fftscale

    im_out = fftshift(im_out, axis=-1)
    im_out = fftshift(im_out, axis=-2)
    im_out = torch.transpose(im_out, tpdims)

    return im_out


def fft2c(im, name="fft2c", do_orthonorm=True):
    """Centered FFT2 on second and third dimensions."""    
    im_out = im
    dims = torch.shape(im_out)
    if do_orthonorm:
        fftscale = torch.sqrt(torch.cast(dims[1] * dims[2], dtype=torch.float32))
    else:
        fftscale = 1.0
    fftscale = torch.cast(fftscale, dtype=torch.complex64)

    # permute FFT dimensions to be the last (faster!)
    tpdims = list(range(len(im_out.get_shape().as_list())))
    tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
    tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

    im_out = torch.transpose(im_out, tpdims)
    im_out = fftshift(im_out, axis=-1)
    im_out = fftshift(im_out, axis=-2)

    # with torch.device('/gpu:0'):
    im_out = torch.fft2d(im_out) / fftscale

    im_out = fftshift(im_out, axis=-1)
    im_out = fftshift(im_out, axis=-2)
    im_out = torch.transpose(im_out, tpdims)

    return im_out


def sumofsq(image_in, keep_dims=False, axis=-1, name="sumofsq", type="mag"):
    """Compute square root of sum of squares."""
    if type == "mag":
        image_out = torch.square(torch.abs(image_in))
    else:
        image_out = torch.square(torch.angle(image_in))
    image_out = torch.sum(image_out, keep_dims=keep_dims, axis=axis)
    image_out = torch.sqrt(image_out)

    return image_out


def conj_kspace(image_in, name="kspace_conj"):
    """Conjugate k-space data."""
    image_out = torch.reverse(image_in, axis=[1])
    image_out = torch.reverse(image_out, axis=[2])
    mod = np.zeros((1, 1, 1, image_in.get_shape().as_list()[-1]))
    mod[:, :, :, 1::2] = -1
    mod = torch.constant(mod, dtype=torch.float32)
    image_out = torch.multiply(image_out, mod)

    return image_out


def replace_kspace(image_orig, image_cur, name="replace_kspace"):
    """Replace k-space with known values."""    
    mask_x = kspace_mask(image_orig)
    image_out = torch.add(
        torch.multiply(mask_x, image_orig), torch.multiply(
            (1 - mask_x), image_cur)
    )

    return image_out


def kspace_mask(image_orig, name="kspace_mask", dtype=None):
    """Find k-space mask."""
    mask_x = torch.not_equal(image_orig, 0)
    if dtype is not None:
        mask_x = torch.cast(mask_x, dtype=dtype)
    return mask_x


def kspace_threshhold(image_orig, threshhold=1e-8, name="kspace_threshhold"):
    """Find k-space mask based on threshhold.
    Anything less the specified threshhold is set to 0.
    Anything above the specified threshhold is set to 1.
    """
    mask_x = torch.greater(torch.abs(image_orig), threshhold)
    mask_x = torch.cast(mask_x, dtype=torch.float32)
    return mask_x


def kspace_location(image_size):
    """Construct matrix with k-space normalized location."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    out = np.stack((xg.T, yg.T))
    return out


def torch_kspace_location(shape_y, shape_x):
    """Construct matrix with k-psace normalized location as tensor."""
    y = torch.cast(torch.range(shape_y), torch.float32)
    y = y / torch.cast(shape_y, torch.float32) - 0.5
    x = torch.cast(torch.range(shape_x), torch.float32)
    x = x / torch.cast(shape_x, torch.float32) - 0.5

    [yg, xg] = torch.meshgrid(y, x)
    yg = torch.transpose(yg, [1, 0])
    xg = torch.transpose(xg, [1, 0])
    out = torch.stack((yg, xg))
    return out


def create_window(out_shape, pad_shape=10):
    """Create 2D window mask."""
    g_std = pad_shape / 10
    window_z = np.ones(out_shape[0] - pad_shape)
    window_z = np.convolve(
        window_z, scipy.signal.gaussian(pad_shape + 1, g_std), mode="full"
    )

    window_z = np.expand_dims(window_z, axis=1)
    window_y = np.ones(out_shape[1] - pad_shape)
    window_y = np.convolve(
        window_y, scipy.signal.gaussian(pad_shape + 1, g_std), mode="full"
    )
    window_y = np.expand_dims(window_y, axis=0)

    window = np.expand_dims(window_z * window_y, axis=2)
    window = window / np.max(window)

    return window


def kspace_radius(image_size):
    """Construct matrix with k-space radius."""
    x = np.arange(image_size[0], dtype=np.float32) / image_size[0] - 0.5
    y = np.arange(image_size[1], dtype=np.float32) / image_size[1] - 0.5
    xg, yg = np.meshgrid(x, y)
    kr = np.sqrt(xg * xg + yg * yg)

    return kr.T


def sensemap_model(x, sensemap, name="sensemap_model", do_transpose=False):
    """Apply sensitivity maps."""
    if do_transpose:
        x_shape = x.get_shape().as_list()
        x = torch.expand_dims(x, axis=-2)
        x = torch.multiply(torch.conj(sensemap), x)
        x = torch.sum(x, axis=-1)
    else:
        x = torch.expand_dims(x, axis=-1)
        x = torch.multiply(x, sensemap)
        x = torch.sum(x, axis=3)
    return x


def model_forward(x, sensemap, name="model_forward"):
    """Apply forward model.
    Image domain to k-space domain.
    """
    if sensemap is not None:
        x = sensemap_model(x, sensemap, do_transpose=False)
    x = fft2c(x)
    return x


def model_transpose(x, sensemap, name="model_transpose"):
    """Apply transpose model.
    k-Space domain to image domain
    """    
    x = ifft2c(x)
    if sensemap is not None:
        x = sensemap_model(x, sensemap, do_transpose=True)
    return x
