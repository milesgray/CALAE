# Utility Methods 

These are misc supporting methods for the many different components/algorithms that I pulled into this repository.  Off the top of my head:

- [alae.py](alae.py) - Adversarial Latent AutoEncoder, the model implementation this repo was built around
- [balance.py](balance.py) - Methods that support finding optimal loss weights during multi objective training (aka most of the time)
- [bending.py](bending.py) - Style Bending, very cool system for manipulating well trained generative models, never fully integrated in
- [coilcomp.py](coilcomp.py) - Coil Compression? Coil compression for accelerated imaging with Cartesian sampling.
- [color.py](color.py) - Color space conversion functionality
- [complex.py](complex.py) - Complex Number enabled layer implementations
- [density.py](density.py) - Disentangling-VAE
- [diffaugment.py](diffaugment.py) - [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/pdf/2006.10738)
- [distances.py](distances.py) - "All" of the distance metrics in numpy and pytorch
- [divergences.py](divergences.py) - Divergence calculations for comparing samples from arbitrary distributions, such as KL and JS
- [fftc.py](fftc.py) - Fast Fourier Transform flavors adaptor for numpy, from before pytorch had native implementations
- [files](files.py) - Basic file system utils
- [importance.py](importance.py) - Disentangling-VAE related
- [interp.py](interp.py) - Interpolation algorithms
- [latent_projecting.py](latent_projecting.py) - Data objects used in [related loss](../loss/latent_projecting.py)
- [matrix_sqrt.py](matrix_sqrt.py) - Some crazy math
- [noise.py](noise.py) - A few different algorithm specific noise distributions
- [pixelnerf.py](pixelnerf.py) - Pixel Neural Radience Fields related code that was never fully integrated into this codebase
- [sample.py](sample.py) - Latent manipulation related code from [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
- [sdf.py](sdf.py) - Density Function related code, from NeRF
- [sr.py](sr.py) - Super Resolution losses related code
- [style_mixing.py](style_mixing.py) - Logic for combining two style features before applying them to Adaptive Instance Normalization style generators
- [tiles.py](tiles.py) - Related to NeRF
- [tunit.py](tunit.py) - TUNIT related code, worked in another repo/notebook
- [unet.py](unet.py) - UNet related code, image to image/segmentation model, from wavelet transform codebase
- [video.py](video.py) - Code for proessing videos with pytorch
- [wt.py](wt.py) - Wavelet Transform, interesting frequency based training strategy
