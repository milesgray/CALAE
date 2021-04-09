# Contrastive Adversarial Latent AutoEncoder (CALAE) Implementation

**Author**: Miles Gray

This is a set of experiments surrounding an image/video dataset of highly complex "fractal" patterns that were originally made using a variety of actual fractal formulas that the I discovered at a time when I had very little mathmematical training.  They are mostly highly parameterized mixtures of the so-called "Kali" set and Mandlebrot/Julia sets, which is where the I started from. However, it seems as though these formula produce significantly more diverse patterns than any other single formula varients of Julia sets that I have seen, though until recently I had not rigorously looked into the established mathematical literature. 

## NOTE

**There is a colab notebook that ties this code together and provides an example of usage (I need to clean all passwords/tokens and make sure it still works). This is primarily being made public for resume purposes, it currently lacks the polish of a production ready library - use with caution. This work was done prior to the rise of fourier layers and SIREN style coordinate translation.**

## Julia Sets

[This](http://www.juliasets.dk/Pictures_of_Julia_and_Mandelbrot_sets.htm) is a very good math-based walkthrough of the theories behind generating fractal imagery from Julia set based equations using computers. My ultimate goal is to fully merge the fractal distribution as a prior for a deep AutoEncoder in place of the usual gaussian distribution. 

# Adversarial Latent Autoencoder (ALAE) Implementation

**Authors**: Grigorii Sotnikov, Vladimir Gogoryan, Dmitry Smorchkov and Ivan Vovk (all have equal contribution)

This repository was originally a fork of the third party reimplemntation fo ALAE by the above authors. I have since then scoured the web to find all of the latest tricks and strategies that may be helpful to modeling a generative network that can reproduce fractal imagery.  Currently, at the core of the algorithm is still an ALAE and it seems like an appropriate starting point.  

## Generative Network Overview

Based on my survey of the state of the art in generative modeling, the lines between GAN and AutoEncoder have completely blurred and the most successful approaches use at least the following models:

- `Encoder` - This model covnerts images into a latent space so that the latent space can be learned based off the gradients from models that consume the latent space
- `Generator` - A model that takes a latent code as input and outputs an RGB image. It is rarely found using purely random noise as inputs in recent works since there is so much benefit from using a sample from the latent space instead.  
- `Discriminator` - An endpoint of the system that works as a binary classifier to determine whether or not an image is real or if it was produced by the `Generator`. This is one of the main sources of gradient signals for many recent systems.
- `Projector` - This is a recent addition that seem to be a key component to high performant generative systems that rely on latent spaces. It is a model that maps the output of the Encoder into a special latent space specifically for the `Generator` to use. 

In the ALAE setup, for a `real image` to pass through the system, it will go through all 4 of these models during training.  

1. Get `real image` from DataLoader
2. Input `real image` into `Encoder` to obtain `latent code`
3. Input `latent code` into `Projector` to transform into `generative code`
4. Input `generative code` into `Generator` to obtain a `fake image`
5. Input `fake image` into the `Discriminator` which attempts to classify it as a fake image (0 label) as opposed to the `real image` from the DataLoader 

## Other Repositories that are influential

- [Hessian Penalty](https://github.com/wpeebles/hessian_penalty)
- [Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation)
- [pyro](https://github.com/pyro-ppl/pyro)
- [network bending](https://github.com/terrybroad/network-bending)
- [style-gan](https://github.com/SiskonEmilia/StyleGAN-PyTorch)
- [Fully Adversarial Mosaics](https://github.com/zalandoresearch/famos)
- [PyContrast](https://github.com/HobbitLong/PyContrast)
- [RepDistiller](https://github.com/HobbitLong/RepDistiller)
