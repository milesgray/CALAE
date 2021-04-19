import math
from math import log2, exp
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
import torch.nn.functional as F
from torch.autograd import grad
 
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')
Module = TypeVar('torch.nn.module')
from CALAE.loss.hessian_penalty import hessian_penalty
from CALAE.metrics.perceptual import PerceptualLoss
import lpips
import piq


class ALAEDiscriminatorLoss(nn.Module):
    """ `A`LAE '`A`dversarial' loss
    Applies either BCE or Non Saturating loss as an adversarial
    target on a pair of real samples and their recreation from a generative
    model. Also applies a Zero-Centered Gradient Penalty on the Discriminator's
    predictions for the real samples as regularization.

    Non Saturating Loss - softplus goes closer to 0 as values decrease (past 0 as well),
        this provides a more evenly distributed spread of values when very close to 0 or 1 
        compared to binary cross entropy - one of the early tricks to help GAN models learn.
            - Negate real predictions as they go down to -1 instead of up to 1, 
            add fake predictions as they go down to 0.
            - This gives an inverse relationship that can be directly minimized.
    Adversarial Loss - Binary Cross Entropy loss with a target of 1 for real
        predictions and 0 for fake, naive way of training Discriminator
        Models from original GAN paper.

    Zero-Centered Gradient Penalty - I don't know which paper this came from
        but it is well named. By taking the mean of the squared output 
        gradients in respect to the expected output (1) of real predictions, 
        which is presumably easier than predicting a constantly changing
        generated target, as a minimization target it prevents the model
        from being too confident in its real predictions to give the generator
        a chance to fool it more easily as a nudge towards nash equilibrium in
        the ill-posed min/max game being played between the discriminator and
        generator.  It is similar to label smoothing, but makes more sense
        in this context (label is always 1).
            - Note, in the case of ALAE the 'generator' here is the
                'Encoder' model, the discriminator is evaluating the latent space
                representation of the encoder for real samples vs normal distribution
                as input. 
                - There is also a reconstruction loss seperate from this
                    that is applied to the actual Generator model which compares the
                    input to the Encoder with the output of the Generator. This is
                    the 'Auto Encoder' aspect of Adversarial Latent Auto Encoder (ALAE)
                    and also shapes the same latent space as this adversarial loss does.
    """
    def __init__(self):
        super().__init__()

    def zero_centered_gradient_penalty(self, real_samples, real_prediction):
        """
        Computes zero-centered gradient penalty for E, D predictions on real samples (label of 1)
        """    
        grad_outputs = torch.ones_like(real_prediction, requires_grad=True)
        squared_grad_wrt_x = torch.autograd.grad(outputs=real_prediction, \
                                                 inputs=real_samples, \
                                                 grad_outputs=grad_outputs,\
                                                 create_graph=True, \
                                                 retain_graph=True)[0] \
                                                    .pow(2)
        
        return squared_grad_wrt_x. \
            view(squared_grad_wrt_x.shape[0], -1). \
                sum(dim=1). \
                    mean()

    def adv_loss(self, logits, target):
        """Adversarial Loss
        Binary Cross Entropy loss on either all 1 (real) 
        or all 0 (fake) as labels.

        Args:
            logits (Tensor): Prediction output from model
            target (int): Label value for all predictions, 
                either 0 or 1 

        Returns:
            Tensor: Loss output with grads
        """
        assert target in [1, 0]

        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        return loss

    def forward(self, E: Module, D: Module, alpha: float,
                real_samples: Tensor, fake_samples: Tensor, 
                gamma=10, 
                use_bce=False) -> Tensor:
        # Encode real and fake images to feature vectors
        E_r, E_f = E(real_samples, alpha), E(fake_samples, alpha)
        # Use Discriminator to classify real and fake feature vectors
        real_prediction, fake_prediction = D(E_r), D(E_f)

        if use_bce:
            # Binary Cross Entropy loss - 1 as label for real, 0 for fake
            loss = self.adv_loss(real_prediction, 1)
            loss += self.adv_loss(fake_prediction, 0)
        else:
            # Non Saturating Loss
            # Minimize negative = Maximize positive (Minimize incorrect D predictions for real data,
            #                                        minimize incorrect D predictions for fake data)
            loss = (F.softplus(-real_prediction) + F.softplus(fake_prediction)) \
                .mean()

        if gamma > 0:
            loss += self.zero_centered_gradient_penalty(real_samples, \
                                                        real_prediction) \
                .mul(gamma/2)

        return loss

class ALAEGeneratorLoss(nn.Module):
    """ "Generator" Loss
    This is badly named as it doesn't apply to the Generator model - it applies
    to the `E`ncoder model, which is a "generator" from a certain perspective of 
    the overall ALAE model (it generates... latent space encodings).
    It seems to be a regularization term for the discriminator that oddly does the 
    opposite of the non-saturating adversarial loss applied in the `ALAEDiscriminatorLoss`.  
    The output of the D model for fake data is negated before going into softplus, 
    whereas in the D loss it is the real data output that is negated.  I may be 
    misunderstanding the double inverse relationship. 
    This is the original comment:

    Minimize negative = Maximize positive (Minimize correct D predictions for fake data)

    Note:
        - I added on the option to apply the hessian penalty to E here
            as the first change to the ALAE model before I fully understood it.
        - Hessian Penalty is a regularization term based on the 2nd order
            derivatives that are cleverly estimated within the algorithm.
            - see: CALAE.loss.hessian_penalty.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, E: Module, D: Module, alpha: float, fake_samples: Tensor, 
                enable_hessian=True, 
                hessian_layers=[-1,-2], 
                current_layer=[-1], 
                hessian_weight=0.01):
        # Hessian applied to E here
        # Minimize negative = Maximize positive (Minimize correct D predictions for fake data)
        E_z = E(fake_samples, alpha)
        loss = F.softplus(-D(E_z)).mean()
        if enable_hessian:
            for layer in hessian_layers:
                # CALAE.loss.hessian_penalty
                h_loss = hessian_penalty(E, z=fake_samples, alpha=alpha, return_norm=layer) * hessian_weight
                if layer in current_layer:
                    h_loss = h_loss * alpha
                loss += h_loss
        return loss

class ALAEAutoEncoderLoss(nn.Module):
    """AL`AE` `A`uto`E`ncoder loss
    
    The AutoEncoder here is reconstructing the projected representation 
    of the input created by `F` model - it is inverted compared to the
    usual AE setup as it takes in the 'latent space' feature and expands
    it to a full image with the `G`enerator model. The generated full image
    is then compressed down back to a space that this loss encourages to be
    similar to the input.
    
    Note: 
        - I added a reconstruction loss between the input image and the
            output of the `G` model, but I changed this to be a metric learning
            problem with an odd noise-selected single dimensional regularization 
            (see code)
        - If no `labels` are passed to forward, `loss_fn` is assumed to be
            a reconstruction style loss that directly compares the output of
            `F` to `E`.
        - I added a Total Variation based penalty to the `G`enerator output
            to encourage it to be totally variational (higher variance)
    """
    def total_variation(self, y):
        # absolute difference in X and Y directions
        return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

    def forward(self, F: Module, G: Module, E: Module, 
                scale: float, alpha: float, z: Tensor, loss_fn: Func, 
                labels=None, 
                use_tv=False, 
                tv_weight=0.001):
        # Hessian applied to G here
        F_z = F(z, scale, z2=None, p_mix=0)
        
        # Autoencoding loss in latent space
        G_z = G(F_z, scale, alpha)
        E_z = E(G_z, alpha)
                
        F_x = F_z[:,0,:]
        if labels is not None:
            # I don't remember if I made this up or not,
            # but it is a noise-based regularization strategy that might
            # encourage variance through order invariance 
            # by duplicating a single element at the same index in both 
            # the projection space (`F`) and the latent space (`E`)
            # when using a distance based loss (metric learning)
            perm = torch.randperm(E_z.shape[0], device=E_z.device)
            E_z_hat = torch.index_select(E_z, 0, perm)
            F_x_hat = torch.index_select(F_x, 0, perm)
            F_hat = torch.cat([F_x, F_x_hat], 0)
            E_hat = torch.cat([E_z, E_z_hat], 0)
            loss = loss_fn(F_hat, E_hat, labels)
        else:
            loss = loss_fn(F_x, E_z)

        if use_tv:
            loss += self.total_variation(G_z) * tv_weight
        return loss
