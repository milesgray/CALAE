# -*- coding: utf-8 -*-
"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains the tensorflow implementation of different loss functions for each dissimilarity that are required in DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
Converted to PyTorch by Miles Gray
"""

import torch
from torch import nn
from torvh.nn import functional as F
from torch import linalg as LA

def euclidean_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Euclidean loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return torch.sum(torch.sqrt(torch.sum((X-mu_tilde)**2, axis=1))-torch.log(pi_tilde)/alpha)

def cosine_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Cosine loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    X_norm = LA.norm(X, 2, axis=1)
    mu_tilde_norm = LA.norm(mu_tilde, 2, axis=1)
    return torch.sum((1-torch.sum(X_norm*mu_tilde_norm, axis=1))-torch.log(pi_tilde)/alpha)

def correlation_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Correlation loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    centered_X = X-torch.reshape(torch.mean(X, axis=0),(1, -1))
    centered_mu_tilde = mu_tilde-torch.reshape(torch.mean(mu_tilde, axis=1), (-1,1))
    return cosine_loss(centered_X, centered_mu_tilde, pi_tilde, alpha)

def manhattan_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Manhattan loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return torch.sum(torch.sum(torch.abs(X-mu_tilde), axis=1)-torch.log(pi_tilde)/alpha)

def minkowsky_loss(X, mu_tilde, pi_tilde, alpha, p):
    """
    Computes the Manhattan loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        p: float
            Order of the Minkowski distance.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return torch.sum(torch.sum(torch.abs(X-mu_tilde)**p, axis=1)**(1/p)-torch.log(pi_tilde)/alpha)

def chebyshev_loss(X, mu_tilde, pi_tilde, alpha, p):
    """
    Computes the Chebyshev loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return torch.sum(torch.max(torch.abs(X-mu_tilde), axis=1)-torch.log(pi_tilde)/alpha)

def mahalanobis_loss(X, mu_tilde, Cov_tilde, pi_tilde, alpha):
    """
    Computes the Mahalanobis loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        Cov_tilde: array-like, shape=(batch_size, n_features, n_features)
            Tensor with the assigned covariances.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    diff = torch.unsqueeze(X-mu_tilde, axis=1)
    return torch.sum(torch.squeeze(torch.mm(torch.mm(diff, Cov_tilde), torch.transpose(diff, perm = [0, 2, 1])))\
                         -torch.log(pi_tilde)/alpha)

def mahalanobis_loss_decomp(X, mu_tilde, Cov_tilde, pi_tilde, alpha):
    """
    Computes the Mahalanobis loss using the decomposition of the covariance matrices.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        Cov_tilde: array-like, shape=(batch_size, n_features, n_features)
            Tensor with the assigned decomposition.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    cov = torch.mm(Cov_tilde, torch.transpose(Cov_tilde, [0, 2, 1]))
    diff = torch.unsqueeze(X-mu_tilde, axis=1)
    return torch.sum(torch.squeeze(torch.mm(torch.mm(diff, cov), torch.transpose(diff, perm = [0, 2, 1])))\
                         -torch.log(pi_tilde)/alpha)

class losses():
    def __init__(self):
        self.euclidean_loss = euclidean_loss
        self.cosine_loss = cosine_loss
        self.manhattan_loss =manhattan_loss
        self.minkowsky_loss = minkowsky_loss
        self.correlation_loss = correlation_loss
        self.mahalanobis_loss = mahalanobis_loss
        self.chebyshev_loss = chebyshev_loss
        self.mahalanobis_loss_decomp = mahalanobis_loss_decomp
