import torch
import torch.nn as nn

def euclidean_loss(X, mu_tilde):
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
    
    return torch.sqrt(torch.sum((X-mu_tilde)**2, axis=1))

def cosine_loss(X, mu_tilde):
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
    
    X_norm = torch.nn.l2_normalize(X, axis=1)
    mu_tilde_norm = torch.nn.l2_normalize(mu_tilde, axis=1)
    return 1-torch.sum(X_norm*mu_tilde_norm, axis=1)

def correlation_loss(X, mu_tilde):
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
    return cosine_loss(centered_X, centered_mu_tilde)

def manhattan_loss(X, mu_tilde):
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
    
    return torch.sum(torch.abs(X-mu_tilde), axis=1)

def minkowsky_loss(X, mu_tilde, p):
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
    
    return torch.sum(torch.abs(X-mu_tilde)**p, axis=1)**(1/p)

def chebyshev_loss(X, mu_tilde):
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
    
    return torch.max(torch.abs(X-mu_tilde), axis=1)

def mahalanobis_loss(X, mu_tilde, Cov_tilde):
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
    return torch.squeeze(torch.matmul(torch.matmul(diff, Cov_tilde), torch.transpose(diff, perm = [0, 2, 1])))

def mahalanobis_loss_decomp(X, mu_tilde, Cov_tilde):
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
    
    cov = torch.matmul(Cov_tilde, torch.transpose(Cov_tilde, [0, 2, 1]))
    diff = torch.unsqueeze(X-mu_tilde, axis=1)
    return torch.squeeze(torch.matmul(torch.matmul(diff, cov), torch.transpose(diff, perm = [0, 2, 1])))