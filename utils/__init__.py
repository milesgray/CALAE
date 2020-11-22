""" Allows for access to these methods and classes through the .utils module.
"""
from .alae import find_alpha, allow_gradient
from .alae import adjust_lr, linear_scale_lr
from .alae import save_batch, save_reconstructions
from .noise import sample_noise, set_noise
from .density import matrix_log_density_gaussian, log_density_gaussian
from .importance import log_importance_weight_matrix
from .files import ensure_dir, ensure_parent_dir
from .alae import img_from_tensor

# General utility methods

def query_gpu(indices=[0]):
    import os
    for i in indices:
        os.system(f'nvidia-smi -i {i} --query-gpu=memory.free --format=csv')
