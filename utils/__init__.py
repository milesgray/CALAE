""" Allows for access to these methods and classes through the .utils module.
"""
from .alae import find_alpha, allow_gradient
from .alae import adjust_lr, linear_scale_lr
from .alae import save_batch, save_reconstructions
from .noise import sample_noise, FAMOS_Noise
from .density import matrix_log_density_gaussian, log_density_gaussian
from .importance import log_importance_weight_matrix
from .files import ensure_dir, ensure_parent_dir
from .alae import img_from_tensor

# General utility methods

def query_gpu(indices=[0]):
    import os
    for i in indices:
        os.system(f'nvidia-smi -i {i} --query-gpu=memory.free --format=csv')

def clean(string_value):
    """Standardizes string values for lookup by removing case and special characters and spaces.

    Args:
        string_value (str): The lookup key to be transformed.

    Returns:
        str: The original value, but lowercase and without spaces, underscores or dashes.
    """
    return string_value.lower().strip().replace("_","").replace("-","").replace(" ","")
