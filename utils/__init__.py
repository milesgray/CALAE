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

def summary(model, logger=print):
    linfo_name = "LAYER NAME"
    linfo_shape = "LAYER SHAPE"
    linfo_params = "TOTAL PARAMS"
    linfo_mean = "MEAN"
    linfo_var = "VARIANCE"
    linfo_norm = "NORM"
    linfo_nansum = "NANSUM"
    logger(f"[{linfo_name:45s}]\t{linfo_shape:25s}\t|{linfo_params:18s}|{linfo_mean:8s}|{linfo_var:8s}|{linfo_norm:8s}|{linfo_nansum:8s}")
    for layer in list(model.keys()):
        if "opt" in layer: continue
        try:
            real_layer = model[layer]
            if isinstance(real_layer, (dict, list)):
                summary(real_layer)
            else:
                linfo = munch.DefaultMunch()
                linfo.shape = real_layer.shape
                linfo.name = layer
                linfo.params = linfo.shape.numel()
                linfo.mean = float(real_layer.mean().cpu().numpy())
                linfo.var = float(real_layer.var().cpu().numpy())
                linfo.norm = float(real_layer.norm().cpu().numpy())
                linfo.nansum = float(real_layer.nansum().cpu().numpy())
                logger(f"[{linfo.name:45s}]\t{str(linfo.shape):25s}\t|{linfo.params:17d} |{linfo.mean:5.7f} |{linfo.var:5.7f} |{linfo.norm:5.7f} |{linfo.nansum:5.7f} ")
        except:
            logger(f"[{linfo.name:45s}]\t{str(linfo.shape):25s}\t|{linfo.params:17d}")
