import pathlib

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from torchvision.utils import save_image

def find_alpha(tracked, limit):
    return min(tracked/max(limit, 1), 1)

def allow_gradient(module, permission=True):
    for block in module.parameters():
        block.requires_grad = permission

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
        
def linear_scale_lr(tracked, total_items, start=5e-6, end=1.5e-4):
    coef = tracked/total_items
    return (1 - coef) * start + coef * end

def get_total_elements(obj, verbose=False):
    try:
        if torch.is_tensor(obj):
            total = fake.size()[0]
        elif isinstance(obj, tuple):
            total = len(fake)
        elif isinstance(obj, np.ndarray):
            total = fake.shape[0]

        return total
    except Exception as e:
        if verbose: print(f"[ERROR]\t get_total_elements:: {e}")
        return 0

def save_batch(name, fake, real, nrows=6, split=(4,2), verbose=False):
    fake_total = get_total_elements(fake)
    real_total = get_total_elements(real)
    if verbose: print(f"Saving: fake: {fake_total} real: {real_total}")
    try:

        fake, real = fake.split(split[0]), real.split(split[1])
        save_image(torch.cat([torch.cat([fake[i], real[i]], dim=0) for i in range(nrows)], dim=0), name, nrow=nrows, padding=1,
                normalize=True, range=(-1, 1))
    except Exception as e:
        if verbose: print(f"[ERROR]\t Couldn't save! shape fake: {len(fake)}, shape real: {len(real)}, nrows: {nrows} \n\t\t{e}")
    
def save_reconstructions(name, original, reconstruction, nrows=6):
    """
    original, reconstruction - type: list, e.g. original = [x, x_hat], reconstruction = [G(E(x)), G(E(x_hat))]
    
    [[orig_x, rec_x], [orig_x, rec_x], [orig_x, rec_x]]
    [[orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat]]
    
    """
    tensor = []
    for orig, rec in zip(original, reconstruction):        
        tensor.append(torch.cat([torch.cat([orig.split(1)[i], rec.split(1)[i]], dim=0) for i in range(nrows//2)], dim=0))
    
    save_image(torch.cat(tensor, dim=0), name, nrow=nrows, padding=1, normalize=True, range=(-1, 1))


def img_from_tensor(x):
    if len(x.shape) > 3:
        x = x.squeeze().cpu().detach()
    return ((x * 0.5 + 0.5) * 255) \
                .type(torch.long) \
                    .clamp(0, 255) \
                        .cpu() \
                            .type(torch.uint8) \
                                .transpose(0, 2) \
                                    .transpose(0, 1) \
                                        .numpy()
