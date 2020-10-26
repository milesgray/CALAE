import argparse
import math
import torch
import yaml
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm
from util import *
import numpy as np

def get_noise_list(size):
    log_size = int(math.log(size, 2))
    noise_list = [[1, 1, 2 ** 2, 2 ** 2]]
    for i in range(3, log_size + 1):
        for j in range(2):
                noise_list.append([1, 1, 2 ** i, 2 ** i])
    return noise_list


def slerp(val, low, high):
    '''
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    '''
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def get_slerp_interp(nb_latents, nb_interp):
    low = np.random.randn(512)
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for _ in range(nb_latents):
            high = np.random.randn(512)#low + np.random.randn(512) * 0.7

            interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
            latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                            dtype=np.float32)

            latent_interps = np.vstack((latent_interps, latent_interp))
            low = high

    return latent_interps

def get_slerp_loop(nb_latents, nb_interp):
        low = np.random.randn(512)
        og_low = low
        latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
        for _ in range(nb_latents):
                high = np.random.randn(512)#low + np.random.randn(512) * 0.7

                interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
                latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                                dtype=np.float32)

                latent_interps = np.vstack((latent_interps, latent_interp))
                low = high
        
        interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, og_low) for v in interp_vals],
                                                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        return latent_interps

def get_slerp_loop_noise(nb_latents, nb_interp, shape):
        low = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        og_low = low
        latent_interps = np.empty(shape=(shape[0],shape[1],shape[2],shape[3]), dtype=np.float32)
        for _ in range(nb_latents):
                high = np.random.randn(shape[0],shape[1],shape[2],shape[3])#low + np.random.randn(512) * 0.7

                interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
                latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                                dtype=np.float32)

                latent_interps = np.vstack((latent_interps, latent_interp))
                low = high
        
        interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, og_low) for v in interp_vals],
                                                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        return latent_interps

#1 min slerps = get_slerp_loop(32, 45)
def interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
    slerps = get_slerp_loop(32, 45)
    # noise_slerps = []
    # noise_shape_list = get_noise_list(args.size)
    # for shape in noise_shape_list:
    #     noise_slerps.append(get_slerp_loop_noise(32, 35, shape)) 
    t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
    for i in range(len(slerps)):
        print('generating frame: ' + str(i))
        input = torch.tensor(slerps[i])
        input = input.view(1,512)
        input = input.to(device)
        # noises = []
        # for layer_n in noise_slerps:
        #     noises.append(torch.tensor(layer_n[i].to(device)))
        image, _ = g_ema([input],truncation=args.truncation, randomize_noise=False, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

        if not os.path.exists('interp'):
            os.makedirs('interp')
        
        utils.save_image( 
                    image,
                    'interp/'+str(i + 1).zfill(6)+'.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                    padding=0)

def multiple_transform_interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
    slerps = get_slerp_loop(32, 45)
    
    transform_dict_list_list = create_transforms_dict_list_list(yaml_config,cluster_config, layer_channel_dims)
    t_index = 0
    for t_dict_list in transform_dict_list_list:
        for i in range(len(slerps)):
            print('generating frame: ' + str(i))
            input = torch.tensor(slerps[i])
            input = input.view(1,512)
            input = input.to(device)
            image, _ = g_ema([input],truncation=args.truncation, randomize_noise=False, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

            if not os.path.exists('interp/'+str(t_index)+'/'):
                os.makedirs('interp/'+str(t_index)+'/')
            
            utils.save_image( 
                        image,
                        'interp/'+str(t_index)+'/'+str(i + 1).zfill(6)+'.png',
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                        padding=0)
        t_index += 1
