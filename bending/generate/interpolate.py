import argparse
import math
import torch
import yaml
import os
import random

from torchvision import utils
from CALAE.models.stylegan2_bending import Generator
from tqdm import tqdm
from CALAE.utils.bending import *
from CALAE.utils.interp import *
import numpy as np

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--config', type=str, default="configs/example_transform_config.yaml")
    parser.add_argument('--load_latent', type=str, default="") 
    parser.add_argument('--load_clusters', type=str, default="configs/example_cluster_dict.yaml")
    parser.add_argument('--multiple_transforms',type=int, default=0)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    yaml_config = {}
    with open(args.config, 'r') as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cluster_config = {}
    if args.load_clusters != "":
        with open(args.load_clusters, 'r') as stream:
            try:
                cluster_config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    new_state_dict = g_ema.state_dict()
    checkpoint = torch.load(args.ckpt)
    
    ext_state_dict  = torch.load(args.ckpt)['g_ema']
    g_ema.load_state_dict(checkpoint['g_ema'])
    new_state_dict.update(ext_state_dict)
    g_ema.load_state_dict(new_state_dict)
    g_ema.eval()
    g_ema.to(device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    layer_channel_dims = create_layer_channel_dim_dict(args.channel_multiplier)
    
    if args.multiple_transforms == 1:
        transform_dict_list = create_transforms_dict_list_list(yaml_config, cluster_config, layer_channel_dims)
    else:
        transform_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)

    if args.load_latent == "":
        if args.multiple_transforms == 1:
            multiple_transform_interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
        else:
            interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
