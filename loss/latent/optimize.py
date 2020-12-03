import numpy as np

#from layers.model_ops import snlinear, linear

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import autograd

def latent_optimise(zs, fake_labels, gen_model, dis_model, conditional_strategy, 
                    latent_op_step, latent_op_rate,
                    latent_op_alpha, latent_op_beta, 
                    trans_cost, default_device):
    batch_size = zs.shape[0]
    for step in range(latent_op_step):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - latent_op_rate).to(default_device)
        z_gradients, z_gradients_norm = calc_derv(zs, fake_labels, dis_model, conditional_strategy, default_device, gen_model)
        delta_z = latent_op_alpha*z_gradients/(latent_op_beta + z_gradients_norm)
        zs = torch.clamp(zs + drop_mask*delta_z, -1.0, 1.0)

        if trans_cost:
            if step == 0:
                transport_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                transport_cost += (delta_z.norm(2, dim=1)**2).mean()
            return zs, trans_cost
        else:
            return zs


def calc_derv(inputs, labels, netD, conditional_strategy, device, netG=None):
    zs = autograd.Variable(inputs, requires_grad=True)
    fake_images = netG(zs, labels)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy == 'ACGAN':
        _, dis_out_fake = netD(fake_images, labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=dis_out_fake, inputs=zs,
                              grad_outputs=torch.ones(dis_out_fake.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm
