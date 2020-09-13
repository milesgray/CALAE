import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from ..utils.wt import zero_mask, zero_pad, postprocess_low_freq
import numpy as np
import pywt

def truncated_normal_(tensor, mean=0, std=0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        with torch.no_grad():
            truncated_normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Sequential):
        for sub_m in m:
            weights_init(sub_m)

def iwt(vres, inv_filters, levels=1):
    bs = vres.shape[0]
    h = vres.size(2)
    w = vres.size(3)
    vres = vres.reshape(-1, 1, h, w)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res.reshape(bs, -1, h, w)

def iwt_haar(vres, inv_filters, levels=1):
    bs = vres.shape[0]
    h = vres.size(2)
    w = vres.size(3)
    vres = vres.reshape(-1, 1, h, w)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt_haar(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)

    return res.reshape(bs, -1, h, w)

def wt(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1], filters, levels-1)
        res[:,:1,32:,:] = res[:,:1,32:,:]*1.
        res[:,:1,:,32:] = res[:,:1,:,32:]*1.
        res[:,1:] = res[:,1:]*1.
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)

    return res.reshape(bs, -1, h, w)

def wt_haar(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(0,0,0,0))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt_haar(res[:,:1], filters, levels-1)
        res[:,:1,32:,:] = res[:,:1,32:,:]*1.
        res[:,:1,:,32:] = res[:,:1,:,32:]*1.
        res[:,1:] = res[:,1:]*1.
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)

    return res.reshape(bs, -1, h, w)

def get_upsampling_layer(name, res, bottleneck_dim=100):
    layer = None
    if name == 'linear':
        layer = nn.Linear(3 * res * res, 3 * res * res)
    elif name == 'conv1d':
        layer = nn.Conv1d(res*res, res*res, kernel_size=1, stride=1)
    elif name == 'conv2d':
        layer = nn.Conv2d(3, 3, kernel_size=1, stride=1)
    elif name == 'bottleneck':
        layer = nn.Sequential(nn.Linear(res*res*3, 1024),
                                nn.Linear(1024, bottleneck_dim),
                                nn.Linear(bottleneck_dim, 1024),
                                nn.Linear(1024, res*res*3)
                             )  
    
    return layer

def get_upsampling_dims(name, res):
    sizes = None
    if name == 'linear' or name == 'bottleneck':
        sizes = (-1, 3 * res * res)
    elif name == 'conv1d':
        sizes = (-1, res*res, 3)
    elif name == 'conv2d':
        sizes = (-1, 3, res, res)
    
    return sizes

class WT(nn.Module):
    def __init__(self, wt, num_wt=2):
        super(WT, self).__init__()

        self.num_wt = num_wt
        self.filters = None
        self.device = None
        self.wt = wt
        
    def forward(self, input):
        return self.wt(input, filters=self.filters, levels=self.num_wt)

    def set_filters(self, filters):
        self.filters = filters     

    def set_device(self, device):
        self.device = device

class IWT(nn.Module):
    def __init__(self, iwt, num_iwt=2):
        super(IWT, self).__init__()

        self.num_iwt = num_iwt
        self.inv_filters = None
        self.device = None
        self.iwt = iwt

    def forward(self, input):
        return self.iwt(input, inv_filters=self.inv_filters, levels=self.num_iwt)
    
    def set_filters(self, filters):
        self.inv_filters = filters

    def set_device(self, device):
        self.device = device
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1)

class UnFlatten1(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 2, 2) 

class UnFlatten4(nn.Module):
    def forward(self, input):
        return input.view(input.shape(0), -1, 4, 4)

# Mask substracting from image
class Mask_Sub(nn.Module):
    def forward(self, img, mask):
        return img - mask.unsqueeze(1)

# Mask adding to image
class Mask_Add(nn.Module):
    def forward(self, img, mask):
        return img + mask.unsqueeze(1)

# Variational Autoencoder Code
class WTVAE_32(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*2*2, z_dim=100, num_wt=2):
        super(WTVAE_32, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 32, 16, 16]
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 8, 8]
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 4, 4]
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 2, 2]
            nn.BatchNorm2d(256),
            self.relu
        )

        # Initializing weights for encoder conv layers
        weights_init(self.encoder)

        # Flatten after this maxpool for linear layer
        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)
        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),      #[b, 128, 4, 4]
            nn.BatchNorm2d(128),
            self.relu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),       #[b, 64, 8, 8]
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True),        #[b, 32, 16, 16]
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True),          #[b, 3, 32, 32]
        )

        # Initializing weights for decoder conv layers
        weights_init(self.decoder)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)                                                         #[b, 256, 2, 2]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (256*2*2)]
        z = self.decoder(z.reshape(-1, 256, 2, 2))                                  #[b, 3, 32, 32]
        
        return z
    
    def sample(self, batch_size):
        sample = torch.randn(batch_size, self.z_dim, device=self.device)
        z_sample = self.decode(sample)

        return z_sample

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        
        return z, mu, logvar

    def loss_function(self, x_wt_hat, x_512, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = self.wt(x_512)
        x_wt = x_wt[:, :, :32, :32]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.mse_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x_512.shape[0] * 32 * 32

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.wt = WT(wt=wt, num_wt=self.num_wt)
        self.wt.set_filters(filters)
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# WTVAE for 64 x 64 images
# num_wt of WT layers (default: 2)
# Using Unflatten for decoder (N * 2048 * 1 * 1), instead of Unflatten1
class WTVAE_64(nn.Module):
    def __init__(self, image_channels=3, h_dim=2048, z_dim=100, unflatten=0, num_wt=2):
        super(WTVAE_64, self).__init__()

        self.inv_filters = None
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), # N * 32 * 31 * 31
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # N * 64 * 14 * 14,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # N * 128 * 6 * 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # N * 256 * 2 * 2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1), # N * 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder with Flatten (N * 2048 * 1 * 1)
        if unflatten == 0:
            
            self.fct_decode_1 = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2), # N * 128 * 5 * 5
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), # N * 64 * 13 * 13
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), # N * 64 * 30 * 30
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2), # N * 3 * 64 * 64
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
            )

        # Decoder with Flatten1 (N * 512 * 2 * 2) and 1 more layer of convolutions
        elif unflatten == 1:
            self.fct_decode_1 = nn.Sequential(
                UnFlatten1(),                                          # N * 512 * 2 * 2
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), # N * 32 * 62 * 62
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1), # N * 3 * 64 * 64 
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Sigmoid(),
            )
        
        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)) # N * 3 * 64 * 64
            self.wt.add_module('wt{}_bn'.format(i), nn.BatchNorm2d(image_channels))
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.fct_decode_1(z)
        z = self.wt(z)
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,64,64)
        x_recon = iwt(wt_x, self.inv_filters, levels=3)
        x_recon = x_recon.view(-1,3,64,64)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD
    
    def set_inv_filters(self, inv_filters):
        self.inv_filters = inv_filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device


# WTVAE for 128 x 128 images
# 2 WT layers
# Using Unflatten1 for decoder (N * 512 * 2 * 2), instead of Flatten (N * 2048 * 1 * 1)
class WTVAE_128(nn.Module):
    def __init__(self, image_channels=3, h_dim=2048, z_dim=100, num_wt=2):
        super(WTVAE_128, self).__init__()

        self.inv_filters = None
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), # N * 32 * 63 * 63
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2), # N * 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder
        self.fct_decode_1 = nn.Sequential(
            UnFlatten1(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), # N * 64 * 62 * 62
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2, output_padding=1), # N * 3 * 128 * 128
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        
        self.wt1 = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2), # N * 3 * 128 * 128
            nn.BatchNorm2d(image_channels)
        )
        
        self.wt2 = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2), # N * 3 * 128 * 128
            nn.BatchNorm2d(image_channels)
        )
        
        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)) # N * 3 * 64 * 64
            self.wt.add_module('wt{}_bn'.format(i), nn.BatchNorm2d(image_channels))
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.fct_decode_1(z)
        z = self.wt(z)
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
       

    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,64,64)
        x_recon = iwt(wt_x, self.inv_filters, levels=3)
        x_recon = x_recon.view(-1,3,64,64)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD
    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,128,128)
        x_recon = iwt(wt_x, levels=2)
        x_recon = x_recon.view(-1,3,128,128)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 128 * 128), x.view(-1, 3 * 128 * 128))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= BATCH_SIZE * 3 * 128 * 128

        return BCE + KLD

    def set_inv_filters(self, inv_filters):
        self.inv_filters = inv_filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

class WTVAE_64_1(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*4*4, z_dim=100, num_wt=2):
        super(WTVAE_64_1, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 32, 32, 32]
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 16, 16]
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 8, 8]
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 4, 4]
            nn.BatchNorm2d(256),
            self.relu
        )

        # Initializing weights for encoder conv layers
        weights_init(self.encoder)

        # Flatten after this maxpool for linear layer
        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)
        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),      #[b, 128, 8, 8]
            nn.BatchNorm2d(128),
            self.relu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),       #[b, 64, 16, 16]
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True),        #[b, 32, 32, 32]
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True),          #[b, 3, 64, 64]
            nn.BatchNorm2d(3),
            self.sigmoid
        )

        # Initializing weights for decoder conv layers
        weights_init(self.decoder)

        self.wt_layer = nn.Sequential()
        for i in range(self.num_wt):
            self.wt_layer.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=3, stride=1, padding=1)) # N * 3 * 64 * 64, when num_wt=2
            self.wt_layer.add_module('wt{}_in'.format(i), nn.BatchNorm2d(image_channels))
        
        weights_init(self.wt_layer)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)                                                         #[b, 256, 4, 4]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (256*4*4)]
        z = self.decoder(z.reshape(-1, 256, 4, 4))                                  #[b, 3, 64, 64]
        z = self.wt_layer(z)                                                        #[b, 3, 64, 64], when num_wt=2
        
        return z
    
    def sample(self, batch_size):
        sample = torch.randn(batch_size, self.z_dim, device=self.device)
        z_sample = self.decode(sample)

        return z_sample

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, x_512, x_wt_hat, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = self.wt(x_512)
        x_wt = x_wt[:, :, :64, :64]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.mse_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x_512.shape[0] * 64 * 64

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.wt = WT(wt=wt, num_wt=self.num_wt)
        self.wt.set_filters(filters)
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

class WTVAE_128_1(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=100, num_wt=2):
        super(WTVAE_128_1, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 32, 64, 64]
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 32, 32]
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 16, 16]
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 8, 8]
            nn.BatchNorm2d(256),
            self.relu
        )

        # Initializing weights for encoder conv layers
        weights_init(self.encoder)

        # Flatten after this maxpool for linear layer
        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)
        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),      #[b, 128, 16, 16]
            nn.BatchNorm2d(128),
            self.relu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),       #[b, 64, 32, 32]
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True),        #[b, 32, 64, 64]
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True),          #[b, 3, 128, 128]
            nn.BatchNorm2d(3),
            self.sigmoid
        )

        # Initializing weights for decoder conv layers
        weights_init(self.decoder)

        self.wt_layer = nn.Sequential()
        for i in range(self.num_wt):
            self.wt_layer.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=3, stride=1, padding=1)) # N * 3 * 128 * 128, when num_wt=2
            self.wt_layer.add_module('wt{}_in'.format(i), nn.BatchNorm2d(image_channels))
        
        weights_init(self.wt_layer)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)                                                         #[b, 128, 32, 32]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (256*8*8)]
        z = self.decoder(z.reshape(-1, 256, 8, 8))                                  #[b, 3, 512, 512]
        z = self.wt_layer(z)                                                        #[b, 3, 128, 128], when num_wt=2
        
        return z
    
    def sample(self, batch_size):
        sample = torch.randn(batch_size, self.z_dim, device=self.device)
        z_sample = self.decode(sample)

        return z_sample

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, x_512, x_wt_hat, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = self.wt(x_512)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.mse_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x_512.shape[0] * 128 * 128

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.wt = WT(wt=wt, num_wt=self.num_wt)
        self.wt.set_filters(filters)
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

class WTVAE_128_FixedWT(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=100, num_wt=2):
        super(WTVAE_128_FixedWT, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 32, 64, 64]
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 32, 32]
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 16, 16]
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 8, 8]
            nn.BatchNorm2d(256),
            self.relu
        )

        # Initializing weights for encoder conv layers
        weights_init(self.encoder)

        # Flatten after this maxpool for linear layer
        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)
        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),      #[b, 128, 16, 16]
            nn.BatchNorm2d(128),
            self.relu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),       #[b, 64, 32, 32]
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True),        #[b, 32, 64, 64]
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True),          #[b, 3, 128, 128]
            nn.BatchNorm2d(3),
            self.sigmoid
        )

        # Initializing weights for decoder conv layers
        weights_init(self.decoder)

        self.wt = WT(num_wt=num_wt)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)                                                         #[b, 128, 32, 32]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (256*8*8)]
        z = self.decoder(z.reshape(-1, 256, 8, 8))                                  #[b, 3, 128, 128]
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        z = self.wt(z)

        return z, mu, logvar

    def loss_function(self, x_512, x_wt_hat, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = wt(x_512.reshape(x_512.shape[0] * x_512.shape[1], 1, x_512.shape[2], x_512.shape[3]), self.filters, levels=3)
        x_wt = x_wt.reshape(x_512.shape)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x_512.shape[0] * 128 * 128

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.filters = filters
        self.wt.set_filters(filters)
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

class WTVAE_512(nn.Module):
    def __init__(self, image_channels=3, h_dim=512*4*4, z_dim=100, num_wt=2):
        super(WTVAE_512, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
        self.e1 = nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 32, 256, 256]
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=32, affine=False)
        weights_init(self.e1)

        self.e2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 128, 128]
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=64, affine=False)
        weights_init(self.e2)

        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 64, 64, 64]
        
        self.e3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 32, 32]
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=128, affine=False)
        weights_init(self.e3)

        self.e4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 16, 16]
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=256, affine=False)
        weights_init(self.e4)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 256, 8, 8]

        self.e5 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        self.instance_norm_e5 = nn.InstanceNorm2d(num_features=512, affine=False)
        weights_init(self.e5)

        # Flatten after this maxpool for linear layer

        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)

        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True)       #[b, 256, 8, 8]
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=256, affine=False)
        weights_init(self.d1)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)                    #[b, 256, 16, 16]

        self.d2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True)       #[b, 128, 32, 32]
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)
        weights_init(self.d2)

        self.d3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True)        #[b, 64, 64, 64]
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=64, affine=False)
        weights_init(self.d3)

        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)                    #[b, 64, 128, 128]

        self.d4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True)             #[b, 3, 256, 256]
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=64, affine=False)         
        weights_init(self.d4)

        self.d5 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True)             #[b, 3, 512, 512]
        self.instance_norm_d5 = nn.InstanceNorm2d(num_features=64, affine=False)         
        weights_init(self.d5)

        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=4, stride=2, padding=1)) # N * 3 * 128 * 128, when num_wt=2
            self.wt.add_module('wt{}_in'.format(i), nn.InstanceNorm2d(image_channels))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 32, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 64, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 64, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 128, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 256, 16, 16]
        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)     
        h = self.leakyrelu(self.instance_norm_e5(self.e5(h)))                       #[b, 512, 4, 4]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar, m1_idx, m2_idx

    def decode(self, z, m1_idx, m2_idx):
        z = self.fc_dec(z)                                                          #[b, h_dim (512*4*4)]
        z = self.leakyrelu(self.instance_norm_d1(self.d1(z.reshape(-1, 512, 4, 4))))#[b, 256, 8, 8]
        z = self.leakyrelu(self.u1(z, indices=m2_idx))                              #[b, 256, 16, 16]
        z = self.leakyrelu(self.instance_norm_d2(self.d2(z)))                       #[b, 128, 32, 32]
        z = self.leakyrelu(self.instance_norm_d3(self.d3(z)))                       #[b, 64, 64, 64]
        z = self.leakyrelu(self.u2(z, indices=m1_idx))                              #[b, 64, 128, 128]
        z = self.leakyrelu(self.instance_norm_d4(self.d4(z)))                       #[b, 32, 256, 256]
        z = self.relu(self.instance_norm_d5(self.d5(z)))                            #[b, 3, 512, 512]
        z = self.wt(z)                                                              #[b, 3, 128, 128], when num_wt=2
        
        return z

    def forward(self, x):
        z, mu, logvar, m1_idx, m2_idx = self.encode(x)
        z = self.decode(z, m1_idx, m2_idx)
        return z, mu, logvar

    def loss_function(self, x, x_wt_hat, mu, logvar) -> Variable:
        
        x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), self.filters, levels=2)
        x_wt = x_wt.reshape(x.shape)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
        KLD /= x.shape[0] * 3 * 128 * 128

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.filters = filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# Testing out more of a similar architecture to WTVAE_64
class WTVAE_512_1(nn.Module):
    def __init__(self, image_channels=3, h_dim=512*8*8, z_dim=100, num_wt=2):
        super(WTVAE_512_1, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.e1 = nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 32, 256, 256]
        self.instance_norm_e1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 128, 128]
        self.instance_norm_e2 = nn.BatchNorm2d(64)
        
        self.e3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 64, 64]
        self.instance_norm_e3 = nn.BatchNorm2d(128)

        self.e4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        self.instance_norm_e4 = nn.BatchNorm2d(256)

        self.e5 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        self.instance_norm_e5 = nn.BatchNorm2d(512)

        self.e6 = nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 8, 8]
        self.instance_norm_e6 = nn.BatchNorm2d(512)

        # Flatten after this maxpool for linear layer

        self.fc_mean = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        self.fc_dec = nn.Linear(z_dim, h_dim)

        # Unflatten before going through layers of decoder

        self.d1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=True)       #[b, 512, 16, 16]
        self.instance_norm_d1 = nn.BatchNorm2d(512)

        self.d2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True)       #[b, 256, 32, 32]
        self.instance_norm_d2 = nn.BatchNorm2d(256)

        self.d3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True)       #[b, 128, 64, 64]
        self.instance_norm_d3 = nn.BatchNorm2d(128)

        self.d4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True)        #[b, 64, 128, 128]
        self.instance_norm_d4 = nn.BatchNorm2d(64)

        self.d5 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True)         #[b, 32, 256, 256]
        self.instance_norm_d5 = nn.BatchNorm2d(32)

        self.d6 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True)          #[b, 3, 512, 512]
        self.instance_norm_d6 = nn.BatchNorm2d(3)
        # weights_init(self.d5)

        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=4, stride=2, padding=1)) # N * 3 * 128 * 128, when num_wt=2
            self.wt.add_module('wt{}_in'.format(i), nn.BatchNorm2d(image_channels))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.relu(self.instance_norm_e1(self.e1(x)))                       #[b, 32, 256, 256]
        h = self.relu(self.instance_norm_e2(self.e2(h)))                       #[b, 64, 128, 128]
        h = self.relu(self.instance_norm_e3(self.e3(h)))                       #[b, 128, 64, 64]                                                       
        h = self.relu(self.instance_norm_e4(self.e4(h)))                       #[b, 256, 32, 32]
        h = self.relu(self.instance_norm_e5(self.e5(h)))                       #[b, 512, 16, 16]
        h = self.relu(self.instance_norm_e6(self.e6(h)))                       #[b, 512, 8, 8]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (512*8*8)]
        z = self.relu(self.instance_norm_d1(self.d1(z.reshape(-1, 512, 8, 8))))     #[b, 512, 16, 16]
        z = self.relu(self.instance_norm_d2(self.d2(z)))                            #[b, 256, 32, 32]
        z = self.relu(self.instance_norm_d3(self.d3(z)))                            #[b, 128, 64, 64]
        z = self.relu(self.instance_norm_d4(self.d4(z)))                            #[b, 64, 128, 128]
        z = self.relu(self.instance_norm_d5(self.d5(z)))                            #[b, 32, 256, 256]
        z = self.sigmoid(self.instance_norm_d6(self.d6(z)))                         #[b, 3, 512, 512]
        z = self.wt(z)                                                              #[b, 3, 128, 128], when num_wt=2
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, x, x_wt_hat, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), self.filters, levels=2)
        x_wt = x_wt.reshape(x.shape)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x.shape[0] * 3 * 512 * 512

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.filters = filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# Version 2: with 4 layers in each encoder + decoder
class WTVAE_512_2(nn.Module):
    def __init__(self, image_channels=3, h_dim=128*32*32, z_dim=100, num_wt=2):
        super(WTVAE_512_2, self).__init__()
        
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 32, 256, 256]
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 128, 128]
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 64, 64]
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 32, 32]
            nn.BatchNorm2d(128),
            self.relu
        )

        # Initializing weights for encoder conv layers
        weights_init(self.encoder)

        # Flatten after this maxpool for linear layer
        self.fc_mean = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        weights_init(self.fc_logvar)
        self.fc_dec = nn.Linear(z_dim, h_dim)
        weights_init(self.fc_dec)

        # Unflatten before going through layers of decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=True),      #[b, 128, 64, 64]
            nn.BatchNorm2d(128),
            self.relu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),       #[b, 64, 128, 128]
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=True),        #[b, 32, 256, 256]
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True),          #[b, 3, 512, 512]
            nn.BatchNorm2d(3),
            self.sigmoid
        )

        # Initializing weights for decoder conv layers
        weights_init(self.decoder)

        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=4, stride=2, padding=1)) # N * 3 * 128 * 128, when num_wt=2
            self.wt.add_module('wt{}_in'.format(i), nn.BatchNorm2d(image_channels))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)                                                         #[b, 128, 32, 32]

        z, mu, logvar = self.bottleneck(h.reshape(h.shape[0], -1))                  #[b, z_dim]

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_dec(z)                                                          #[b, h_dim (128*32*32)]
        z = self.decoder(z.reshape(-1, 128, 32, 32))                                #[b, 3, 512, 512]
        z = self.wt(z)                                                              #[b, 3, 128, 128], when num_wt=2
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, x, x_wt_hat, mu, logvar, kl_weight=1.0) -> Variable:
        
        x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), self.filters, levels=2)
        x_wt = x_wt.reshape(x.shape)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x.shape[0]

        return BCE + KLD, BCE, KLD

    def set_filters(self, filters):
        self.filters = filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# Simple 4 layer CNN
class WTCNN_512(nn.Module):
    def __init__(self, image_channels=3, h_dim=512*8*8, z_dim=100, num_wt=2):
        super(WTCNN_512, self).__init__()
        
        self.cuda = False
        self.device = None
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.wt = nn.Sequential(
            nn.Conv2d(3, 3, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 3, 256, 256]
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, 5, stride=1, padding=2, bias=True, padding_mode='zeros'), #[b, 3, 256, 256]
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 3, 128, 128]
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, 5, stride=1, padding=2, bias=True, padding_mode='zeros'), #[b, 3, 128, 128]
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        h = self.wt(x)
        
        return h

    def loss_function(self, x, x_wt_hat) -> Variable:
        
        x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), self.filters, levels=2)
        x_wt = x_wt.reshape(x.shape)
        x_wt = x_wt[:, :, :128, :128]
        
        # Loss btw original WT 1st patch & reconstructed 1st patch
        BCE = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))

        return BCE

    def set_filters(self, filters):
        self.filters = filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=100, upsampling='linear', num_upsampling=2, reuse=False):
        super(IWTVAE_64, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.upsampling = upsampling
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
                weights_init(self.d2)

            self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        upsampling_sizes = get_upsampling_dims(self.upsampling, self.res)
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt1(h))                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 3, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
            h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

class IWTVAE_64_FreezeIWT(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=0, upsampling='linear', num_upsampling=2, reuse=False):
        super(IWTVAE_64_FreezeIWT, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.upsampling = upsampling
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = IWT()
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
                weights_init(self.d2)

            self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = self.iwt1
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        upsampling_sizes = get_upsampling_dims(self.upsampling, self.res)
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y.reshape(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.iwt1(h)                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 3, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.reshape(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
            h = self.iwt2(h)                               #[b, 3, 64, 64]
        
        return h
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64_Bottleneck(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=100):
        super(IWTVAE_64_Bottleneck, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer('bottleneck', self.res)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        # if self.num_upsampling > 1:
        #     if self.reuse:
        #         self.d2 = self.d1
        #     else:
        #         self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        #         weights_init(self.d2)

        # self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
        # self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
        # self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(y + mu)) #[b, 3, 64, 64]
        h = self.d1(h.reshape(-1, 3*64*64))                                                 #[b, 3*64*64]
        h = self.leakyrelu(self.iwt1(h.reshape(-1, 3, 64, 64)))                               #[b, 3, 64, 64]
        
        # mu = self.mu2(z).reshape(-1, 3, 64, 64)
        # var = self.var2(z).reshape(-1, 3, 64, 64)
        # h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

class AE_Mask_64(nn.Module):
    def __init__(self, image_channels=3, z_dim=100):
        super(AE_Mask_64, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 64, 64]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.BatchNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.BatchNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.BatchNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.BatchNorm2d(num_features=512, affine=False)

        self.e5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 1024, 2, 2]
        weights_init(self.e5)
        self.instance_norm_e5 = nn.BatchNorm2d(num_features=1024, affine=False)
        
        self.fc_enc = nn.Linear(1024 * 2 * 2, z_dim)
        weights_init(self.fc_enc)
        
        self.fc_dec = nn.Linear(z_dim, 1024 * 2 * 2)
        weights_init(self.fc_dec)

        self.d1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True) #[b, 512, 4, 4]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.BatchNorm2d(num_features=512, affine=False)
        
        self.d2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 8, 8]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.BatchNorm2d(num_features=256, affine=False)

        self.d3= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 16, 16]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.BatchNorm2d(num_features=128, affine=False)

        self.d4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True) #[b, 64, 32, 32]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.BatchNorm2d(num_features=64, affine=False)

        self.d5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True) #[b, 3, 64, 64]
        weights_init(self.d5)
        self.instance_norm_d5 = nn.BatchNorm2d(num_features=3, affine=False)
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 4, 4]
        h = self.leakyrelu(self.instance_norm_e5(self.e5(h)))                       #[b, 1024, 2, 2]

        h = self.leakyrelu(self.fc_enc(h.reshape(-1,1024*2*2)))                     #[b, z_dim]

        return h
    
    def decode(self, x):
        h = self.leakyrelu(self.fc_dec(x))                                          #[b, 1024*2*2]

        h = self.leakyrelu(self.instance_norm_d1(self.d1(h.reshape(-1, 1024, 2, 2))))#[b, 512, 4, 4]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                       #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                       #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_d4(self.d4(h)))                       #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_d5(self.d5(h)))                       #[b, 3, 64, 64]

        return h
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        
        return x

    def loss_function(self, x, x_hat, criterion):
        loss = criterion(x_hat.reshape(-1), x.reshape(-1))

        return loss
    
    def set_device(self, device):
        self.device = device

class AE_Mask_128(nn.Module):
    def __init__(self, image_channels=3, z_dim=100):
        super(AE_Mask_128, self).__init__()
        # Resolution of images (128 x 128)
        self.res = 128
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 128, 128]
        self.e1 = nn.Conv2d(image_channels, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 64, 64]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.BatchNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 32, 32]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.BatchNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 16, 16]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.BatchNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 8, 8]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.BatchNorm2d(num_features=512, affine=False)

        self.e5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 1024, 4, 4]
        weights_init(self.e5)
        self.instance_norm_e5 = nn.BatchNorm2d(num_features=1024, affine=False)
        
        self.fc_enc = nn.Linear(1024 * 4 * 4, z_dim)
        weights_init(self.fc_enc)
        
        self.fc_dec = nn.Linear(z_dim, 1024 * 4 * 4)
        weights_init(self.fc_dec)

        self.d1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True) #[b, 512, 8, 8]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.BatchNorm2d(num_features=512, affine=False)
        
        self.d2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 16, 16]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.BatchNorm2d(num_features=256, affine=False)

        self.d3= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 32, 32]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.BatchNorm2d(num_features=128, affine=False)

        self.d4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True) #[b, 64, 64, 64]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.BatchNorm2d(num_features=64, affine=False)

        self.d5 = nn.ConvTranspose2d(64, image_channels, 4, stride=2, padding=1, bias=True) #[b, 3, 128, 128]
        weights_init(self.d5)
        self.instance_norm_d5 = nn.BatchNorm2d(num_features=3, affine=False)
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 64, 64]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 32, 32]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 16, 16]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 8, 8]
        h = self.leakyrelu(self.instance_norm_e5(self.e5(h)))                       #[b, 1024, 4, 4]

        h = self.leakyrelu(self.fc_enc(h.reshape(-1,1024*4*4)))                     #[b, z_dim]

        return h
    
    def decode(self, x):
        h = self.leakyrelu(self.fc_dec(x))                                          #[b, 1024*4*4]

        h = self.leakyrelu(self.instance_norm_d1(self.d1(h.reshape(-1, 1024, 4, 4))))#[b, 512, 8, 8]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                       #[b, 256, 16, 16]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                       #[b, 128, 32, 32]
        h = self.leakyrelu(self.instance_norm_d4(self.d4(h)))                       #[b, 64, 64, 64]
        h = self.leakyrelu(self.instance_norm_d5(self.d5(h)))                       #[b, 3, 128, 128]

        return h
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        
        return x

    def loss_function(self, x, x_hat, criterion):
        loss = criterion(x_hat.reshape(-1), x.reshape(-1))

        return loss
    
    def set_device(self, device):
        self.device = device

class AE_Mask_128_Channels(nn.Module):
    def __init__(self, image_channels=9, z_dim=100):
        super(AE_Mask_128_Channels, self).__init__()
        # Resolution of images (128 x 128), but patches concatenated into channels (each 3 x 64 x 64)
        self.res = 64
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 64, 64]
        self.e1 = nn.Conv2d(image_channels, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.BatchNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.BatchNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.BatchNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.BatchNorm2d(num_features=512, affine=False)

        self.e5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 1024, 2, 2]
        weights_init(self.e5)
        self.instance_norm_e5 = nn.BatchNorm2d(num_features=1024, affine=False)
        
        self.fc_enc = nn.Linear(1024 * 2 * 2, z_dim)
        weights_init(self.fc_enc)
        
        self.fc_dec = nn.Linear(z_dim, 1024 * 2 * 2)
        weights_init(self.fc_dec)

        self.d1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True) #[b, 512, 4, 4]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.BatchNorm2d(num_features=512, affine=False)
        
        self.d2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 8, 8]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.BatchNorm2d(num_features=256, affine=False)

        self.d3= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 16, 16]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.BatchNorm2d(num_features=128, affine=False)

        self.d4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True) #[b, 64, 32, 32]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.BatchNorm2d(num_features=64, affine=False)

        self.d5 = nn.ConvTranspose2d(64, image_channels, 4, stride=2, padding=1, bias=True) #[b, 9, 64, 64]
        weights_init(self.d5)
        self.instance_norm_d5 = nn.BatchNorm2d(num_features=9, affine=False)
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 4, 4]
        h = self.leakyrelu(self.instance_norm_e5(self.e5(h)))                       #[b, 1024, 2, 2]

        h = self.leakyrelu(self.fc_enc(h.reshape(-1,1024*2*2)))                     #[b, z_dim]

        return h
    
    def decode(self, x):
        h = self.leakyrelu(self.fc_dec(x))                                          #[b, 1024*2*2]

        h = self.leakyrelu(self.instance_norm_d1(self.d1(h.reshape(-1, 1024, 2, 2))))#[b, 512, 4, 4]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                       #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                       #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_d4(self.d4(h)))                       #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_d5(self.d5(h)))                       #[b, 9, 64, 64]

        return h
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        
        return x

    def loss_function(self, x, x_hat, criterion):
        loss = criterion(x_hat.reshape(-1), x.reshape(-1))

        return loss
    
    def set_device(self, device):
        self.device = device

class AE_Mask_512(nn.Module):
    def __init__(self, image_channels=3, z_dim=100):
        super(AE_Mask, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 128, 128]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 128, 64, 64]

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 512, 8, 8]
        
        self.fc_enc = nn.Linear(512 * 8 * 8, z_dim)
        weights_init(self.fc_enc)
        
        self.fc_dec = nn.Linear(z_dim, 512 * 8 * 8)
        weights_init(self.fc_dec)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 512, 16, 16]

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 32, 32]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.d2= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 64, 64]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)
    
        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 128, 128, 128]

        self.d3 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=32, affine=False)

        self.d4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True) #[b, 1, 512, 512]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=3, affine=False)
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 128, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 16, 16]

        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)
        h = self.leakyrelu(self.fc_enc(h.reshape(-1,512*8*8)))                      #[b, z_dim]

        return h, m1_idx, m2_idx                                                    #[b, z_dim]
    
    def decode(self, x, m1_idx, m2_idx):
        h = self.leakyrelu(self.fc_dec(x))                                      #[b, 512*8*8]
        h = self.leakyrelu(self.u1(h.reshape(-1, 512, 8, 8), indices=m2_idx))   #[b, 512, 16, 16]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h)))                   #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                   #[b, 128, 64, 64]
        h = self.leakyrelu(self.u2(h, indices=m1_idx))                          #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                   #[b, 32, 256, 512]
        h = self.instance_norm_d4(self.d4(h))                                   #[b, 1, 256, 512]

        return h
    
    def forward(self, x):
        x, m1_idx, m2_idx = self.encode(x)
        x = self.decode(x, m1_idx, m2_idx)
        
        return x

    def loss_function(self, x, x_hat, criterion):
        loss = criterion(x_hat.reshape(-1), x.reshape(-1))

        return loss
    
    def set_device(self, device):
        self.device = device


class AE_Mask_512_1(nn.Module):
    def __init__(self, image_channels=3, z_dim=100):
        super(AE_Mask, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.BatchNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.BatchNorm2d(num_features=128, affine=False)

        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 128, 64, 64]

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.BatchNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.BatchNorm2d(num_features=512, affine=False)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 512, 8, 8]
        
        self.fc_enc = nn.Linear(512 * 8 * 8, z_dim)
        weights_init(self.fc_enc)
        
        self.fc_dec = nn.Linear(z_dim, 512 * 8 * 8)
        weights_init(self.fc_dec)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 512, 16, 16]

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 32, 32]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.BatchNorm2d(num_features=256, affine=False)

        self.d2= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 64, 64]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.BatchNorm2d(num_features=128, affine=False)
    
        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 128, 128, 128]

        self.d3 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.BatchNorm2d(num_features=32, affine=False)

        self.d4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=True) #[b, 1, 512, 512]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.BatchNorm2d(num_features=3, affine=False)
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 128, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 16, 16]

        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)
        h = self.leakyrelu(self.fc_enc(h.reshape(-1,512*8*8)))                      #[b, z_dim]

        return h, m1_idx, m2_idx                                                    #[b, z_dim]
    
    def decode(self, x, m1_idx, m2_idx):
        h = self.leakyrelu(self.fc_dec(x))                                      #[b, 512*8*8]
        h = self.leakyrelu(self.u1(h.reshape(-1, 512, 8, 8), indices=m2_idx))   #[b, 512, 16, 16]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h)))                   #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                   #[b, 128, 64, 64]
        h = self.leakyrelu(self.u2(h, indices=m1_idx))                          #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                   #[b, 32, 256, 512]
        h = self.instance_norm_d4(self.d4(h))                                   #[b, 1, 256, 512]

        return h
    
    def forward(self, x):
        x, m1_idx, m2_idx = self.encode(x)
        x = self.decode(x, m1_idx, m2_idx)
        
        return x

    def loss_function(self, x, x_hat, criterion):
        loss = criterion(x_hat.reshape(-1), x.reshape(-1))

        return loss
    
    def set_device(self, device):
        self.device = device



# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64_Mask(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, num_upsampling=2, reuse=False):
        super(IWTVAE_64_Mask, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder
        self.d1 = Mask()    
        self.mu1 = nn.Linear(z_dim, 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = Mask()

            self.mu2 = nn.Linear(z_dim, 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x-y)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        mu = self.mu1(z).reshape(-1, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y, mu))) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt1(h))                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(y, mu))) #[b, 3, 64, 64]
            h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

# Reconstructing 3 masks separately (each 128 x 128)
class IWTVAE_128_3Masks(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_128_3Masks, self).__init__()
        # Resolution of images (128 x 128)
        self.res = 128
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 2, 2]
            # nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 2 * 2, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec1 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec1)

        self.fc_dec2 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec2)

        self.fc_dec3 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec3)


        # Z Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )

        # Initializing weights of decoder
        weights_init(self.decoder1)
        weights_init(self.decoder2)
        weights_init(self.decoder3)
        
        self.iwt = None
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 2, 2]
        h = self.fc_enc(h.reshape(-1,2048*2*2))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, z):
        z1 = self.leakyrelu(self.fc_dec1(z))                       #[b, 2048*2*2]
        z1 = self.decoder1(z1.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        z2 = self.leakyrelu(self.fc_dec2(z))                       #[b, 2048*2*2]
        z2 = self.decoder2(z2.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        z3 = self.leakyrelu(self.fc_dec3(z))                       #[b, 2048*2*2]
        z3 = self.decoder3(z3.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        # Returns mask
        return z1, z2, z3

    def sample(self, batch_size):
        z_sample = torch.randn(batch_size, self.z_dim, device=self.device)
        sample1, sample2, sample3 = self.decode(z_sample)
        
        return sample1, sample2, sample3
        
    def forward(self, y):
        mu, var = self.encode(y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask1, mask2, mask3  = self.decode(z)
        
        return mask1, mask2, mask3, mu, var
        
    def loss_function(self, mask1, mask1_hat, mask2, mask2_hat, mask3, mask3_hat, mu, var) -> Variable:

        # Computing loss on each of the masks
        BCE_wt = F.mse_loss(mask1_hat.reshape(-1), mask1.reshape(-1))
        BCE_wt += F.mse_loss(mask2_hat.reshape(-1), mask2.reshape(-1))
        BCE_wt += F.mse_loss(mask3_hat.reshape(-1), mask3.reshape(-1))

        BCE_wt *= (mask1.shape[1] * mask1.shape[2] * mask1.shape[3])
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= mask1.shape[0]

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# Reconstructing 3 masks separately (each 128 x 128) in AE version without KLD
class IWTVAE_128_3Masks_1(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_128_3Masks_1, self).__init__()
        # Resolution of images (128 x 128)
        self.res = 128
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 64, 64]
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 32, 32]
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 16, 16]
            nn.BatchNorm2d(256),
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 8, 8]
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 1024, 4, 4]
            nn.BatchNorm2d(1024),
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 2, 2]
            nn.BatchNorm2d(2048),
            # nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 2 * 2, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec1 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec1)

        self.fc_dec2 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec2)

        self.fc_dec3 = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec3)


        # Z Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            nn.BatchNorm2d(1024),
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            nn.BatchNorm2d(256),
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            nn.BatchNorm2d(1024),
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            nn.BatchNorm2d(256),
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            nn.BatchNorm2d(1024),
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            nn.BatchNorm2d(256),
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            nn.BatchNorm2d(128),
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            nn.BatchNorm2d(64),
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 128, 128]
        )

        # Initializing weights of decoder
        weights_init(self.decoder1)
        weights_init(self.decoder2)
        weights_init(self.decoder3)
        
        self.iwt = None
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 2, 2]
        h = self.fc_enc(h.reshape(-1,2048*2*2))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, z):
        z1 = self.leakyrelu(self.fc_dec1(z))                       #[b, 2048*2*2]
        z1 = self.decoder1(z1.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        z2 = self.leakyrelu(self.fc_dec2(z))                       #[b, 2048*2*2]
        z2 = self.decoder2(z2.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        z3 = self.leakyrelu(self.fc_dec3(z))                       #[b, 2048*2*2]
        z3 = self.decoder3(z3.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]

        # Returns mask
        return z1, z2, z3

    def sample(self, batch_size):
        z_sample = torch.randn(batch_size, self.z_dim, device=self.device)
        sample1, sample2, sample3 = self.decode(z_sample)
        
        return sample1, sample2, sample3
        
    def forward(self, y):
        mu, var = self.encode(y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask1, mask2, mask3  = self.decode(z)
        
        return mask1, mask2, mask3, mu, var
        
    def loss_function(self, mask1, mask1_hat, mask2, mask2_hat, mask3, mask3_hat, mu, var) -> Variable:

        # Computing loss on each of the masks
        BCE_wt = F.mse_loss(mask1_hat.reshape(-1), mask1.reshape(-1))
        BCE_wt += F.mse_loss(mask2_hat.reshape(-1), mask2.reshape(-1))
        BCE_wt += F.mse_loss(mask3_hat.reshape(-1), mask3.reshape(-1))

        BCE_wt *= (mask1.shape[1] * mask1.shape[2] * mask1.shape[3])
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= mask1.shape[0]

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

class IWTVAE_128_Mask_LearnIWT(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_128_Mask_LearnIWT, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 128
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 128, 128]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 128, 64, 64]

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 512, 8, 8]
        
        self.fc_enc = nn.Linear(512 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 512 * 8 * 8)
        weights_init(self.fc_dec)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 512, 16, 16]

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 32, 32]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.d2= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 64, 64]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)
    
        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 128, 128, 128]

        self.d3 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=32, affine=False)

        self.d4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=True) #[b, 1, 512, 512]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=3, affine=False)
        
        self.iwt = None
    
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x-self.iwt(y))))           #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 128, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 16, 16]

        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)
        h = self.leakyrelu(self.fc_enc(h.reshape(-1,512*8*8)))                      #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h)), m1_idx, m2_idx          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z, m1_idx, m2_idx):
        h = self.leakyrelu(self.fc_dec(z))                                      #[b, 512*8*8]
        h = self.leakyrelu(self.u1(h.reshape(-1, 512, 8, 8), indices=m2_idx))   #[b, 512, 16, 16]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h)))                   #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                   #[b, 128, 64, 64]
        h = self.leakyrelu(self.u2(h, indices=m1_idx))                          #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                   #[b, 32, 256, 512]
        h = self.sigmoid(self.instance_norm_d4(self.d4(h)))                     #[b, 1, 256, 512]
        h = h.clone()

        # Dynamic masks (covering all irrelevant patches at each IWT)
        # for i in range(self.num_iwt):
        #     with torch.no_grad():
        #         mask = mask_og.clone().detach()
        #         mask = zero_mask(mask.squeeze(1), self.num_iwt, i+1)
        #     h = y - mask.unsqueeze(1)
        #     h = self.iwt(h)

        # Static mask (covering the first patch)
        with torch.no_grad():
            h = zero_mask(h.squeeze(1), self.num_iwt, 1)
            assert (h[:, :128, :128] == 0).all()
        
        assert((y[:, :, 128:, 128:] == 0).all())
        h = y + h.unsqueeze(1)
        # h = postprocess_low_freq(h)
        # h = self.iwt(h)
        
        return h
        
    def forward(self, x, y):
        mu, var, m1_idx, m2_idx = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z, m1_idx, m2_idx)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, x_wt, x_wt_hat, mu, var, img_loss=False) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = 0
        if img_loss:
            BCE = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level other than 1st patch
        # BCE_wt = F.l1_loss(x_wt_hat[:, :, 128:, 128:].reshape(-1), x_wt[:, :, 128:, 128:].reshape(-1))
        BCE_wt = F.binary_cross_entropy(x_wt_hat[:, :, 128:, 128:].reshape(-1), x_wt[:, :, 128:, 128:].reshape(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= x.shape[0] * 3 * 128 * 128

        return BCE + BCE_wt + KLD, BCE + BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

class IWTVAE_512_Mask(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_512_Mask, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 128, 128]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        ### Maxpool??? Test this out
        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 128, 64, 64]

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 512, 8, 8]
        
        self.fc_enc = nn.Linear(512 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 512 * 8 * 8)
        weights_init(self.fc_dec)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 512, 16, 16]

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 32, 32]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.d2= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 64, 64]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)
    
        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 128, 128, 128]

        self.d3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=32, affine=False)

        # Maybe try 3 channel mask, instead of 1
        self.d4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True) #[b, 3, 512, 512]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=3, affine=False)
        
        self.iwt = None
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 128, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 16, 16]

        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)
        h = self.leakyrelu(self.fc_enc(h.reshape(-1,512*8*8)))                      #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h)), m1_idx, m2_idx          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z, m1_idx, m2_idx):
        h = self.leakyrelu(self.fc_dec(z))                                      #[b, 512*8*8]
        h = self.leakyrelu(self.u1(h.reshape(-1, 512, 8, 8), indices=m2_idx))   #[b, 512, 16, 16]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h)))                   #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                   #[b, 128, 64, 64]
        h = self.leakyrelu(self.u2(h, indices=m1_idx))                          #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                   #[b, 32, 256, 256]
        # Make linear? or normalize and then use sigmoid
        h = self.instance_norm_d4(self.d4(h))                                   #[b, 3, 512, 512]
        
        # Returns mask
        return h
        
    def forward(self, x, y_full, y):
        mu, var, m1_idx, m2_idx = self.encode(y_full - y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask = self.decode(y, z, m1_idx, m2_idx)
        
        return mask, mu, var
        
    def loss_function(self, x, x_hat, x_wt, x_wt_hat, mu, var, img_loss=False, kl_weight=1.0) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = 0
        # BCE instead of mse
        if img_loss:
            BCE = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level other than 1st patch
        BCE_wt = F.l1_loss(x_wt_hat[:, :, 128:, :].reshape(-1), x_wt[:, :, 128:, :].reshape(-1), reduction='sum') + F.l1_loss(x_wt_hat[:, :, :128, 128:].reshape(-1), x_wt[:, :, :128, 128:].reshape(-1), reduction='sum')
        BCE_wt /= x_wt_hat.numel()
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x.shape[0] * 3 * 512 * 512

        return BCE + BCE_wt + KLD, BCE + BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)


class IWTVAE_512_Mask_1(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_512_Mask_1, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.BatchNorm2d(num_features=64)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 128, 128]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.BatchNorm2d(num_features=128)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 64, 64]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.BatchNorm2d(num_features=256)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 32, 32]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.BatchNorm2d(num_features=512)

        self.e5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e5)
        self.instance_norm_e5 = nn.BatchNorm2d(num_features=1024)

        self.e6 = nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 2048, 8, 8]
        weights_init(self.e6)
        self.instance_norm_e6 = nn.BatchNorm2d(num_features=2048)
        
        self.fc_enc = nn.Linear(2048 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 8 * 8)
        weights_init(self.fc_dec)

        self.d1 = nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True) #[b, 1024, 16, 16]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.BatchNorm2d(num_features=1024)

        self.d2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True) #[b, 512, 32, 32]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.BatchNorm2d(num_features=512)

        self.d3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 64, 64]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.BatchNorm2d(num_features=256)

        self.d4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 128, 128]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.BatchNorm2d(num_features=128)

        self.d5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d5)
        self.instance_norm_d5 = nn.BatchNorm2d(num_features=64)

        self.d6 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True) #[b, 3, 512, 512]
        weights_init(self.d6)
        self.instance_norm_d6 = nn.BatchNorm2d(num_features=3)
        
        self.iwt = None
    
      
    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 64, 64]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 32, 32]
        h = self.leakyrelu(self.instance_norm_e5(self.e5(h)))                       #[b, 1024, 16, 16]
        h = self.leakyrelu(self.instance_norm_e6(self.e6(h)))                       #[b, 1024, 8, 8]

        h = self.leakyrelu(self.fc_enc(h.reshape(-1,2048*8*8)))                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        h = self.leakyrelu(self.fc_dec(z))                                              #[b, 1024*8*8]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h.reshape(-1, 2048, 8, 8))))   #[b, 1024, 16, 16]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                           #[b, 512, 32, 32]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                           #[b, 256, 64, 64]
        h = self.leakyrelu(self.instance_norm_d4(self.d4(h)))                           #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d5(self.d5(h)))                           #[b, 64, 256, 256]
        h = self.instance_norm_d6(self.d6(h))                                           #[b, 3, 512, 512]
        
        # Returns mask
        return h
        
    def forward(self, x, y_full, y):
        mu, var = self.encode(y_full - y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask = self.decode(y, z)
        
        return mask, mu, var
        
    def loss_function(self, x, x_hat, x_wt, x_wt_hat, mu, var, img_loss=False, kl_weight=1.0) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = 0
        # BCE instead of mse
        if img_loss:
            BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level other than 1st patch
        BCE_wt = F.l1_loss(x_wt_hat.reshape(-1), x_wt.reshape(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kl_weight
        KLD /= x.shape[0] * 3 * 512 * 512

        return BCE + BCE_wt + KLD, BCE + BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# Reconstructing IWT'ed mask
class IWTVAE_512_Mask_2(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_512_Mask_2, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 256, 256]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 128, 128]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 64, 64]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 32, 32]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 16, 16]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 8, 8]
            # nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 8 * 8)
        weights_init(self.fc_dec)

        # Z Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 16, 16]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 32, 32]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 64, 64]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 128, 128]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 256, 256]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 512, 512]
        )
        
        # Initializing weights of decoder
        weights_init(self.decoder)
        
        self.iwt = None
    
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 8, 8]
        h = self.fc_enc(h.reshape(-1,2048*8*8))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, z):
        h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*8*8]
        h = self.decoder(h.reshape(-1, 2048, 8, 8))              #[b, 3, 512, 512]
        
        # Returns mask
        return h
        
    def forward(self, y):
        mu, var = self.encode(y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask = self.decode(z)
        
        return mask, mu, var
        
    def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        # BCE = 0
        # BCE instead of mse
        # if img_loss:
        #     BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level (x_wt already has first patch all 0's)
        BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= mask.shape[0]

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# Conditional masking network that produces high frequency components for upscaling to 256
class IWTAE_256_Mask_Conditional(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, num_iwt=2):
        super(IWTAE_128_Mask_2, self).__init__()
        # Resolution of images (256 x 256)
        self.res = 256
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 128, 128]
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 64, 64]
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 32, 32]
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 16, 16]
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 1024, 8, 8]
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 4, 4]
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 4 * 4, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 4 * 4)
        weights_init(self.fc_dec)

        # Z Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 8, 8]
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 16, 16]
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 32, 32]
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 64, 64]
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 128, 128]
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 256, 256]
        )
        
        # Initializing weights of decoder
        weights_init(self.decoder)
        
        self.iwt = None
    
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 4, 4]
        h = self.fc_enc(h.reshape(-1,2048*4*4))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]

    
    def decode(self, z):
        h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*4*4]
        h = self.decoder(h.reshape(-1, 2048, 4, 4))              #[b, 3, 256, 256]
        
        # Returns mask
        return h
        
    def forward(self, y):
        mu, var = self.encode(y)
        mask = self.decode(mu)
        
        return mask, mu, var
        
    def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        # WT-space loss on patch level (x_wt already has first patch all 0's)
        BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        KLD = torch.tensor(0)

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# Reconstructing IWT'ed mask
class IWTAE_128_Mask_2(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTAE_128_Mask_2, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 2, 2]
            # nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 2 * 2, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 2 * 2)
        weights_init(self.fc_dec)

        # Z Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 4, 4]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 8, 8]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 16, 16]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 32, 32]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 64, 64]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 64, 64]
        )
        
        # Initializing weights of decoder
        weights_init(self.decoder)
        
        self.iwt = None
    
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 2, 2]
        h = self.fc_enc(h.reshape(-1,2048*2*2))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]

    
    def decode(self, z):
        h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*2*2]
        h = self.decoder(h.reshape(-1, 2048, 2, 2))              #[b, 3, 128, 128]
        
        # Returns mask
        return h
        
    def forward(self, y):
        mu, var = self.encode(y)
        mask = self.decode(mu)
        
        return mask, mu, var
        
    def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        # BCE = 0
        # BCE instead of mse
        # if img_loss:
        #     BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level (x_wt already has first patch all 0's)
        BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        logvar = torch.log(var)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD /= mask.shape[0]
        KLD = torch.tensor(0)

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# Reconstructing IWT'ed mask
class IWTAE_512_Mask_2(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTAE_512_Mask_2, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 256, 256]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 128, 128]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 64, 64]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 32, 32]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 16, 16]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 8, 8]
            # nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 8 * 8)
        weights_init(self.fc_dec)

        # Z Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 16, 16]
            # nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 32, 32]
            # nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 64, 64]
            # nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 128, 128]
            # nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 256, 256]
            # nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 512, 512]
        )
        
        # Initializing weights of decoder
        weights_init(self.decoder)
        
        self.iwt = None
    
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 8, 8]
        h = self.fc_enc(h.reshape(-1,2048*8*8))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]

    
    def decode(self, z):
        h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*8*8]
        h = self.decoder(h.reshape(-1, 2048, 8, 8))              #[b, 3, 512, 512]
        
        # Returns mask
        return h
        
    def forward(self, y):
        mu, var = self.encode(y)
        mask = self.decode(mu)
        
        return mask, mu, var
        
    def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        # BCE = 0
        # BCE instead of mse
        # if img_loss:
        #     BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level (x_wt already has first patch all 0's)
        BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        logvar = torch.log(var)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD /= mask.shape[0]
        KLD = torch.tensor(0)

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

# class IWTAE_512_Mask_3(nn.Module):
#     def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
#         super(IWTAE_512_Mask_3, self).__init__()
#         # Resolution of images (512 x 512)
#         self.res = 512
#         self.device = None
#         self.cuda = False
        
#         self.z_dim = z_dim
#         self.num_iwt = num_iwt
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.sigmoid = nn.Sigmoid()

#         # Z Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 256, 256]
#             # nn.InstanceNorm2d(64),
#             self.leakyrelu,
#             nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 128, 128]
#             # nn.InstanceNorm2d(128),
#             self.leakyrelu,
#             nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 64, 64]
#             # nn.InstanceNorm2d(256),
#             self.leakyrelu,
#             nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 32, 32]
#             # nn.InstanceNorm2d(512),
#             self.leakyrelu,
#             nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 16, 16]
#             # nn.InstanceNorm2d(1024),
#             self.leakyrelu,
#             nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 8, 8]
#             # nn.InstanceNorm2d(2048),
#             self.leakyrelu
#         )

#         # Initializing weights of encoder                                      
#         weights_init(self.encoder)
        
#         self.fc_enc = nn.Linear(2048 * 8 * 8, 1024)
#         weights_init(self.fc_enc)
        
#         self.fc_mean = nn.Linear(1024, z_dim)
#         weights_init(self.fc_mean)
        
#         self.fc_var = nn.Linear(1024, z_dim)
#         weights_init(self.fc_var)
        
#         self.fc_dec = nn.Linear(z_dim, 2048 * 8 * 8)
#         weights_init(self.fc_dec)

#         # Z Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 16, 16]
#             # nn.InstanceNorm2d(1024),
#             self.leakyrelu,
#             nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 32, 32]
#             # nn.InstanceNorm2d(512),
#             self.leakyrelu,
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 64, 64]
#             # nn.InstanceNorm2d(256),
#             self.leakyrelu,
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 128, 128]
#             # nn.InstanceNorm2d(128),
#             self.leakyrelu,
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 256, 256]
#             # nn.InstanceNorm2d(64),
#             self.leakyrelu,
#             nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 512, 512]
#         )
        
#         # Initializing weights of decoder
#         weights_init(self.decoder)
        
#         self.iwt = None
    
      
#     def encode(self, y):
#         h = self.encoder(y)                                                         #[b, 2048, 8, 8]
#         h = self.fc_enc(h.reshape(-1,2048*8*8))                                     #[b, z_dim]

#         return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]

    
#     def decode(self, z):
#         h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*8*8]
#         h = self.decoder(h.reshape(-1, 2048, 8, 8))              #[b, 3, 512, 512]
        
#         # Returns mask
#         return h
        
#     def forward(self, y):
#         mu, var = self.encode(y)
#         mask = self.decode(mu)
        
#         return mask, mu, var
        
#     def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        
#         # Loss btw reconstructed img and original img
#         # BCE = 0
#         # BCE instead of mse
#         # if img_loss:
#         #     BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

#         # WT-space loss on patch level (x_wt already has first patch all 0's)
#         BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
#         logvar = torch.log(var)
#         # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         # KLD /= mask.shape[0]
#         KLD = torch.tensor(0)

#         return BCE_wt + KLD, BCE_wt, KLD

#     def set_device(self, device):
#         self.device = device
#         if 'cuda' in self.device:
#             self.cuda = True
    
#     def set_filters(self, filters):
#         self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
#         self.iwt.set_filters(filters)

class IWTVAE_512_Mask_3(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_512_Mask_3, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.device = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Z Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 64, 256, 256]
            nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 128, 128, 128]
            nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 256, 64, 64]
            nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 32, 32]
            nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 512, 16, 16]
            nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=True, padding_mode='zeros'), #[b, 2048, 8, 8]
            nn.InstanceNorm2d(2048),
            self.leakyrelu
        )

        # Initializing weights of encoder                                      
        weights_init(self.encoder)
        
        self.fc_enc = nn.Linear(2048 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 2048 * 8 * 8)
        weights_init(self.fc_dec)

        # Z Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=True), #[b, 1024, 16, 16]
            nn.InstanceNorm2d(1024),
            self.leakyrelu,
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=True), #[b, 512, 32, 32]
            nn.InstanceNorm2d(512),
            self.leakyrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), #[b, 256, 64, 64]
            nn.InstanceNorm2d(256),
            self.leakyrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True), #[b, 128, 128, 128]
            nn.InstanceNorm2d(128),
            self.leakyrelu,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), #[b, 32, 256, 256]
            nn.InstanceNorm2d(64),
            self.leakyrelu,
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True), #[b, 3, 512, 512]
        )
        
        # Initializing weights of decoder
        weights_init(self.decoder)
        
        self.iwt = None
    
      
    def encode(self, y):
        h = self.encoder(y)                                                         #[b, 2048, 8, 8]
        h = self.fc_enc(h.reshape(-1,2048*8*8))                                     #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h))                          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, z):
        h = self.leakyrelu(self.fc_dec(z))                       #[b, 2048*8*8]
        h = self.decoder(h.reshape(-1, 2048, 8, 8))              #[b, 3, 512, 512]
        
        # Returns mask
        return h
        
    def forward(self, y):
        mu, var = self.encode(y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        mask = self.decode(z)
        
        return mask, mu, var
        
    def loss_function(self, mask, mask_recon, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        # BCE = 0
        # BCE instead of mse
        # if img_loss:
        #     BCE = F.binary_cross_entropy(x_hat.reshape(-1), x.reshape(-1))

        # WT-space loss on patch level (x_wt already has first patch all 0's)
        BCE_wt = F.mse_loss(mask_recon.reshape(-1), mask.reshape(-1)) * mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= mask.shape[0]

        return BCE_wt + KLD, BCE_wt, KLD

    def set_device(self, device):
        self.device = device
        if 'cuda' in self.device:
            self.cuda = True
    
    def set_filters(self, filters):
        self.iwt = IWT(iwt=iwt, num_iwt=self.num_iwt)
        self.iwt.set_filters(filters)

class Full_WTVAE128_IWTAE512(nn.Module):
    def __init__(self, wt_model, iwt_model, devices):
        super(Full_WTVAE128_IWTAE512, self).__init__()
        self.devices = devices

        self.wt_model = wt_model
        self.iwt_model = iwt_model

    def encode_wt(self, X_128):
        z, mu, logvar = self.wt_model.encode(X_128)

        return z, mu, logvar

    def decode_wt(self, z):
        z = self.wt_model.decode(z)

        return z
    
    def forward_iwt(self, Y_low):
        Y_low_padded = zero_pad(Y_low, target_dim=512, device=self.iwt_model.device)
        Y_low_iwt = self.iwt_model.iwt(Y_low_padded)

        mask, _, _ = self.iwt_model(Y_low_iwt)
        mask_wt = self.wt_model.wt(mask.to(self.wt_model.device))

        # Add the reconstructed first patch with mask and apply IWT to get image
        X_wt = Y_low_padded + mask_wt.to(self.iwt_model.device)
        X = self.iwt_model.iwt(X_wt)
        
        return mask, X
    
    def sample(self, batch_size):
        Y_low_sample = self.wt_model.sample(batch_size)
        mask_sample, X_sample = self.forward_iwt(Y_low_sample.to(self.iwt_model.device))

        return Y_low_sample, mask_sample, X_sample

    def forward(self, X_128):
        # Create first patch (low frequency), pad with zeros to dim 512 x 512, and run through IWT
        Y_low, mu, logvar = self.wt_model(X_128.to(self.wt_model.device))
        mask, X = self.forward_iwt(Y_low.to(self.iwt_model.device))

        return Y_low, mask, X, mu, logvar

    def loss_function(self, X_512, Y_low_hat, X_hat, mu, logvar, kl_weight=1.0) -> Variable:
        loss_wt, loss_wt_bce, loss_wt_kld = self.wt_model.loss_function(X_512.to(self.wt_model.device), Y_low_hat, mu, logvar, kl_weight)
        loss_img = F.mse_loss(X_hat, X_512.to(self.iwt_model.device))

        total_loss = loss_wt + loss_img.to(self.wt_model.device)
        total_loss_bce = loss_wt_bce + loss_img.to(self.wt_model.device)
        total_loss_kld = loss_wt_kld

        return total_loss, total_loss_bce, total_loss_kld


class FullVAE_512(nn.Module):
    def __init__(self, wt_model, iwt_model, devices):
        super(FullVAE_512, self).__init__()
        self.devices = devices

        # Setting up filters for loss function of WTVAE model
        w = pywt.Wavelet('bior2.2')
        dec_hi = torch.Tensor(w.dec_hi[::-1]).to(devices[0]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1]).to(devices[0])
        rec_hi = torch.Tensor(w.rec_hi).to(devices[0])
        rec_lo = torch.Tensor(w.rec_lo).to(devices[0])
        filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

        self.wt_model = wt_model.to(devices[0])
        self.wt_model.set_device(devices[0])
        self.iwt_model = iwt_model.to(devices[1])
        self.iwt_model.set_device(devices[1])

    def forward(self, x):
        # Produces 128 x 128 reconstructed 1st patch 
        y, y_padded, mu_wt, logvar_wt = self.wt(x)
        x_hat, mu, var = self.iwt(x, y_padded)

        return y, mu_wt, logvar_wt, x_hat, mu, var
    
    def wt(self, x):
        y, mu_wt, logvar_wt = self.wt_model(x.to(self.devices[0]))                   #[b, 3, 128, 128]
        y_padded = zero_pad(y, target_dim=512, device=self.devices[1])            #[b, 3, 512, 512]

        return y, y_padded, mu_wt, logvar_wt

    def iwt(self, x, y_padded):
        x_hat, mu, var = self.iwt_model(x.to(self.devices[1]), y_padded)             #[b, 3, 512, 512]

        return x_hat, mu, var

        
    def loss_function(self, x, y, mu_wt, logvar_wt, x_hat, mu, var):
        # Returns a loss tuple of (KLD+BCE, BCE, KLD)
        wt_loss = self.wt_model.loss_function(x.to(self.devices[0]), y, mu_wt, logvar_wt)
        iwt_loss = self.iwt_model.loss_function(x.to(self.devices[1]), x_hat, mu, var)

        # Summing each elements (KLD+BCE, BCE, KLD)
        return wt_loss[0].to(self.devices[1]) + iwt_loss[0], wt_loss[1].to(self.devices[1]) + iwt_loss[1], wt_loss[2].to(self.devices[1]) + iwt_loss[2]
