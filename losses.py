from math import log2, exp
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
import torch.nn.functional as F
from torch.autograd import grad
 
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')
from hessian_penalty import hessian_penalty
from metrics.perceptual import PerceptualLoss

def zero_centered_gradient_penalty(real_samples, real_prediction):
    """
    Computes zero-centered gradient penalty for E, D
    """    
    grad_outputs = torch.ones_like(real_prediction, requires_grad=True)    
    squared_grad_wrt_x = grad(outputs=real_prediction, inputs=real_samples, grad_outputs=grad_outputs,\
                              create_graph=True, retain_graph=True)[0].pow(2)
    
    return squared_grad_wrt_x.view(squared_grad_wrt_x.shape[0], -1).sum(dim=1).mean()

def loss_discriminator(E, D, alpha, real_samples, fake_samples, gamma=10, use_bce=False,
                       enable_hessian_real=False, enable_hessian_fake=False, 
                       hessian_layers_fake=[-2], hessian_layers_real=[-2]):
    E_r = E(real_samples, alpha)
    E_f = E(fake_samples, alpha)
    real_prediction, fake_prediction = D(E_r), D(E_f)

    if use_bce:
        loss = adv_loss(real_prediction, 1)
        loss += adv_loss(fake_prediction, 0)
    else:
        # Minimize negative = Maximize positive (Minimize incorrect D predictions for real data,
        #                                        minimize incorrect D predictions for fake data)
        loss = (F.softplus(-real_prediction) + F.softplus(fake_prediction)).mean()

    if gamma > 0:
        loss += zero_centered_gradient_penalty(real_samples, real_prediction).mul(gamma/2)

    return loss

def loss_discriminator_patch(D, real_samples, fake_samples, gamma=10, use_bce=False,
                       enable_hessian_real=False, enable_hessian_fake=False, 
                       hessian_layers_fake=[-2], hessian_layers_real=[-2]):
    real_prediction = D(real_samples)
    fake_prediction = D(fake_samples)

    if use_bce:
        loss = adv_loss(real_prediction, 1)
        loss += adv_loss(fake_prediction, 0)
    else:
        # Minimize negative = Maximize positive (Minimize incorrect D predictions for real data,
        #                                        minimize incorrect D predictions for fake data)
        loss = (F.softplus(-real_prediction) + F.softplus(fake_prediction)).mean()

    if gamma > 0:
        loss += zero_centered_gradient_penalty(real_samples, real_prediction).mul(gamma/2)
    return loss

def loss_generator(E, D, alpha, fake_samples, enable_hessian=True, hessian_layers=[-1,-2], current_layer=[-1], hessian_weight=0.01):
    # Hessian applied to E here
    # Minimize negative = Maximize positive (Minimize correct D predictions for fake data)
    E_z = E(fake_samples, alpha)
    loss = softplus(-D(E_z)).mean()
    if enable_hessian:
        for layer in hessian_layers:
            h_loss = hessian_penalty(E, z=fake_samples, alpha=alpha, return_norm=layer) * hessian_weight
            if layer in current_layer:
                h_loss = h_loss * alpha
            loss += h_loss
    return loss

def loss_avg_generator(G, G_avg, F_z, scale, alpha, loss_fn):
    # Hessian applied to G here
    G_z = G(F_z, scale, alpha)
    G_avg_z = G_avg(F_z, scale, alpha)

    loss = loss_fn(G_z, G_avg_z)

    return loss

def loss_generator_consistency(fake, real, loss_fn=None, use_perceptual=False,
                               use_ssim=True, ssim_weight=1, use_ssim_tv=False,
                               use_sobel=True, sobel_weight=1,
                               use_sobel_tv=False, sobel_fn=None):
    if loss_fn:
        if use_perceptual:
            scale = fake.shape[2]
            p_func = perceptual_loss[scale if scale < 32 else 32]
            loss = loss_fn(p_func(fake), p_func(real))
        else:
            loss = loss_fn(fake, real)
    else:
        loss = 0       
    if use_ssim:
        s_loss = ssim_loss(fake, real)
        if use_ssim_tv:
            s_loss = s_loss / total_variation(fake)
        loss *= s_loss * ssim_weight
    if use_sobel:
        sobel_real = sobel(real)
        sobel_fake = sobel(fake)
        if use_sobel_tv:
            sobel_real = sobel_real / total_variation(fake) 
            sobel_fake = sobel_fake / total_variation(fake)

        if sobel_fn:
            sobel_loss = sobel_fn(sobel_real, sobel_fake)
        else:
            sim, cs = ssim(sobel_real, sobel_fake, window_size=11, size_average=True, full=True, val_range=2)
            sim = (1 - sim) / 2
            cs = (1 - cs) / 2

            sobel_loss = (sim + cs) ** cs
        loss += sobel_loss * sobel_weight

    return loss

def loss_autoencoder(F, G, E, scale, alpha, z, loss_fn, labels=None, use_tv=False, tv_weight=0.001):
    # Hessian applied to G here
    F_z = F(z, scale, z2=None, p_mix=0)
    
    # Autoencoding loss in latent space
    G_z = G(F_z, scale, alpha)
    E_z = E(G_z, alpha)
    
    #E_z = E_z.reshape(E_z.shape[0], 1, E_z.shape[1]).repeat(1, F_z.shape[1], 1)
    F_x = F_z[:,0,:]
    if labels is not None:
        perm = torch.randperm(E_z.shape[0], device=E_z.device)
        E_z_hat = torch.index_select(E_z, 0, perm)
        F_x_hat = torch.index_select(F_x, 0, perm)
        F_hat = torch.cat([F_x, F_x_hat], 0)
        E_hat = torch.cat([E_z, E_z_hat], 0)
        loss = loss_fn(F_hat, E_hat, labels)
    else:
        loss = loss_fn(F_x, E_z)

    if use_tv:
        loss += total_variation(G_z) * tv_weight
    return loss 

###########################################
#### H E S S I A N ########################
#------------------ GENERATOR
    
def loss_generator_hessian(G, F, z, scale, alpha, 
                           hessian_layers=[3], 
                           current_layer=[0]):
    loss = hessian_penalty(G, z=F(z, scale, z2=None, p_mix=0), scale=scale, alpha=alpha, return_norm=hessian_layers)
    if current_layer in hessian_layers:
        loss = h_loss * alpha    
    return loss
#------------------ ENCODER
def loss_encoder_hessian(E, samples, scale, alpha, 
                         real_samples, fake_samples, 
                         hessian_layers_fake=[-2], 
                         hessian_layers_real=[-2], 
                         current_layer=[-1]):
    loss = 0
    loss += hessian_penalty(E, z=real_samples, G_z=E_r, alpha=alpha, return_norm=hessian_layers_real)
    loss += hessian_penalty(E, z=fake_samples, G_z=E_f, alpha=alpha, return_norm=hessian_layers_real)
    return loss

def loss_encoder_hessian(E, samples, scale, alpha, 
                         hessian_layers=[-1,-2], current_layer=[-1]):
    loss = hessian_penalty(E, z=samples, alpha=alpha, return_norm=hessian_layers)
    if layer in current_layer:
        loss = loss * alpha
    return loss

#############################################
#### S T A N D A R D ########################
#####################------------------------

def msle(x, y):
    return (torch.log(x) - torch.log(y)).pow(2).mean()

def mse(x, y):
    return (x - y).pow(2).mean()

def mae(x, y):
    return torch.abs(x - y).mean()

def logcosh(x, y):
    diff = x - y
    loss = (diff + 1e-12).cosh().log()
    return loss.mean()

def xtanh(x, y):
    diff = x - y
    loss = diff.tanh() * diff
    return loss.mean()

def xsigmoid(x, y):
    diff = x - y
    loss = 1 + (-diff).exp()
    loss = loss - diff
    loss = 2 * diff / loss
    return loss.mean()
    #return torch.mean(2 * diff / (1 + torch.exp(-diff)) - diff)

def correlation(x, y):
    delta = torch.abs(x - y)
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

# Simple BCE Discriminator target
def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

#############################################
#### P E R C E P T U A L ####################
#########################--------------------    
## Perceptual Loss
percep_layer_lookup = {
    4: 4,
    8: 9,
    16: 16,
    32: 23
}
perceptual_loss = {
    4: PerceptualLoss(ilayer=percep_layer_lookup[4]),
    8: PerceptualLoss(ilayer=percep_layer_lookup[8]),
    16: PerceptualLoss(ilayer=percep_layer_lookup[16]),
    32: PerceptualLoss(ilayer=percep_layer_lookup[32]),
}
def percep_loss(x, y, scale):
    loss = perceptual_loss[scale if scale < 32 else 32](x) - perceptual_loss[scale if scale < 32 else 32](y)
    loss = loss.pow(2)
    loss = loss.mean()
    return loss

##########################################################################################
### FAMOS losses - https://github.com/zalandoresearch/famos/blob/master/utils.py
##some image level content loss
def contentLoss(a, b, netR, loss_type):
    def nr(x):
        return (x**2).mean()
        return x.abs().mean()

    if loss_type==0:
        a = avgG(a)
        b = avgG(b)
        return nr(a.mean(1) - b.mean(1))
    if loss_type==1:
        a = avgP(a)
        b = avgP(b)
        return nr(a.mean(1) - b.mean(1))

    if loss_type==10:
        return nr(netR(a)-netR(b))

    if loss_type==100:
        return nr(netR(a)-b)
    if loss_type == 101:
        return nr(avgG(netR(a)) - avgG(b))
    if loss_type == 102:
        return nr(avgP(netR(a)) - avgP(b))
    if loss_type == 103:
        return nr(avgG(netR(a)).mean(1) - avgG(b).mean(1))

    raise Exception("NYI")

def GaussKernel(sigma,wid=None):
    if wid is None:
        wid =2 * 2 * sigma + 1+10

    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))
    def make_kernel(sigma):
        # kernel radius = 2*sigma, but minimum 3x3 matrix
        kernel_size = max(3, int(wid))
        kernel_size = min(kernel_size,150)
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        # make 2D kernel
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=np.float32)
        # normalize kernel by sum of elements
        kernel = np_kernel / np.sum(np_kernel)
        return kernel
    ker = make_kernel(sigma)
  
    a = np.zeros((3,3,ker.shape[0],ker.shape[0])).astype(dtype=np.float32)
    for i in range(3):
        a[i,i] = ker
    return a
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gsigma=1.##how much to blur - larger blurs more ##+"_sig"+str(gsigma)
gwid=61
kernel = torch.FloatTensor(GaussKernel(gsigma,wid=gwid)).to(device)
def avgP(x):
    return nn.functional.avg_pool2d(x,int(16))
def avgG(x):
    pad=nn.functional.pad(x,(gwid//2,gwid//2,gwid//2,gwid//2),'reflect')##last 2 dimensions padded
    return nn.functional.conv2d(pad,kernel)##reflect pad should avoid border artifacts 

#############################################
#### T O T A L V A R I A T I O N ############
#################################------------   

def tv_loss(x, y, loss_fn):
    loss = loss_fn(total_variation(x), total_variation(y))    
    return loss

#absolute difference in X and Y directions
def total_variation(y):
    return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

##2D array of the edges of C channels image
def tvArray(x):
    border1 = x[:, :, :-1] - x[:, :, 1:]
    border1 = torch.cat([border1.abs().sum(1).unsqueeze(1), x[:, :1, :1] * 0], 2)  ##so square with extra 0 line
    border2 = x[:, :, :, :-1] - x[:, :, :, 1:]
    border2 = torch.cat([border2.abs().sum(1).unsqueeze(1), x[:, :1, :, :1] * 0], 3)
    border = torch.cat([border1, border2], 1)
    return border

#############################################
#### G R A M ################################
#############--------------------------------  

def gram_loss(x, y):
    loss = gramMatrix(x, x).exp() - gramMatrix(y, y).exp()
    loss = loss.abs()
    loss = loss.mean()
    return loss

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

##negative gram matrix
def gramMatrix(x,y=None,sq=True,bEnergy=False):
    if y is None:
        y = x

    B, CE, width, height = x.size()
    hw = width * height

    energy = torch.bmm(x.permute(2, 3, 0, 1).view(hw, B, CE),
                       y.permute(2, 3, 1, 0).view(hw, CE, B), )
    energy = energy.permute(1, 2, 0).view(B, B, width, height)
    if bEnergy:
        return energy
    sqX = (x ** 2).sum(1).unsqueeze(0)
    sqY = (y ** 2).sum(1).unsqueeze(1)
    d=-2 * energy + sqX + sqY
    if not sq:
        return d##debugging
    gram = -torch.clamp(d, min=1e-10)#.sqrt()
    return gram

##########################################################################################
#### P E A K - S I G N A L - N O I S E - R A T I O #######################################
###################################################---------------------------------------  
## PSNR
def psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(diff ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

##########################################################################################
#### S S I M #############################################################################
###################################################---------------------------------------
## SSIM 
def ssim_loss(x, y):
    loss = 1 - ssim(x, y)
    loss = loss / 2
    return loss

def ssim_yuv_loss(x, y):
    loss = 1 - ssim(x, y)
    loss = loss / 2
    return loss

def msssim_loss(x, y):
    loss = 1 - msssim(x, y)
    loss = loss / 2
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=2):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=2, normalize=True):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    mssim = []
    mcs = []
    for i in range(5):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        if img1.shape[2] > 1 and i + 1 < 5:
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2)) 
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

##########################################################################################
#### S O B E L ###########################################################################
###################################################---------------------------------------
## Sobel
def ssim_sobel_loss(x, y, window_size=11, size_average=True, val_range=2, normalize=True):
    x_sobel = sobel(x)
    y_sobel = sobel(y)   
    sim, cs = ssim(x_sobel, y_sobel, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    sim = (1 - sim) / 2
    cs = (1 - cs) / 2

    loss = (sim + cs) ** cs
    return loss

def ssim_sobel_loss_broke(x, y, window_size=11, size_average=True, val_range=2, normalize=True):
    x_sobel = sobel(x)
    y_sobel = sobel(y)
    mssim = []
    mcs = []    
    sim, cs = ssim(x, y, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    mssim.append(((sim + 1) / 2))
    mcs.append(((cs + 1) / 2))    
    sim, cs = ssim(x_sobel, y_sobel, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    mssim.append(((sim + 1) / 2))
    mcs.append(((cs + 1) / 2))
    x_sobel_0 = x * x_sobel[:, 0, ...].reshape(x_sobel.shape[0], 1, x_sobel.shape[2], x_sobel.shape[3])
    y_sobel_0 = y * y_sobel[:, 0, ...].reshape(y_sobel.shape[0], 1, y_sobel.shape[2], y_sobel.shape[3])
    sim, cs = ssim(x_sobel_0, y_sobel_0, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    mssim.append(((sim + 1) / 2))
    mcs.append(((cs + 1) / 2))
    x_sobel_1 = x * x_sobel[:, 1, ...].reshape(x_sobel.shape[0], 1, x_sobel.shape[2], x_sobel.shape[3])
    y_sobel_1 = y * y_sobel[:, 1, ...].reshape(y_sobel.shape[0], 1, y_sobel.shape[2], y_sobel.shape[3])
    sim, cs = ssim(x_sobel_1, y_sobel_1, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    mssim.append(((sim + 1) / 2))
    mcs.append(((cs + 1) / 2))
    x_sobel_3 = x - (x_sobel_0 * x_sobel_1)
    y_sobel_3 = y - (y_sobel_0 * y_sobel_1)
    sim, cs = ssim(x_sobel_3, y_sobel_3, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
    mssim.append(((sim + 1) / 2))
    mcs.append(((cs + 1) / 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    loss = torch.prod(mssim ** mcs)
    return loss

def sobel_correlation_loss(x, y):
    x_sobel = sobel(x)
    y_sobel = sobel(y)
    return correlation(x_sobel, y_sobel)

def sobel(img):
    #N,C,_,_ = img.size()
    grad_y, grad_x = sobel_grad(img)
    return torch.cat((grad_y, grad_x), dim=1)
def sobel_grad(img, stride=1, padding=1):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

##########################################################################################
#### O R I G I N A L - A L A E ###########################################################
###################################################---------------------------------------
## Original 

def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x)**2)


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (torch.nn.functional.softplus(d_result_fake) + torch.nn.functional.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


####################################################################################################################
############### M O D U L E S ##################--------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        return losses.logcosh(y_t, y_prime_t)

class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):        
        return losses.xtanh(y_t, y_prime_t)

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        return losses.xsigmoid(y_t, y_prime_t)

class ContentLoss(nn.Module):
    def __init__(self, wid=61, sigma=1, loss_type=0):
        super().__init__()
        self.wid = wid
        self.sigma = sigma
        self.loss_type = loss_type

    def forward(self, a, b, netR, loss_type=None):
        loss_type = loss_type if loss_type else self.loss_type
        def nr(x):
            return (x**2).mean()

        if loss_type==0:
            a = self.avgG(a)
            b = self.avgG(b)
            return nr(a.mean(1) - b.mean(1))
        if loss_type==1:
            a = self.avgP(a)
            b = self.avgP(b)
            return nr(a.mean(1) - b.mean(1))

        if loss_type==10:
            return nr(netR(a)-netR(b))

        if loss_type==100:
            return nr(netR(a)-b)
        if loss_type == 101:
            return nr(self.avgG(netR(a)) - self.avgG(b))
        if loss_type == 102:
            return nr(self.avgP(netR(a)) - self.avgP(b))
        if loss_type == 103:
            return nr(self.avgG(netR(a)).mean(1) - self.avgG(b).mean(1))

        raise Exception("NYI")

    def avgP(self, x):
        return nn.functional.avg_pool2d(x, int(16))
    
    def avgG(self, x):
        kernel = torch.FloatTensor(self.make_gauss_kernel()).to(x.device)
        n = self.wid//2        
        pad = nn.functional.pad(x,(n,n,n,n),'reflect')
        return nn.functional.conv2d(pad,kernel)

    def make_gauss_kernel(self):
        if self.wid is None:
            wid = 2 * 2 * self.sigma + 1+10
        else:
            wid = self.wid

        def gaussian(x, mu, sigma):
            return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

        def make_kernel(sigma):
            # kernel radius = 2*sigma, but minimum 3x3 matrix
            kernel_size = max(3, int(wid))
            kernel_size = min(kernel_size,150)
            mean = np.floor(0.5 * kernel_size)
            kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
            # make 2D kernel
            np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=np.float32)
            # normalize kernel by sum of elements
            kernel = np_kernel / np.sum(np_kernel)
            return kernel
        ker = make_kernel(self.sigma)
    
        a = np.zeros((3, 3, ker.shape[0], ker.shape[0])).astype(dtype=np.float32)
        for i in range(3):
            a[i,i] = ker
        return a

class GramMatrixLoss(nn.Module):
    def __init__(self, sq=True, bEnergy=False):
        super().__init__()
        self.sq = sq
        self.energy = bEnergy

    def forward(self, x, y=None):
        if y is None:
            y = x

        B, CE, width, height = x.size()
        hw = width * height

        energy = torch.bmm(x.permute(2, 3, 0, 1).view(hw, B, CE),
                        y.permute(2, 3, 1, 0).view(hw, CE, B), )
        energy = energy.permute(1, 2, 0).view(B, B, width, height)
        if self.energy:
            return energy
        sqX = (x ** 2).sum(1).unsqueeze(0)
        sqY = (y ** 2).sum(1).unsqueeze(1)
        d = -2 * energy + sqX + sqY
        if not self.sq:
            return d##debugging
        gram = -torch.clamp(d, min=1e-10)#.sqrt()
        return gram


class AutoEncoderLoss(nn.Module):
    def __init__(self, loss_fn, use_tv=False, use_dist=False, enable_hessian=False, hessian_weight=0.01):
        super().__init__()
        self.loss_fn = loss_fn
        self.use_tv = use_tv
        self.use_dist = use_dist
        self.enable_hessian = enable_hessian
        self.hessian_weight = hessian_weight
        self.p_dist = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

    def forward(self, F, G, E, scale, alpha, z, 
                labels=None, hessian_layers=[3], 
                current_layer=[0]):        
        F_z = F(z, scale, z2=None, p_mix=0)
        
        # Autoencoding loss in latent space
        G_z = G(F_z, scale, alpha)
        E_z = E(G_z, alpha)
        
        if labels is not None:
            E_z = E_z.reshape(E_z.shape[0], 1, E_z.shape[1]).repeat(1, F_z.shape[1], 1)
            if self.use_dist:
                x = self.p_dist(F_z, E_z)
                y = torch.eq(labels, labels.T).float().to(x.device)
                loss = self.loss_fn(x, y)
            else:
                perm = torch.randperm(E_z.shape[0], device=E_z.device)
                E_z_hat = torch.index_select(E_z, 0, perm)
                F_z_hat = torch.index_select(F_z, 0, perm)
                F_hat = torch.cat([F_z, F_x_hat], 0)
                E_hat = torch.cat([E_z, E_z_hat], 0)
                loss = self.loss_fn(F_hat, E_hat, labels)
        else:
            F_x = F_z[:,0,:]
            loss = self.loss_fn(F_x, E_z)

        if self.use_tv:
            loss += self.total_variation(G_z)
        
        # Hessian applied to G here
        if self.enable_hessian:
            h_loss = hessian_penalty(G, z=F_z, scale=scale, alpha=alpha, return_norm=hessian_layers) 
            h_loss *= self.hessian_weight
            if current_layer in hessian_layers:
                h_loss = h_loss * alpha
            loss += h_loss
        return loss
    
    @staticmethod
    def total_variation(y):
        return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) \
            + torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

class BaseLoss(torch.nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.loss_stats_ = dict()

    @property
    def loss_stats(self):
        return self.loss_stats_


class CriticLoss(BaseLoss):
    def __init__(self):
        super(CriticLoss, self).__init__()

    def forward(self, critic_outputs_t, critic_outputs_s):
        loss = (critic_outputs_t - critic_outputs_s).mean()
        self.loss_stats_['total'] = loss
        return loss


class CycleLoss(BaseLoss):
    def __init__(self):
        super(CycleLoss, self).__init__()

    def forward(self, z_t, z_t_hat, z_s, z_s_restored):
        z_t_l1_loss = torch.nn.L1Loss()(z_t_hat, z_t)
        z_s_l1_loss = torch.nn.L1Loss()(z_s_restored, z_s)
        loss = z_t_l1_loss + z_s_l1_loss

        self.loss_stats_ = dict()
        self.loss_stats_['z_t_l1_loss'] = z_t_l1_loss
        self.loss_stats_['z_s_l1_loss'] = z_s_l1_loss
        self.loss_stats_['total'] = loss
        return loss


class GeneratorLoss(BaseLoss):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.cycle_criterion = CycleLoss()

    def forward(self, critic_outputs_t, z_t, z_t_hat, z_s, z_s_restored):
        cycle_loss = self.cycle_criterion(z_t, z_t_hat, z_s, z_s_restored)
        loss = -critic_outputs_t.mean() + cycle_loss

        self.loss_stats_ = dict()
        cycle_loss_stats = self.cycle_criterion.loss_stats
        self.loss_stats_.update({
            f'cycle_loss/{key}': cycle_loss_stats[key] for key in cycle_loss_stats.keys()
        })
        self.loss_stats_['total'] = loss
        return loss


class FaceRotationModelLoss(BaseLoss):
    def __init__(self):
        super(FaceRotationModelLoss, self).__init__()
        self.critic_criterion = CriticLoss()
        self.generator_criterion = GeneratorLoss()

    def forward(self, outputs, x):
        critic_outputs_t, critic_outputs_s, z_t, z_t_hat, z_s, z_s_restored = \
            outputs['critic_outputs_t'], outputs['critic_outputs_s'], outputs['z_t'], outputs['z_t_hat'], \
            outputs['z_s'], outputs['z_s_restored']

        critic_loss = self.critic_criterion(critic_outputs_t, critic_outputs_s)
        generator_loss = self.generator_criterion(critic_outputs_t, z_t, z_t_hat, z_s, z_s_restored)
        losses = (critic_loss, generator_loss)

        self.loss_stats_ = dict()
        self.loss_stats_['critic_t'] = critic_outputs_t.mean()
        self.loss_stats_['critic_s'] = critic_outputs_s.mean()
        critic_loss_stats = self.critic_criterion.loss_stats
        self.loss_stats_.update({
            f'critic_loss/{key}': critic_loss_stats[key] for key in critic_loss_stats.keys()
        })
        generator_loss_stats = self.generator_criterion.loss_stats
        self.loss_stats_.update({
            f'generator_loss/{key}': generator_loss_stats[key] for key in generator_loss_stats.keys()
        })
        return losses