import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from .utils import fspecial_gauss

# Algorithm 2
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_2(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# Algorithm 1
def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out

def ssim_1(X, Y, win, get_ssim_map=False, get_cs=False, get_weight=False):
    C1 = 0.01**2
    C2 = 0.03**2

    win = win.to(X.device)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12   = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) 
    cs_map = F.relu(cs_map) #force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_val = ssim_map.mean([1,2,3])

    if get_weight:
        weights = torch.log((1+sigma1_sq/C2)*(1+sigma2_sq/C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1,2,3])
        
    return ssim_val

class SSIM(torch.nn.Module):
    def __init__(self, channels=3, window_size=11, algorithm=1):
        super(SSIM, self).__init__()
        self.algorithm = algorithm
        self.window_size = window_size
        self.channels = channels
        if algorithm == 1:
            self.win = fspecial_gauss(window_size, 1.5, channels)
        elif algorithm == 2:
            self.win = create_window(window_size, channels)
        else:
            print(f"WARNING: Only 2 algorithms available, defaulting to algorithm 2")
            self.win = create_window(window_size, channels)

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if self.algorithm == 1:            
            if as_loss:
                score = ssim_1(X, Y, win=self.win)
                return 1 - score.mean()
            else:
                with torch.no_grad():
                    score = ssim_1(X, Y, win=self.win)
                return score
        else:
            if as_loss:
                (_, channels, _, _) = X.size()

                if channels == self.channels and self.window.data.type() == X.data.type():
                    window = self.window
                else:
                    window = create_window(self.window_size, channels)
                    
                    if X.is_cuda:
                        window = window.cuda(X.get_device())
                    window = window.type_as(X)
                    
                    self.window = window
                    self.channels = channels

                return  1 - ssim_2(X, Y, window, self.window_size, channels, self.size_average)
            else:
                with torch.no_grad():
                    (_, channels, _, _) = X.size()
                    window =  self.window(window_size, channels)

                    if X.is_cuda:
                        window = window.cuda(X.get_device())
                    window = window.type_as(X)

                    score = ssim_2(X, Y, window, window_size, channels, size_average)
                return score

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/r0.png')
    parser.add_argument('--dist', type=str, default='images/r1.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)
    
    model = SSIM(channels=3)

    score = model(dist, ref, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.6717

    # model = SSIM(channels=1)
    # score = 0
    # for i in range(3):
    #     ref1 = ref[:,i,:,:].unsqueeze(1)
    #     dist1= dist[:,i,:,:].unsqueeze(1)
    #     score = score + model(ref1, dist1).item()
    # print('score: %.4f' % score)
