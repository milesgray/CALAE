import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn

### ALAE Noise
def sample_noise(bs, code=512, device='cpu'):
    return torch.randn(bs, code).to(device)

### FAMOS Noise - https://github.com/zalandoresearch/famos/blob/master/utils.py#L232
class Waver(nn.Module):
    def __init__(self, zPeriodic, zGL):
        """
        zPeriodic - int
            'periodic spatial waves'
        zGL - int 
            'noise channels, identical on every spatial position'
        """
        super().__init__()
        self.zGL = zGL

        if self.zGL > 0:
            K = 50
            layers = [nn.Conv2d(zGL, K, 1)]
            layers += [nn.ReLU(True)]
            layers += [nn.Conv2d(K, 2 * zPeriodic, 1)]
            self.learnedWN = nn.Sequential(*layers)
        else:##static
            self.learnedWN = nn.Parameter(torch.zeros(zPeriodic * 2).uniform_(-1, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * 0.2)

    def forward(self, c,GLZ=None):
        if self.zGL > 0:
            return (waveNumbers + 5 * self.learnedWN(GLZ)) * c

        return (waveNumbers + self.learnedWN) * c

class FAMOS_Noise:
    def __init__(self, z_periodic=2, z_gl=20):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.z_periodic = z_periodic
        self.z_gl = z_gl
        # 2*nPeriodic initial spread values
        # slowest wave 0.5 pi-- full cycle after 4 steps in noise tensor
        # fastest wave 1.5pi step -- full cycle in 0.66 steps
        self.waveNumbers = self.initWave().to(self.device)
        self.learnedWN = Waver(self.z_periodic, self.z_gl)

    def initWave(self):
        buf = []
        for i in range(self.z_periodic // 4+1):
            v = 0.5 + i / float(self.z_periodic//4+1e-10)
            buf += [0, v, v, 0]
            buf += [0, -v, v, 0]  # #so from other quadrants as well..
        buf=buf[:2*self.z_periodic]
        awave = np.array(buf, dtype=np.float32) * np.pi
        awave = torch.FloatTensor(awave).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return awave

    def set_noise(self, noise, use_gl=True, use_periodic=True):
        noise = noise.detach() # *1 .0
        noise.uniform_(-1, 1)  # normal_(0, 1)
        if use_gl:
            noise[:, :self.z_gl] = noise[:, :self.z_gl, :1, :1].repeat(1, 1, noise.shape[2], noise.shape[3])
        if use_periodic:
            xv, yv = np.meshgrid(np.arange(noise.shape[2]), np.arange(noise.shape[3]), indexing='ij')
            c = torch.FloatTensor(np.concatenate([xv[np.newaxis], yv[np.newaxis]], 0)[np.newaxis])
            c = c.repeat(noise.shape[0], self.z_periodic, 1, 1)
            c = c.to(device)
            # #now c has canonic coordinate system -- multiply by wave numbers
            raw = self.learnedWN(c, noise[:, :self.z_gl])
            #random offset
            offset = (noise[:, -self.z_periodic:, :1, :1] * 1.0).uniform_(-1, 1) * 6.28
            offset = offset.repeat(1, 1, noise.shape[2], noise.shape[3])
            wave = torch.sin(raw[:, ::2] + raw[:, 1::2] + offset)
            noise[:, -self.z_periodic:] = wave
        return noise
