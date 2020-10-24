import torch
import torch.nn as nn

from .spectralnorm import SNConv2d

class SAGAN_Attention(nn.Module):    
    def __init__(self, ch, which_conv=SNConv2d):
        """A non-local block as used in SA-GAN.

        Args:
            ch (int): Number of channels/filters
            which_conv (nn.Module, optional): A 2D Convolutional module to use. Defaults to SNConv2d.
        """
        super().__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

# ATT_U-Net's Attention module
class Attention(nn.Module):
    # 128*128*512
    # F_g, F_l are equal in size and larger than the output, F_int channel is half of them (512, 512, 256)
    def  __init__(self, F_g, F_l, F_int):  # Channel F_g: Large size input F_l: Front-level input F_int: half of their channel
        super().__init__()
        self.W_g = nn.Sequential(          # 1*1 convolution BN with a step size of 1
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )                                  # Output: Hg*Wg*F_int
        
        self.W_x = nn.Sequential(          # 1*1 convolution BN with a step size of 1
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )                                  # Output: Hg*Wg*F_int

        self.psi = nn.Sequential(          # 1*1 convolution BN with a step size of 1
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self,g,x):
        # g,x 128*128*512
        g1 = self.W_g(g)                            # g branch output 128*128*256
        x1 = self.W_x(x)                            # Xl branch output 128*128*256    
        psi = self.relu(g1 + x1)                    # Add 2 channels of information 128*128*256
        psi = self.psi(psi)                         # output       128*128*1 
        return x * psi                             # Multiply the feature map by 128*128*512

# self-attention module (use standard convolution instead of spectral normalization)
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def  __init__(self, in_dim, activation):                                                      # constructor
        super().__init__()
        self.chanel_in = in_dim                                                                   # Number of input channels
        self.activation = activation                                                              # Attribute in the parent class, activation function? ? ?
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # The number of channels output by the Q channel is one-eighth of the original
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)    # The number of channels output by the K channel is one-eighth of the original
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)       # The number of channels output by the V channel remains unchanged
        self.gamma = nn.Parameter(torch.zeros(1))                                                 # att graph weight parameter

        self.softmax = nn.Softmax(dim=-1)  # softmax is formed after att_map
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()    # X: B*C*W*H Get the dimension information of B, C, W, H
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height). permute(0, 2, 1) # B X (C /. 8) X (* W is H) feature the convolution Q Stretched to two dimensions, B*C*(H*W), then transposed B*(H*W)*C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)    # B X (C /. 8) X (* W is H) after the characteristic K for the two-dimensional convolution of elongated, B * C*(H*W)
        energy = torch.bmm(proj_query, proj_key)    # transpose check  B*(H*W)*(H*W)
        attention = self.softmax(energy)            # B x (N) x (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)     # B X C X N V characteristics after the 2-dimensional convolution of elongated, B * C * (H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))    # Multiply V and attention_map (B*C*N)(B*N*N) = B*C*N
        out = out.view(m_batchsize, C, width, height)       # Restore to the original image B*C*H*W
        
        out = self.gamma * out + x  # Calculate the residual weight parameter as self.gamma If the residual is 0, it is the identity mapping
        return out     #attention # Return 1.attention residual result 2.N*N attention map (what's the point)


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        query = x.view(m_batchsize, C, -1)
        key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
