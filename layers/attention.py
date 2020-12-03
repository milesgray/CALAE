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
                                                           self.ch // 2, 
                                                           x.shape[2], 
                                                           x.shape[3]))
        return self.gamma * o + x

# ATT_U-Net's Attention module
class UNetAttention(nn.Module):
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
        return x * psi                              # Multiply the feature map by 128*128*512

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

class Self_Attn(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_channels, spectral_norm):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels

        if spectral_norm:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma*attn_g
        
# CAM module
class ChannelAttention(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

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

############################################
## Squeeze Excite style Channel Attention
## https://github.com/bmycheez/C3Net/blob/master/Burst/models.py#L42

class SqueezeExciteAttention(nn.Module):
    def __init__(self, channel, reduction=16, pool_size=1):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


#####################################
## Triplet Attention

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
