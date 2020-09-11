import torch
import torch.nn as nn


# ATT_U-Net's Attention module
class Attention_block(nn.Module):
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
        return  x * psi                             # Multiply the feature map by 128*128*512

# self-attention module (use standard convolution instead of spectral normalization)
class Self_Attn(nn.Module):
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
    
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()    # X: B*C*W*H Get the dimension information of B, C, W, H
        proj_query  = Self.query_conv(X). View(m_batchsize, -.1, width * height). permute(0, 2, .1) # B X (C /. 8) X (* W is H) feature the convolution Q Stretched to two dimensions, B*C*(H*W), then transposed B*(H*W)*C
        proj_key =  Self.key_conv(X). View(m_batchsize, -.1, width * height)    # B X (C /. 8) X (* W is H) after the characteristic K for the two-dimensional convolution of elongated, B * C*(H*W)
        energy =  torch.bmm(proj_query,proj_key)    # transpose check  B*(H*W)*(H*W)
        attention = self.softmax(energy)            # B x (N) x (N) 
        proj_value = Self.value_conv(X). View(m_batchsize, -.1, width * height)     # B X C X N V characteristics after the 2-dimensional convolution of elongated, B * C * (H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))    # Multiply V and attention_map (B*C*N)(B*N*N) = B*C*N
        out = out.view(m_batchsize, C, width, height)       # Restore to the original image B*C*H*W
        
        out = self.gamma * out  +  x  # Calculate the residual weight parameter as self.gamma If the residual is 0, it is the identity mapping
        return  out     #attention # Return 1.attention residual result 2.N*N attention map (what's the point)
