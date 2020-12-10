from CALAE.layers import lreq


####################################################################################################################
####### D I S C R I M I N A T O R ########--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Progressive Discriminator Network from ALAE, customized
# ------------------------------------------------------------------------------------------------------------------          
class Discriminator(nn.Module):
    def __init__(self, code=512, depth=3, norm='layer', act='mish', verbose=False):
        super().__init__()

        self.norm = norm
        self.act = act
        
        self.disc = []
        for i in range(depth - 1):
            if verbose: print(f"[Discriminator]\t Block {i} for {code}d code using norm {norm}, act {act} and a residual skip delay of {skip_delay} (only applies if non zero)")   
            self.disc.extend(self.build_layer(code))
        self.disc = self.disc + [lreq.Linear(code, 1)]
        self.disc = nn.Sequential(*self.disc)

    def build_layer(self, code):
        layer = []
        layer.append(lreq.Linear(code, code))

        if self.norm:
            layer.append(Factory.get_normalization(Factory.make_norm_1d(self.norm)))
        if self.act:
            layer.append(Factory.get_activation(self.act))  
            
        return layer
        
    def forward(self, x):
        return self.disc(x)
    
# ------------------------------------------------------------------------------------------------------------------
# Discriminator Block from original ALAE 
# https://github.com/podgorskiy/ALAE/blob/master/net.py#L129
# ------------------------------------------------------------------------------------------------------------------
class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=False, dense=False):
        super().__init__()
        self.conv_1 = lreq.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = BlurSimple(inputs)
        self.last = last
        self.dense_ = dense
        self.fused_scale = fused_scale
        if self.dense_:
            self.dense = lreq.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = lreq.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = lreq.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.dense_:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)

        return x

# ------------------------------------------------------------------------------------------------------------------
# StyleGAN2 Discriminator model from the `one-model-to-reconstruct-them-all` repo
# https://github.com/Bartzi/one-model-to-reconstruct-them-all/blob/main/networks/stylegan2/model.py#L623
# ------------------------------------------------------------------------------------------------------------------
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [stylegan2.ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(stylegan2.ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = stylegan2.ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            stylegan2.EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            stylegan2.EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


# ------------------------------------------------------------------------------------------------------------------
# N-Layer Discriminator Base for Patch-GAN Style
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1285
# ------------------------------------------------------------------------------------------------------------------
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, scale, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        size = scale
        kw = 3
        padw = 1
        stride = 2 if (size % 2) == 0 else 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
            if stride > 1:
                sequence += [Downsample(ndf)]
        size = max(size // stride, 1)
        stride = 2 if (size % 2) == 0 else 1
        nf_mult = 1
        nf_mult_prev = 1
        if n_layers > 1:
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                if(no_antialias):
                    sequence += [
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)]
                    if stride > 1:
                        sequence += [Downsample(ndf * nf_mult)]
                
                size = max(size // stride, 1)
                stride = 2 if (size % 2) == 0 else 1

                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n_layers, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)

# ------------------------------------------------------------------------------------------------------------------
# Pixel Segmentation Discriminator
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1343
# ------------------------------------------------------------------------------------------------------------------
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

# ------------------------------------------------------------------------------------------------------------------
# Patch Discriminator for GAN Contrastive Learning using Patches from same image
# https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py#L1375
# ------------------------------------------------------------------------------------------------------------------
class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, patch_size, scale, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc=input_nc, scale=scale, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, no_antialias=no_antialias)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        size = self.patch_size
        Y = H // size
        X = W // size
        x = x.view(B, C, Y, size, X, size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(x)
