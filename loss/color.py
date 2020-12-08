import torch
from torch import nn

from CALAE.util import clean

class ColorVectLoss(nn.Module):
    def __init__(self, color_axes=[0,1,2]):
        super().__init__()
        self.color_axes = color_axes

    def forward(self, x, y):
        try:
            x.to(y.device)
            diff_vectors = []
            for i in [0,1,2]:
                mean_diff = torch.abs(x[:,i,...].mean() - y[:,i,...].mean())
                std_diff = torch.abs(x[:,i,...].std() - y[:,i,...].std())
                var_diff = torch.abs(x[:,i,...].var() - y[:,i,...].var())
                cos_diff = torch.abs(stand(x[:,i,...], override="tanh").cos() - \
                            stand(y[:,i,...], override="tanh").cos()).std()
                sin_diff = torch.abs(stand(x[:,i,...], override="tanh").sin() - \
                            stand(y[:,i,...], override="tanh").sin()).std()
                
                diff_vectors.append(torch.stack([mean_diff, std_diff, var_diff, cos_diff, sin_diff]))
            color_vect = torch.stack(diff_vectors)
            dist = torch.nn.HingeEmbeddingLoss()(color_vect, -torch.ones_like(color_vect))
            return dist.mean()
        except:
            r = torch.square(x[:,0,...].mean() - y[:,0,...].mean())
            g = torch.square(x[:,1,...].mean() - y[:,1,...].mean())
            b = torch.square(x[:,2,...].mean() - y[:,2,...].mean())
            r_loss = r.exp() / (g.exp() + b.exp()).log()
            g_loss = g.exp() / (r.exp() + b.exp()).log()
            b_loss = b.exp() / (g.exp() + r.exp()).log()
            loss = torch.sqrt(r_loss + g_loss + b_loss)
            return loss

def color_vect_loss(x, y):
    try:
        diff_vectors = []
        for i in [0,1,2]:
            mean_diff = torch.abs(x[:,i,...].mean() - y[:,i,...].mean())
            std_diff = torch.abs(x[:,i,...].std() - y[:,i,...].std())
            var_diff = torch.abs(x[:,i,...].var() - y[:,i,...].var())
            cos_diff = torch.abs(stand(x[:,i,...], override="tanh").cos() - stand(y[:,i,...], override="tanh").cos()).std()
            sin_diff = torch.abs(stand(x[:,i,...], override="tanh").sin() - stand(y[:,i,...], override="tanh").sin()).std()
            
            diff_vectors.append(torch.stack([mean_diff, std_diff, var_diff, cos_diff, sin_diff]))
        color_vect = torch.stack(diff_vectors)
        dist = torch.nn.PairwiseDistance()(color_vect, torch.zeros_like(color_vect))
        return dist.mean()
    except:
        r = torch.square(x[:,0,...].mean() - y[:,0,...].mean())
        g = torch.square(x[:,1,...].mean() - y[:,1,...].mean())
        b = torch.square(x[:,2,...].mean() - y[:,2,...].mean())
        r_loss = r.exp() / (g.exp() + b.exp()).log()
        g_loss = g.exp() / (r.exp() + b.exp()).log()
        b_loss = b.exp() / (g.exp() + r.exp()).log()
        loss = torch.sqrt(r_loss + g_loss + b_loss)
        return loss

class ColorLoss(nn.Module):
    def __init__(self, color_axes=[0,1,2]):
        super().__init__()
        self.color_axes = color_axes

    def forward(self, x, y):
        try:
            x.to(y.device)
            diff_vectors = []
            for i in [0,1,2]:
                mean_diff = torch.abs(x[:,i,...].mean() - y[:,i,...].mean())
                std_diff = torch.abs(x[:,i,...].std() - y[:,i,...].std())
                var_diff = torch.abs(x[:,i,...].var() - y[:,i,...].var())
                cos_diff = torch.abs(stand(x[:,i,...], override="tanh").cos() - \
                            stand(y[:,i,...], override="tanh").cos()).std()
                sin_diff = torch.abs(stand(x[:,i,...], override="tanh").sin() - \
                            stand(y[:,i,...], override="tanh").sin()).std()
                
                diff_vectors.append(torch.stack([mean_diff, std_diff, var_diff, cos_diff, sin_diff]))
            color_vect = torch.stack(diff_vectors)
            dist = torch.nn.HingeEmbeddingLoss()(color_vect, -torch.ones_like(color_vect))
            return dist.mean()
        except:
            r = torch.square(x[:,0,...].mean() - y[:,0,...].mean())
            g = torch.square(x[:,1,...].mean() - y[:,1,...].mean())
            b = torch.square(x[:,2,...].mean() - y[:,2,...].mean())
            r_loss = r.exp() / (g.exp() + b.exp()).log()
            g_loss = g.exp() / (r.exp() + b.exp()).log()
            b_loss = b.exp() / (g.exp() + r.exp()).log()
            loss = torch.sqrt(r_loss + g_loss + b_loss)
            return loss

def color_loss(out, target):
    out_yuv = rgb_to_yuv(out)
    out_u = out_yuv[:, 1, :, :]
    out_v = out_yuv[:, 2, :, :]
    target_yuv = rgb_to_yuv(target)
    target_u = target_yuv[:, 1, :, :]
    target_v = target_yuv[:, 2, :, :]

    return torch.div(torch.mean((out_u - target_u).pow(1)).abs() + torch.mean((out_v - target_v).pow(1)).abs(), 2)


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17

    from https://github.com/sxyu/pixel-nerf/blob/master/model/loss.py#L51
    """

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = (
            torch.mean(self.element_loss(outputs, targets), -1) / betas
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))

class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (
            1 + lambda_bg
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))


def get(name, *args, **kwargs):
    name = clean(name)
    if name in ["colorvectloss","colorvect"]:
        color_axes = kwargs.get("color_axes", [0,1,2])
        return ColorVectLoss(color_axes=color_axes)
    elif name in ["colorloss","color"]:
        color_axes = kwargs.get("color_axes", [0,1,2])
        return ColorLoss(color_axes=color_axes)
    elif name in ["rgbwithuncertainty","rgbuncertainty","uncertainty"]:
        conf = kwargs.get("conf")
