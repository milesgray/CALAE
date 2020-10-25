import torch
from torch import nn

class ColorVectLoss(nn.Module):
    def __init__(self, color_axes=[0,1,2]):
        super().__init__()
        self.color_axes = color_axes

    def forward(self, x, y):
        try:
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
