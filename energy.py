import torch


def w1(z): 
    return torch.sin(2 * math.pi * z[:,0] / 4)

def w2(z): 
    return 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)

def w3(z): 
    return 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

def u_z1(z): 
    return 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                 torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
def u_z2(z): 
    return 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
def u_z3(z): 
    return - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
def u_z4(z): 
    return - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)