"""https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
This is the unofficial PyTorch implementation of the paper 
"Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" 
in NIPS 2018.
https://arxiv.org/abs/1805.07836
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        """'Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels'

        Args:
            q (float, optional): Weight for labels. Defaults to 0.7.
            k (float, optional): Weight for mainly logits, but not only.. Defaults to 0.5.
            trainset_size (int, optional): Size of the internal weight tensor, 
                needs shape [DS, 1] (DS = total samples being trained on). 
                Defaults to 50000.
        """
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(
            data=torch.ones(trainset_size, 1), requires_grad=False
        )

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - (
            (1 - (self.k ** self.q)) / self.q
        ) * self.weight[indexes]
        loss = torch.mean(loss)

        return loss
    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = (1 - (Yg ** self.q)) / self.q
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
