import numpy as np
import torch

class DistancesNumpy:
    """A collection of nearly all known distance functions implemented with numpy operators"""
    
    def braycurtis(self, a, b):
        return np.sum(np.fabs(a - b)) / np.sum(np.fabs(a + b))

    def canberra(self, a, b):
        return np.sum(np.fabs(a - b) / (np.fabs(a) + np.fabs(b)))

    def chebyshev(self, a, b):
        return np.amax(a - b)

    def cityblock(self, a, b):
        return self.manhattan(a, b)

    def correlation(self, a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        return 1.0 - np.mean(a * b) / np.sqrt(np.mean(np.square(a)) * np.mean(np.square(b)))

    def cosine(self, a, b):
        return 1 - np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

    def dice(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))

    def euclidean(self, a, b):
        return np.sqrt(np.sum(np.dot((a - b), (a - b))))

    def hamming(self, a, b, w = None):
        if w is None:
            w = np.ones(a.shape[0])
        return np.average(a != b, weights = w)

    def jaccard(self, u, v):
        return np.double(np.bitwise_and((u != v), np.bitwise_or(u != 0, v != 0)).sum()) / np.double(np.bitwise_or(u != 0, v != 0).sum())

    def kulsinski(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (ntf + nft - ntt + len(a)) / (ntf + nft + len(a))

    def mahalanobis(self, a, b, vi):
        return np.sqrt(np.dot(np.dot((a - b), vi),(a - b).T))

    def manhattan(self, a, b):
        return np.sum(np.fabs(a - b))

    def matching(self, a, b):
        return self.hamming(a, b)

    def minkowski(self, a, b, p):
        return np.power(np.sum(np.power(np.fabs(a - b), p)), 1 / p)

    def rogerstanimoto(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))

    def russellrao(self, a, b):
        return float(len(a) - (a * b).sum()) / len(a)

    def seuclidean(self, a, b, V):
        return np.sqrt(np.sum((a - b) ** 2 / V))

    def sokalmichener(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))

    def sokalsneath(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / np.array(ntt + 2.0 * (ntf + nft))

    def sqeuclidean(self, a, b):
        return np.sum(np.dot((a - b), (a - b)))

    def wminkowski(self, a, b, p, w):
        return np.power(np.sum(np.power(np.fabs(w * (a - b)), p)), 1 / p)

    def yule(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))

class DistancesTorch:
    """A collection of nearly all known distance functions implemented with torch operators"""
    
    def braycurtis(self, a, b):
        return torch.sum(torch.abs(a - b)) / torch.sum(torch.abs(a + b))

    def canberra(self, a, b):
        return torch.sum(torch.abs(a - b) / (torch.abs(a) + torch.abs(b)))

    def chebyshev(self, a, b):
        return torch.max(a - b)

    def cityblock(self, a, b):
        return self.manhattan(a, b)

    def correlation(self, a, b):
        a = a - torch.mean(a)
        b = b - torch.mean(b)
        return 1.0 - torch.mean(a * b) / torch.sqrt(torch.mean(torch.square(a)) * torch.mean(torch.square(b)))

    def cosine(self, a, b):
        return 1 - torch.dot(a, b) / (torch.sqrt(torch.dot(a, a)) * torch.sqrt(torch.dot(b, b)))

    def dice(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return ((ntf + nft) / torch.tensor(2.0 * ntt + ntf + nft, dtype=torch.float32)).type(dtype=torch.float32)

    def euclidean(self, a, b):
        return torch.sqrt(torch.sum(torch.dot((a - b), (a - b))))

    def hamming(self, a, b):
        return torch.cdist(a, b, p=0)

    def jaccard(self, u, v):
        return (torch.bitwise_and((u != v), torch.bitwise_or(u != 0, v != 0)).sum()).type(dtype=torch.float64) \
            / (torch.bitwise_or(u != 0, v != 0).sum()).type(dtype=torch.float64)

    def kulsinski(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (ntf + nft - ntt + a.shape[0]) / (ntf + nft + a.shape[0])

    def mahalanobis(self, a, b, vi):
        return torch.sqrt(torch.dot(torch.dot((a - b), vi),(a - b).T))

    def manhattan(self, a, b):
        return torch.sum(torch.abs(a - b))

    def matching(self, a, b):
        return self.hamming(a, b)

    def minkowski(self, a, b, p):
        return torch.power(torch.sum(torch.pow(torch.abs(a - b), p)), 1 / p)

    def rogerstanimoto(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / (ntt + nff + (2.0 * (ntf + nft))).type(dtype=torch.float32)

    def russellrao(self, a, b):
        return (a.shape[0] - (a * b).sum()).type(dtype=torch.float32) / len(a)

    def seuclidean(self, a, b, V):
        return torch.sqrt(torch.sum((a - b) ** 2 / V))

    def sokalmichener(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / (ntt + nff + 2.0 * (ntf + nft)).type(dtype=torch.float32)

    def sokalsneath(self, a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / torch.tensor(ntt + 2.0 * (ntf + nft), dtype=torch.float32)

    def sqeuclidean(self, a, b):
        return torch.sum(torch.dot((a - b), (a - b)))

    def wminkowski(self, a, b, p, w):
        return torch.pow(torch.sum(torch.pow(torch.abs(w * (a - b)), p)), 1 / p)

    def yule(self, a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * ntf * nft / torch.tensor(ntt * nff + ntf * nft)).type(dtype=torch.float32)
