import numpy as np
import torch

class DistancesNumpy:
    """A collection of nearly all known distance functions implemented with numpy operators"""
    @staticmethod
    def braycurtis(a, b):
        return np.sum(np.fabs(a - b)) / np.sum(np.fabs(a + b))
    @staticmethod
    def canberra(a, b):
        return np.sum(np.fabs(a - b) / (np.fabs(a) + np.fabs(b)))
    @staticmethod
    def chebyshev(a, b):
        return np.amax(a - b)
    @staticmethod
    def cityblock(a, b):
        return self.manhattan(a, b)
    @staticmethod
    def correlation(a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        return 1.0 - np.mean(a * b) / np.sqrt(np.mean(np.square(a)) * np.mean(np.square(b)))
    @staticmethod
    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
    @staticmethod
    def dice(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
    @staticmethod
    def euclidean(a, b):
        return np.sqrt(np.sum(np.dot((a - b), (a - b))))
    @staticmethod
    def hamming(a, b, w = None):
        if w is None:
            w = np.ones(a.shape[0])
        return np.average(a != b, weights = w)
    @staticmethod
    def hellinger_distance(p, q):
        return  np.linalg.norm((np.sqrt(p) - np.sqrt(q)), ord=2, axis=1) / np.sqrt(2)
    @staticmethod
    def jaccard(u, v):
        return np.double(np.bitwise_and((u != v), np.bitwise_or(u != 0, v != 0)).sum()) / np.double(np.bitwise_or(u != 0, v != 0).sum())
    @staticmethod
    def kulsinski(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (ntf + nft - ntt + len(a)) / (ntf + nft + len(a))
    @staticmethod
    def mahalanobis(a, b, vi):
        return np.sqrt(np.dot(np.dot((a - b), vi),(a - b).T))
    @staticmethod
    def manhattan(a, b):
        return np.sum(np.fabs(a - b))
    @staticmethod
    def matching(a, b):
        return self.hamming(a, b)
    @staticmethod
    def minkowski(a, b, p):
        return np.power(np.sum(np.power(np.fabs(a - b), p)), 1 / p)
    @staticmethod
    def rogerstanimoto(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))
    @staticmethod
    def russellrao(a, b):
        return float(len(a) - (a * b).sum()) / len(a)
    @staticmethod
    def seuclidean(a, b, V):
        return np.sqrt(np.sum((a - b) ** 2 / V))
    @staticmethod
    def sokalmichener(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))
    @staticmethod
    def sokalsneath(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * (ntf + nft)) / np.array(ntt + 2.0 * (ntf + nft))
    @staticmethod
    def sqeuclidean(a, b):
        return np.sum(np.dot((a - b), (a - b)))
    @staticmethod
    def wminkowski(a, b, p, w):
        return np.power(np.sum(np.power(np.fabs(w * (a - b)), p)), 1 / p)
    @staticmethod
    def yule(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))

class DistancesTorch:
    """A collection of nearly all known distance functions implemented with torch operators"""
    @staticmethod
    def braycurtis(a, b):
        return torch.sum(torch.abs(a - b)) / torch.sum(torch.abs(a + b))
    
    @staticmethod
    def canberra(a, b):
        return torch.sum(torch.abs(a - b) / (torch.abs(a) + torch.abs(b)))

    @staticmethod
    def chebyshev(a, b):
        return torch.max(a - b)

    @staticmethod
    def cityblock(a, b):
        return self.manhattan(a, b)

    @staticmethod
    def correlation(a, b):
        a = a - torch.mean(a)
        b = b - torch.mean(b)
        return 1.0 - torch.mean(a * b) / torch.sqrt(torch.mean(torch.square(a)) * torch.mean(torch.square(b)))

    @staticmethod
    def cosine(a, b):
        return 1 - torch.dot(a, b) / (torch.sqrt(torch.dot(a, a)) * torch.sqrt(torch.dot(b, b)))

    @staticmethod
    def dice(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return ((ntf + nft) / torch.tensor(2.0 * ntt + ntf + nft, dtype=torch.float32)).type(dtype=torch.float32)

    @staticmethod
    def euclidean(a, b):
        return torch.sqrt(torch.sum(torch.dot((a - b), (a - b))))

    @staticmethod
    def hamming(a, b):
        return torch.cdist(a, b, p=0)

    @staticmethod
    def hellinger(p, q):
        return  torch.norm((torch.sqrt(p) - torch.sqrt(q)), p=2, dim=1) / np.sqrt(2)

    @staticmethod
    def jaccard(u, v):
        return (torch.bitwise_and((u != v), torch.bitwise_or(u != 0, v != 0)).sum()).type(dtype=torch.float64) \
            / (torch.bitwise_or(u != 0, v != 0).sum()).type(dtype=torch.float64)

    @staticmethod
    def kulsinski(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (ntf + nft - ntt + a.shape[0]) / (ntf + nft + a.shape[0])

    @staticmethod
    def mahalanobis(a, b, vi):
        return torch.sqrt(torch.dot(torch.dot((a - b), vi),(a - b).T))

    @staticmethod
    def manhattan(a, b):
        return torch.sum(torch.abs(a - b))

    @staticmethod
    def matching(a, b):
        return self.hamming(a, b)

    @staticmethod
    def minkowski(a, b, p):
        return torch.power(torch.sum(torch.pow(torch.abs(a - b), p)), 1 / p)

    @staticmethod
    def rogerstanimoto(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / (ntt + nff + (2.0 * (ntf + nft))).type(dtype=torch.float32)

    @staticmethod
    def russellrao(a, b):
        return (a.shape[0] - (a * b).sum()).type(dtype=torch.float32) / a.shape[0]

    @staticmethod
    def seuclidean(a, b, V):
        return torch.sqrt(torch.sum((a - b) ** 2 / V))

    @staticmethod
    def sokalmichener(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / (ntt + nff + 2.0 * (ntf + nft)).type(dtype=torch.float32)

    @staticmethod
    def sokalsneath(a, b):
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * (ntf + nft)).type(dtype=torch.float32) \
            / torch.tensor(ntt + 2.0 * (ntf + nft), dtype=torch.float32)

    @staticmethod
    def sqeuclidean(a, b):
        return torch.sum(torch.dot((a - b), (a - b)))

    @staticmethod
    def wminkowski(a, b, p, w):
        return torch.pow(torch.sum(torch.pow(torch.abs(w * (a - b)), p)), 1 / p)

    @staticmethod
    def yule(a, b):
        nff = ((1 - a) * (1 - b)).sum()
        nft = ((1 - a) * b).sum()
        ntf = (a * (1 - b)).sum()
        ntt = (a * b).sum()
        return (2.0 * ntf * nft / torch.tensor(ntt * nff + ntf * nft)).type(dtype=torch.float32)
