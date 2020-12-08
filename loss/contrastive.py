import torch
from torch import nn

# Contrastive Unsupervised Image to Image Translation (CUT) Gen's Patch based Contrastive task
class PatchNCELoss(nn.Module):
    def __init__(self, nce_T, batch_size):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        print(f"Feat q: {feat_q.shape}, feat_k: {feat_k.shape}")
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        #if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
        batch_dim_for_bmm = 1
        #else:
         #   batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

# SupCLR mixed with Contrastive Patch 
class PatchSupCLRLoss(nn.Module):
    def __init__(self, temp, batch_size, base_T=0.07, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.T = temp
        self.base_T = base_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.verbose = verbose

    def forward(self, feat_q, feat_k, labels):
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T)
        neg_mask = torch.logical_not(pos_mask)
        pos_mask = pos_mask.float().to(feat_q.device)
        neg_mask = neg_mask.float().to(feat_q.device)

        # pos logit
        if self.verbose: print(f"Feat q: {feat_q.shape}, feat_k: {feat_k.shape}")
        dist = torch.div(torch.matmul(feat_q.view(-1, 1), feat_k.view(-1, 1).T), self.T)
        logits_max, _ = torch.max(dist, dim=1, keepdim=True)
        logits = dist - logits_max.detach()
        exp_logits = torch.exp(logits)
        if self.verbose: print(f"logits: {logits.shape}, exp_logits: {exp_logits.shape}")
        pos_logits = exp_logits * pos_mask
        neg_logits = exp_logits * neg_mask
        if self.verbose: print(f"pos_logits: {pos_logits.shape}, neg_logits: {neg_logits.shape}")
        log_prob = neg_logits - torch.log(pos_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        loss = - (self.T / self.base_T) * mean_log_prob_pos
        if self.verbose: print(f"loss: {loss.shape}")
        loss = loss.mean()

        return loss

    def old(self):
        #l_pos = torch.bmm(feat_q.view(batch_size, 1, -1), feat_k.view(batch_size, -1, 1))
        #l_pos = l_pos.view(batch_size, 1)

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(batch_size, -1, dim)
        feat_k = feat_k.view(batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        #print(f"l_pos: {l_pos.shape}, l_neg: {l_neg.shape}")
        out = torch.cat((l_pos, l_neg), dim=1) / self.T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


# Conditional Contrastive
# https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/losses.py#L107

class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp((self.remove_diag(similarity_matrix) - margin)/temperature)

        inst2proxy_positive = torch.exp((self.cosine_similarity(inst_embed, proxy) - margin)/temperature)
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive

        denomerator = torch.cat([torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature*(numerator/denomerator)).mean()
        return criterion

class Conditional_Contrastive_loss_plus(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        p2i_similarity_matrix = self.calculate_similarity_matrix(proxy, inst_embed)
        i2i_similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        p2i_similarity_zone = torch.exp((p2i_similarity_matrix - margin)/temperature)
        i2i_similarity_zone = torch.exp((i2i_similarity_matrix - margin)/temperature)

        mask_4_remove_negatives = negative_mask[labels]
        p2i_positives = p2i_similarity_zone*mask_4_remove_negatives
        i2i_positives = i2i_similarity_zone*mask_4_remove_negatives

        p2i_numerator = p2i_positives.sum(dim=1)
        i2i_numerator = i2i_positives.sum(dim=1)
        p2i_denomerator = p2i_similarity_zone.sum(dim=1)
        i2i_denomerator = i2i_similarity_zone.sum(dim=1)

        p2i_contra_loss = -torch.log(temperature*(p2i_numerator/p2i_denomerator)).mean()
        i2i_contra_loss = -torch.log(temperature*(i2i_numerator/i2i_denomerator)).mean()
        return p2i_contra_loss + i2i_contra_loss

class Proxy_NCA_loss(torch.nn.Module):
    def __init__(self, device, embedding_layer, num_classes, batch_size):
        super().__init__()
        self.device = device
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _get_positive_proxy_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        rvs_one_hot_target = np.ones([self.num_classes, self.num_classes]) - np.eye(self.num_classes)
        rvs_one_hot_target = rvs_one_hot_target[labels]
        mask = torch.from_numpy((rvs_one_hot_target)).type(torch.bool)
        return mask.to(self.device)

    def forward(self, inst_embed, proxy, labels):
        all_labels = torch.tensor([c for c in range(self.num_classes)]).type(torch.long).to(self.device)
        positive_proxy_mask = self._get_positive_proxy_mask(labels)
        negative_proxies = torch.exp(torch.mm(inst_embed, self.embedding_layer(all_labels).T))*positive_proxy_mask

        inst2proxy_positive = torch.exp(self.cosine_similarity(inst_embed, proxy))
        numerator = inst2proxy_positive
        denomerator = negative_proxies.sum(dim=1)
        criterion = -torch.log(numerator/denomerator).mean()
        return criterion

class NT_Xent_loss(torch.nn.Module):
    def __init__(self, device, batch_size, use_cosine_similarity=True):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, temperature):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)
