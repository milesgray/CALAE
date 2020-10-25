import torch
from torch import nn


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

class PatchSupCLRLoss(nn.Module):
    def __init__(self, T, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.T = T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

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
        #print(f"Feat q: {feat_q.shape}, feat_k: {feat_k.shape}")
        dist = torch.div(torch.matmul(feat_q.view(-1, 1), feat_k.view(-1, 1).T), self.T)
        logits_max, _ = torch.max(dist, dim=1, keepdim=True)
        logits = dist - logits_max.detach()
        exp_logits = torch.exp(logits)
        #print(f"logits: {logits.shape}, exp_logits: {exp_logits.shape}")
        pos_logits = exp_logits * pos_mask
        neg_logits = exp_logits * neg_mask
        #print(f"pos_logits: {pos_logits.shape}, neg_logits: {neg_logits.shape}")
        log_prob = neg_logits - torch.log(pos_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        loss = - (self.T / 0.07) * mean_log_prob_pos
        #print(f"loss: {loss.shape}")
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
