"""
Supervised Contrastive Loss for TopoCon-MP
------------------------------------------
Pairwise contrastive loss between image embeddings and
topology embeddings for each sample in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, topo_emb):
        """
        img_emb  : [B, d]
        topo_emb : [B, d]
        """
        # L2 normalize embeddings
        zi = F.normalize(img_emb, dim=1)
        zt = F.normalize(topo_emb, dim=1)

        B = zi.size(0)
        # Stack to shape [2B, d]
        z = torch.cat([zi, zt], dim=0)

        # Similarity matrix
        sim = z @ z.T / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values  # stability

        # Remove diagonal entries (self-similarity)
        eye = torch.eye(2 * B, device=sim.device)
        exp_sim = torch.exp(sim) * (1 - eye)

        # Positive pairs
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        idx = torch.arange(B, device=sim.device)
        pos_mask[idx, idx + B] = True
        pos_mask[idx + B, idx] = True

        pos_sim = exp_sim[pos_mask]
        denom = exp_sim.sum(dim=1) + 1e-12

        loss = -torch.log(pos_sim / denom)
        return loss.mean()
