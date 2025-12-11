"""
Fusion Module for TopoCon-MP
-----------------------------
Combines the Swin-T image embedding and the topology embedding,
and outputs class logits.
"""

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, emb_dim: int = 768, num_classes: int = 2, p_drop: float = 0.3):
        """
        Parameters
        ----------
        emb_dim : int
            Dimension of each branch (Swin and topology encoder).
        num_classes : int
            Number of output classes.
        p_drop : float
            Dropout probability.
        """
        super().__init__()

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * emb_dim),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # Final classifier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, img_emb, topo_emb):
        """
        img_emb  : [B, emb_dim]  from Swin
        topo_emb : [B, emb_dim]  from topology encoder
        returns  : [B, num_classes]
        """
        fused = torch.cat([img_emb, topo_emb], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)
