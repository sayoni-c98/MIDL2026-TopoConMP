"""
Image Encoder (Swin-Tiny) for TopoCon-MP
----------------------------------------
This module wraps torchvision Swin-Tiny and returns
the image embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class ImageEncoder(nn.Module):
    def __init__(self, use_pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        # Load Swin-T backbone
        self.backbone = swin_t(
            weights=Swin_T_Weights.IMAGENET1K_V1 if use_pretrained else None
        )

        # Swin-T embedding dimension
        self.emb_dim = self.backbone.head.in_features

        # Remove classifier head
        self.backbone.head = nn.Identity()

        # Optionally freeze backbone weights
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, 3, H, W] normalized RGB image
        returns: [B, emb_dim] image embedding
        """
        return self.backbone(x)
