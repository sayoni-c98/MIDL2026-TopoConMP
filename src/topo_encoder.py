"""
Topology Encoder (CNN) for TopoCon-MP
-------------------------------------
We treat the 3×20×20 multipersistence representation
(β0, β1, activated pixels) as a 3-channel topological image.
"""

import torch
import torch.nn as nn


class TopologyEncoder(nn.Module):
    def __init__(self, emb_dim: int = 768):
        """
        Parameters
        ----------
        emb_dim : int
            Output embedding dimension.
            Swin-Tiny uses emb_dim = 768.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # Input: [B, 3, 20, 20]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # → [B, 32, 10, 10]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),      # → [B, 64, 5, 5]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),         # → [B, 128, 1, 1]

            nn.Flatten(),                    # → [B, 128]
            nn.Linear(128, emb_dim),         # → [B, emb_dim]
        )

    def forward(self, x):
        """
        x : torch.Tensor
            Shape [B, 3, 20, 20]
        Returns
        -------
        torch.Tensor : [B, emb_dim]
        """
        return self.encoder(x)
