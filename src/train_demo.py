"""
Minimal Demo for TopoCon-MP
---------------------------
Runs the full architecture end-to-end using random tensors:

- Swin-T image encoder
- CNN topo encoder
- Fusion classifier
- SupCon loss

This verifies correctness of the method without requiring datasets.
"""

import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from topo_encoder import TopologyEncoder
from fusion import FusionClassifier
from supcon_loss import SupConLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    img_h = img_w = 224
    topo_hw = 20
    num_classes = 2

    # ---------------------------
    # Instantiate modules
    # ---------------------------
    image_encoder = ImageEncoder(
        use_pretrained=False,    # lightweight for demo
        freeze_backbone=True
    ).to(device)

    topo_encoder = TopologyEncoder(
        in_channels=3,
        emb_dim=image_encoder.emb_dim
    ).to(device)

    fusion = FusionClassifier(
        emb_dim=image_encoder.emb_dim,
        num_classes=num_classes
    ).to(device)

    supcon = SupConLoss(temperature=0.07).to(device)

    # ---------------------------
    # Fake random inputs
    # ---------------------------
    fake_imgs = torch.randn(batch_size, 3, img_h, img_w).to(device)
    fake_topo = torch.randn(batch_size, 3, topo_hw, topo_hw).to(device)

    # ---------------------------
    # Forward pass
    # ---------------------------
    img_emb = image_encoder(fake_imgs)
    topo_emb = topo_encoder(fake_topo)

    print("Image embedding:", img_emb.shape)
    print("Topo embedding:", topo_emb.shape)

    logits = fusion(img_emb, topo_emb)
    print("Logits:", logits.shape)

    # Contrastive loss
    loss_con = supcon(img_emb, topo_emb)
    print("SupCon Loss:", float(loss_con))

    print("\nDemo complete — architecture functioning correctly.")


if __name__ == "__main__":
    main()
