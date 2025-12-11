# TopoCon-MP: Multipersistence-Topology-Aware Contrastive Learning for Dermoscopic Image Classification

This repository contains a clean and modular reference implementation of **TopoCon-MP**,  
the topology-infused contrastive learning framework introduced in our MIDL 2026 submission.

**TopoCon-MP** integrates:
- **Multipersistence descriptors** computed on red–green intensity grids  
- **3×20×20 topological images** (β0, β1, activated pixels)  
- **Topology Encoder** (CNN)  
- **Image Encoder** (Swin-Tiny)  
- **Fusion Module**  
- **Supervised Contrastive Loss** (TopoSupCon)  

All components are presented in a minimal, reproducible form consistent with the methodology described in the paper.

---

## 📁 Repository Structure

MIDL2026-TopoConMP/
│
├── src/
│ ├── multipersistence.py # Compute (β0, β1, activated pixels) grids
│ ├── topo_encoder.py # CNN topology encoder
│ ├── image_encoder.py # Swin-T image encoder
│ ├── fusion.py # Fusion block + classifier
│ ├── supcon_loss.py # Supervised contrastive loss
│ └── train_demo.py # Minimal runnable demo
│
├── requirements.txt
└── README.md


---

## 🔧 Installation

Install dependencies:

```bash
pip install -r requirements.txt


python src/train_demo.py
