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


##  Installation

### 1) Create a new environment

```bash
conda create -n topoconmp python=3.10 -y
conda activate topoconmp
```

### 2) Install dependencies
pip install -r requirements.txt

### 3) Quick Start (No Dataset Required)
Run the minimal demo to verify that the full architecture works end-to-end:

python src/train_demo.py

## Datasets
Experiments were performed on three dermoscopic datasets:
- **DermaMNIST**
- **MILK-10K**
- **PAD-UFES-20**

## Repository Structure

```
MIDL2026-TopoConMP/
│
├── src/
│   ├── multipersistence.py      # Compute multipersistence (β0, β1, activated pixels)
│   ├── topo_encoder.py          # CNN topology encoder
│   ├── image_encoder.py         # Swin-T image encoder
│   ├── fusion.py                # Fusion block + classifier
│   ├── supcon_loss.py           # Supervised contrastive loss
│   └── train_demo.py            # Minimal runnable demo (no dataset required)
│
├── requirements.txt
└── README.md
```

## Acknowledgements

We thank the creators and maintainers of the following resources:

- **MedMNIST**, **ISIC Archive**, and **PAD-UFES-20** for providing open dermoscopic datasets  
- **PyTorch** and **Torchvision** teams for Swin Transformer implementations and pretrained weights  
- The authors of **Supervised Contrastive Learning** for foundational contrastive learning methodology  
- The medical imaging and computational topology communities for their continued contributions to open research


