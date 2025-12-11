"""
Multipersistence Descriptor for TopoCon-MP (MIDL 2026)

Computes 20×20×3 grid:
  [β0, β1, activated_pixels]
"""

import numpy as np
from scipy import ndimage as ndi


# ============================================================
# --- Betti utilities ----------------------------------------
# ============================================================

def betti2d(mask: np.ndarray):
    """
    Compute β0 (connected components) and β1 (holes) for a binary mask.
    """
    mask = mask.astype(bool, copy=False)

    struct8 = np.ones((3, 3), int)
    struct4 = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], int)

    # β0 = foreground connected components
    _, n_fg = ndi.label(mask, structure=struct8)

    # β1 = holes = background components - 1
    bg = np.pad(~mask, 1, constant_values=True)
    _, n_bg = ndi.label(bg, structure=struct4)

    return int(n_fg), int(n_bg - 1)


# ============================================================
# --- Threshold utilities ------------------------------------
# ============================================================

def make_thresholds(arr: np.ndarray, grid_size: int, use_percentiles: bool):
    """Compute threshold grid along a single channel."""
    if use_percentiles:
        qs = np.linspace(0.01, 0.99, grid_size)
        return np.quantile(arr.astype(np.float64), qs)
    return np.linspace(0, 255, grid_size, dtype=np.float64)


# ============================================================
# --- Core multipersistence descriptor ------------------------
# ============================================================

def compute_multipersistence(rgb: np.ndarray,
                             grid_size: int = 20,
                             use_percentiles: bool = False):
    """
    Compute the 3 × grid × grid multipersistence descriptor:
      channel 0 → β0
      channel 1 → β1
      channel 2 → activated pixels

    rgb: uint8 RGB image [H, W, 3].
    """
    R = rgb[..., 0]
    G = rgb[..., 1]

    r_grid = make_thresholds(R, grid_size, use_percentiles)
    g_grid = make_thresholds(G, grid_size, use_percentiles)

    B0  = np.zeros((grid_size, grid_size), np.float32)
    B1  = np.zeros((grid_size, grid_size), np.float32)
    ACT = np.zeros((grid_size, grid_size), np.float32)

    for i, r_thr in enumerate(r_grid):
        rmask = (R <= r_thr)

        for j, g_thr in enumerate(g_grid):
            mask = rmask & (G <= g_thr)

            β0, β1 = betti2d(mask)
            B0[i, j] = β0
            B1[i, j] = β1
            ACT[i, j] = mask.sum()   # activated pixels

    # Output: 3×20×20 topological image
    return np.stack([B0, B1, ACT], axis=0).astype(np.float32)


"""
# Example usage (not required to run now):

from PIL import Image
import numpy as np

img = Image.open("example.jpg").resize((224,224))
rgb = np.array(img)

feat = compute_multipersistence(rgb)
print(feat.shape)  # (3, 20, 20)
"""
