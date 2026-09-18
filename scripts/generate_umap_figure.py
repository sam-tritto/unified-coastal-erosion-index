"""Reformat Figure 3 (4-Panel UMAP Latent Space Projection) into a 2x2 grid.

Converts the wide 1x4 horizontal strip into an aligned 2x2 publication layout:
- Top-Left: (a) Dean number Omega
- Top-Right: (b) Hydrodynamic Kinetic Energy log10(Upsilon)
- Bottom-Left: (c) Iribarren number xi (with y-axis alignment)
- Bottom-Right: (d) Universal Cross-Site Invariance

Outputs:
  reports/figures/figure_umap_2x2.png
"""

import os
from PIL import Image


def generate_umap_2x2():
    src_path = "reports/figures/figure_umap_4panel_final.png"
    out_path = "reports/figures/figure_umap_2x2.png"

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source UMAP figure not found at {src_path}")

    im = Image.open(src_path)

    # Clean panel split boundaries from white divider gaps
    p1 = im.crop((0, 0, 2048, 1600))         # (a) Dean number
    p2 = im.crop((2048, 0, 3952, 1600))      # (b) Kinetic energy
    p3_raw = im.crop((3952, 0, 5857, 1769))  # (c) Iribarren number
    p4 = im.crop((5857, 0, 7768, 1769))      # (d) Cross-site invariance

    # Extract y-axis label strip ("UMAP Dimension 2") from p1 to align p3
    ylabel_strip = p1.crop((0, 0, 144, 1600))

    # Pad p3 with the matching y-axis label strip so the left column is perfectly aligned
    p3_full = Image.new("RGB", (144 + p3_raw.size[0], 1769), (255, 255, 255))
    p3_full.paste(ylabel_strip, (0, 0))
    p3_full.paste(p3_raw, (144, 0))

    w_col1 = max(p1.size[0], p3_full.size[0])
    w_col2 = max(p2.size[0], p4.size[0])
    total_w = w_col1 + w_col2

    h_row1 = p1.size[1]
    h_row2 = p3_full.size[1]
    total_h = h_row1 + h_row2

    grid_2x2 = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    grid_2x2.paste(p1, (0, 0))
    grid_2x2.paste(p2, (w_col1, 0))
    grid_2x2.paste(p3_full, (0, h_row1))
    grid_2x2.paste(p4, (w_col1, h_row1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid_2x2.save(out_path, dpi=(300, 300))
    print(f"✓ Successfully generated 2x2 UMAP figure at: {out_path} (size={grid_2x2.size})")


if __name__ == "__main__":
    generate_umap_2x2()
