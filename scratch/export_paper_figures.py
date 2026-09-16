import json
import base64
import os
from io import BytesIO
from PIL import Image

os.makedirs("reports/figures", exist_ok=True)

def export_image_from_notebook(nb_path, cell_idx, out_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    cell = nb["cells"][cell_idx]
    png_b64 = None
    for out in cell.get("outputs", []):
        if "data" in out and "image/png" in out["data"]:
            png_b64 = out["data"]["image/png"]
            break
            
    if not png_b64:
        raise ValueError(f"No image/png found in {nb_path} cell {cell_idx}")
        
    img_bytes = base64.b64decode(png_b64)
    im = Image.open(BytesIO(img_bytes))
    
    # Ensure clean white background if RGBA
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
        
    im.save(out_path, format="PNG", dpi=(300, 300))
    print(f"✓ Successfully exported {out_path} (size={im.size}) from {nb_path} cell {cell_idx}")

# 1. Continuous ±72-Hour Parent Sequence Hydration (Cell 4 of Notebook 04)
export_image_from_notebook(
    "04_Physics_RAG_Deployment.ipynb",
    4,
    "reports/figures/figure_hydrated_parent_sequence.png",
)

# 2. 72-Hour SciML Autonomous Storm Rollouts (Cell 11 of Notebook 03)
export_image_from_notebook(
    "03_SciML_Architectural_Ablation.ipynb",
    11,
    "reports/figures/figure_sciml_72h_storm_rollouts.png",
)

# 3. Real-World Field Survey Anomaly Audits via Sequence Hydration (Cell 22 of Notebook 04)
export_image_from_notebook(
    "04_Physics_RAG_Deployment.ipynb",
    22,
    "reports/figures/figure_anomaly_audits_hydration.png",
)
