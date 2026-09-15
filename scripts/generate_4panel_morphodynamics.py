"""
Generate Publication-Grade 4-Panel Morphodynamic Regime Plot.

Panels:
  (A) Morphodynamic Regime Parameterization (Dean Number Omega)
  (B) Deep-Water Wave Steepness Distribution (s0)
  (C) Tidal Dissipation & Energy Smearing (Psi_tide)
  (D) Scale-Invariant Calibration Invariance (Upsilon* ≈ 1.0)

Outputs:
  reports/figures/figure_morphodynamic_regimes_4panel.pdf
  reports/figures/figure_morphodynamic_regimes_4panel.png
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.physics import COASTAL_SITE_PROFILES

def main():
    sites = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]
    site_datasets = {}
    
    print("Loading 5-year sensor parquets (2012-2016)...")
    for s_key in sites:
        parquet_path = os.path.join(repo_root, f"data/processed/{s_key}_2012_2016_sensors.parquet")
        site_datasets[s_key] = pd.read_parquet(parquet_path)

    # Publication style
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    site_labels = [COASTAL_SITE_PROFILES[s]["name"].split(",")[0] for s in sites]

    # Harmonious, domain-aligned site color palette (consistent across all subplots)
    site_palette = {
        "duck_nc": "#1f77b4",          # Atlantic Ocean Blue (Barrier Island)
        "narrabeen_aus": "#e76f51",     # Warm Coral / Terracotta (Embayed Beach)
        "torrey_pines_ca": "#2a9d8f",   # California Coastal Teal (Pocket Beach)
        "truc_vert_france": "#7b2cbf",  # French Atlantic Amethyst (Macrotidal)
    }
    site_colors = [site_palette[s] for s in sites]

    # Panel A: Morphodynamic Regime Parameterization (Dean Number Omega)
    dean_data = [site_datasets[s]["dean_number"].values for s in sites]
    sns.boxplot(data=dean_data, ax=axes[0, 0], palette=site_colors)
    axes[0, 0].set_xticks(range(len(site_labels)))
    axes[0, 0].set_xticklabels(site_labels, rotation=15, fontweight="bold")
    axes[0, 0].set_ylabel(r"Dean Number $\Omega = H_s / (w_s T_p)$", fontweight="bold")
    axes[0, 0].set_title("(A) Morphodynamic Regime Across Global Sites", fontweight="bold", fontsize=14)
    axes[0, 0].axhline(1.0, color="#27ae60", linestyle="--", alpha=0.8, linewidth=1.5, label=r"Reflective ($\Omega < 1$)")
    axes[0, 0].axhline(6.0, color="#e74c3c", linestyle="--", alpha=0.8, linewidth=1.5, label=r"Dissipative ($\Omega > 6$)")
    axes[0, 0].legend(loc="upper right", fontsize=10, frameon=True)

    # Panel B: Deep-Water Wave Steepness Distribution (s0)
    s0_data = [site_datasets[s]["wave_steepness_s0"].values for s in sites]
    sns.boxplot(data=s0_data, ax=axes[0, 1], palette=site_colors)
    axes[0, 1].set_xticks(range(len(site_labels)))
    axes[0, 1].set_xticklabels(site_labels, rotation=15, fontweight="bold")
    axes[0, 1].set_ylabel(r"Wave Steepness $s_0 = H_s / L_0$", fontweight="bold")
    axes[0, 1].set_title("(B) Wave Steepness Across Sites", fontweight="bold", fontsize=14)
    axes[0, 1].axhline(0.025, color="#c0392b", linestyle="--", linewidth=1.8, label=r"Crit Steepness $s_{0,\mathrm{crit}}=0.025$")
    axes[0, 1].legend(loc="upper right", fontsize=10, frameon=True)

    # Panel C: Tidal Dissipation & Energy Smearing Factor (Psi_tide)
    psi_tide_data = [site_datasets[s]["psi_tide"].values for s in sites]
    sns.boxplot(data=psi_tide_data, ax=axes[1, 0], palette=site_colors)
    axes[1, 0].set_xticks(range(len(site_labels)))
    axes[1, 0].set_xticklabels(site_labels, rotation=15, fontweight="bold")
    axes[1, 0].set_ylabel(r"Tidal Smearing Factor $\Psi_{\mathrm{tide}}$ [-]", fontweight="bold")
    axes[1, 0].set_title(r"(C) Tidal Dissipation & Energy Smearing ($\Psi_{tide}$)", fontweight="bold", fontsize=14)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].axhline(1.0, color="#27ae60", linestyle="--", alpha=0.8, linewidth=1.5, label=r"Microtidal Limit ($\Psi_{\mathrm{tide}} \to 1.0$)")
    axes[1, 0].axhline(0.5, color="#e74c3c", linestyle=":", alpha=0.8, linewidth=1.5, label=r"Macrotidal Damping ($\Psi_{\mathrm{tide}} \leq 0.5$)")
    axes[1, 0].legend(loc="lower left", fontsize=10, frameon=True)

    # Panel D: Scale-Invariant Collapse (Normalized Upsilon*)
    u_star_data = [np.clip(site_datasets[s]["upsilon_star"].values, 0, 6) for s in sites]
    sns.boxplot(data=u_star_data, ax=axes[1, 1], palette=site_colors)
    axes[1, 1].set_xticks(range(len(site_labels)))
    axes[1, 1].set_xticklabels(site_labels, rotation=15, fontweight="bold")
    axes[1, 1].set_ylabel(r"Dimensionless Relative Forcing $\Upsilon^*$", fontweight="bold")
    axes[1, 1].set_title(r"(D) Scale-Invariant Calibration Invariance ($\Upsilon^* \approx 1.0$)", fontweight="bold", fontsize=14)
    axes[1, 1].axhline(1.0, color="#2c3e50", linestyle="-", linewidth=2.0, label=r"Equilibrium Reference ($\Upsilon^* = 1.0$)")
    axes[1, 1].legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    out_dir = os.path.join(repo_root, "reports/figures")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "figure_morphodynamic_regimes_4panel.png")
    pdf_path = os.path.join(out_dir, "figure_morphodynamic_regimes_4panel.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ Figure successfully exported to:\n  - {png_path}\n  - {pdf_path}")

if __name__ == "__main__":
    main()
