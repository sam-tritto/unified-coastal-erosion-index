"""Generate publication-quality Figure 5 for the IEEE paper.

Refined with:
- Project-wide cohesive architectural color palette matching Figure 4:
  * Baseline 1: Raw LLM (#64748B - Slate Gray)
  * Baseline 2: Flat SQL (#D97706 - Technical Amber)
  * Baseline 3: Pure Physics (#0D9488 - Deep Sea Teal)
  * Proposed: Hybrid Cascade (#1D4ED8 - Royal Cobalt Blue)
- Clean typography and crisp annotations across all 4 panels
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def generate_figure_5():
    csv_path = "reports/tables/peak_impact_disequilibrium_benchmark.csv"
    df = pd.read_csv(csv_path)

    # Architectural color mapping (identical to Fig 4)
    arch_color_map = {
        "Baseline 1: Raw-Data LLM": "#64748B",
        "Baseline 2: Flat SQL": "#D97706",
        "Baseline 3: Pure Physics": "#0D9488",
        "Proposed: Hybrid Cascade": "#1D4ED8",
    }

    colors = [arch_color_map.get(a, "#334155") for a in df["Architecture"]]
    labels_short = ["Raw LLM", "Flat SQL", "Pure Phys", "Hybrid\nCascade"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), dpi=300)

    # ----------------------------------------------------
    # Panel (a): Graded NDCG@5 (Peak Hydrodynamic Work)
    # ----------------------------------------------------
    ndcg_vals = df["Graded NDCG@5"].values
    bars0 = axes[0].bar(range(len(df)), ndcg_vals, color=colors, alpha=0.92, edgecolor="#1E293B", linewidth=0.7)
    for b in bars0:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(labels_short, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Graded NDCG@5 (Peak Work)", fontweight="bold", fontsize=10.5)
    axes[0].set_title("(a) Peak Hydrodynamic Work NDCG@5", fontweight="bold", fontsize=11.5)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0, 1.15)

    # ----------------------------------------------------
    # Panel (b): Median Temporal Peak Error (hrs)
    # ----------------------------------------------------
    dt_errors = df["Median |Δt_peak| Error (hrs)"].values
    bars1 = axes[1].bar(range(len(df)), dt_errors, color=colors, alpha=0.92, edgecolor="#1E293B", linewidth=0.7)
    for b in bars1:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2.0, h + 3.0, f"{h:.0f}h", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(labels_short, fontweight="bold", fontsize=10)
    axes[1].set_ylabel("Median Temporal Error to Peak (hours)", fontweight="bold", fontsize=10.5)
    axes[1].set_title("(b) Temporal Peak Error (|Δt_peak|)", fontweight="bold", fontsize=11.5)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].set_axisbelow(True)
    axes[1].set_ylim(0, max(dt_errors) * 1.25)

    # ----------------------------------------------------
    # Panel (c): Retrieved Disequilibrium Energy (Upsilon)
    # ----------------------------------------------------
    ups_top1 = df["Median Top-1 Upsilon"].values / 1000.0
    ups_top5 = df["Mean Top-5 Upsilon"].values / 1000.0
    x = np.arange(len(df))
    w = 0.35

    b_top1 = axes[2].bar(x - w / 2, ups_top1, w, label="Top-1 Record", color=colors, alpha=0.95, edgecolor="#1E293B", linewidth=0.7)
    b_top5 = axes[2].bar(x + w / 2, ups_top5, w, label="Mean Top-5", color=colors, alpha=0.55, edgecolor="#1E293B", linewidth=0.7, hatch="//")

    for b in b_top1:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for b in b_top5:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}k", ha="center", va="bottom", fontsize=8.5)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_short, fontweight="bold", fontsize=10)
    axes[2].set_ylabel(r"Disequilibrium Energy $\Upsilon$ ($10^3$)", fontweight="bold", fontsize=10.5)
    axes[2].set_title(r"(c) Disequilibrium Energy ($\Upsilon$)", fontweight="bold", fontsize=11.5)
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper left", fontsize=8.8)
    axes[2].set_ylim(0, max(max(ups_top1), max(ups_top5)) * 1.25)

    # ----------------------------------------------------
    # Panel (d): Wave Power Flux & Peak Precision@5
    # ----------------------------------------------------
    p5_vals = df["Peak Precision@5"].values
    power_vals = df["Mean Top-5 Wave Power (kW/m)"].values

    ax4_twin = axes[3].twinx()
    b_p5 = axes[3].bar(x - w / 2, p5_vals, w, label="Peak Precision@5", color=colors, alpha=0.92, edgecolor="#1E293B", linewidth=0.7)
    b_pwr = ax4_twin.bar(x + w / 2, power_vals, w, label="Mean Wave Power", color=colors, alpha=0.45, edgecolor="#1E293B", linewidth=0.7, hatch="xx")

    for b in b_p5:
        h = b.get_height()
        axes[3].text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9.0)
    for b in b_pwr:
        h = b.get_height()
        ax4_twin.text(b.get_x() + b.get_width() / 2.0, h + 3.0, f"{h:.0f}", ha="center", va="bottom", fontsize=8.8)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels_short, fontweight="bold", fontsize=10)
    axes[3].set_ylabel("Peak Precision@5 (Fraction)", fontweight="bold", fontsize=10.5)
    ax4_twin.set_ylabel("Retrieved Wave Power $P$ (kW/m)", fontweight="bold", fontsize=10.5)
    axes[3].set_title("(d) Peak Precision & Wave Power", fontweight="bold", fontsize=11.5)
    axes[3].set_ylim(0, 1.15)
    ax4_twin.set_ylim(0, max(power_vals) * 1.3)
    axes[3].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[3].set_axisbelow(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#64748B")
        ax.spines["bottom"].set_color("#64748B")

    plt.tight_layout()
    out_path = "reports/figures/figure_peak_impact_disequilibrium_advantage.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Successfully generated {out_path}")


if __name__ == "__main__":
    generate_figure_5()
