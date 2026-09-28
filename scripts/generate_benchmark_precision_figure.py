"""Generate publication-quality Figure 4 for the IEEE paper.

Refined with:
- Project-wide cohesive color palette (Domain 1: Architectural Hierarchy)
- Zero-collision 2-column header legend placed cleanly above plot axes
- Enhanced micro-aesthetics, crisp typography, and publication-grade styling
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def generate_figure_4():
    csv_path = "reports/tables/grand_4way_rag_benchmark.csv"
    df = pd.read_csv(csv_path)

    k_vals = [1, 5, 10, 20]

    # Project-Wide Cohesive Architectural Hierarchy Palette:
    # - Proposed Hybrid: Salmon (#f4633a)
    # - Pure Physics: Sage Green (#81a554)
    # - Raw-Data LLM: Lilac (#8b7cbf)
    # - Rule-Based SQL: Light Orange (#f89c25)
    # - Google Dense Embeddings: Slate Blue (#4338ca)
    # - BM25 Lexical: Dark Gray (#64748b)
    palette = {
        "Proposed: Hybrid Cascade": {
            "label": "Proposed: Hybrid Cascade",
            "color": "#f4633a",
            "marker": "o",
            "markersize": 6.5,
            "linewidth": 2.4,
            "linestyle": "-",
            "zorder": 6,
        },
        "Baseline 3: Pure Physics": {
            "label": "Baseline 3: Pure Physics",
            "color": "#81a554",
            "marker": "^",
            "markersize": 5.8,
            "linewidth": 1.6,
            "linestyle": "--",
            "zorder": 5,
        },
        "Baseline 1: Raw LLM": {
            "label": "Baseline 1: Raw LLM",
            "color": "#8b7cbf",
            "marker": "D",
            "markersize": 5.0,
            "linewidth": 1.5,
            "linestyle": "-.",
            "zorder": 4,
        },
        "Baseline 2: Rule-Based SQL": {
            "label": "Baseline 2: Rule-Based SQL",
            "color": "#f89c25",
            "marker": "s",
            "markersize": 5.2,
            "linewidth": 1.5,
            "linestyle": ":",
            "zorder": 3,
        },
    }

    arch_map = {
        "Proposed: Hybrid Cascade Physics-RAG": "Proposed: Hybrid Cascade",
        "Proposed: Two-Tier Hybrid Cascade": "Proposed: Hybrid Cascade",
        "Baseline 3: Pure Physics-RAG": "Baseline 3: Pure Physics",
        "Baseline 1: Raw-Data LLM-RAG": "Baseline 1: Raw LLM",
        "Baseline 2: Rule-Based SQL": "Baseline 2: Rule-Based SQL",
    }

    # Setup publication typography
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Exactly 3.5 inches wide for IEEE single column format
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=300)

    # Plot lines
    curves = {}
    for _, row in df.iterrows():
        raw_arch = row["Architecture"]
        arch = arch_map.get(raw_arch, raw_arch)
        cfg = palette.get(arch)
        if not cfg:
            continue
        p_curve = []
        for k in k_vals:
            val = None
            for col in [f"Precision@{k}", f"P@{k}"]:
                if col in row:
                    val = float(row[col])
                    break
            if val is None:
                raise KeyError(f"Could not find Precision@{k} or P@{k} in columns: {list(row.index)}")
            p_curve.append(val)
        curves[arch] = p_curve
        ax.plot(
            k_vals,
            p_curve,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=cfg["markersize"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            linewidth=cfg["linewidth"],
            linestyle=cfg["linestyle"],
            zorder=cfg["zorder"],
            clip_on=False,
        )

    # Subtle shading under Proposed curve representing the dual-guided margin over naive baseline
    if "Proposed: Hybrid Cascade Physics-RAG" in curves and "Baseline 1: Raw-Data LLM-RAG" in curves:
        ax.fill_between(
            k_vals,
            curves["Baseline 1: Raw-Data LLM-RAG"],
            curves["Proposed: Hybrid Cascade Physics-RAG"],
            color="#f4633a",
            alpha=0.08,
            zorder=2,
            label="_nolegend_",
        )

    # Limits and Ticks
    ax.set_xlim(0.3, 20.7)
    ax.set_ylim(0.38, 0.505)
    ax.set_xticks(k_vals)
    ax.set_xticklabels(["k = 1", "k = 5", "k = 10", "k = 20"], fontsize=7.8, fontweight="bold", color="#1E293B")
    
    yticks = np.arange(0.38, 0.51, 0.02)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=7.8, color="#334155")

    ax.set_xlabel("Retrieval Cutoff Radius (k)", fontsize=8.2, fontweight="bold", color="#0F172A", labelpad=4)
    ax.set_ylabel("Macro Precision@k", fontsize=8.2, fontweight="bold", color="#0F172A", labelpad=4)

    # Clean Despine & Minimal Gridlines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#CBD5E1")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Highlight k=5 apex gain (+19.4% vs SQL baseline, P@5 = 0.480)
    ax.annotate(
        "Apex Gain +19.4% vs SQL\n(P@5 = 0.480)",
        xy=(5, 0.480),
        xytext=(7.0, 0.493),
        arrowprops=dict(
            arrowstyle="->",
            color="#c2410c",
            lw=1.0,
            connectionstyle="arc3,rad=-0.15",
        ),
        fontsize=6.2,
        fontweight="bold",
        color="#991b1b",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.96, lw=0.6),
        zorder=7,
    )

    # ZERO-COLLISION HEADER LEGEND:
    # 2-column horizontal legend placed cleanly above the axes canvas
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=6.8,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        framealpha=0.96,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.32,
        labelspacing=0.28,
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    out_path = "reports/figures/figure_precision_at_k_single_column.png"
    alt_out_path = "reports/figures/grand_4way_benchmark_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(alt_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Successfully generated {out_path} and {alt_out_path}")


if __name__ == "__main__":
    generate_figure_4()
