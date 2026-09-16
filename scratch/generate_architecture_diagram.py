import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs("reports/figures", exist_ok=True)

# Set high-resolution publication styling
plt.rcParams['font.sans-serif'] = 'Helvetica', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(20, 8.5), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Color Palette (Curated Modern Scientific Palette)
c_bg = "#f8f9fa"
c_query = "#3a0ca3"       # Deep Indigo
c_parser = "#4361ee"      # Royal Blue
c_tier1 = "#0077b6"       # Ocean Blue
c_tier1_box = "#e0f2fe"   # Soft Cyan
c_tier2 = "#b5179e"       # Magenta / Coral
c_tier2_box = "#fdf2f8"   # Soft Rose
c_hydrate = "#059669"     # Emerald Green
c_hydrate_box = "#ecfdf5" # Soft Emerald
c_sciml = "#1e293b"       # Dark Slate
c_sciml_box = "#f1f5f9"   # Light Slate
c_badge_border = "#94a3b8"

# Helper for drawing rounded boxes
def draw_card(ax, x, y, w, h, bg_color, border_color, lw=1.8, radius=0.25, zorder=1):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(box)
    return box

# Helper for drawing title headers on cards
def draw_header(ax, x, y, w, h, title_text, bg_color, text_color="white", subtitle=None):
    draw_card(ax, x, y, w, h, bg_color, bg_color, radius=0.2, zorder=2)
    ax.text(x + w/2, y + h*0.58 if subtitle else y + h*0.5, title_text,
            ha="center", va="center", color=text_color, fontsize=12, fontweight="bold", zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h*0.22, subtitle,
                ha="center", va="center", color=text_color, fontsize=9.5, alpha=0.9, zorder=3)

# Helper for badges (Pathology resolutions)
def draw_badge(ax, x, y, w, h, text, bg_color="#fee2e2", border_color="#ef4444", text_color="#991b1b"):
    draw_card(ax, x, y, w, h, bg_color, border_color, lw=1.2, radius=0.15, zorder=4)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=text_color,
            fontsize=8.5, fontweight="bold", zorder=5)

# -------------------------------------------------------------
# STAGE 1: Natural Language Query & Hybrid Parsing
# -------------------------------------------------------------
s1_x, s1_w = 0.5, 3.4
draw_card(ax, s1_x, 0.5, s1_w, 7.5, "#ffffff", c_query, lw=2.0)
draw_header(ax, s1_x, 7.2, s1_w, 0.8, "Stage 1: Query & Parsing", c_query, subtitle="Hybrid Intent Decomposition")

# User Query Box
draw_card(ax, s1_x + 0.2, 5.2, s1_w - 0.4, 1.7, "#f8fafc", "#cbd5e1", radius=0.15)
ax.text(s1_x + 0.35, 6.6, "Operational Query Input:", fontsize=9.5, fontweight="bold", color="#334155")
query_quote = '"Retrieve extreme storm collisions at\n USACE FRF Duck, NC in Oct 2012\n with severe kinetic energy excess\n and deepwater wave steepness $s_0 \\geq 0.035$"'
ax.text(s1_x + 0.35, 5.9, query_quote, fontsize=8.5, color="#1e293b", style="italic")

# Parser Box
draw_card(ax, s1_x + 0.2, 1.8, s1_w - 0.4, 3.0, "#eef2ff", c_parser, radius=0.15)
ax.text(s1_x + s1_w/2, 4.45, "Hybrid Query Parser", ha="center", fontsize=10.5, fontweight="bold", color=c_parser)
ax.text(s1_x + s1_w/2, 4.15, "(Zero-Shot Dual Intent Mapping)", ha="center", fontsize=8.5, color="#475569")

# Decoupled outputs
# Path A: Relational
draw_card(ax, s1_x + 0.35, 2.9, s1_w - 0.7, 1.0, "#ffffff", "#38bdf8", radius=0.1)
ax.text(s1_x + 0.45, 3.65, "Path A: Columnar SQL Predicates", fontsize=8.5, fontweight="bold", color="#0369a1")
ax.text(s1_x + 0.45, 3.2, "• site_id = 'duck_nc'\n• 2012-10-01 ≤ t ≤ 2012-10-31\n• hs_m ≥ 3.0 m, steepness ≥ 0.035", fontsize=7.5, color="#334155")

# Path B: Physical Target
draw_card(ax, s1_x + 0.35, 1.95, s1_w - 0.7, 0.85, "#ffffff", "#f43f5e", radius=0.1)
ax.text(s1_x + 0.45, 2.55, "Path B: 14D Invariant Target", fontsize=8.5, fontweight="bold", color="#be123c")
ax.text(s1_x + 0.45, 2.15, "• $\\mathbf{z}_{\\mathrm{target}} = [\\Omega, s_0, \\xi, \\mathrm{TWL}/Z_c, \\Delta\\Upsilon^*, \\dots]^T$\n• Sallenger Collision (Regime 2)", fontsize=7.5, color="#334155")

draw_badge(ax, s1_x + 0.2, 0.75, s1_w - 0.4, 0.75, "Decouples Semantic Tokens\nfrom Physical Similitude", "#ede9fe", "#7c3aed", "#5b21b6")


# -------------------------------------------------------------
# STAGE 2: Tier 1 Columnar Relational Pre-Filtering (DuckDB)
# -------------------------------------------------------------
s2_x, s2_w = 4.3, 3.4
draw_card(ax, s2_x, 0.5, s2_w, 7.5, "#ffffff", c_tier1, lw=2.0)
draw_header(ax, s2_x, 7.2, s2_w, 0.8, "Stage 2: Tier 1 SQL Pre-Filter", c_tier1, subtitle="Columnar Predicate Pushdown")

# Parquet Storage Icon & Details
draw_card(ax, s2_x + 0.2, 5.2, s2_w - 0.4, 1.7, c_tier1_box, "#0284c7", radius=0.15)
ax.text(s2_x + s2_w/2, 6.6, "Master Sensor Database", ha="center", fontsize=10, fontweight="bold", color="#0369a1")
ax.text(s2_x + s2_w/2, 6.25, "175,392 Contiguous Hourly Records", ha="center", fontsize=8.5, color="#0c4a6e")
ax.text(s2_x + s2_w/2, 5.85, "Zero-Copy Apache Parquet Storage\n(Duck, Narrabeen, Torrey Pines, Truc Vert)", ha="center", fontsize=8, color="#334155")
ax.text(s2_x + s2_w/2, 5.4, "5-Year In-Situ Validation Archive (2012–2016)", ha="center", fontsize=7.5, color="#0369a1", style="italic")

# DuckDB Engine Card
draw_card(ax, s2_x + 0.2, 2.3, s2_w - 0.4, 2.6, "#ffffff", "#0284c7", radius=0.15)
ax.text(s2_x + s2_w/2, 4.55, "DuckDB Vectorized Engine", ha="center", fontsize=10.5, fontweight="bold", color="#0369a1")
ax.text(s2_x + s2_w/2, 4.25, "Sub-15ms Columnar Memory Scans", ha="center", fontsize=8.5, color="#0284c7", fontweight="bold")

code_text = "SELECT * FROM master_parquet\nWHERE site_id = 'duck_nc'\n  AND timestamp BETWEEN ...\n  AND hs_m >= 3.0\n  AND sallenger_regime >= 2"
draw_card(ax, s2_x + 0.35, 2.7, s2_w - 0.7, 1.35, "#f1f5f9", "#cbd5e1", radius=0.1)
ax.text(s2_x + 0.5, 3.4, code_text, fontsize=7.2, fontfamily="monospace", color="#0f172a")

ax.text(s2_x + s2_w/2, 2.45, "Pruned Candidate Pool: $\\mathcal{C}_{\\mathrm{filtered}} \\ll \\mathcal{D}_{\\mathrm{master}}$", ha="center", fontsize=8.5, fontweight="bold", color="#0369a1")

# Pathology Badge
draw_badge(ax, s2_x + 0.2, 0.75, s2_w - 0.4, 1.25, "PATHOLOGY 1 RESOLVED:\nEliminates Cross-Decadal\nManifold Degeneracy\n(Anchors Calendar & Spatial Epochs)", "#dbeafe", "#2563eb", "#1e40af")


# -------------------------------------------------------------
# STAGE 3: Tier 2 14D Invariant Manifold & Pareto Ranking
# -------------------------------------------------------------
s3_x, s3_w = 8.1, 3.6
draw_card(ax, s3_x, 0.5, s3_w, 7.5, "#ffffff", c_tier2, lw=2.0)
draw_header(ax, s3_x, 7.2, s3_w, 0.8, "Stage 3: Tier 2 Pareto Ranking", c_tier2, subtitle="14D Non-Euclidean Manifold")

# 14D Space Card
draw_card(ax, s3_x + 0.2, 5.0, s3_w - 0.4, 1.9, c_tier2_box, "#db2777", radius=0.15)
ax.text(s3_x + s3_w/2, 6.6, "14D Dimensionless Manifold ($\\mathbf{z}$)", ha="center", fontsize=10, fontweight="bold", color="#9d174d")
dim_text = "• Dean Number $\\Omega = H_s / (w_s T_p)$\n• Wave Steepness $s_0 = 2\\pi H_s / (g T_p^2)$\n• Surf Similarity $\\xi = \\tan\\beta / \\sqrt{s_0}$\n• Water Level Ratio $\\mathrm{TWL} / Z_{\\mathrm{crest}}$\n• Disequilibrium $\\Delta\\Upsilon^* = (\\Upsilon - \\Upsilon_{\\mathrm{eq}}) / \\Upsilon_{\\mathrm{crit}}$\n• Radiation Stresses $S_{xx}, S_{xy} / (\\rho g H_s^2)$"
ax.text(s3_x + 0.4, 5.75, dim_text, fontsize=7.5, color="#334155")

# Directional Pareto Card
draw_card(ax, s3_x + 0.2, 2.3, s3_w - 0.4, 2.45, "#ffffff", "#db2777", radius=0.15)
ax.text(s3_x + s3_w/2, 4.45, "Directional Pareto Objective", ha="center", fontsize=10.5, fontweight="bold", color="#9d174d")
pareto_eq = "$\\mathcal{R}(\\mathbf{z}_i) = (1 - \\alpha) S_C(\\mathbf{z}_i, \\mathbf{z}_{\\mathrm{tgt}})$\n$+\\, \\alpha \\operatorname{ReLU}\\left(\\frac{H_{s,i} - H_{s,\\mathrm{tgt}}}{H_{s,\\mathrm{tgt}}}\\right) + \\beta \\max(0, \\Delta\\Upsilon_i^*)$"
draw_card(ax, s3_x + 0.35, 3.25, s3_w - 0.7, 1.05, "#fff1f2", "#fecdd3", radius=0.1)
ax.text(s3_x + s3_w/2, 3.75, pareto_eq, ha="center", fontsize=8.2, color="#881337")

ax.text(s3_x + s3_w/2, 2.9, "• Penalizes Sub-Threshold Sea States\n• Rewards Extreme Kinetic Excess ($\\Delta\\Upsilon^*$)\n• Locks onto Peak Storm Work Apexes ($t^*$)", ha="center", fontsize=7.8, color="#334155")
ax.text(s3_x + s3_w/2, 2.45, "Hydro-Apex Sync: $P_{\\mathrm{apex}}@5 = 0.12$ (0–14h Error)", ha="center", fontsize=8.2, fontweight="bold", color="#9d174d")

# Pathology Badge
draw_badge(ax, s3_x + 0.2, 0.75, s3_w - 0.4, 1.25, "PATHOLOGY 2 RESOLVED:\nEliminates Chronological Blindness\n& Offshore Over-Retrieval\n(Captures Destructive Nearshore Work)", "#fce7f3", "#ec4899", "#9d174d")


# -------------------------------------------------------------
# STAGE 4: Continuous Dynamical Sequence Hydration (±72h)
# -------------------------------------------------------------
s4_x, s4_w = 12.1, 3.6
draw_card(ax, s4_x, 0.5, s4_w, 7.5, "#ffffff", c_hydrate, lw=2.0)
draw_header(ax, s4_x, 7.2, s4_w, 0.8, "Stage 4: Sequence Hydration", c_hydrate, subtitle="Zero-Copy Pointer Retrieval")

# Hydration Card
draw_card(ax, s4_x + 0.2, 4.4, s4_w - 0.4, 2.5, c_hydrate_box, "#059669", radius=0.15)
ax.text(s4_x + s4_w/2, 6.6, "±72-Hour Continuous Parent Tensor", ha="center", fontsize=10, fontweight="bold", color="#065f46")
ax.text(s4_x + s4_w/2, 6.25, "$\\mathbf{X}_{\\mathrm{parent}} \\in \\mathbb{R}^{145 \\times 14}$ centered on $t^*$", ha="center", fontsize=9.5, fontweight="bold", color="#047857")

hyd_details = "• Antecedent Impulse Integration:\n  $E_{\\mathrm{impulse}} = \\int_{t_0-72\\mathrm{h}}^{t_0} \\max(0, \\Upsilon - \\Upsilon_{\\mathrm{crit}})\\,dt$\n• $26.5\\times$ Exponential Kinetic Amplification:\n  $\\Upsilon: 6{,}500 \\to 172{,}275\\ \\mathrm{kg/s}$\n• Total Water Level Confrontation:\n  $\\mathrm{TWL} = \\eta_{\\mathrm{tide}} + S_{\\mathrm{surge}} + R_{2\\%} > Z_{\\mathrm{crest}}$"
ax.text(s4_x + 0.35, 5.2, hyd_details, fontsize=7.8, color="#064e3b")

# Mini Wave Diagram inside Card
draw_card(ax, s4_x + 0.35, 2.3, s4_w - 0.7, 1.85, "#ffffff", "#10b981", radius=0.1)
ax.text(s4_x + s4_w/2, 3.85, "Continuous Multi-Channel Waveforms", ha="center", fontsize=8.5, fontweight="bold", color="#047857")

# Draw mini synthetic hydrograph
t_pts = np.linspace(s4_x + 0.55, s4_x + s4_w - 0.55, 60)
y_hs = 2.7 + 0.8 * np.exp(-((t_pts - (s4_x + s4_w/2))**2) / 0.3)
y_ups = 2.6 + 1.0 * np.exp(-((t_pts - (s4_x + s4_w/2))**2) / 0.15)
ax.plot(t_pts, y_hs, color="#0284c7", lw=1.8, label="$H_s$ (m)")
ax.plot(t_pts, y_ups, color="#e11d48", lw=1.8, ls="--", label="$\\Upsilon$ (kg/s)")
# Confine storm apex line strictly within the mini-waveform card (ymin=2.55, ymax=3.75)
ax.plot([s4_x + s4_w/2, s4_x + s4_w/2], [2.55, 3.75], color="crimson", ls=":", lw=1.6, zorder=5)
ax.text(s4_x + s4_w/2, 2.4, "Storm Apex ($t^* = 0$)", ha="center", fontsize=7.5, fontweight="bold", color="crimson")
ax.legend(loc="upper right", bbox_to_anchor=(s4_x + s4_w - 0.4, 3.8), bbox_transform=ax.transData, fontsize=7, frameon=False)

# Pathology Badge
draw_badge(ax, s4_x + 0.2, 0.75, s4_w - 0.4, 1.25, "PATHOLOGY 3 RESOLVED:\nEliminates Markovian Memorylessness\n(Supplies Antecedent Wave Impulse\n& Berm Scarping History)", "#d1fae5", "#10b981", "#065f46")


# -------------------------------------------------------------
# STAGE 5: Downstream Diagnostic & SciML Consumers
# -------------------------------------------------------------
s5_x, s5_w = 16.1, 3.4
draw_card(ax, s5_x, 0.5, s5_w, 7.5, "#ffffff", c_sciml, lw=2.0)
draw_header(ax, s5_x, 7.2, s5_w, 0.8, "Stage 5: SciML & Diagnostics", c_sciml, subtitle="Autonomous Proving Ground Consumers")

# Consumer 1: SciML Storm Rollouts
draw_card(ax, s5_x + 0.2, 4.0, s5_w - 0.4, 2.95, c_sciml_box, "#475569", radius=0.15)
ax.text(s5_x + 0.35, 6.72, "1. Autonomous SciML Rollouts", va="top", fontsize=9.8, fontweight="bold", color="#1e293b")
ax.text(s5_x + 0.35, 6.38, "(72-Hour Continuous Rollouts, Table III)", va="top", fontsize=8.2, color="#475569", style="italic")
c1_text = (
    "• Model B (Loss-Guided PINN):\n"
    "  Stabilizes acute shock rollouts;\n"
    "  suppresses false storm accretion\n"
    "  (Storm RMSE = 0.342 m, BSS = +0.108)\n"
    "• Model C (Feature-Guided Invariants):\n"
    "  Dominates multi-year representation\n"
    "  tracking across global observatories\n"
    "  (Mean Test r = 0.574, up to 0.839)\n"
    "• Model D (Dual-Guided Formulation):\n"
    "  Combines representations for minimal\n"
    "  multi-site test error (RMSE = 0.509 m)"
)
ax.text(s5_x + 0.35, 5.98, c1_text, va="top", fontsize=7.6, color="#334155", linespacing=1.26)

# Consumer 2: Real-World Field Anomaly Audits
draw_card(ax, s5_x + 0.2, 0.75, s5_w - 0.4, 3.05, c_sciml_box, "#475569", radius=0.15)
ax.text(s5_x + 0.35, 3.60, "2. Real-World Failure Audits", va="top", fontsize=9.8, fontweight="bold", color="#1e293b")
ax.text(s5_x + 0.35, 3.25, "(Sequence Hydration Post-Mortems, Table IV)", va="top", fontsize=8.2, color="#475569", style="italic")
c2_text = (
    "• Torrey Pines, CA (Case 1):\n"
    "  Infragravity resonance & surf beat\n"
    "  ($T_p = 16.0\\,\\mathrm{s}, s_0 = 0.0052, \\Delta X = -9.80\\,\\mathrm{m}$)\n"
    "• Narrabeen-Collaroy, AUS (Case 2):\n"
    "  2D Embayment rotation & oblique\n"
    "  radiation stress ($S_{xy} > 120\\,\\mathrm{N/m}$)\n"
    "• USACE FRF Duck, NC (Case 3):\n"
    "  Antecedent berm stripping ($E_{\\mathrm{impulse}}$)\n"
    "  leaves dune scarp unbuffered\n"
    "• Truc Vert, France (Case 4):\n"
    "  Multi-year sediment sequestration\n"
    "  in deep offshore sandbars"
)
ax.text(s5_x + 0.35, 2.85, c2_text, va="top", fontsize=7.6, color="#334155", linespacing=1.26)


# -------------------------------------------------------------
# Inter-Stage Connecting Arrows & Dataflow Annotations
# -------------------------------------------------------------
arrow_kw = dict(arrowstyle="->,head_width=0.45,head_length=0.55", lw=2.5, color="#475569")
accent_arrow_kw = dict(arrowstyle="->,head_width=0.45,head_length=0.55", lw=2.5, color="#0284c7")

# Stage 1 to Stage 2: SQL Predicates (Top Arrow)
ax.annotate("", xy=(s2_x, 3.4), xytext=(s1_x + s1_w, 3.4),
            arrowprops=accent_arrow_kw, zorder=6)
ax.text((s1_x + s1_w + s2_x)/2, 3.65, "SQL Predicates\n(Metadata bounds)", ha="center", fontsize=8.0, fontweight="bold", color="#0369a1")

# Stage 1 to Stage 3: 14D Target Vector (Curved bottom bypass)
ax.annotate("", xy=(s3_x, 2.5), xytext=(s1_x + s1_w, 2.3),
            arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.55", lw=2.2, color="#be123c",
                            connectionstyle="arc3,rad=-0.15"), zorder=6)
ax.text((s1_x + s1_w + s3_x)/2, 1.45, "Target State $\\mathbf{z}_{\\mathrm{target}} \\in \\mathbb{R}^{14}$", ha="center", fontsize=8.0, fontweight="bold", color="#be123c")

# Stage 2 to Stage 3: Candidate Pool
ax.annotate("", xy=(s3_x, 4.4), xytext=(s2_x + s2_w, 4.4),
            arrowprops=arrow_kw, zorder=6)
ax.text((s2_x + s2_w + s3_x)/2, 4.65, "Candidate Records\n$\\mathcal{C}_{\\mathrm{filtered}}$", ha="center", fontsize=8.0, fontweight="bold", color="#334155")

# Stage 3 to Stage 4: Top-k Apex Timestamps
ax.annotate("", xy=(s4_x, 4.4), xytext=(s3_x + s3_w, 4.4),
            arrowprops=arrow_kw, zorder=6)
ax.text((s3_x + s3_w + s4_x)/2, 4.65, "Top-$k$ Apex\nTimestamps ($t^*$)", ha="center", fontsize=8.0, fontweight="bold", color="#334155")

# Stage 4 to Stage 5: Hydrated Matrix [145 x 14]
ax.annotate("", xy=(s5_x, 4.4), xytext=(s4_x + s4_w, 4.4),
            arrowprops=arrow_kw, zorder=6)
ax.text((s4_x + s4_w + s5_x)/2, 4.65, "Hydrated Parent\nMatrix [145 × 14]", ha="center", fontsize=8.0, fontweight="bold", color="#047857")

# Save publication figure
plt.tight_layout()
out_path = "reports/figures/figure_physics_rag_architecture.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✓ Saved high-resolution architecture diagram to {out_path}")
