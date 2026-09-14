"""
build_notebooklm_bundle.py

Regenerates all notebook markdown files from latest notebook state and
creates comprehensive research-grade markdown files for relevant Python modules,
packaging everything into a single, clean directory for Google NotebookLM ingestion:
`notebooklm_bundle/`.
"""

import os
import sys
import json
import zipfile

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "notebooklm_bundle")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Workspace: {WORKSPACE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# Helper: Extract Notebook Markdown
# =============================================================================
def extract_notebook_markdown(nb_path, title, summary_cell_idx=None):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    doc_lines = []
    doc_lines.append(f"# {title}\n\n")
    doc_lines.append(f"> **Source Document**: [`{os.path.basename(nb_path)}`](file:///{nb_path.replace(' ', '%20')})\n")
    doc_lines.append("> **Ingestion Target**: Google NotebookLM & Scientific Research Reference\n\n")
    doc_lines.append("---\n\n")

    # If there is an executive synthesis cell, extract and highlight it at top
    if summary_cell_idx is not None and summary_cell_idx < len(nb["cells"]):
        summary_cell = nb["cells"][summary_cell_idx]
        summary_src = "".join(summary_cell.get("source", [])).strip()
        if summary_src:
            doc_lines.append(summary_src)
            doc_lines.append("\n\n---\n\n")

    doc_lines.append("## Appendix: Complete Cell-by-Cell Execution Trajectory & Technical Methodology\n\n")

    for i, cell in enumerate(nb["cells"]):
        if summary_cell_idx is not None and i == summary_cell_idx:
            continue

        c_type = cell["cell_type"]
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue

        if c_type == "markdown":
            doc_lines.append(f"### Cell {i} (Markdown Documentation)\n\n{src}\n\n")
        elif c_type == "code":
            first_line = src.split("\n")[0]
            doc_lines.append(f"#### Cell {i} (Code Execution): `{first_line[:80]}`\n\n")
            doc_lines.append("```python\n" + src + "\n```\n\n")

            # Collect outputs
            outputs = cell.get("outputs", [])
            for out in outputs:
                out_type = out.get("output_type", "")
                if out_type == "stream":
                    stream_text = "".join(out.get("text", [])).strip()
                    if stream_text:
                        truncated = stream_text[:3000] + ("\n... [Output Truncated for Brevity]" if len(stream_text) > 3000 else "")
                        doc_lines.append(f"**Execution Log / Output**:\n```text\n{truncated}\n```\n\n")
                elif "text/plain" in out.get("data", {}):
                    txt = "".join(out["data"]["text/plain"]).strip()
                    if txt and not txt.startswith("<Figure") and not txt.startswith("<matplotlib"):
                        truncated = txt[:2500] + ("\n... [Output Truncated for Brevity]" if len(txt) > 2500 else "")
                        doc_lines.append(f"**Result Data**:\n```text\n{truncated}\n```\n\n")
                elif "image/png" in out.get("data", {}):
                    doc_lines.append("*[Visual Matplotlib Figure Generated in Execution]*\n\n")

    return "".join(doc_lines)


# =============================================================================
# 1. Regenerate Notebook Markdown Files
# =============================================================================
notebook_specs = [
    {
        "filename": "01_Unified_Feature_Engineering.ipynb",
        "title": "Notebook 01: Unified Coastal Morphodynamics Feature Engineering Pipeline",
        "summary_idx": 23,
        "dest_root": "01_Unified_Feature_Engineering.md",
        "dest_bundle": "01_Unified_Feature_Engineering.md",
    },
    {
        "filename": "02_Diagnostic_Baselines.ipynb",
        "title": "Notebook 02: Diagnostic Morphodynamic Baselines & Cross-Paradigm Benchmarking",
        "summary_idx": 32,
        "dest_root": "02_Diagnostic_Baselines.md",
        "dest_bundle": "02_Diagnostic_Baselines.md",
    },
    {
        "filename": "03_SciML_Architectural_Ablation.ipynb",
        "title": "Notebook 03: SciML Architectural Ablation (Observing System Simulation Experiment)",
        "summary_idx": 27,
        "dest_root": "03_SciML_Architectural_Ablation.md",
        "dest_bundle": "03_SciML_Architectural_Ablation.md",
    },
    {
        "filename": "04_Physics_RAG_Deployment.ipynb",
        "title": "Notebook 04: Unified Global Physics-RAG Diagnostic Engine",
        "summary_idx": 23,
        "dest_root": "04_Physics_RAG_Deployment.md",
        "dest_bundle": "04_Physics_RAG_Deployment.md",
    },
    {
        "filename": "unified_coastal_vulnerability_index.ipynb",
        "title": "Notebook 00: Unified Physics-Informed Coastal Vulnerability Index (Proof of Concept & Foundation)",
        "summary_idx": None,
        "dest_root": None,
        "dest_bundle": "00_Unified_Coastal_Vulnerability_Index_Overview.md",
    },
]

print("\n--- Regenerating Notebook Markdown Files ---")
for spec in notebook_specs:
    nb_path = os.path.join(WORKSPACE_DIR, spec["filename"])
    if not os.path.exists(nb_path):
        print(f"Warning: {nb_path} does not exist. Skipping.")
        continue

    print(f"Processing {spec['filename']}...")
    md_content = extract_notebook_markdown(nb_path, spec["title"], spec["summary_idx"])

    # Write to root if specified
    if spec["dest_root"]:
        root_path = os.path.join(WORKSPACE_DIR, spec["dest_root"])
        with open(root_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  -> Updated root: {spec['dest_root']}")

    # Write to bundle
    bundle_path = os.path.join(OUTPUT_DIR, spec["dest_bundle"])
    with open(bundle_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  -> Written to bundle: {spec['dest_bundle']} ({len(md_content)} chars)")


# =============================================================================
# 2. Generate Research Markdown Files for Relevant Python Modules
# =============================================================================

python_module_specs = [
    {
        "source_rel": "src/physics.py",
        "bundle_fname": "python_physics_engine.md",
        "title": "Core Physics Engine & Coastal Morphodynamic Formulations (`src/physics.py`)",
        "role": "Foundational first-principles coastal physics library containing the Unified Coastal Vulnerability Index ($\\Upsilon$), dimensionless disequilibrium ($\\Delta\\Upsilon^*$), dynamic equilibrium shoreline ODEs, radiation stress tensors, wave runup, and multi-scale antecedent memory convolutions.",
        "theoretical_sections": """## Theoretical & Physical Foundations

### 1. Master Continuous Kinetic Formulation ($\\Upsilon$)
Notebook 01 and `src/physics.py` establish $\\Upsilon$ as a continuous kinetic mass-flux rate in SI units of **$[\\text{kg}/\\text{s}]$**:
$$\\Upsilon = \\left(\\frac{\\rho g^{3/2}}{16(2\\pi)^{3/2}}\\right) \\cdot \\tan\\beta \\cdot \\frac{H_{\\text{shore}}^{2.5} T_p}{w_s(D_{50}, T)} \\cdot \\left[ 1 + \\frac{\\max(S, 0)}{s_{\\text{ref}}} + \\frac{\\max(R_{2\\%}, 0)}{r_{\\text{ref}}} \\right] \\cdot \\Psi_{\\text{tide}} \\cdot \\Psi_{\\text{phase}} \\cdot \\Psi_{\\text{dir}} \\cdot \\Psi_{\\text{wind}}$$

### 2. The Constructive 4-Tier Progression
1. **Tier 1 (Base Core Flux)**: Evaluates deepwater wave energy flux transformed to nearshore breaking conditions, scaled by foreshore slope $\\tan\\beta$ and normalized by sediment settling velocity $w_s(D_{50}, T)$.
2. **Tier 2 (+ Infragravity / TWL Amplification)**: Modulates forcing via dynamic storm surge $S$ and Stockdon et al. (2006) $R_{2\\%}$ infragravity wave runup, capturing water level elevation above the berm crest.
3. **Tier 3 (+ Radiation Stress Tensor & Longshore Divergence)**: Incorporates the Longuet-Higgins & Stewart (1964) radiation stress tensor components ($S_{xx}, S_{yy}, S_{xy}$) and alongshore transport gradients.
4. **Tier 4 (+ Directional Collimation & Wave Steepness)**: Modulates forcing via deepwater wave steepness $s_0 = H_s / L_0$ (critical threshold $s_{0,\\text{crit}} = 0.025$ separating swell accretion from storm erosive cutbacks) and shore-normal wind stress projection.

### 3. Scale-Invariant Relative Forcing Index ($\\Upsilon^*$)
To eliminate orders-of-magnitude site-specific scaling disparities across contrasting microtidal and macrotidal environments, the dimensionless index is defined as:
$$\\Upsilon^* = \\frac{\\Upsilon}{\\Upsilon_{\\text{ref}}}, \\quad \\Upsilon_{\\text{ref}} = \\text{median}(\\Upsilon_{\\text{annual}})$$
This maps all global sites onto a scale-invariant distribution centered near $\\mathcal{O}(1)$, with catastrophic storms cleanly identifiable at $\\Upsilon^* \\gg 10$.

### 4. Dynamic Shoreline Evolution Models & Boundary Damping
`src/physics.py` provides multiple continuous 1D shoreline evolution models:
1. **Yates et al. (2009) Wave Energy Disequilibrium**:
   $$\\frac{dx}{dt} = C^{\\pm} E^{1/2} (E - E_{\\text{eq}}(x)), \\quad E_{\\text{eq}}(x) = a x + b$$
2. **Signed Equilibrium $\\Upsilon$ Formulation**:
   $$\\frac{dx}{dt} = -k \\cdot \\text{sign}(\\Delta \\Upsilon_\\tau) \\cdot |\\Delta \\Upsilon_\\tau|^p$$
3. **Dimensionless Normalized Equilibrium Upsilon ODE (`simulate_dimensionless_normalized_upsilon_evolution`)**:
   $$\\frac{dx}{dt} = -k \\cdot \\text{sign}(\\Delta \\Upsilon^*_\\tau) \\cdot |\\Delta \\Upsilon^*_\\tau|^p$$
4. **Four Boundary Damping Modes (`boundary_damping`)**:
   - `'hard'`: Rigid box clipping at $[x_{\\min}, x_{\\max}]$ (produces artificial horizontal saturation plateaus).
   - `'asymptotic'`: Hyperbolic tangent $\\tanh$ taper:
     $$\\text{factor} = \\begin{cases} \\tanh\\left(\\frac{x_{\\max} - x}{\\delta_{\\text{accretion}}}\\right), & \\dot{x} > 0 \\\\ \\tanh\\left(\\frac{x - x_{\\min}}{\\delta_{\\text{erosion}}}\\right), & \\dot{x} < 0 \\end{cases}$$
     with $\\delta_{\\text{accretion}} = 5.0\\text{ m}$ (berm accommodation limit) and $\\delta_{\\text{erosion}} = 8.0\\text{ m}$ (scarping resistance limit). Completely eliminates flat-top saturation artifacts while maintaining sensitivity to high-frequency storm cuts!
   - `'exponential'`: Soft exponential repulsion as shoreline approaches limits.
   - `None`: Free, unconstrained numerical integration.
5. **Davidson et al. (2013) ShoreFor Model**: Memory kernel with exponential response time.
6. **Miller & Dean (2004) Model**: Analytical equilibrium beach profile approach.

### 5. Multi-Site Climatological Archetypes (`COASTAL_SITE_PROFILES`)
Contains verified physical parameters for four global benchmark observatories:
- **USACE FRF Duck, NC, USA**: $\\tan\\beta = 0.080, D_{50} = 0.35\\text{ mm}, TR = 1.0\\text{ m}, Z_{\\text{berm}} = 2.4\\text{ m}, \\Upsilon_{\\text{ref}} \\approx 1,964\\text{ kg/s}$.
- **Narrabeen-Collaroy, NSW, Australia**: $\\tan\\beta = 0.085, D_{50} = 0.30\\text{ mm}, TR = 1.2\\text{ m}, Z_{\\text{berm}} = 3.2\\text{ m}, \\Upsilon_{\\text{ref}} \\approx 5,783\\text{ kg/s}$.
- **Torrey Pines State Beach, CA, USA**: $\\tan\\beta = 0.040, D_{50} = 0.22\\text{ mm}, TR = 1.5\\text{ m}, Z_{\\text{berm}} = 2.0\\text{ m}, \\Upsilon_{\\text{ref}} \\approx 3,374\\text{ kg/s}$.
- **Truc Vert Beach, Gironde, France**: $\\tan\\beta = 0.065, D_{50} = 0.40\\text{ mm}, TR = 4.5\\text{ m}, Z_{\\text{berm}} = 3.8\\text{ m}, \\Upsilon_{\\text{ref}} \\approx 6,874\\text{ kg/s}$.
""",
    },
    {
        "source_rel": "src/physics_rag.py",
        "bundle_fname": "python_physics_rag.md",
        "title": "Two-Tier Hybrid Cascade Physics-RAG Diagnostic Engine (`src/physics_rag.py`)",
        "role": "Production search, retrieval, and failure diagnosis engine coupling DuckDB relational SQL pre-filtering with a 14D invariant manifold vector search and parent sequence time-series hydration.",
        "theoretical_sections": """## Theoretical & Engineering Foundations

### 1. The Three Classical Search Pathologies in Earth Observation
1. **Baseline 1 (Raw-Data LLM-RAG)**: Standard dense vector embeddings on raw telemetry ($H_s, T_p, \\text{Surge}$). Lacks hydrodynamic domain grounding, confuses pre-storm onset hours with peak destruction, and suffers from ungrounded numerical hallucinations.
2. **Baseline 2 (Rule-Based SQL)**: Metadata filtering (`WHERE site = 'duck_nc' AND hs_m > 3.0`). While precise for explicit constraints, SQL is **chronologically blind**: standard sorting/limiting retrieves calm early hours at storm onset rather than the peak erosive apex.
3. **Baseline 3 (Pure Physics-RAG)**: 14D manifold vector search without relational filtering suffers from **spatial and temporal degeneracy**, unable to distinguish identical wave states across disparate geographic observatories without metadata slicing.

### 2. The Two-Tier Hybrid Cascade Architecture
- **Tier 1 (DuckDB Relational Pre-Filtering)**: Fast columnar SQL evaluation over $175,392$ hourly records in $< 15\\text{ ms}$, filtering by site, date ranges, and hydrodynamic bounds.
- **Parent Sequence Hydration**: Instantly extracts continuous $\\pm 72$-hour multi-channel arrays around retrieved timestamps for dynamic SciML rollout and morphological failure diagnosis.
- **Tier 2 (14D Invariant Manifold Search & Disequilibrium Re-Ranking)**: Computes cosine similarity over dimensionless hydrodynamic states and applies direction-aware disequilibrium scoring:
  $$\\mathcal{S}_{\\text{hybrid}} = \\cos(\\mathbf{z}_q, \\mathbf{z}_d) \\cdot \\left[ 1 + \\alpha \\cdot \\text{sign}(\\Delta\\Upsilon) \\cdot \\frac{|\\Delta\\Upsilon|}{\\Upsilon_{\\text{ref}}} \\right]$$

### 3. Rigorous Hypothesis Testing (1-Tailed & 2-Tailed + Bootstrap CIs)
To satisfy computational science peer-review standards, the evaluation suite reports both directional 1-tailed tests, conservative 2-tailed tests, and non-parametric bootstrap 95% confidence intervals ($B = 10,000$ resamples) across $N = 60$ diagnostic benchmark queries:
- **Precision@5**: Hybrid Cascade achieves **$0.7000$**, significantly outperforming Raw LLM-RAG ($0.4667$, $+23.33\\%$, $t = 2.808$, 1-tail $p = 0.0044$, 2-tail $p = 0.0087$, CI $[+0.0634, +0.4032]$) and outperforming SQL ($0.5867$, $+11.33\\%$, $t = 1.628$, 1-tail $p = 0.0571$, 2-tail $p = 0.1143$, CI $[-0.0290, +0.2556]$).
- **MRR (Mean Reciprocal Rank)**: **$0.7261$** vs SQL $0.5833$ ($+14.28\\%$, 1-tail $p = 0.0439$, 2-tail $p = 0.0879$, Wilcoxon 1-tail $p = 0.0308$).
- **NDCG@5**: **$0.7180$** vs SQL $0.6134$ ($+10.46\\%$, 1-tail $p = 0.0491$, 2-tail $p = 0.0982$).
- **Regime Purity**: **$0.7267$** vs LLM $0.5233$ ($+20.34\\%$).
- **Peak Precision@5**: **$0.1200$** (Hybrid Cascade is the **only architecture with non-zero peak precision**, eliminating the $15,000\\text{ h}$ temporal drift of pure vector search).
""",
    },
    {
        "source_rel": "src/data.py",
        "bundle_fname": "python_data_pipeline.md",
        "title": "Institutional Multi-Site Data Harmonization & Survey Alignment (`src/data.py`)",
        "role": "Data ingestion, coordinate transformation, linear wave shoaling/refraction, harmonic astronomical tide extraction, and in-situ survey alignment pipeline.",
        "theoretical_sections": """## Theoretical & Data Engineering Foundations

### 1. The 5-Year Authentic Golden Window (2012–2016)
Consists exclusively of 100% authentic sensor and survey records ($N = 43,848$ contiguous hours per site; $N = 175,392$ total hourly records; $N = 1,228$ in-situ topographic profile surveys):
1. **USACE FRF Duck, NC, USA**: NOAA 8651370 tide gauge, 8m shallow array, 17m waverider, and bi-weekly CRAB/LARC profile surveys.
2. **Narrabeen-Collaroy, Sydney, Australia**: Middle Head tide gauge, Sydney/Wamberal waverider buoys, and RTK-GPS profile surveys along transects PF1–PF8.
3. **Torrey Pines State Beach, CA, USA**: NOAA 9410230 La Jolla tide gauge, CDIP 100 Torrey Pines Outer buoy, and ATV/LIDAR profile surveys.
4. **Truc Vert Beach, Gironde, France**: Arcachon tide gauge, CANDHIS 03301 directional buoy / ERA5 atmospheric reanalysis, and DGPS surveys along TV_NORTH and TV_SOUTH.

### 2. Wave Mechanics & Shoaling Transformations
Implements linear wave theory to transform deepwater wave buoy observations to local nearshore breaker coordinates:
- Wavelength dispersion relation: $L = \\frac{g T^2}{2\\pi} \\tanh(k d)$
- Group velocity: $C_g = n C$
- Shoaling coefficient: $K_s = \\sqrt{C_{g,0} / C_{g}}$
- Snell's Law refraction coefficient: $K_r = \\sqrt{\\cos\\theta_0 / \\cos\\theta}$
- Radiation stress tensor components ($S_{xx}, S_{yy}, S_{xy}$) per Longuet-Higgins & Stewart (1964).
""",
    },
    {
        "source_rel": "src/models.py",
        "bundle_fname": "python_sciml_models.md",
        "title": "Scientific Machine Learning (SciML) Architectures & Bayesian Inference (`src/models.py`)",
        "role": "SciML neural network paradigms, physics-informed disequilibrium loss functions, Bayesian NUTS MCMC latent memory kernels, and feature attribution.",
        "theoretical_sections": """## Theoretical & Modeling Foundations

### 1. The 2×2 SciML Architectural Matrix
1. **Model A (Pure Data-Driven)**: Standard multi-layer perceptron (MLP) mapping raw hydrodynamics ($H_s, T_p, \\text{Surge}, W_a$) directly to cross-shore displacement.
2. **Model B (Loss-Guided / PINN)**: Ingests raw hydrodynamics but optimizes an asymmetric soft physics penalty:
   $$\\mathcal{L}_{\\text{PINN}} = \\mathcal{L}_{\\text{MSE}} + \\lambda_{\\text{physics}} \\cdot \\mathbb{E}\\left[ \\max(0, \\dot{x}_{\\text{pred}})^2 \\cdot \\mathbb{I}_{\\Upsilon > \\Upsilon_{\\text{p80}}} \\right]$$
   penalizing unphysical seaward accretion during high-energy storm events. Achieves best-in-class storm RMSE ($0.354\\text{ m}$) and Storm BSS ($-0.147$).
3. **Model C (Feature-Guided)**: Maps inputs through $\\Upsilon$'s invariant dimensionless manifold (core wave-sediment flux, $\\Psi$ modulators, and multi-scale temporal memory). Delivers superior long-horizon out-of-sample correlation ($r = 0.570$).
4. **Model D (Dual-Guided / Hybrid Residual)**: Unites the 14D feature manifold projection with PINN loss regularization, achieving optimal balance across seasonal cycles and extreme storms.

### 2. Autonomous Multi-Step Storm Rollout Simulation (72-Hour Windows)
Evaluates neural stability without continuous teacher-forcing, measuring cumulative drift and storm-event fidelity over continuous 72-hour storm windows.

### 3. Williams Test for Dependent Correlations
Tests whether the correlation of shoreline displacement with kinetic index $\\Upsilon$ ($r_1$) is statistically significantly higher than with raw wave height $H_s$ ($r_2$), accounting for the mutual collinearity $r_{12}$:
$$t = (r_1 - r_2) \\sqrt{\\frac{(n - 1)(1 + r_{12})}{2 \\frac{n - 1}{n - 3} |R| + \\bar{r}^2 (1 - r_{12})^3}}$$

### 4. Bayesian Latent Memory Models (PyMC NUTS MCMC)
Performs Bayesian posterior parameter estimation over memory relaxation timescales $\\tau$, disequilibrium power $p$, and morphodynamic transport coefficients $k$.
""",
    },
    {
        "source_rel": "src/neural_baselines.py",
        "bundle_fname": "python_neural_baselines.md",
        "title": "Deep Learning Neural Shoreline Baselines (`src/neural_baselines.py`)",
        "role": "PyTorch deep learning baseline implementations including CoastalMLP, CoastalLSTM, and PhysicsGuidedLSTM for shoreline evolution.",
        "theoretical_sections": """## Theoretical & Neural Architecture Foundations

### 1. PyTorch Model Implementations
- `CoastalMLP`: Multi-layer feedforward network with LayerNorm and Dropout for cross-shore displacement regression.
- `CoastalLSTM`: Recurrent neural network capturing temporal sequences and long-term memory.
- `PhysicsGuidedLSTM`: Recurrent architecture integrating physical feature inputs and physics-informed loss penalties.
- `PhysicsGuidedLoss`: PyTorch loss module incorporating asymmetric storm penalty terms.

### 2. Dual-Horizon Evaluation Framework
Evaluates models across both continuous long-term seasonal horizons ($> 1\\text{ year}$) and short-term event horizons ($72\\text{ hours}$ centered on storm apex $t^*$).
""",
    },
    {
        "source_rel": "src/embeddings.py",
        "bundle_fname": "python_embeddings_manifold.md",
        "title": "14D Invariant Feature Manifold & UMAP Projection (`src/embeddings.py`)",
        "role": "Manifold construction, dimensionless normalization, regime classification, and 2D UMAP projection routines.",
        "theoretical_sections": """## Theoretical Foundations

### 1. 14D Invariant Latent Space Formulation
Projects disparate multi-sensor hydrodynamic observations into a scale-invariant 14-dimensional representation:
$$\\mathbf{z} = \\big[H_s,\\; T_p,\\; S,\\; R_{2\\%},\\; \\Upsilon,\\; \\Upsilon^*,\\; \\Omega,\\; s_0,\\; S_{xx},\\; S_{xy},\\; \\text{TWL},\\; \\Delta\\Upsilon^*_{7d},\\; \\Delta\\Upsilon^*_{30d},\\; \\Delta\\Upsilon^*_{90d}\\big]$$

### 2. 2D UMAP Dimensionality Reduction
Proves that global coastal breaking regimes form a **continuous physical spectrum** rather than isolated clusters, with disparate global sites (Duck, Narrabeen, Torrey Pines, Truc Vert) interleaving along a unified manifold governed by wave steepness and sediment mobility.
""",
    },
    {
        "source_rel": "src/acquire_real_surveys.py",
        "bundle_fname": "python_survey_acquisition.md",
        "title": "In-Situ Field Survey Acquisition & Quality Control Pipeline (`src/acquire_real_surveys.py`)",
        "role": "Automated data harvesting and parsing for USACE FRF Duck NetCDF surveys, Narrabeen RTK-GPS surveys, Torrey Pines ATV/LIDAR transects, and Truc Vert DGPS surveys.",
        "theoretical_sections": """## Engineering & Survey Processing Methodology

### 1. In-Situ Observational Ground Truth
- Automated harvesting of USACE Field Research Facility (FRF) cross-shore profile surveys from THREDDS/NetCDF servers.
- Elevation zero-crossing / Mean High Water (MHW) shoreline position extraction ($x_{\\text{MHW}}$) relative to established baseline monuments.
- Quality control filtering: rejecting surveys with profile gaps, unrealistic slopes ($> 0.25$), or datum shifts.
""",
    },
    {
        "source_rel": "scripts/run_parser_ablation.py",
        "bundle_fname": "python_parser_ablation.md",
        "title": "Query Parser Ablation Benchmark (`scripts/run_parser_ablation.py`)",
        "role": "Ablation study evaluating rule-based regex parsing vs hybrid LLM query understanding across 60 multi-modal coastal queries.",
        "theoretical_sections": """## Experimental Methodology & Ablation Results

### 1. Experimental Setup
Evaluates 60 complex scientific queries covering compound storm searches, regime transitions, disequilibrium queries, and multi-sensor failure audits.

### 2. Key Findings
- Rule-based regex achieves $>95\\%$ precision on canonical queries with explicit syntax, executing in $<1\\text{ ms}$.
- Hybrid query understanding resolves semantic ambiguities (e.g. \"extreme nor'easter after beach drawdown\") without introducing latency bottlenecks into Tier 1 relational pre-filtering.
""",
    },
    {
        "source_rel": "scripts/validate_duck_real_surveys.py",
        "bundle_fname": "scripts_validate_duck_surveys.md",
        "title": "Validation of FRF Duck In-Situ Surveys (`scripts/validate_duck_real_surveys.py`)",
        "role": "Rigorous validation script comparing USACE FRF cross-shore profile surveys against NOAA tide gauge water levels, wave buoy records, and historic storm markers.",
        "theoretical_sections": """## Validation Methodology & Quality Control

### 1. Validation Protocol
Cross-references FRF profile survey epochs against NOAA 8651370 tide gauge elevations and 17m waverider records during landmark events (Hurricane Sandy 2012, Winter Nor'easter 2014) to ensure sub-meter positional accuracy.
""",
    },
]

print("\n--- Generating Research Markdown Files for Python Modules ---")
for mod in python_module_specs:
    src_path = os.path.join(WORKSPACE_DIR, mod["source_rel"])
    if not os.path.exists(src_path):
        print(f"Warning: {src_path} does not exist. Skipping.")
        continue

    print(f"Processing {mod['source_rel']} -> {mod['bundle_fname']}...")
    with open(src_path, "r", encoding="utf-8") as f:
        code_text = f.read()

    lines = []
    lines.append(f"# {mod['title']}\n\n")
    lines.append(f"> **Source File**: [`{mod['source_rel']}`](file:///{src_path.replace(' ', '%20')})\n")
    lines.append("> **Ingestion Target**: Google NotebookLM & Research Paper Reference\n\n")
    lines.append("---\n\n")
    lines.append("## Executive Purpose & Role\n\n")
    lines.append(f"{mod['role']}\n\n")
    lines.append("---\n\n")
    lines.append(mod["theoretical_sections"].strip() + "\n\n")
    lines.append("---\n\n")
    lines.append("## Complete Annotated Source Code\n\n")
    lines.append(f"```python\n{code_text}\n```\n")

    full_md = "".join(lines)
    dest_path = os.path.join(OUTPUT_DIR, mod["bundle_fname"])
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"  -> Written to {dest_path} ({len(full_md)} chars)")


# =============================================================================
# 3. Generate Master Index & Guide for NotebookLM
# =============================================================================
print("\n--- Generating Master NotebookLM Research Guide (README) ---")

readme_content = """# Unified Coastal Vulnerability Index ($\\Upsilon$) Research Knowledge Base
> **Ingestion Target**: Google NotebookLM Research Agent & Research Paper Synthesis
> **Repository**: `unified-coastal-erosion-index`
> **Primary Window**: 2012–2016 Multi-Site Golden Window ($N = 175,392$ Contiguous Sensor Hours, $N = 1,228$ Field Surveys)

---

## Overview & Directory Manifest
This directory (`notebooklm_bundle/`) contains the complete scientific corpus of markdown documents generated specifically for ingestion into **Google NotebookLM** or other LLM research agents. It packages executive scientific findings, visual diagnostics transcriptions, mathematical equations in LaTeX, quantitative benchmark tables, and complete annotated Python source code.

### 1. Research Notebooks
| File | Title | Core Contents & Key Scientific Takeaways |
| :--- | :--- | :--- |
| [`01_Unified_Feature_Engineering.md`](01_Unified_Feature_Engineering.md) | **Notebook 01: Feature Engineering Pipeline** | 4-tier constructive physical $\\Upsilon$ progression; exact radiation stress tensor; wave steepness $s_0$; Sallenger storm regimes; 4-scale exponential memory; in-situ survey coupling diagnostic across Duck, Narrabeen, Torrey Pines, Truc Vert. |
| [`02_Diagnostic_Baselines.md`](02_Diagnostic_Baselines.md) | **Notebook 02: Morphodynamic Baselines** | Cross-paradigm benchmark (Yates, ShoreFor, XBeach 1D surrogate, Signed Equilibrium $\\Upsilon$, Dimensionless Normalized $\\Delta\\Upsilon^*$). Identifies empirical survey noise floor ($R^2 \\le 0.023$) and documents the elimination of saturation plateaus via **asymptotic boundary damping**. |
| [`03_SciML_Architectural_Ablation.md`](03_SciML_Architectural_Ablation.md) | **Notebook 03: SciML Architectural Ablation** | Observing System Simulation Experiment (OSSE) evaluating the $2 \\times 2$ SciML matrix (Model A: Data-Driven, Model B: Loss-Guided PINN, Model C: Feature-Guided, Model D: Dual-Guided). Autonomous 72-hour storm rollouts; Bayesian XAI and SHAP kinetic attributions. |
| [`04_Physics_RAG_Deployment.md`](04_Physics_RAG_Deployment.md) | **Notebook 04: Physics-RAG Diagnostic Engine** | Two-Tier Hybrid Cascade Physics-RAG (DuckDB SQL pre-filter + 14D invariant manifold vector search + $\\pm 72$h parent sequence hydration). Precision@5 ($0.700$ vs $0.467$ LLM, $+23.3\\%, p=0.0044$), NDCG@5 ($0.718$), MRR ($0.726$), 1-tailed & 2-tailed tests, and bootstrap 95% CIs. |
| [`00_Unified_Coastal_Vulnerability_Index_Overview.md`](00_Unified_Coastal_Vulnerability_Index_Overview.md) | **Notebook 00: Foundation & Proof of Concept** | Foundational proof of concept notebook illustrating the evolution of the kinetic index and early empirical evaluations. |

### 2. Core Python Architecture Modules
| File | Module | Role & Foundational Implementations |
| :--- | :--- | :--- |
| [`python_physics_engine.md`](python_physics_engine.md) | `src/physics.py` | Closed-form kinetic mass-flux $\\Upsilon$ ($[\\text{kg}/\\text{s}]$), dimensionless forcing $\\Upsilon^*$, dimensionless disequilibrium $\\Delta\\Upsilon^*$, Yates/ShoreFor ODEs, asymptotic/exponential boundary damping, radiation stress, Dean number $\\Omega$, Soulsby settling velocity, `COASTAL_SITE_PROFILES`. |
| [`python_physics_rag.md`](python_physics_rag.md) | `src/physics_rag.py` | Two-Tier Hybrid Cascade engine, DuckDB pre-filtering, $\\pm 72$h sequence hydration, 14D invariant latent manifold, directional disequilibrium scoring, intent classification, 60-query benchmark suite, 1-tailed/2-tailed paired t-tests, bootstrap 95% CIs. |
| [`python_data_pipeline.md`](python_data_pipeline.md) | `src/data.py` | Multi-site data ingestion (Duck, Narrabeen, Torrey Pines, Truc Vert), harmonic astronomical tides, NOAA/NDBC/CANDHIS/CDIP wave shoaling and refraction, 1,228 GPS survey transects, datum conversions. |
| [`python_sciml_models.md`](python_sciml_models.md) | `src/models.py` | $2 \\times 2$ SciML architectures (Data-Driven, PINN with $\\mathcal{L}_{\\text{physics}}$, Feature-Guided, Hybrid Residual), 72h autonomous storm rollouts, Brier Skill Score (BSS), Williams dependent correlation test, PyMC NUTS Bayesian MCMC. |
| [`python_neural_baselines.md`](python_neural_baselines.md) | `src/neural_baselines.py` | PyTorch architectures (`CoastalMLP`, `CoastalLSTM`, `PhysicsGuidedLSTM`, `PhysicsGuidedLoss`), sliding window sequence processing, dual-horizon evaluations. |
| [`python_embeddings_manifold.md`](python_embeddings_manifold.md) | `src/embeddings.py` | 14D invariant latent manifold construction, scale-invariant normalization, regime classification, 2D UMAP projection and silhouette scoring. |
| [`python_survey_acquisition.md`](python_survey_acquisition.md) | `src/acquire_real_surveys.py` | USACE FRF NetCDF harvesting, MHW cross-shore shoreline intercept extraction, RTK-GPS/DGPS survey harmonization. |
| [`python_parser_ablation.md`](scripts_run_parser_ablation.py) | `scripts/run_parser_ablation.py` | Parser ablation study comparing rule-based regex vs hybrid LLM query understanding across 60 scientific queries. |
| [`scripts_validate_duck_surveys.md`](scripts/validate_duck_real_surveys.py) | `scripts/validate_duck_real_surveys.py` | Quality control script validating FRF Duck surveys against NOAA tide gauge water levels and buoy observations. |

---

## Suggested Research Queries for NotebookLM
When interacting with this corpus in Google NotebookLM, try asking:
1. *"What is the exact physical formulation and dimensions of the Unified Coastal Vulnerability Index ($\\Upsilon$), and how do the four constructive tiers modify it?"*
2. *"Why does raw $\\Upsilon$ exhibit negative correlation with shoreline displacement at Narrabeen and Truc Vert, but positive correlation at Torrey Pines and near-zero correlation at Duck?"*
3. *"How does the Two-Tier Hybrid Cascade Physics-RAG eliminate the 15,000-hour temporal drift of standard vector search, and what are its 1-tailed and 2-tailed statistical significance values against SQL and LLM baselines?"*
4. *"What is the Observing System Simulation Experiment (OSSE) design in Notebook 03, and what are the quantitative trade-offs between Model A (Data-Driven), Model B (PINN), Model C (Feature-Guided), and Model D (Dual-Guided) during 72-hour storm rollouts?"*
5. *"How does asymptotic boundary damping in `src/physics.py` prevent unphysical saturation plateaus while preserving dynamic storm cuts?"*
"""

readme_path = os.path.join(OUTPUT_DIR, "README_NotebookLM_Guide.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"  -> Written to {readme_path}")

# =============================================================================
# 4. Create a Convenient Zip Archive of the Entire Bundle
# =============================================================================
zip_path = os.path.join(WORKSPACE_DIR, "notebooklm_bundle.zip")
print(f"\n--- Packaging zip archive: {zip_path} ---")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_fpath = os.path.join(root, file)
            rel_fpath = os.path.relpath(full_fpath, OUTPUT_DIR)
            zipf.write(full_fpath, arcname=rel_fpath)
print(f"Zip archive created successfully ({os.path.getsize(zip_path)} bytes).")
print("\nAll tasks completed successfully!")
