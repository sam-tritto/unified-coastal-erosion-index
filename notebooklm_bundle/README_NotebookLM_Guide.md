# Unified Coastal Vulnerability Index ($\Upsilon$) Research Knowledge Base
> **Ingestion Target**: Google NotebookLM Research Agent & Research Paper Synthesis
> **Repository**: `unified-coastal-erosion-index`
> **Primary Window**: 2012–2016 Multi-Site Golden Window ($N = 175,392$ Contiguous Sensor Hours, $N = 1,228$ Field Surveys)

---

## Overview & Directory Manifest
This directory (`notebooklm_bundle/`) contains the complete scientific corpus of markdown documents generated specifically for ingestion into **Google NotebookLM** or other LLM research agents. It packages executive scientific findings, visual diagnostics transcriptions, mathematical equations in LaTeX, quantitative benchmark tables, and complete annotated Python source code.

### 1. Research Notebooks
| File | Title | Core Contents & Key Scientific Takeaways |
| :--- | :--- | :--- |
| [`01_Unified_Feature_Engineering.md`](01_Unified_Feature_Engineering.md) | **Notebook 01: Feature Engineering Pipeline** | 4-tier constructive physical $\Upsilon$ progression; exact radiation stress tensor; wave steepness $s_0$; Sallenger storm regimes; 4-scale exponential memory; in-situ survey coupling diagnostic across Duck, Narrabeen, Torrey Pines, Truc Vert. |
| [`02_Diagnostic_Baselines.md`](02_Diagnostic_Baselines.md) | **Notebook 02: Morphodynamic Baselines** | Cross-paradigm benchmark (Yates, ShoreFor, XBeach 1D surrogate, Signed Equilibrium $\Upsilon$, Dimensionless Normalized $\Delta\Upsilon^*$). Identifies empirical survey noise floor ($R^2 \le 0.023$) and documents the elimination of saturation plateaus via **asymptotic boundary damping**. |
| [`03_SciML_Architectural_Ablation.md`](03_SciML_Architectural_Ablation.md) | **Notebook 03: SciML Architectural Ablation** | Observing System Simulation Experiment (OSSE) evaluating the $2 \times 2$ SciML matrix (Model A: Data-Driven, Model B: Loss-Guided PINN, Model C: Feature-Guided, Model D: Dual-Guided). Autonomous 72-hour storm rollouts; Bayesian XAI and SHAP kinetic attributions. |
| [`04_Physics_RAG_Deployment.md`](04_Physics_RAG_Deployment.md) | **Notebook 04: Physics-RAG Diagnostic Engine** | Two-Tier Hybrid Cascade Physics-RAG (DuckDB SQL pre-filter + 14D invariant manifold vector search + $\pm 72$h parent sequence hydration). Precision@5 ($0.700$ vs $0.467$ LLM, $+23.3\%, p=0.0044$), NDCG@5 ($0.718$), MRR ($0.726$), 1-tailed & 2-tailed tests, and bootstrap 95% CIs. |
| [`00_Unified_Coastal_Vulnerability_Index_Overview.md`](00_Unified_Coastal_Vulnerability_Index_Overview.md) | **Notebook 00: Foundation & Proof of Concept** | Foundational proof of concept notebook illustrating the evolution of the kinetic index and early empirical evaluations. |

### 2. Core Python Architecture Modules
| File | Module | Role & Foundational Implementations |
| :--- | :--- | :--- |
| [`python_physics_engine.md`](python_physics_engine.md) | `src/physics.py` | Closed-form kinetic mass-flux $\Upsilon$ ($[\text{kg}/\text{s}]$), dimensionless forcing $\Upsilon^*$, dimensionless disequilibrium $\Delta\Upsilon^*$, Yates/ShoreFor ODEs, asymptotic/exponential boundary damping, radiation stress, Dean number $\Omega$, Soulsby settling velocity, `COASTAL_SITE_PROFILES`. |
| [`python_physics_rag.md`](python_physics_rag.md) | `src/physics_rag.py` | Two-Tier Hybrid Cascade engine, DuckDB pre-filtering, $\pm 72$h sequence hydration, 14D invariant latent manifold, directional disequilibrium scoring, intent classification, 60-query benchmark suite, 1-tailed/2-tailed paired t-tests, bootstrap 95% CIs. |
| [`python_data_pipeline.md`](python_data_pipeline.md) | `src/data.py` | Multi-site data ingestion (Duck, Narrabeen, Torrey Pines, Truc Vert), harmonic astronomical tides, NOAA/NDBC/CANDHIS/CDIP wave shoaling and refraction, 1,228 GPS survey transects, datum conversions. |
| [`python_sciml_models.md`](python_sciml_models.md) | `src/models.py` | $2 \times 2$ SciML architectures (Data-Driven, PINN with $\mathcal{L}_{\text{physics}}$, Feature-Guided, Hybrid Residual), 72h autonomous storm rollouts, Brier Skill Score (BSS), Williams dependent correlation test, PyMC NUTS Bayesian MCMC. |
| [`python_neural_baselines.md`](python_neural_baselines.md) | `src/neural_baselines.py` | PyTorch architectures (`CoastalMLP`, `CoastalLSTM`, `PhysicsGuidedLSTM`, `PhysicsGuidedLoss`), sliding window sequence processing, dual-horizon evaluations. |
| [`python_embeddings_manifold.md`](python_embeddings_manifold.md) | `src/embeddings.py` | 14D invariant latent manifold construction, scale-invariant normalization, regime classification, 2D UMAP projection and silhouette scoring. |
| [`python_survey_acquisition.md`](python_survey_acquisition.md) | `src/acquire_real_surveys.py` | USACE FRF NetCDF harvesting, MHW cross-shore shoreline intercept extraction, RTK-GPS/DGPS survey harmonization. |
| [`python_parser_ablation.md`](scripts_run_parser_ablation.py) | `scripts/run_parser_ablation.py` | Parser ablation study comparing rule-based regex vs hybrid LLM query understanding across 60 scientific queries. |
| [`scripts_validate_duck_surveys.md`](scripts/validate_duck_real_surveys.py) | `scripts/validate_duck_real_surveys.py` | Quality control script validating FRF Duck surveys against NOAA tide gauge water levels and buoy observations. |

---

## Suggested Research Queries for NotebookLM
When interacting with this corpus in Google NotebookLM, try asking:
1. *"What is the exact physical formulation and dimensions of the Unified Coastal Vulnerability Index ($\Upsilon$), and how do the four constructive tiers modify it?"*
2. *"Why does raw $\Upsilon$ exhibit negative correlation with shoreline displacement at Narrabeen and Truc Vert, but positive correlation at Torrey Pines and near-zero correlation at Duck?"*
3. *"How does the Two-Tier Hybrid Cascade Physics-RAG eliminate the 15,000-hour temporal drift of standard vector search, and what are its 1-tailed and 2-tailed statistical significance values against SQL and LLM baselines?"*
4. *"What is the Observing System Simulation Experiment (OSSE) design in Notebook 03, and what are the quantitative trade-offs between Model A (Data-Driven), Model B (PINN), Model C (Feature-Guided), and Model D (Dual-Guided) during 72-hour storm rollouts?"*
5. *"How does asymptotic boundary damping in `src/physics.py` prevent unphysical saturation plateaus while preserving dynamic storm cuts?"*
