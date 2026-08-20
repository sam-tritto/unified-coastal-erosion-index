# Unified Coastal Vulnerability Index ($\Upsilon$) & Physics-Anchored Embeddings

A physics-informed framework for multi-modal coastal forcing synchronization, Bayesian morphodynamic forecasting with PyMC, and physics-anchored latent embeddings.

## Overview
This proof of concept validates the **Unified Coastal Vulnerability Index ($\Upsilon$)** against real-world observations from the USACE Field Research Facility (FRF) at Duck, NC during the 2022 season (including Hurricane Ian).

### Key Components
- **Multi-Modal Data Ingestion (`src/data.py`)**: Synchronized wave parameters ($H_s, T_p$), storm surge residual ($S$), wind speed & direction projected onto shore-normal azimuth, Soulsby (1997) sediment settling velocity ($w_s$), and cross-shore profile change ($\Delta X$).
- **Physics Formulations (`src/physics.py`)**: Vectorized Iribarren number ($\xi$), wave energy flux ($P$), Dean number ($\Omega$), and the closed-form Unified Coastal Vulnerability Index ($\Upsilon$).
- **Bayesian Modeling (`src/models.py`)**: PyMC probabilistic model leveraging $\Upsilon$ as an informative anchor with antecedent storm memory / exponential lag decay kernels.
- **Physics-Anchored Embeddings (`src/embeddings.py`)**: High-dimensional state mapping into latent space preserving morphodynamic similarity and breaker regimes (Spilling, Plunging, Surging).
- **Interactive Notebook (`unified_coastal_vulnerability_index.ipynb`)**: Complete end-to-end execution, diagnostic charts, posterior distributions, and latent space visualizations.

## Quickstart

```bash
# Install dependencies with uv
uv sync

# Run automated tests
uv run pytest tests/

# Launch Jupyter
uv run jupyter lab
```
