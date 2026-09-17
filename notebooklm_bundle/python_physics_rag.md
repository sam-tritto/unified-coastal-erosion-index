# Two-Tier Hybrid Cascade Physics-RAG Diagnostic Engine (`src/physics_rag.py`)

> **Source File**: [`src/physics_rag.py`](file:////Users/sam/Locals%20Only/unified-coastal-erosion-index/src/physics_rag.py)
> **Ingestion Target**: Google NotebookLM & Research Paper Reference

---

## Executive Purpose & Role

Production search, retrieval, and failure diagnosis engine coupling DuckDB relational SQL pre-filtering with a 14D invariant manifold vector search and parent sequence time-series hydration.

---

## Theoretical & Engineering Foundations

### 1. The Three Classical Search Pathologies in Earth Observation
1. **Baseline 1 (Raw-Data LLM-RAG)**: Standard dense vector embeddings on raw telemetry ($H_s, T_p, \text{Surge}$). Lacks hydrodynamic domain grounding, confuses pre-storm onset hours with peak destruction, and suffers from ungrounded numerical hallucinations.
2. **Baseline 2 (Rule-Based SQL)**: Metadata filtering (`WHERE site = 'duck_nc' AND hs_m > 3.0`). While precise for explicit constraints, SQL is **chronologically blind**: standard sorting/limiting retrieves calm early hours at storm onset rather than the peak erosive apex.
3. **Baseline 3 (Pure Physics-RAG)**: 14D manifold vector search without relational filtering suffers from **spatial and temporal degeneracy**, unable to distinguish identical wave states across disparate geographic observatories without metadata slicing.

### 2. The Two-Tier Hybrid Cascade Architecture
- **Tier 1 (DuckDB Relational Pre-Filtering)**: Fast columnar SQL evaluation over $175,392$ hourly records in $< 15\text{ ms}$, filtering by site, date ranges, and hydrodynamic bounds.
- **Parent Sequence Hydration**: Instantly extracts continuous $\pm 72$-hour multi-channel arrays around retrieved timestamps for dynamic SciML rollout and morphological failure diagnosis.
- **Tier 2 (14D Invariant Manifold Search & Disequilibrium Re-Ranking)**: Computes cosine similarity over dimensionless hydrodynamic states and applies direction-aware disequilibrium scoring:
  $$\mathcal{S}_{\text{hybrid}} = \cos(\mathbf{z}_q, \mathbf{z}_d) \cdot \left[ 1 + \alpha \cdot \text{sign}(\Delta\Upsilon) \cdot \frac{|\Delta\Upsilon|}{\Upsilon_{\text{ref}}} \right]$$

### 3. Rigorous Hypothesis Testing (1-Tailed & 2-Tailed + Bootstrap CIs)
To satisfy computational science peer-review standards, the evaluation suite reports both directional 1-tailed tests, conservative 2-tailed tests, and non-parametric bootstrap 95% confidence intervals ($B = 10,000$ resamples) across $N = 60$ diagnostic benchmark queries:
- **Precision@5**: Hybrid Cascade achieves **$0.7000$**, significantly outperforming Raw LLM-RAG ($0.4667$, $+23.33\%$, $t = 2.808$, 1-tail $p = 0.0044$, 2-tail $p = 0.0087$, CI $[+0.0634, +0.4032]$) and outperforming SQL ($0.5867$, $+11.33\%$, $t = 1.628$, 1-tail $p = 0.0571$, 2-tail $p = 0.1143$, CI $[-0.0290, +0.2556]$).
- **MRR (Mean Reciprocal Rank)**: **$0.7261$** vs SQL $0.5833$ ($+14.28\%$, 1-tail $p = 0.0439$, 2-tail $p = 0.0879$, Wilcoxon 1-tail $p = 0.0308$).
- **NDCG@5**: **$0.7180$** vs SQL $0.6134$ ($+10.46\%$, 1-tail $p = 0.0491$, 2-tail $p = 0.0982$).
- **Regime Purity**: **$0.7267$** vs LLM $0.5233$ ($+20.34\%$).
- **Peak Precision@5**: **$0.1200$** (Hybrid Cascade is the **only architecture with non-zero peak precision**, eliminating the $15,000\text{ h}$ temporal drift of pure vector search).

---

## Complete Annotated Source Code

```python
"""Physics-Inspired Retrieval-Augmented Generation (Physics-RAG) Engine with Gemini Flash.

Enables natural language querying of multi-modal coastal datasets and latent embeddings.
Combines Gemini-powered structured query parsing, physics-anchored vector similarity search,
dynamic synthetic vector projections, physical regime filtering, and grounded oceanographic narrative synthesis.
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


def sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively prune unsupported schema fields (such as 'additionalProperties') for Google GenAI SDK."""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in ("additionalProperties", "additional_properties"):
                continue
            cleaned[k] = sanitize_gemini_schema(v)
        return cleaned
    elif isinstance(schema, list):
        return [sanitize_gemini_schema(item) for item in schema]
    return schema


class TargetSyntheticValues(BaseModel):
    """Explicitly typed synthetic physical target parameters with no additional properties allowed."""

    model_config = ConfigDict(extra="forbid")

    hs_m: Optional[float] = Field(default=None, description="Significant wave height in meters.")
    tp_s: Optional[float] = Field(default=None, description="Peak wave period in seconds.")
    tan_beta: Optional[float] = Field(default=None, description="Foreshore beach slope.")
    total_water_level_m: Optional[float] = Field(default=None, description="Total water level elevation in meters.")
    surge_m: Optional[float] = Field(default=None, description="Storm surge residual elevation in meters.")
    wind_speed_ms: Optional[float] = Field(default=None, description="Wind speed in m/s.")
    wind_normal_ms: Optional[float] = Field(default=None, description="Shore-normal wind speed in m/s.")
    dean_number: Optional[float] = Field(default=None, description="Dean dimensionless fall velocity parameter.")
    xi_iribarren: Optional[float] = Field(default=None, description="Iribarren surf similarity number.")
    delta_x_m: Optional[float] = Field(default=None, description="Shoreline displacement delta_X in meters.")
    wave_dir_deg: Optional[float] = Field(default=None, description="Wave direction in degrees.")
    runup_r2_m: Optional[float] = Field(default=None, description="2% wave runup excursion in meters.")
    runup_r_ig_m: Optional[float] = Field(default=None, description="Infragravity swash runup excursion in meters.")
    wave_steepness: Optional[float] = Field(default=None, description="Deepwater wave steepness H0/L0.")
    p_flux_Wm: Optional[float] = Field(default=None, description="Wave power flux in W/m.")
    upsilon_diseq: Optional[float] = Field(default=None, description="Signed disequilibrium metric.")
    accretion_potential: Optional[float] = Field(default=None, description="Accretion potential metric.")
    upsilon: Optional[float] = Field(default=None, description="Unified Coastal Vulnerability Index.")
    breaker_regime: Optional[str] = Field(default=None, description="Breaker regime string.")

    def to_dict(self) -> Dict[str, float]:
        """Convert non-None fields to dict."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def items(self):
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, default)
        return val if val is not None else default

    def __getitem__(self, item: str) -> Any:
        val = getattr(self, item, None)
        if val is None:
            raise KeyError(item)
        return val

    def __contains__(self, item: str) -> bool:
        return getattr(self, item, None) is not None

    def __bool__(self) -> bool:
        return bool(self.to_dict())


def _extract_target_synthetic_dict(target_synthetic_values: Any) -> Dict[str, float]:
    """Extract a dictionary of non-None target synthetic parameters from a dict or model."""
    if target_synthetic_values is None:
        return {}
    if hasattr(target_synthetic_values, "to_dict"):
        try:
            return target_synthetic_values.to_dict()
        except Exception:
            pass
    if hasattr(target_synthetic_values, "model_dump"):
        try:
            return {k: v for k, v in target_synthetic_values.model_dump().items() if v is not None}
        except Exception:
            pass
    if isinstance(target_synthetic_values, dict):
        return {k: v for k, v in target_synthetic_values.items() if v is not None}
    return {}


try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

try:
    from src.embeddings import (
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
        compute_regime_centroids,
        synthesize_target_physics_vector,
        classify_agentic_latent_phases,
        prune_subcritical_noise_for_agent,
        extract_recovery_timeframes,
        compute_site_equilibrium_thresholds,
        compute_site_normalized_disequilibrium,
    )
except ImportError:
    from embeddings import (  # type: ignore
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
        compute_regime_centroids,
        synthesize_target_physics_vector,
        classify_agentic_latent_phases,
        prune_subcritical_noise_for_agent,
        extract_recovery_timeframes,
        compute_site_equilibrium_thresholds,
        compute_site_normalized_disequilibrium,
    )


class PhysicsQuerySpec(BaseModel):
    """Structured physical constraints and target embedding specifications."""

    model_config = ConfigDict(extra="forbid")

    raw_query: str = Field(description="The original natural language query string.")
    intent_summary: str = Field(
        default="",
        description="Summary of physical intent and oceanographic phenomena requested."
    )
    breaker_regime: Optional[Literal["Spilling", "Plunging", "Surging"]] = Field(
        default=None,
        description="Target breaking wave regime (Spilling for ξ < 0.5, Plunging for 0.5 <= ξ < 1.0, Surging for ξ >= 1.0)."
    )
    morphology: Optional[Literal["Erosive", "Accretive", "Equilibrium"]] = Field(
        default=None,
        description="Morphodynamic response (Erosive: shoreline recession, Accretive: berm buildup)."
    )
    phase: Optional[Literal["Erosion", "Recovery", "Noise", "Equilibrium"]] = Field(
        default=None,
        description="Suggestion 8: Discrete agentic state phase (Erosion: Upsilon > Upsilon_eq, Recovery: Upsilon < Upsilon_eq with +dX, Noise: Upsilon < Upsilon_crit)."
    )
    filter_background_noise: Optional[bool] = Field(
        default=None,
        description="Suggestion 8: If True, prune subcritical background noise (Upsilon < Upsilon_crit) to prevent LLM context-window bloat."
    )
    query_recovery_timeframe: Optional[bool] = Field(
        default=None,
        description="Suggestion 8: If True, extract post-storm recovery duration, timeframe metrics, and swash accretion rates."
    )
    query_storm_evolution: Optional[bool] = Field(
        default=None,
        description="If True, evaluate antecedent storm growth and temporal wave energy buildup."
    )
    hs_min: Optional[float] = Field(default=None, description="Minimum significant wave height in meters (Hs >= min).")
    hs_max: Optional[float] = Field(default=None, description="Maximum significant wave height in meters (Hs <= max).")
    tp_min: Optional[float] = Field(default=None, description="Minimum peak wave period in seconds (Tp >= min).")
    tp_max: Optional[float] = Field(default=None, description="Maximum peak wave period in seconds (Tp <= max).")
    surge_min: Optional[float] = Field(default=None, description="Minimum storm surge elevation in meters (Surge >= min).")
    surge_max: Optional[float] = Field(default=None, description="Maximum storm surge elevation in meters (Surge <= max).")
    wind_normal_min: Optional[float] = Field(default=None, description="Minimum shore-normal wind speed in m/s (positive for onshore).")
    wind_normal_max: Optional[float] = Field(default=None, description="Maximum shore-normal wind speed in m/s.")
    xi_min: Optional[float] = Field(default=None, description="Minimum Iribarren number.")
    xi_max: Optional[float] = Field(default=None, description="Maximum Iribarren number.")
    delta_x_min: Optional[float] = Field(default=None, description="Minimum shoreline displacement delta_X in meters.")
    delta_x_max: Optional[float] = Field(default=None, description="Maximum shoreline displacement delta_X in meters.")
    min_upsilon_percentile: Optional[float] = Field(default=None, description="Minimum percentile of Unified Index Upsilon (e.g. 75 for extreme high).")
    max_upsilon_percentile: Optional[float] = Field(default=None, description="Maximum percentile of Unified Index Upsilon (e.g. 25 for calm/low).")
    date_start: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD) if query refers to specific storm/date.")
    date_end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD) if query refers to specific storm/date.")
    target_synthetic_values: Optional[TargetSyntheticValues] = Field(
        default=None,
        description="Target numerical values to synthesize an explicit query vector (e.g. {'hs_m': 4.5, 'tp_s': 14.0, 'surge_m': 1.0})."
    )
    site_id: Optional[str] = Field(
        default=None,
        description="Target site identifier for pre-filtering (e.g. duck_nc, narrabeen_aus, oahu_north_shore_hi, torrey_pines_ca, truc_vert_france)."
    )
    geomorphic_regime: Optional[str] = Field(
        default=None,
        description="Geomorphic coastal regime tag for pre-filtering (e.g. macrotidal, microtidal, embayed, reef, narrow_shelf)."
    )
    target_state: Optional[str] = Field(
        default=None,
        description="Canonical reference state name (State_ExtremeStormPeak, State_CalmFairWeather, State_PostStormRecovery, State_LongPeriodSwell, Breaker_Spilling, etc.)"
    )
    focus: Optional[str] = Field(default=None, description="Focal phenomenon (e.g. 'surge', 'swell', 'wind', 'calm', 'storm', 'recovery').")
    is_failure_mode: Optional[bool] = Field(
        default=None,
        description="If True, query seeks hydrodynamic failure modes or model under-prediction episodes (e.g. 'When does Upsilon under-predict erosion?')."
    )


# Improvement 5: Physical Energy-Flux Latent Weighting (w_Upsilon metric tensor)
# Implement a physics-weighted metric tensor: Sim(q, x) = (q W x^T) / (||q||_W ||x||_W)
# Prioritizes primary hydrodynamic drivers over ambient secondary variables.
DEFAULT_UPSILON_FEATURE_WEIGHTS: Dict[str, float] = {
    # Primary Hydrodynamic Drivers
    "log_Upsilon": 2.5,
    "log_upsilon": 2.5,
    "upsilon": 2.5,
    "upsilon_disequilibrium": 2.5,

    "Dean_Omega": 2.0,
    "dean_number": 2.0,
    "Omega_Over_Crit": 2.0,

    "Iribarren_xi": 2.0,
    "xi_iribarren": 2.0,

    "Bar_Dissipation_Ratio": 1.5,
    "dissipation_ratio": 1.5,
    "delta_p_dissip": 1.5,

    "Wave_Power_P": 2.0,
    "log_p_flux": 2.0,
    "p_flux_Wm": 2.0,
    "Accretion_Potential": 1.5,
    "recovery_potential": 1.5,

    # Secondary Variables (Default 1.0)
    "Hs": 1.0,
    "Tp": 1.0,
    "Surge_S": 1.0,
    "relative_surge": 1.0,
    "Wind_Wa": 1.0,
    "wind_shear_stress": 1.0,
    "tan_alpha": 1.0,
    "w_s": 1.0,
    "wave_steepness": 1.0,
    "radiation_stress_gradient": 1.0,
    "water_level_excursion": 1.0,
    "sediment_mobility": 1.0,
    "infragravity_proxy": 1.0,
    "Water_Temp_C": 0.2,
}


def determine_upsilon_target_intent(
    spec: Optional[PhysicsQuerySpec] = None,
    query_text: str = "",
) -> str:
    """Classify physical query intent into one of three direction-aware target regimes:
    - 'erosion': Storms, beach cut, scarping, overwash, high wave energy (S_Upsilon = ReLU(Upsilon* - 1.0)).
    - 'recovery': Constructive swash, berm accretion, healing, post-storm calm (S_Upsilon = ReLU(1.0 - Upsilon*)).
    - 'equilibrium': Mean carrying capacity, steady-state background, stability (S_Upsilon = exp(- (Upsilon* - 1.0)^2 / (2*sigma^2))).

    Parameters
    ----------
    spec : PhysicsQuerySpec, optional
        Structured query specification.
    query_text : str, default ""
        Raw query text string.

    Returns
    -------
    target_intent : str
        One of 'erosion', 'recovery', or 'equilibrium'.
    """
    q_lower = (query_text or "").lower()

    # 1. Recovery / Accretion Intent
    is_recovery = False
    if spec is not None:
        if getattr(spec, "morphology", None) == "Accretive":
            is_recovery = True
        elif getattr(spec, "phase", None) == "Recovery":
            is_recovery = True
        elif getattr(spec, "query_recovery_timeframe", False):
            is_recovery = True
        elif getattr(spec, "target_state", None) in ["State_PostStormRecovery", "State_CalmFairWeather"]:
            is_recovery = True
        elif getattr(spec, "focus", None) in ["calm", "recovery"]:
            is_recovery = True
        elif getattr(spec, "max_upsilon_percentile", None) is not None:
            is_recovery = True

    if not is_recovery:
        recovery_keywords = [
            "accretion", "accretive", "recovery", "berm welding", "berm building",
            "berm accretion", "constructive swash", "constructive wave", "swash welding",
            "healing", "fair weather", "calm", "mild swell", "low wave", "low energy"
        ]
        if any(k in q_lower for k in recovery_keywords):
            is_recovery = True

    if is_recovery:
        return "recovery"

    # 2. Equilibrium Background Intent
    is_equilibrium = False
    if spec is not None:
        if getattr(spec, "morphology", None) == "Equilibrium":
            is_equilibrium = True
        elif getattr(spec, "phase", None) == "Equilibrium":
            is_equilibrium = True
        elif getattr(spec, "target_state", None) in ["State_Equilibrium", "State_Background"]:
            is_equilibrium = True

    if not is_equilibrium:
        eq_keywords = [
            "equilibrium", "carrying capacity", "background flux",
            "steady state", "stable beach", "quasi-equilibrium"
        ]
        if any(k in q_lower for k in eq_keywords):
            is_equilibrium = True

    if is_equilibrium:
        return "equilibrium"

    # 3. Default: Erosion / Storm
    return "erosion"


def compute_direction_aware_upsilon_disequilibrium(
    upsilon: Union[float, np.ndarray],
    upsilon_ref: Union[float, np.ndarray],
    upsilon_crit: Optional[Union[float, np.ndarray]] = None,
    target: str = "erosion",
    sigma: float = 0.35,
    recovery_mode: str = "capacity_deficit",
) -> Union[float, np.ndarray]:
    """Compute Direction-Aware Signed Upsilon Disequilibrium Score S_Upsilon(t).

    Theoretical Formulation:
    - Erosive / Storm Queries (target = 'erosion'):
        S_Upsilon(t) = ReLU(Upsilon*(t) - 1.0)
    - Constructive Swash / Accretion Queries (target = 'recovery'):
        S_Upsilon(t) = ReLU(1.0 - Upsilon*(t))  [if recovery_mode == 'capacity_deficit']
        or ReLU((Upsilon_crit - Upsilon(t)) / Upsilon_ref) [if recovery_mode == 'critical_deficit']
        or max(ReLU(1.0 - Upsilon*), ReLU((Upsilon_crit - Upsilon) / Upsilon_ref)) [if recovery_mode == 'combined']
    - Equilibrium Background Queries (target = 'equilibrium'):
        S_Upsilon(t) = exp(- (Upsilon*(t) - 1.0)^2 / (2 * sigma^2))

    Parameters
    ----------
    upsilon : float or np.ndarray
        Dimensional Unified Coastal Vulnerability Index Upsilon [kg/s].
    upsilon_ref : float or np.ndarray
        Site reference kinetic forcing flux Upsilon_ref [kg/s].
    upsilon_crit : float or np.ndarray, optional
        Critical threshold shear activation flux Upsilon_crit [kg/s].
    target : str, default 'erosion'
        Target intent: 'erosion', 'recovery', or 'equilibrium'.
    sigma : float, default 0.35
        Gaussian scale parameter for equilibrium background centering.
    recovery_mode : str, default 'capacity_deficit'
        Mode for recovery calculation: 'capacity_deficit' (ReLU(1.0 - Upsilon*)),
        'critical_deficit' (ReLU((Upsilon_crit - Upsilon) / Upsilon_ref)),
        or 'combined' (maximum of both).

    Returns
    -------
    s_upsilon : float or np.ndarray
        Direction-aware disequilibrium score S_Upsilon(t).
    """
    u_arr = np.asarray(upsilon, dtype=float)
    ref_arr = np.maximum(np.asarray(upsilon_ref, dtype=float), 1e-6)
    u_star = u_arr / ref_arr

    target_clean = str(target).lower().strip()
    if target_clean in ["recovery", "accretion", "accretive", "swash", "berm", "calm"]:
        if recovery_mode in ["critical_deficit", "crit"] and upsilon_crit is not None:
            crit_arr = np.asarray(upsilon_crit, dtype=float)
            score = np.maximum(0.0, (crit_arr - u_arr) / ref_arr)
        elif recovery_mode == "combined" and upsilon_crit is not None:
            crit_arr = np.asarray(upsilon_crit, dtype=float)
            score_cap = np.maximum(0.0, 1.0 - u_star)
            score_crit = np.maximum(0.0, (crit_arr - u_arr) / ref_arr)
            score = np.maximum(score_cap, score_crit)
        else:
            score = np.maximum(0.0, 1.0 - u_star)
    elif target_clean in ["equilibrium", "steady_state", "stable", "background", "neutral"]:
        score = np.exp(-((u_star - 1.0) ** 2) / (2.0 * (sigma ** 2)))
    else:
        # Default: erosion / storm / peak
        score = np.maximum(0.0, u_star - 1.0)

    if np.isscalar(upsilon) and np.isscalar(upsilon_ref):
        return float(score)
    return score


def compute_ranks(scores: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Compute 1-based ranks from scores in descending order (highest score -> rank 1).

    Handles ties by assigning the average rank to tied elements, matching
    `scipy.stats.rankdata(-scores, method='average')`.

    Parameters
    ----------
    scores : Union[np.ndarray, List[float]]
        1D array of scores to rank.

    Returns
    -------
    ranks : np.ndarray
        1-based ranks (1.0 to N).
    """
    scores_arr = np.asarray(scores, dtype=float)
    n = len(scores_arr)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    scores_arr = np.nan_to_num(scores_arr, nan=-1e9, posinf=1e9, neginf=-1e9)

    if np.all(scores_arr == scores_arr[0]):
        return np.full(n, (1.0 + n) / 2.0, dtype=float)

    try:
        from scipy.stats import rankdata
        return rankdata(-scores_arr, method="average")
    except ImportError:
        order = np.argsort(-scores_arr, kind="mergesort")
        sorted_scores = scores_arr[order]
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1
        return ranks


def compute_reciprocal_rank_fusion(
    scores_dict: Dict[str, Union[np.ndarray, List[float]]],
    weights: Optional[Dict[str, float]] = None,
    k0: int = 60,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    r"""Compute Reciprocal Rank Fusion (RRF) scores across multiple ranking channels.

    Formula (Cormack, Clarke, & Buettcher, SIGIR 2009):
    $$\text{RRF\_Score}(d) = \sum_{c \in C} \frac{w_c}{k_0 + \text{rank}_c(d)}$$

    RRF is scale-invariant, immune to outlier distortion, and provides a principled,
    peer-reviewed consensus ranking across heterogeneous retrieval channels
    (e.g., dense latent embeddings, physical relational constraints, and hydrodynamic disequilibrium).

    Parameters
    ----------
    scores_dict : Dict[str, Union[np.ndarray, List[float]]]
        Dictionary mapping channel names (e.g. 'cos', 'constraints', 'upsilon')
        to 1D score arrays where higher values indicate stronger relevance.
    weights : Optional[Dict[str, float]], default None
        Channel weights $w_c$. If None or missing for a channel, defaults to 1.0.
    k0 : int, default 60
        RRF smoothing constant preventing top-ranked items from excessively dominating.
        Standard IR constant is 60.

    Returns
    -------
    rrf_scores : np.ndarray
        Total fused RRF score for each candidate.
    ranks_dict : Dict[str, np.ndarray]
        Dictionary of 1-based ranks for each channel.
    """
    if not scores_dict:
        return np.array([], dtype=float), {}

    weights = weights or {}
    first_arr = next(iter(scores_dict.values()))
    n = len(first_arr)
    rrf_scores = np.zeros(n, dtype=float)
    ranks_dict: Dict[str, np.ndarray] = {}

    for channel, scores in scores_dict.items():
        w = float(weights.get(channel, 1.0))
        ranks = compute_ranks(scores)
        ranks_dict[channel] = ranks
        rrf_scores += w / (float(k0) + ranks)

    return rrf_scores, ranks_dict


class PhysicsQueryParser:
    """Deterministic rule-based parser for physical criteria and target embedding profiles."""

    REGIME_KEYWORDS = {
        "spilling": {"breaker_regime": "Spilling", "xi_max": 0.5},
        "plunging": {"breaker_regime": "Plunging", "xi_min": 0.5, "xi_max": 1.0},
        "surging": {"breaker_regime": "Surging", "xi_min": 1.0},
        "collapsing": {"breaker_regime": "Surging", "xi_min": 1.0},
        "dissipative": {"breaker_regime": "Spilling", "xi_max": 0.5},
        "reflective": {"breaker_regime": "Surging", "xi_min": 1.0},
    }

    MORPHO_KEYWORDS = {
        "erosion": {"morphology": "Erosive", "phase": "Erosion", "delta_x_max": 0.0},
        "erosive": {"morphology": "Erosive", "phase": "Erosion", "delta_x_max": 0.0},
        "beach cut": {"morphology": "Erosive", "phase": "Erosion", "delta_x_max": -0.1},
        "scarping": {"morphology": "Erosive", "phase": "Erosion", "delta_x_max": -0.1},
        "accretion": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0},
        "accretive": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0},
        "recovery": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0, "query_recovery_timeframe": True},
        "recovery phase": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0, "query_recovery_timeframe": True},
        "recovery timeframe": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0, "query_recovery_timeframe": True},
        "berm building": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0},
        "berm accretion": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0},
        "berm": {"morphology": "Accretive", "phase": "Recovery", "delta_x_min": 0.0},
    }

    EVENT_KEYWORDS = {
        # Historical storm events & survey epochs
        "june 2016": {"event": "narrabeen_ecl_2016", "high_vuln": True, "date_start": "2016-06-01", "date_end": "2016-06-15"},
        "east coast low": {"event": "east_coast_low", "high_vuln": True},
        "matthew": {"event": "hurricane_matthew", "high_vuln": True, "date_start": "2016-10-01", "date_end": "2016-10-15"},
        "october 2016": {"event": "matthew_oct_2016", "high_vuln": True, "date_start": "2016-10-01", "date_end": "2016-10-15"},
        "sandy": {"event": "hurricane_sandy", "high_vuln": True, "date_start": "2012-10-25", "date_end": "2012-11-05"},
        "october 2012": {"event": "sandy_oct_2012", "high_vuln": True, "date_start": "2012-10-25", "date_end": "2012-11-05"},
        "april 2015": {"event": "narrabeen_ecl_2015", "high_vuln": True, "date_start": "2015-04-20", "date_end": "2015-04-25"},
        "hercules": {"event": "megastorm_hercules", "high_vuln": True, "date_start": "2013-12-15", "date_end": "2014-03-15"},
        "christine": {"event": "megastorm_christine", "high_vuln": True, "date_start": "2013-12-15", "date_end": "2014-03-15"},
        "winter 2013-2014": {"event": "atlantic_megastorms_2014", "high_vuln": True, "date_start": "2013-12-15", "date_end": "2014-03-15"},
        "early 2014": {"event": "atlantic_megastorms_2014", "high_vuln": True, "date_start": "2013-12-15", "date_end": "2014-03-15"},
        "el nino": {"event": "el_nino_2015_2016", "high_vuln": True, "date_start": "2015-12-01", "date_end": "2016-03-01"},
        "el niño": {"event": "el_nino_2015_2016", "high_vuln": True, "date_start": "2015-12-01", "date_end": "2016-03-01"},
        "2015-2016": {"event": "el_nino_2015_2016", "date_start": "2015-12-01", "date_end": "2016-03-01"},
        "december 2015": {"event": "tp_dec_2015", "date_start": "2015-12-01", "date_end": "2015-12-31"},
        "summer 2016": {"event": "tv_summer_2016", "date_start": "2016-06-01", "date_end": "2016-08-31"},
        "january 2016": {"event": "jonas_jan_2016", "date_start": "2016-01-15", "date_end": "2016-01-31", "high_vuln": True},
        "jonas": {"event": "jonas_jan_2016", "date_start": "2016-01-15", "date_end": "2016-01-31", "high_vuln": True},
        "august 2014": {"event": "narrabeen_aug_2014", "date_start": "2014-08-10", "date_end": "2014-08-25", "high_vuln": True},
        "february 2014": {"event": "ulla_feb_2014", "date_start": "2014-02-10", "date_end": "2014-02-20", "high_vuln": True},
        "ulla": {"event": "ulla_feb_2014", "date_start": "2014-02-10", "date_end": "2014-02-20", "high_vuln": True},
        "march 2014": {"event": "duck_mar_2014", "date_start": "2014-03-20", "date_end": "2014-03-31", "high_vuln": True},
        "june 2012": {"event": "narrabeen_jun_2012", "date_start": "2012-06-01", "date_end": "2012-06-15", "high_vuln": True},
        "december 2013": {"event": "dirk_dec_2013", "date_start": "2013-12-20", "date_end": "2013-12-31", "high_vuln": True},
        "dirk": {"event": "dirk_dec_2013", "date_start": "2013-12-20", "date_end": "2013-12-31", "high_vuln": True},
        "february 2016": {"event": "tp_feb_2016", "date_start": "2016-01-25", "date_end": "2016-02-15", "high_vuln": True},
        "october 2015": {"event": "joaquin_oct_2015", "date_start": "2015-10-01", "date_end": "2015-10-10", "high_vuln": True},
        "joaquin": {"event": "joaquin_oct_2015", "date_start": "2015-10-01", "date_end": "2015-10-10", "high_vuln": True},
        "november 2013": {"event": "narrabeen_nov_2013", "date_start": "2013-11-05", "date_end": "2013-11-25", "high_vuln": True},
        "january 2015": {"event": "tv_jan_2015", "date_start": "2015-01-05", "date_end": "2015-01-25", "high_vuln": True},
        "july 2014": {"event": "tp_jul_2014", "date_start": "2014-07-01", "date_end": "2014-07-31"},
        "september 2014": {"event": "duck_sep_2014", "date_start": "2014-09-01", "date_end": "2014-09-30"},
        "april 2014": {"event": "narrabeen_apr_2014", "date_start": "2014-04-05", "date_end": "2014-04-25", "high_vuln": True},
        "summer 2015": {"event": "tv_summer_2015", "date_start": "2015-06-01", "date_end": "2015-08-31"},
        "hurricane ian": {"event": "hurricane_ian", "high_vuln": True, "date_start": "2022-09-27", "date_end": "2022-10-02"},
        "ian peak": {"event": "hurricane_ian_peak", "high_vuln": True, "date_start": "2022-09-27", "date_end": "2022-10-02"},
        "storm peak": {"high_vuln": True},
        "extreme": {"high_vuln": True},
        "major storm": {"high_vuln": True},
        "surge": {"focus": "surge"},
        "swell": {"focus": "swell"},
        "onshore wind": {"focus": "onshore_wind", "wind_normal_min": 0.0},
        "offshore wind": {"focus": "offshore_wind", "wind_normal_max": 0.0},
        "gale": {"focus": "wind_gale"},
        "calm": {"focus": "calm", "low_vuln": True},
        "fair weather": {"focus": "calm", "low_vuln": True},
        "mild": {"focus": "calm", "low_vuln": True},
        "reef": {"focus": "reef_dissipation"},
        "coral reef": {"focus": "reef_dissipation"},
    }

    @classmethod
    def parse(cls, query_text: str) -> Dict[str, Any]:
        """Parse natural language query into physical search parameters dict."""
        text = query_text.lower().strip()
        parsed: Dict[str, Any] = {
            "raw_query": query_text,
            "breaker_regime": None,
            "morphology": None,
            "phase": None,
            "filter_background_noise": None,
            "query_recovery_timeframe": None,
            "high_vuln": False,
            "low_vuln": False,
            "focus": None,
            "site_id": None,
            "geomorphic_regime": None,
            "filters": {},
            "target_state": None,
            "target_synthetic_values": None,
        }

        # Check site & geomorphic regime keywords for dynamic vector pre-filtering
        SITE_KEYWORDS = {
            "truc vert": {"site_id": "truc_vert_france", "geomorphic_regime": "macrotidal"},
            "france": {"site_id": "truc_vert_france", "geomorphic_regime": "macrotidal"},
            "macrotidal": {"site_id": "truc_vert_france", "geomorphic_regime": "macrotidal"},
            "biscay": {"site_id": "truc_vert_france", "geomorphic_regime": "macrotidal"},
            "narrabeen": {"site_id": "narrabeen_aus", "geomorphic_regime": "embayed"},
            "australia": {"site_id": "narrabeen_aus", "geomorphic_regime": "embayed"},
            "embayed": {"site_id": "narrabeen_aus", "geomorphic_regime": "embayed"},
            "oahu": {"site_id": "oahu_north_shore_hi", "geomorphic_regime": "reef"},
            "hawaii": {"site_id": "oahu_north_shore_hi", "geomorphic_regime": "reef"},
            "fringing reef": {"site_id": "oahu_north_shore_hi", "geomorphic_regime": "reef"},
            "torrey": {"site_id": "torrey_pines_ca", "geomorphic_regime": "narrow_shelf"},
            "california": {"site_id": "torrey_pines_ca", "geomorphic_regime": "narrow_shelf"},
            "narrow shelf": {"site_id": "torrey_pines_ca", "geomorphic_regime": "narrow_shelf"},
            "duck": {"site_id": "duck_nc", "geomorphic_regime": "microtidal"},
            "microtidal": {"site_id": "duck_nc", "geomorphic_regime": "microtidal"},
        }
        for kw, specs in SITE_KEYWORDS.items():
            if kw in text:
                parsed["site_id"] = specs["site_id"]
                parsed["geomorphic_regime"] = specs["geomorphic_regime"]
                parsed["filters"]["site_id"] = specs["site_id"]
                parsed["filters"]["geomorphic_regime"] = specs["geomorphic_regime"]
                break

        # Check breaker regime keywords
        for kw, specs in cls.REGIME_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", text):
                parsed["breaker_regime"] = specs["breaker_regime"]
                if "xi_max" in specs:
                    parsed["filters"]["xi_max"] = specs["xi_max"]
                if "xi_min" in specs:
                    parsed["filters"]["xi_min"] = specs["xi_min"]
                break

        # Check morphodynamic keywords (avoid false positives on 'buffered erosion' or 'under-predict erosion')
        is_negated_or_dissip = any(p in text for p in ["buffered erosion", "under-predict", "underpredict", "underprediction", "failure mode"])
        for kw, specs in cls.MORPHO_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", text) and not (kw in ["erosion", "erosive"] and is_negated_or_dissip):
                parsed["morphology"] = specs["morphology"]
                if "phase" in specs:
                    parsed["phase"] = specs["phase"]
                if "query_recovery_timeframe" in specs:
                    parsed["query_recovery_timeframe"] = specs["query_recovery_timeframe"]
                if "delta_x_max" in specs:
                    parsed["filters"]["delta_x_max"] = specs["delta_x_max"]
                if "delta_x_min" in specs:
                    parsed["filters"]["delta_x_min"] = specs["delta_x_min"]
                break

        # Suggestion 8: Threshold anchor & Noise filtering keywords
        if any(w in text for w in ["ignore noise", "filter noise", "above critical", "prune noise", "active forcing", "above critical threshold", "ignoring background noise"]):
            parsed["filter_background_noise"] = True
            parsed["filters"]["filter_background_noise"] = True

        is_ignoring_noise = any(w in text for w in ["ignore noise", "ignoring", "filter noise", "above critical", "prune noise", "active storm"])
        if any(w in text for w in ["background noise", "subcritical noise", "subcritical", "below threshold", "below critical"]) and not is_ignoring_noise:
            parsed["phase"] = "Noise"
            parsed["target_state"] = "State_SubcriticalNoise"

        if any(w in text for w in ["recovery timeframe", "recovery duration", "how long to recover", "recovery phase", "post-storm recovery", "beach recovery"]):
            parsed["phase"] = "Recovery"
            parsed["morphology"] = "Accretive"
            parsed["target_state"] = "State_PostStormRecovery"
            parsed["query_recovery_timeframe"] = True

        # Failure mode & under-prediction queries (e.g. "When does Upsilon under-predict erosion?")
        if any(w in text for w in ["under-predict", "underpredict", "under-prediction", "underprediction", "failure mode", "failure modes", "unexplained erosion", "model failure", "underpredicts", "under-predicts"]):
            parsed["is_failure_mode"] = True
            parsed["morphology"] = "Erosive"
            parsed["phase"] = "Erosion"
            parsed["target_state"] = "State_UnderpredictedErosion"
            parsed["intent_summary"] = "Identify physical episodes where observed erosion was severe despite low-to-moderate Upsilon forcing (model under-prediction failure mode)."

        # Check event / state keywords
        for kw, specs in cls.EVENT_KEYWORDS.items():
            if kw in text:
                if specs.get("high_vuln"):
                    parsed["high_vuln"] = True
                if specs.get("low_vuln"):
                    parsed["low_vuln"] = True
                if "focus" in specs:
                    parsed["focus"] = specs["focus"]
                if "wind_normal_min" in specs:
                    parsed["filters"]["wind_normal_min"] = specs["wind_normal_min"]
                if "wind_normal_max" in specs:
                    parsed["filters"]["wind_normal_max"] = specs["wind_normal_max"]
                if "date_start" in specs:
                    parsed["filters"]["date_start"] = specs["date_start"]
                if "date_end" in specs:
                    parsed["filters"]["date_end"] = specs["date_end"]

        # Check numerical patterns in query for target values
        synth_vals: Dict[str, float] = {}

        # 1. Wave height
        hs_m1 = re.search(r"(\d+(\.\d+)?)\s*m\s*(wave|hs|height)", text)
        hs_m2 = re.search(r"(wave|hs|height)\w*\s*(elevation|height)?\s*(above|over|>|exceed\w*|higher\w*|>=|of)?\s*(\d+(\.\d+)?)\s*m?", text)
        if hs_m1 or hs_m2:
            val = float(hs_m1.group(1)) if hs_m1 else float(hs_m2.group(4))
            synth_vals["hs_m"] = val
            if any(w in text for w in [">", "over", "above", "exceed", "higher"]):
                parsed["filters"]["hs_min"] = val
            elif any(w in text for w in ["<", "under", "below", "less"]):
                parsed["filters"]["hs_max"] = val

        # 2. Wave period
        tp_m1 = re.search(r"(\d+(\.\d+)?)\s*s\s*(period|tp|swell)", text)
        tp_m2 = re.search(r"(period|tp|swell)\w*\s*(above|over|>|exceed\w*|higher\w*|>=|of)?\s*(\d+(\.\d+)?)\s*s?", text)
        if tp_m1 or tp_m2:
            val = float(tp_m1.group(1)) if tp_m1 else float(tp_m2.group(3))
            synth_vals["tp_s"] = val
            if any(w in text for w in [">", "over", "above", "exceed", "higher"]):
                parsed["filters"]["tp_min"] = val
            elif any(w in text for w in ["<", "under", "below", "less"]):
                parsed["filters"]["tp_max"] = val

        # 3. Storm surge
        surge_m1 = re.search(r"(\d+(\.\d+)?)\s*m\s*surge", text)
        surge_m2 = re.search(r"(surge|water level)\w*\s*(elevation|residual)?\s*(above|over|>|exceed\w*|higher\w*|>=|of)?\s*(\d+(\.\d+)?)\s*m?", text)
        if surge_m1 or surge_m2:
            val = float(surge_m1.group(1)) if surge_m1 else float(surge_m2.group(4))
            synth_vals["surge_m"] = val
            if any(w in text for w in [">", "over", "above", "exceed", "higher", "positive"]):
                parsed["filters"]["surge_min"] = val
            elif any(w in text for w in ["<", "under", "below", "less", "negative"]):
                parsed["filters"]["surge_max"] = val

        # 4. Wind speed
        wind_m1 = re.search(r"(\d+(\.\d+)?)\s*(m/s|ms)\s*wind", text)
        wind_m2 = re.search(r"(wind|gale)\w*\s*(speed|velocity)?\s*(above|over|>|exceed\w*|higher\w*|>=|of)?\s*(\d+(\.\d+)?)\s*(m/s|ms)?", text)
        if wind_m1 or wind_m2:
            val = float(wind_m1.group(1)) if wind_m1 else float(wind_m2.group(4))
            synth_vals["wind_normal_ms"] = val
            if any(w in text for w in [">", "over", "above", "exceed", "higher", "strong"]):
                parsed["filters"]["wind_normal_min"] = val

        # Zero-shot parameter synthesis for qualitative oceanographic phenomena if explicit numbers omitted
        if "hs_m" not in synth_vals:
            if any(w in text for w in ["high energy storm", "extreme storm", "catastrophic", "severe storm cut"]):
                synth_vals["hs_m"] = 3.5
                synth_vals["tp_s"] = 11.5
                parsed["filters"]["hs_min"] = 2.7
            elif "compound" in text:
                synth_vals["hs_m"] = 3.0
                synth_vals["tp_s"] = 11.0
                synth_vals["total_water_level_m"] = 1.7
                parsed["filters"]["hs_min"] = 2.2
                parsed["filters"]["surge_min"] = 1.4
            elif "infragravity" in text:
                synth_vals["hs_m"] = 2.5
                synth_vals["tp_s"] = 14.5
                parsed["filters"]["tp_min"] = 13.0
                parsed["filters"]["hs_min"] = 1.5
                parsed["focus"] = "infragravity"
            elif "summer swell" in text or "constructive long-period" in text:
                synth_vals["hs_m"] = 0.9
                synth_vals["tp_s"] = 13.0
                parsed["filters"]["tp_min"] = 11.0
                parsed["filters"]["hs_max"] = 1.3
                parsed["phase"] = "Recovery"
                parsed["morphology"] = "Accretive"
                parsed["target_state"] = "State_PostStormRecovery"
            elif "macrotidal" in text or "tidal smearing" in text:
                synth_vals["hs_m"] = 2.5
                synth_vals["tp_s"] = 9.5
                parsed["site_id"] = "truc_vert_france"
                parsed["geomorphic_regime"] = "macrotidal"
                parsed["filters"]["site_id"] = "truc_vert_france"
                parsed["focus"] = "macrotidal"
            elif "sheltered" in text or "protected embayment" in text:
                synth_vals["hs_m"] = 0.7
                synth_vals["tp_s"] = 12.0
                parsed["geomorphic_regime"] = "embayed"
                parsed["filters"]["hs_max"] = 1.1
                parsed["focus"] = "sheltered_swell"
            elif "oblique" in text or "headland" in text or "beach rotation" in text:
                synth_vals["hs_m"] = 2.8
                synth_vals["tp_s"] = 11.0
                parsed["filters"]["hs_min"] = 2.0
                parsed["focus"] = "oblique"
            elif "sandbar" in text or "buffering" in text or "bar-breach" in text:
                synth_vals["hs_m"] = 3.0
                synth_vals["tp_s"] = 10.5
                synth_vals["dean_number"] = 5.0
                parsed["filters"]["hs_min"] = 2.2
                parsed["focus"] = "sandbar_dissipation"
            elif "tide-gated" in text or "high tide" in text:
                synth_vals["hs_m"] = 2.5
                synth_vals["tp_s"] = 10.0
                synth_vals["total_water_level_m"] = 1.8
                parsed["filters"]["hs_min"] = 2.0
                parsed["filters"]["surge_min"] = 1.5
            elif "steep coarse sand" in text:
                synth_vals["hs_m"] = 2.8
                synth_vals["tp_s"] = 9.0
                synth_vals["dean_number"] = 4.0
                parsed["filters"]["hs_min"] = 2.2
            elif any(w in text for w in ["under-predict", "underpredict", "underprediction", "failure mode"]):
                synth_vals["hs_m"] = 1.8
                synth_vals["tp_s"] = 12.0
                synth_vals["total_water_level_m"] = 1.7
                parsed["is_failure_mode"] = True
                parsed["filters"]["hs_max"] = 2.0
                parsed["filters"]["tp_min"] = 10.0
                parsed["filters"]["surge_min"] = 1.4
            elif parsed.get("breaker_regime") == "Spilling" or "spilling" in text or "dissipative" in text:
                synth_vals["hs_m"] = 3.0
                synth_vals["tp_s"] = 7.5
                synth_vals["tan_beta"] = 0.08
            elif parsed.get("breaker_regime") == "Plunging" or "plunging" in text or "curling" in text:
                synth_vals["hs_m"] = 2.0
                synth_vals["tp_s"] = 9.5
                synth_vals["tan_beta"] = 0.08
            elif parsed.get("breaker_regime") == "Surging" or "surging" in text or "reflective" in text or "shorebreak" in text:
                synth_vals["hs_m"] = 0.6
                synth_vals["tp_s"] = 14.0
                synth_vals["tan_beta"] = 0.08
            elif parsed.get("phase") == "Recovery" or "berm accretion" in text or "swash recovery" in text or "profile healing" in text:
                synth_vals["hs_m"] = 0.8
                synth_vals["tp_s"] = 11.5
                synth_vals["delta_x_m"] = 0.2
            elif "gale" in text:
                synth_vals["hs_m"] = 2.8
                synth_vals["tp_s"] = 9.0
                synth_vals["wind_speed_ms"] = 8.5
                synth_vals["wind_normal_ms"] = 8.0
                parsed["filters"]["wind_normal_min"] = 7.0
            elif "total water level" in text or "dune toe" in text:
                synth_vals["hs_m"] = 2.5
                synth_vals["tp_s"] = 10.0
                synth_vals["total_water_level_m"] = 1.8
                parsed["filters"]["surge_min"] = 1.6
            elif "breaking at low tide" in text:
                synth_vals["hs_m"] = 2.2
                synth_vals["tp_s"] = 10.0
                synth_vals["total_water_level_m"] = 0.9
                parsed["filters"]["hs_min"] = 2.0
                parsed["filters"]["surge_max"] = 1.1

        if synth_vals:
            parsed["target_synthetic_values"] = synth_vals

        # Infer target reference state
        if parsed.get("target_state"):
            pass
        elif parsed["high_vuln"] or "peak" in text or "storm" in text or "ian" in text:
            parsed["target_state"] = "State_ExtremeStormPeak"
        elif parsed.get("phase") == "Recovery" or "recovery" in text:
            parsed["target_state"] = "State_PostStormRecovery"
        elif parsed["low_vuln"] or "calm" in text or "fair" in text:
            parsed["target_state"] = "State_CalmFairWeather"
        elif parsed["focus"] == "swell" or "swell" in text:
            parsed["target_state"] = "State_LongPeriodSwell"
        elif parsed["breaker_regime"]:
            parsed["target_state"] = f"Breaker_{parsed['breaker_regime']}"

        return parsed


_GEMINI_PARSE_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "gemini_query_cache.json"
)
_GEMINI_PARSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_query_cache() -> None:
    global _GEMINI_PARSE_CACHE
    if not _GEMINI_PARSE_CACHE and os.path.exists(_GEMINI_PARSE_CACHE_FILE):
        try:
            with open(_GEMINI_PARSE_CACHE_FILE, "r") as f:
                loaded = json.load(f)
                _GEMINI_PARSE_CACHE.clear()
                _GEMINI_PARSE_CACHE.update(loaded)
        except Exception:
            pass


def _save_query_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_GEMINI_PARSE_CACHE_FILE), exist_ok=True)
        tmp_file = _GEMINI_PARSE_CACHE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(_GEMINI_PARSE_CACHE, f, indent=2)
        os.replace(tmp_file, _GEMINI_PARSE_CACHE_FILE)
    except Exception as e:
        logger.warning(f"Failed to save Gemini query cache: {e}")


def _call_gemini_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: Dict[str, Any],
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Any:
    """Execute a Gemini API call with exponential backoff on transient errors (504, 503, 429, timeouts)."""
    import time
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_transient = (
                "504" in err_str
                or "deadline_exceeded" in err_str
                or "503" in err_str
                or "unavailable" in err_str
                or "429" in err_str
                or "resource_exhausted" in err_str
                or "timeout" in err_str
                or "timed out" in err_str
                or "connection" in err_str
            )
            if is_transient and attempt < max_retries:
                sleep_sec = backoff * (2 ** attempt)
                logger.warning(
                    f"Gemini call hit transient error ({type(e).__name__}: {e}). "
                    f"Retrying in {sleep_sec:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_sec)
                continue
            raise
    raise last_err


class GeminiPhysicsQueryParser:
    """Uses Gemini Flash (or configured model) with structured output to parse natural language queries into PhysicsQuerySpec."""

    SYSTEM_PROMPT = """You are an expert coastal oceanographer and hydrodynamic search specialist.
Your task is to parse a user's natural language query about coastal hydrodynamic and morphodynamic conditions into a strictly typed JSON schema according to PhysicsQuerySpec.

Guidelines:
1. Physical Parameter Ranges & Thresholds:
   - Extract numerical thresholds directly from the query when stated (e.g., "wave height over 3m" -> hs_min=3.0; "waves under 1.2m" -> hs_max=1.2; "period over 12s" -> tp_min=12.0; "TWL over 2.0m" -> surge_min=2.0).
   - If qualitative physical conditions are described without explicit numbers, infer reasonable physical parameter bounds:
     * Extreme storms / high wave energy: hs_min=2.5 to 3.5, target_state="State_ExtremeStormPeak", min_upsilon_percentile=70.0.
     * Calm / mild / fair weather: hs_max=1.3, target_state="State_CalmFairWeather", max_upsilon_percentile=25.0.
     * Elevated water levels / surge / dune toe saturation: surge_min=0.5 to 1.5.
     * Long-period swell: tp_min=11.0.
     * Onshore gale wind: wind_normal_min=7.0, focus="wind_gale".
     * Compound wave-surge: hs_min=2.2, surge_min=0.8, focus="compound".

2. Breaker Regimes:
   - 'spilling' or 'dissipative': breaker_regime="Spilling", xi_max=0.5.
   - 'plunging' or 'curling': breaker_regime="Plunging", xi_min=0.5, xi_max=1.0.
   - 'surging', 'collapsing', or 'reflective': breaker_regime="Surging", xi_min=1.0.

3. Morphology & Morphodynamic Phases:
   - 'erosion', 'beach cut', 'scarping', 'recession' (when not negated): morphology="Erosive", phase="Erosion", delta_x_max=0.0.
   - 'accretion', 'recovery', 'berm building', 'healing', 'berm welding': morphology="Accretive", phase="Recovery", delta_x_min=0.0, target_state="State_PostStormRecovery".
   - 'recovery timeframe' or post-storm duration: query_recovery_timeframe=True, phase="Recovery", morphology="Accretive", target_state="State_PostStormRecovery".
   - 'ignore background noise', 'filter noise', 'active storm only', 'above critical threshold': filter_background_noise=True.
   - 'subcritical', 'below threshold', 'background noise' (when not ignored): phase="Noise", target_state="State_SubcriticalNoise".

4. Geographic Sites & Geomorphic Archetypes (ONLY if explicitly mentioned in query text):
   - Only set site_id or geomorphic_regime if explicitly referenced by location name or geomorphic regime in the query:
     * Duck / North Carolina: site_id="duck_nc", geomorphic_regime="microtidal"
     * Narrabeen / Sydney / Australia: site_id="narrabeen_aus", geomorphic_regime="embayed"
     * Torrey Pines / California / San Diego: site_id="torrey_pines_ca", geomorphic_regime="narrow_shelf"
     * Truc Vert / Biscay / France: site_id="truc_vert_france", geomorphic_regime="macrotidal"
     * Oahu / North Shore / Hawaii: site_id="oahu_north_shore_hi", geomorphic_regime="reef"
   - If the query mentions generic geomorphic types (e.g., 'macrotidal', 'embayed', 'pocket beach', 'fringing reef'), set geomorphic_regime accordingly without forcing site_id unless the location is named.

5. Dates & Temporal Windows (ONLY if explicitly mentioned in query text):
   - Only populate date_start and date_end if a specific year, month, or calendar date range is explicitly stated in the query text.
   - For example:
     * "June 2016" -> date_start="2016-06-01", date_end="2016-06-30"
     * "October 2016" -> date_start="2016-10-01", date_end="2016-10-31"
     * "December 2015" -> date_start="2015-12-01", date_end="2015-12-31"
     * "Winter 2013-2014" -> date_start="2013-12-01", date_end="2014-03-31"
     * "Summer 2016" -> date_start="2016-06-01", date_end="2016-08-31"
   - Do NOT invent arbitrary day ranges or storm dates unless the user explicitly specified the dates in the query text.

6. Diagnostic Failure Modes & Physical Blind Spots:
   - 'under-predict', 'underpredict', 'failure mode', 'model blind spots': is_failure_mode=True, phase="Erosion", morphology="Erosive", target_state="State_UnderpredictedErosion".
   - 'infragravity swash resonance', 'surf-beat resonance': tp_min=13.0, hs_min=1.5, focus="infragravity".
   - 'sandbar dissipation', 'bar-breach': hs_min=2.0, focus="sandbar_dissipation".
"""

    @classmethod
    def parse(
        cls,
        query_text: str,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        parser_mode: str = "pure_gemini",
        allow_regex_backfill: Optional[bool] = None,
    ) -> PhysicsQuerySpec:
        """Parse natural language query with explicit mode control.

        Parameters
        ----------
        query_text : str
            The natural language engineering query.
        model_name : str
            Gemini model name (default: 'gemini-2.5-flash').
        api_key : str, optional
            Gemini API key.
        client : Any, optional
            Pre-instantiated GenAI client.
        parser_mode : str, default 'pure_gemini'
            Parser execution mode:
            - 'pure_gemini': Pure live Gemini Flash structured output with NO regex backfill.
            - 'pure_regex': Offline deterministic keyword/regex parsing (no LLM calls).
            - 'hybrid': Live Gemini Flash with deterministic regex backfill for omitted/null fields.
        allow_regex_backfill : bool, optional
            Legacy boolean toggle. If provided, True maps to 'hybrid', False maps to 'pure_gemini'.
        """
        # Resolve legacy parameter if supplied
        if allow_regex_backfill is not None:
            parser_mode = "hybrid" if allow_regex_backfill else "pure_gemini"

        mode_clean = str(parser_mode).lower().strip()
        if mode_clean in ["pure_regex", "regex", "deterministic", "fallback"]:
            return cls._fallback_parse(query_text)

        # 1. Check persistent disk cache first (0 tokens, <1ms response)
        cache_key = f"{model_name}:{mode_clean}:{query_text.strip()}"
        _load_query_cache()
        if cache_key in _GEMINI_PARSE_CACHE:
            try:
                cached_dict = dict(_GEMINI_PARSE_CACHE[cache_key])
                cached_dict["raw_query"] = query_text
                return PhysicsQuerySpec(**cached_dict)
            except Exception:
                pass

        nl_spec = cls._fallback_parse(query_text)
        try:
            from google import genai

            if client is None:
                k = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if k:
                    client = genai.Client(api_key=k, http_options={"timeout": 45000})
                else:
                    try:
                        client = genai.Client(http_options={"timeout": 45000})
                    except Exception:
                        client = None

            if client is not None:
                schema = sanitize_gemini_schema(PhysicsQuerySpec.model_json_schema())
                response = _call_gemini_with_retry(
                    client=client,
                    model=model_name,
                    contents=f"Parse the following coastal oceanography query into the PhysicsQuerySpec schema:\nQuery: {query_text}",
                    config={
                        "system_instruction": cls.SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "temperature": 0.0,
                        "http_options": {"timeout": 45000},
                    },
                    max_retries=3,
                    backoff=2.0,
                )
                if response and response.text:
                    import json
                    parsed_json = json.loads(response.text)
                    parsed_json["raw_query"] = query_text
                    spec = PhysicsQuerySpec(**parsed_json)

                    # In Pure Gemini mode, cache and return directly without regex backfilling
                    if mode_clean in ["pure_gemini", "gemini_only", "zero_shot"]:
                        _GEMINI_PARSE_CACHE[cache_key] = spec.model_dump()
                        _save_query_cache()
                        return spec

                    # In Hybrid mode, backfill zero-shot synthetic values & omitted fields from deterministic fallback
                    if not spec.target_synthetic_values and nl_spec.target_synthetic_values:
                        spec.target_synthetic_values = nl_spec.target_synthetic_values
                    for attr in ["breaker_regime", "site_id", "geomorphic_regime", "hs_min", "tp_min", "surge_min", "wind_normal_min", "focus", "is_failure_mode"]:
                        if getattr(spec, attr, None) is None and getattr(nl_spec, attr, None) is not None:
                            setattr(spec, attr, getattr(nl_spec, attr))

                    _GEMINI_PARSE_CACHE[cache_key] = spec.model_dump()
                    _save_query_cache()
                    return spec
        except Exception as e:
            logger.error(f"Gemini physics query parsing failed for '{query_text}': {e}")
            if mode_clean in ["pure_gemini", "gemini_only", "zero_shot"]:
                return PhysicsQuerySpec(
                    raw_query=query_text,
                    intent_summary=f"Gemini zero-shot parse failed ({type(e).__name__}): '{query_text}'",
                )

        return nl_spec

    @classmethod
    def _fallback_parse(cls, query_text: str) -> PhysicsQuerySpec:
        """Deterministic rule-based parser fallback."""
        parsed_dict = PhysicsQueryParser.parse(query_text)
        filters = parsed_dict.get("filters", {})

        return PhysicsQuerySpec(
            raw_query=query_text,
            intent_summary=f"Rule-parsed query for '{query_text}'",
            breaker_regime=parsed_dict.get("breaker_regime"),
            morphology=parsed_dict.get("morphology"),
            phase=parsed_dict.get("phase"),
            filter_background_noise=parsed_dict.get("filter_background_noise"),
            query_recovery_timeframe=parsed_dict.get("query_recovery_timeframe"),
            target_state=parsed_dict.get("target_state"),
            focus=parsed_dict.get("focus"),
            site_id=parsed_dict.get("site_id"),
            geomorphic_regime=parsed_dict.get("geomorphic_regime"),
            hs_min=filters.get("hs_min"),
            hs_max=filters.get("hs_max"),
            tp_min=filters.get("tp_min"),
            tp_max=filters.get("tp_max"),
            surge_min=filters.get("surge_min"),
            surge_max=filters.get("surge_max"),
            xi_min=filters.get("xi_min"),
            xi_max=filters.get("xi_max"),
            delta_x_min=filters.get("delta_x_min"),
            delta_x_max=filters.get("delta_x_max"),
            wind_normal_min=filters.get("wind_normal_min"),
            wind_normal_max=filters.get("wind_normal_max"),
            date_start=filters.get("date_start"),
            date_end=filters.get("date_end"),
            min_upsilon_percentile=75.0 if parsed_dict.get("high_vuln") else None,
            max_upsilon_percentile=25.0 if parsed_dict.get("low_vuln") else None,
            target_synthetic_values=parsed_dict.get("target_synthetic_values"),
        )


def explain_with_gemini(
    query_text: str,
    spec: PhysicsQuerySpec,
    top_df: pd.DataFrame,
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Optional[str]:
    """Generate rich, expert-level oceanographic diagnosis from retrieved data using Gemini Flash."""
    if top_df.empty:
        return None

    try:
        from google import genai

        if client is None:
            k = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if k:
                client = genai.Client(api_key=k)
            else:
                try:
                    client = genai.Client()
                except Exception:
                    client = None

        if client is None:
            return None

        display_cols = [
            c for c in [
                "timestamp", "hs_m", "tp_s", "surge_m", "wind_normal_ms",
                "xi_iribarren", "breaker_regime", "upsilon", "upsilon_diseq", "delta_x_m", "similarity"
            ] if c in top_df.columns
        ]
        table_str = top_df[display_cols].to_string(index=False)

        prompt = f"""You are an authoritative senior coastal oceanographer reviewing hydrodynamic search results from the USACE FRF Duck dataset.
Synthesize a concise, 2-to-3 paragraph physical diagnostic narrative for the query: "{query_text}" based strictly on the retrieved observations below.

Query Intent: {spec.intent_summary or spec.raw_query}
Target Regime / Phase: {spec.phase or spec.breaker_regime or 'General Hydrodynamics'}

Retrieved Observation Data:
{table_str}

Summary Statistics:
- Mean Significant Wave Height (Hs): {top_df['hs_m'].mean() if 'hs_m' in top_df.columns else 0:.2f} m
- Mean Peak Period (Tp): {top_df['tp_s'].mean() if 'tp_s' in top_df.columns else 0:.1f} s
- Mean Surge Residual (S): {top_df['surge_m'].mean() if 'surge_m' in top_df.columns else 0:.2f} m
- Mean Iribarren Number (ξ): {top_df['xi_iribarren'].mean() if 'xi_iribarren' in top_df.columns else 0:.2f}
- Mean Unified Index (Υ): {top_df['upsilon'].mean() if 'upsilon' in top_df.columns else 0:.1e}
- Mean Signed Disequilibrium (Υ_diseq): {top_df['upsilon_diseq'].mean() if 'upsilon_diseq' in top_df.columns else 0:.1e}
- Mean Shoreline Displacement (ΔX): {top_df['delta_x_m'].mean() if 'delta_x_m' in top_df.columns else 0:.2f} m

Instructions:
1. Explain the wave dissipation kinetics and surf similarity / breaker regime (Iribarren number ξ).
2. Connect wave power flux and total water level (surge + runup) to morphodynamic response (beach cut vs. constructive berm accretion).
3. If this query concerns recovery phases or recovery timeframes, analyze the Signed Disequilibrium (Υ_diseq < 0) driving constructive swash accretion and estimate recovery timeframes.
4. Cite the quantitative numbers directly from the data. Do NOT speculate or hallucinate outside the retrieved records. Keep tone rigorous, objective, and professional.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
            },
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini explanation synthesis failed for '{query_text}': {e}")
    return None


def explain_hierarchical_with_gemini(
    query_text: str,
    spec: PhysicsQuerySpec,
    child_df: pd.DataFrame,
    parent_windows: List["ParentWindowContext"],
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Optional[str]:
    """Generate rich multi-scale diagnostic narrative synthesizing child matches and hydrated parent time-series."""
    if child_df.empty:
        return None

    try:
        from google import genai

        if client is None:
            k = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if k:
                client = genai.Client(api_key=k)
            else:
                try:
                    client = genai.Client()
                except Exception:
                    client = None

        if client is None:
            return None

        display_cols = [
            c for c in [
                "timestamp", "hs_m", "tp_s", "surge_m", "wind_normal_ms",
                "xi_iribarren", "breaker_regime", "upsilon", "upsilon_diseq", "delta_x_m", "similarity"
            ] if c in child_df.columns
        ]
        child_table_str = child_df[display_cols].to_string(index=False)

        parent_summaries = []
        for idx, pw in enumerate(parent_windows[:3]):
            st = pw.summary_stats
            parent_summaries.append(
                f"Window #{idx+1} [Event: {pw.event_timestamp}, Span: {pw.window_start} to {pw.window_end}, {len(pw.parent_df) if pw.parent_df is not None else 0} timesteps]:\n"
                f"  - Peak Hs: {st.get('peak_hs_m', 0.0):.2f}m (Rise rate: {st.get('hs_rise_rate_m_per_hr', 0.0):.3f} m/hr), Peak Tp: {st.get('peak_tp_s', 0.0):.1f}s\n"
                f"  - Peak Surge: {st.get('peak_surge_m', 0.0):.2f}m (Surge duration: {st.get('surge_duration_hours', 0.0):.1f}h)\n"
                f"  - Peak Upsilon Index: {st.get('peak_upsilon', 0.0):.1e}, Cumulative Power Impulse: {st.get('cumulative_power_impulse_MJ_per_m', 0.0):.1f} MJ/m\n"
                f"  - Net Shoreline Displacement ΔX: {st.get('net_delta_x_m', 0.0):+.2f}m (Start: {st.get('delta_x_start_m', 0.0):.2f}m, End: {st.get('delta_x_end_m', 0.0):.2f}m)\n"
                f"  - Mean Disequilibrium Υ_diseq: {st.get('mean_upsilon_diseq', 0.0):.1e} ({st.get('recovery_phase_fraction', 0.0)*100:.1f}% recovery timesteps)"
            )
        parent_str = "\n".join(parent_summaries)

        prompt = f"""You are an authoritative senior coastal oceanographer reviewing Hierarchical Parent-Child Physics-RAG search results.
The search engine matched instantaneous child hydrodynamic states and hydrated continuous multi-channel parent time-series sequences (±72h context windows via DuckDB) for the query: "{query_text}".

Query Intent: {spec.intent_summary or spec.raw_query}
Target Regime / Phase: {spec.phase or spec.breaker_regime or 'General Hydrodynamics'}

Top Child Vector Matches:
{child_table_str}

Hydrated Parent Time-Series Context Windows (Temporal Multi-Channel Evolution):
{parent_str}

Instructions:
1. Multi-Scale Synthesis: Connect the instantaneous child physical state (peak scour, breaker type, surge elevation) with its antecedent wave energy buildup (pre-storm rise rate) and post-event relaxation trajectory.
2. Hydrodynamic & Morphodynamic Root Cause: Analyze how wave power flux impulse, storm surge duration, and Iribarren breaker dynamics drove the observed shoreline displacement (ΔX) and Unified Index (Υ).
3. Disequilibrium & Recovery: If post-storm recovery or berm accretion is present, explain how the Signed Disequilibrium (Υ_diseq < 0) governed constructive swash welding.
4. Quantitative Grounding: Directly reference the quantitative values from both the child records and parent window summary statistics. Maintain a rigorous, objective, and professional scientific tone.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
            },
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini hierarchical explanation synthesis failed for '{query_text}': {e}")
    return None


class ParentWindowContext(BaseModel):
    """Encapsulates a hydrated parent time-series context window around a child event."""
    model_config = {"arbitrary_types_allowed": True}

    match_index: int = Field(default=0, description="Row index of the child event in the dataset.")
    event_timestamp: Any = Field(description="Timestamp of the matched child physical state.")
    similarity: float = Field(default=1.0, description="Cosine or vector similarity score.")
    window_start: Any = Field(default=None, description="Start timestamp of the parent time-series context window.")
    window_end: Any = Field(default=None, description="End timestamp of the parent time-series context window.")
    pre_window_hours: float = Field(default=72.0, description="Antecedent context window duration in hours.")
    post_window_hours: float = Field(default=24.0, description="Post-event context window duration in hours.")
    parent_df: Any = Field(default=None, description="Hydrated multi-channel sequence DataFrame from DuckDB.")
    summary_stats: Dict[str, Any] = Field(default_factory=dict, description="Extracted temporal dynamics and oceanographic summary metrics.")


class HierarchicalRAGResult:
    """Structured container for Hierarchical Parent-Child RAG results with hydrated multi-channel context."""

    def __init__(
        self,
        query: str,
        child_matches_df: pd.DataFrame,
        parent_windows: List[ParentWindowContext],
        similarities: np.ndarray,
        explanation: str,
        parsed_criteria: Union[Dict[str, Any], PhysicsQuerySpec],
        llm_explanation: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.query = query
        self.child_matches_df = child_matches_df
        self.matches_df = child_matches_df  # Compatibility alias
        self.parent_windows = parent_windows
        self.similarities = similarities
        self.explanation = explanation
        self.parsed_criteria = parsed_criteria
        self.llm_explanation = llm_explanation
        self.query_vector = query_vector
        self.metrics = metrics or {}

    @property
    def spec(self) -> Any:
        """Physical query specification or parsed criteria."""
        return self.parsed_criteria

    @property
    def matches(self) -> List[Dict[str, Any]]:
        """List of top matched child event dictionary records."""
        return self.child_matches_df.to_dict(orient="records")

    def synthesize_narrative(self) -> str:
        """Return synthesized diagnostic narrative (LLM explanation or deterministic physical narrative)."""
        return self.llm_explanation or self.explanation

    def __repr__(self) -> str:
        return f"<HierarchicalRAGResult query='{self.query}', n_children={len(self.child_matches_df)}, n_windows={len(self.parent_windows)}>"

    def summary(self) -> str:
        """Return formatted multi-scale text summary with child matches and hydrated parent sequence statistics."""
        lines = [
            "=== Hierarchical Parent-Child Physics-RAG Results ===",
            f"Query: \"{self.query}\"",
            f"Child Event Matches (Top-k): {len(self.child_matches_df)}",
            f"Hydrated Parent Windows: {len(self.parent_windows)}",
            "",
            "Multi-Scale Physical Diagnostic Narrative:",
            f"{self.llm_explanation or self.explanation}",
            "",
            "Top Matching Child Observations (Instantaneous Physical States):",
        ]

        display_cols = [
            c for c in [
                "timestamp", "hs_m", "tp_s", "surge_m", "wind_normal_ms",
                "xi_iribarren", "breaker_regime", "upsilon", "upsilon_diseq", "delta_x_m", "similarity"
            ] if c in self.child_matches_df.columns
        ]

        table_str = self.child_matches_df[display_cols].to_string(index=False)
        lines.append(table_str)

        if self.parent_windows:
            lines.append("")
            lines.append("Hydrated Parent Context Windows (Temporal Sequence Evolution):")
            for idx, pw in enumerate(self.parent_windows[:3]):
                lines.append(f"\n--- Parent Window #{idx + 1} for Event at {pw.event_timestamp} (Similarity: {pw.similarity:.3f}) ---")
                lines.append(f"Window Span: {pw.window_start} to {pw.window_end} (Antecedent: -{pw.pre_window_hours}h, Post: +{pw.post_window_hours}h, Timesteps: {len(pw.parent_df) if pw.parent_df is not None else 0})")
                stats = pw.summary_stats
                if stats:
                    lines.append("Key Temporal Statistics:")
                    lines.append(
                        f"  * Peak Hs: {stats.get('peak_hs_m', 0.0):.2f}m | Mean Hs: {stats.get('mean_hs_m', 0.0):.2f}m | Hs Rise Rate: {stats.get('hs_rise_rate_m_per_hr', 0.0):.3f} m/hr"
                    )
                    lines.append(
                        f"  * Peak Tp: {stats.get('peak_tp_s', 0.0):.1f}s | Peak Surge: {stats.get('peak_surge_m', 0.0):.2f}m | Surge Duration (>0.3m): {stats.get('surge_duration_hours', 0.0):.1f}h"
                    )
                    lines.append(
                        f"  * Peak Index Υ: {stats.get('peak_upsilon', 0.0):.1e} | Cumulative Wave Energy Impulse: {stats.get('cumulative_power_impulse_MJ_per_m', 0.0):.1f} MJ/m"
                    )
                    lines.append(
                        f"  * Shoreline Displacement ΔX: Start {stats.get('delta_x_start_m', 0.0):.2f}m -> End {stats.get('delta_x_end_m', 0.0):.2f}m (Net: {stats.get('net_delta_x_m', 0.0):+.2f}m)"
                    )
                    lines.append(
                        f"  * Mean Disequilibrium (Υ_diseq): {stats.get('mean_upsilon_diseq', 0.0):.1e} | Recovery Fraction: {stats.get('recovery_phase_fraction', 0.0)*100:.1f}%"
                    )

        return "\n".join(lines)


class PhysicsRAGResult:
    """Structured container for Physics-RAG search results with explainability."""

    def __init__(
        self,
        query: str,
        matches_df: pd.DataFrame,
        similarities: np.ndarray,
        explanation: str,
        parsed_criteria: Union[Dict[str, Any], PhysicsQuerySpec],
        llm_explanation: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
        recovery_episodes: Optional[List[Dict[str, Any]]] = None,
    ):
        self.query = query
        self.matches_df = matches_df
        self.similarities = similarities
        self.explanation = explanation
        self.parsed_criteria = parsed_criteria
        self.llm_explanation = llm_explanation
        self.query_vector = query_vector
        self.metrics = metrics or {}
        self.recovery_episodes = recovery_episodes or []

    def __repr__(self) -> str:
        return f"<PhysicsRAGResult for query='{self.query}', n_matches={len(self.matches_df)}>"

    def summary(self) -> str:
        """Return formatted text summary of the retrieved physics states."""
        lines = [
            f"=== Physics-Inspired RAG Results ===",
            f"Query: \"{self.query}\"",
            f"Retrieved Instances: {len(self.matches_df)}",
            "",
            "Physical Explanation:",
            f"{self.llm_explanation or self.explanation}",
            "",
            "Top Matching Observations:",
        ]

        display_cols = [
            c for c in [
                "timestamp", "hs_m", "tp_s", "surge_m", "wind_normal_ms",
                "xi_iribarren", "breaker_regime", "upsilon", "upsilon_diseq", "delta_x_m",
                "similarity", "rrf_score"
            ] if c in self.matches_df.columns
        ]

        table_str = self.matches_df[display_cols].to_string(index=False)
        lines.append(table_str)
        return "\n".join(lines)


class PhysicsRAG:
    """Physics-Inspired Retrieval-Augmented Generation Engine for Coastal Systems (Suggestion 8)."""

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        scaler_meta: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = None,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        latent_embeddings: Optional[np.ndarray] = None,
        top_k: int = 5,
        similarity_metric: str = "cosine",
        use_weighted_similarity: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        """Initialize the PhysicsRAG engine.

        Parameters
        ----------
        df : pd.DataFrame
            Underlying coastal hydrodynamic & morphodynamic dataset.
        embeddings : np.ndarray, optional
            Physics-anchored latent vectors (N x D).
        feature_names : list[str], optional
            Feature dimension names.
        scaler_meta : dict, optional
            Metadata containing fitted scalers and transforms.
        scaler : Scaler, optional
            Direct fitted scaler object.
        model_name : str
            Gemini model identifier (default: 'gemini-2.5-flash').
        api_key : str, optional
            Gemini API key (defaults to GEMINI_API_KEY / GOOGLE_API_KEY in environment).
        latent_embeddings : np.ndarray, optional
            Alias for embeddings.
        top_k : int
            Default retrieval count.
        similarity_metric : str
            Distance / similarity metric ('cosine', 'euclidean', 'manhattan').
        use_weighted_similarity : bool
            Whether to use Upsilon-weighted cosine similarity (Suggestion 5).
        feature_weights : dict, optional
            Dimension importance weights (defaults to DEFAULT_UPSILON_FEATURE_WEIGHTS).
        """
        self.df = df.copy()
        
        # Improvement 4: Sandy Coast Breaker Regime Harmonization (Wright & Short 1984, Battjes 1974)
        if "xi_iribarren" in self.df.columns and ("breaker_regime" not in self.df.columns or (self.df["breaker_regime"] == "Surging").sum() == 0):
            xi = self.df["xi_iribarren"].values
            self.df["breaker_regime"] = np.where(xi < 0.5, "Spilling", np.where(xi < 1.0, "Plunging", "Surging"))
        
        # Handle embeddings / latent_embeddings alias
        emb = latent_embeddings if latent_embeddings is not None else embeddings
        if emb is None:
            emb, names, meta = compute_latent_embeddings(self.df)
            if feature_names is None:
                feature_names = names
            if scaler_meta is None:
                scaler_meta = meta
                
        self.embeddings = np.asarray(emb)
        self.feature_names = feature_names or []
        self.scaler_meta = scaler_meta or {}
        self.model_name = model_name
        self.api_key = api_key
        self.top_k = top_k
        self.similarity_metric = similarity_metric
        self.use_weighted_similarity = use_weighted_similarity
        self.feature_weights = feature_weights if feature_weights is not None else DEFAULT_UPSILON_FEATURE_WEIGHTS

        # Improvement 5: Physical Energy-Flux Latent Weighting (w_Upsilon metric tensor)
        # Precompute W^{1/2} weight vector for O(1) weighted cosine similarity:
        # Sim(q, x) = (q W x^T) / (||q||_W ||x||_W) == ((q W^{1/2}) . (x W^{1/2})) / (||q W^{1/2}|| ||x W^{1/2}||)
        if self.feature_names and len(self.feature_names) == self.embeddings.shape[1]:
            w_list = []
            for fn in self.feature_names:
                w = self.feature_weights.get(fn)
                if w is None:
                    fn_lower = fn.lower()
                    for k, val in self.feature_weights.items():
                        if k.lower() == fn_lower or k.lower() in fn_lower or fn_lower in k.lower():
                            w = val
                            break
                w_list.append(float(w if w is not None else 1.0))
            w_arr = np.array(w_list, dtype=float)
        elif self.embeddings.shape[1] == 14:
            # Canonical 14D invariant ordering default:
            # [log_p_flux, log_upsilon, dean_number, xi_iribarren, wave_steepness, relative_surge,
            #  wind_shear_stress, recovery_potential, radiation_stress_gradient, water_level_excursion,
            #  dissipation_ratio, sediment_mobility, infragravity_proxy, upsilon_disequilibrium]
            w_arr = np.array([2.0, 2.5, 2.0, 2.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 2.5], dtype=float)
        else:
            w_arr = np.ones(self.embeddings.shape[1], dtype=float)

        self.feature_weight_vec = w_arr
        self.sqrt_feature_weight_vec = np.sqrt(w_arr)
        if self.use_weighted_similarity:
            self.weighted_embeddings = self.embeddings * self.sqrt_feature_weight_vec
        else:
            self.weighted_embeddings = self.embeddings

        self.scaler = scaler if scaler is not None else self.scaler_meta.get("scaler")
        if self.scaler is None and hasattr(self.df, "columns"):
            try:
                if self.feature_names and len(self.feature_names) == self.embeddings.shape[1] and len(self.feature_names) != 14:
                    avail_cols = [c for c in self.feature_names if c in self.df.columns]
                    X_raw = self.df[avail_cols].fillna(0.0).values
                else:
                    X_raw, _ = build_physics_anchored_feature_matrix(self.df)
                scaler_type = self.scaler_meta.get("scaler_type", "robust")
                scaler_obj = RobustScaler() if scaler_type == "robust" else StandardScaler()
                scaler_obj.fit(X_raw)
                self.scaler = scaler_obj
                self.scaler_meta["scaler"] = scaler_obj
            except Exception:
                pass
        self.centroids = compute_regime_centroids(self.embeddings, self.df)
        self.site_equilibrium_stats = compute_site_equilibrium_thresholds(self.df)

    def _compute_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute (optionally Upsilon-weighted) cosine similarity across the latent embedding space (Item 5)."""
        if self.use_weighted_similarity and len(self.sqrt_feature_weight_vec) == self.embeddings.shape[1]:
            weighted_query = (query_vec * self.sqrt_feature_weight_vec).reshape(1, -1)
            return cosine_similarity(weighted_query, self.weighted_embeddings)[0]
        return cosine_similarity(query_vec.reshape(1, -1), self.embeddings)[0]

    @classmethod
    def from_dataset(
        cls,
        df: pd.DataFrame,
        scaler_type: str = "robust",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "PhysicsRAG":
        """Factory method to construct PhysicsRAG directly from a DataFrame."""
        embeddings, feature_names, meta = compute_latent_embeddings(df, scaler_type=scaler_type)
        return cls(
            df=df,
            embeddings=embeddings,
            feature_names=feature_names,
            scaler_meta=meta,
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )

    def _build_query_vector(self, spec: PhysicsQuerySpec) -> np.ndarray:
        """Construct reference latent query vector from structured intent (Suggestion 8)."""
        # 1. Synthesize explicit vector from target values if user specified explicit numbers
        if spec.target_synthetic_values:
            if len(self.feature_names) != 14:
                try:
                    vals = []
                    for fn in self.feature_names:
                        val = None
                        for k, v in spec.target_synthetic_values.items():
                            k_clean = k.lower().replace("_m", "").replace("_s", "").replace("_ms", "")
                            fn_clean = fn.lower().replace("_m", "").replace("_s", "").replace("_ms", "")
                            if k_clean == fn_clean or k_clean in fn_clean or fn_clean in k_clean:
                                val = float(v)
                                break
                        if val is None:
                            val = float(self.df[fn].median()) if fn in self.df.columns else 0.0
                        vals.append(val)
                    raw_vec = np.array([vals])
                    if self.scaler is not None and hasattr(self.scaler, "transform"):
                        return self.scaler.transform(raw_vec)[0]
                    return raw_vec[0]
                except Exception:
                    pass

            try:
                return synthesize_target_physics_vector(
                    target_params=spec.target_synthetic_values,
                    scaler=self.scaler,
                    feature_names=self.feature_names,
                    reference_df=self.df,
                )
            except Exception:
                pass

        # 2. Use named target state centroid if available
        if spec.target_state and spec.target_state in self.centroids:
            return self.centroids[spec.target_state]

        # 3. Item 4: Check phenomenon centroids based on focus, intent, or failure mode
        if spec.focus:
            f_lower = spec.focus.lower()
            if ("infragravity" in f_lower or "surf beat" in f_lower or "surf-beat" in f_lower) and "Centroid_InfragravitySwash" in self.centroids:
                return self.centroids["Centroid_InfragravitySwash"]
            if ("macrotidal" in f_lower or "ridge-and-runnel" in f_lower or "tidal smearing" in f_lower) and "Centroid_MacrotidalDissipation" in self.centroids:
                return self.centroids["Centroid_MacrotidalDissipation"]
            if ("sandbar" in f_lower or "bar dissipation" in f_lower or "bar-breach" in f_lower) and "Centroid_BarDissipation" in self.centroids:
                return self.centroids["Centroid_BarDissipation"]
            if ("compound surge" in f_lower or "dune scarping" in f_lower or "surge scarping" in f_lower) and "Centroid_CompoundSurge" in self.centroids:
                return self.centroids["Centroid_CompoundSurge"]

        if spec.is_failure_mode and "Centroid_CompoundSurge" in self.centroids:
            return self.centroids["Centroid_CompoundSurge"]

        if spec.raw_query:
            q_lower = spec.raw_query.lower()
            if ("infragravity" in q_lower or "surf beat" in q_lower or "surf-beat" in q_lower) and "Centroid_InfragravitySwash" in self.centroids:
                return self.centroids["Centroid_InfragravitySwash"]
            if ("macrotidal" in q_lower or "ridge-and-runnel" in q_lower or "tidal smearing" in q_lower) and "Centroid_MacrotidalDissipation" in self.centroids:
                return self.centroids["Centroid_MacrotidalDissipation"]
            if ("sandbar" in q_lower or "bar dissipation" in q_lower or "bar-breach" in q_lower) and "Centroid_BarDissipation" in self.centroids:
                return self.centroids["Centroid_BarDissipation"]
            if ("compound surge" in q_lower or "dune scarping" in q_lower or "twl scarping" in q_lower) and "Centroid_CompoundSurge" in self.centroids:
                return self.centroids["Centroid_CompoundSurge"]

        # 4. Suggestion 8: Check phase centroid if available
        if spec.phase == "Recovery" and "State_PostStormRecovery" in self.centroids:
            return self.centroids["State_PostStormRecovery"]
        if spec.phase == "Erosion" and "State_StormErosion" in self.centroids:
            return self.centroids["State_StormErosion"]
        if spec.phase == "Noise" and "State_SubcriticalNoise" in self.centroids:
            return self.centroids["State_SubcriticalNoise"]

        # 5. Check if breaker regime centroid is requested
        if spec.breaker_regime:
            key = f"Breaker_{spec.breaker_regime}"
            if key in self.centroids:
                return self.centroids[key]

        # 6. Default to mean dataset embedding
        return np.mean(self.embeddings, axis=0)

    def _apply_hard_and_soft_filters(
        self,
        spec: PhysicsQuerySpec,
        strict_bounds: bool = True,
        disable_spatiotemporal_filters: bool = False,
    ) -> np.ndarray:
        """Generate boolean mask for candidates satisfying physical constraints (Suggestion 8).

        Parameters
        ----------
        spec : PhysicsQuerySpec
            Structured physical query specifications.
        strict_bounds : bool, default True
            If True, enforces zero-leak exact boundaries (disables 10% measurement tolerance padding)
            for explicit diagnostic queries or quantitative thresholds (e.g. Hs < 2.0m, Tp >= 13.0s).
            If False, applies 10% measurement tolerance for fuzzy semantic probing.
        disable_spatiotemporal_filters : bool, default False
            If True (e.g. Mode A: Pure Phenomenological Search), disables site and date filters
            to evaluate pure hydrodynamic similitude across the entire global multi-site dataset.
        """
        mask = np.ones(len(self.df), dtype=bool)

        # Suggestion 8: Threshold Anchor (Upsilon_crit) Background Noise Pruning
        if spec.filter_background_noise:
            if "upsilon_excess" in self.df.columns:
                mask = mask & (self.df["upsilon_excess"].values > 0)
            elif "upsilon_crit" in self.df.columns and "upsilon" in self.df.columns:
                mask = mask & (self.df["upsilon"].values >= self.df["upsilon_crit"].values)
            elif "upsilon" in self.df.columns:
                mask = mask & (self.df["upsilon"].values >= 500.0)

        # Suggestion 8: State-Space Phase Navigation (Recovery vs Erosion vs Noise)
        if spec.phase == "Recovery":
            if "upsilon_diseq" in self.df.columns:
                mask = mask & (self.df["upsilon_diseq"].values < 0)
            elif "signed_equilibrium_rate" in self.df.columns:
                mask = mask & (self.df["signed_equilibrium_rate"].values > 0)
            elif "delta_x_m" in self.df.columns:
                mask = mask & (self.df["delta_x_m"].values >= 0)
        elif spec.phase == "Erosion":
            if "upsilon_diseq" in self.df.columns:
                mask = mask & (self.df["upsilon_diseq"].values > 0)
            elif "signed_equilibrium_rate" in self.df.columns:
                mask = mask & (self.df["signed_equilibrium_rate"].values < 0)
            elif "delta_x_m" in self.df.columns:
                mask = mask & (self.df["delta_x_m"].values < 0)
        elif spec.phase == "Noise":
            if "upsilon_crit" in self.df.columns and "upsilon" in self.df.columns:
                mask = mask & (self.df["upsilon"].values < self.df["upsilon_crit"].values)
            elif "upsilon" in self.df.columns:
                mask = mask & (self.df["upsilon"].values < 500.0)

        # Breaker regime exact match
        if spec.breaker_regime and "breaker_regime" in self.df.columns:
            mask = mask & (self.df["breaker_regime"].values == spec.breaker_regime)

        # Improvement 1: Gated Precision Filtering (Zero-Leak Boundary Mode)
        # If strict_bounds is True, disable 10% measurement tolerance padding for exact threshold gating.
        # Retain 10% tolerance only when strict_bounds is explicitly False.
        low_factor = 1.0 if strict_bounds else 0.9
        high_factor = 1.0 if strict_bounds else 1.1

        # Wave height limits
        if spec.hs_min is not None and "hs_m" in self.df.columns:
            mask = mask & (self.df["hs_m"].values >= spec.hs_min * low_factor)
        if spec.hs_max is not None and "hs_m" in self.df.columns:
            mask = mask & (self.df["hs_m"].values <= spec.hs_max * high_factor)

        # Wave period limits
        if spec.tp_min is not None and "tp_s" in self.df.columns:
            mask = mask & (self.df["tp_s"].values >= spec.tp_min * low_factor)
        if spec.tp_max is not None and "tp_s" in self.df.columns:
            mask = mask & (self.df["tp_s"].values <= spec.tp_max * high_factor)

        # Water level / Surge limits (Improvement 3: Symmetric Water Level & Tidal Phase Routing)
        is_twl_query = (
            ("total_water_level_m" in self.df.columns) and 
            ((spec.surge_min is not None and spec.surge_min > 0.5) or 
             (spec.surge_max is not None and spec.surge_max < 1.5))
        )
        wl_col = "total_water_level_m" if is_twl_query else "surge_m"
        if spec.surge_min is not None and wl_col in self.df.columns:
            mask = mask & (self.df[wl_col].values >= spec.surge_min * low_factor)
        if spec.surge_max is not None and wl_col in self.df.columns:
            mask = mask & (self.df[wl_col].values <= spec.surge_max * high_factor)

        # Wind limits (support wind_speed_ms or wind_normal_ms with measurement tolerance)
        w_col = "wind_speed_ms" if ("wind_speed_ms" in self.df.columns and spec.wind_normal_min is not None and spec.wind_normal_min > 6.0) else "wind_normal_ms"
        if spec.wind_normal_min is not None and w_col in self.df.columns:
            mask = mask & (self.df[w_col].values >= spec.wind_normal_min * low_factor)
        if spec.wind_normal_max is not None and w_col in self.df.columns:
            mask = mask & (self.df[w_col].values <= spec.wind_normal_max * high_factor)

        # Iribarren limits
        if spec.xi_min is not None and "xi_iribarren" in self.df.columns:
            mask = mask & (self.df["xi_iribarren"].values >= spec.xi_min)
        if spec.xi_max is not None and "xi_iribarren" in self.df.columns:
            mask = mask & (self.df["xi_iribarren"].values <= spec.xi_max)

        # Morphology limits
        if spec.delta_x_min is not None and "delta_x_m" in self.df.columns:
            mask = mask & (self.df["delta_x_m"].values >= spec.delta_x_min)
        if spec.delta_x_max is not None and "delta_x_m" in self.df.columns:
            mask = mask & (self.df["delta_x_m"].values <= spec.delta_x_max)

        # Vulnerability percentiles
        if spec.min_upsilon_percentile is not None and "upsilon" in self.df.columns:
            p_val = np.percentile(self.df["upsilon"].values, spec.min_upsilon_percentile)
            mask = mask & (self.df["upsilon"].values >= p_val)
        if spec.max_upsilon_percentile is not None and "upsilon" in self.df.columns:
            p_val = np.percentile(self.df["upsilon"].values, spec.max_upsilon_percentile)
            mask = mask & (self.df["upsilon"].values <= p_val)

        # Date window filter (bypassed in Mode A: Pure Phenomenological Search)
        if not disable_spatiotemporal_filters and (spec.date_start or spec.date_end) and "timestamp" in self.df.columns:
            try:
                ts = pd.to_datetime(self.df["timestamp"], utc=True)
                if spec.date_start:
                    mask = mask & (ts >= pd.to_datetime(spec.date_start, utc=True)).values
                if spec.date_end:
                    mask = mask & (ts <= pd.to_datetime(spec.date_end, utc=True)).values
            except Exception as e:
                logger.error(f"Date filtering error: {e}")

        # Dynamic site_id & geomorphic_regime pre-filtering (bypassed in Mode A: Pure Phenomenological Search)
        site_col = "site_id" if "site_id" in self.df.columns else ("site" if "site" in self.df.columns else None)
        if not disable_spatiotemporal_filters and spec.site_id and site_col:
            site_mask = (self.df[site_col].astype(str).str.lower() == spec.site_id.lower()).values
            if np.any(mask & site_mask):
                mask = mask & site_mask
        if spec.geomorphic_regime:
            target = spec.geomorphic_regime.lower()
            regime_mask = np.zeros(len(self.df), dtype=bool)
            if "geomorphic_regime" in self.df.columns:
                regime_mask = self.df["geomorphic_regime"].astype(str).str.lower().str.contains(target, na=False).values
            if not np.any(regime_mask) and target in ["embayed", "pocket"]:
                if site_col:
                    regime_mask = self.df[site_col].isin(["narrabeen_aus", "torrey_pines_ca"]).values
            if np.any(mask & regime_mask):
                mask = mask & regime_mask

        # Fallback if filters are overly restrictive (return all candidates)
        if not np.any(mask):
            mask = np.ones(len(self.df), dtype=bool)

        return mask

    def extract_recovery_episodes(
        self,
        c_accretion: float = 1.8e-5,
        min_recovery_steps: int = 4,
    ) -> List[Dict[str, Any]]:
        """Extract and summarize all post-storm recovery episodes from the dataset (Suggestion 8)."""
        return extract_recovery_timeframes(
            self.df,
            c_accretion=c_accretion,
            min_recovery_steps=min_recovery_steps,
        )

    def prune_noise_for_llm(
        self,
        upsilon_crit: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """Prune subcritical background noise (Upsilon < Upsilon_crit) for agentic context optimization."""
        return prune_subcritical_noise_for_agent(
            df=self.df,
            embeddings=self.embeddings,
            upsilon_crit=upsilon_crit,
        )

    def _generate_explanation(
        self,
        spec: PhysicsQuerySpec,
        top_df: pd.DataFrame,
    ) -> str:
        """Synthesize deterministic physical explainability narrative for query results (Suggestion 8)."""
        if top_df.empty:
            return "No matching coastal state instances found for the specified physical parameters."

        regime = spec.breaker_regime
        avg_hs = top_df["hs_m"].mean() if "hs_m" in top_df.columns else 0.0
        avg_tp = top_df["tp_s"].mean() if "tp_s" in top_df.columns else 0.0
        avg_xi = top_df["xi_iribarren"].mean() if "xi_iribarren" in top_df.columns else 0.0
        avg_upsilon = top_df["upsilon"].mean() if "upsilon" in top_df.columns else 0.0
        avg_diseq = top_df["upsilon_diseq"].mean() if "upsilon_diseq" in top_df.columns else 0.0
        avg_surge = top_df["surge_m"].mean() if "surge_m" in top_df.columns else 0.0
        avg_dx = top_df["delta_x_m"].mean() if "delta_x_m" in top_df.columns else 0.0

        parts = []

        if spec.query_recovery_timeframe or spec.phase == "Recovery":
            parts.append(
                f"Retrieved {len(top_df)} post-storm constructive recovery states characterized by negative Signed Disequilibrium "
                f"(mean Υ_diseq={avg_diseq:.1e}, Υ={avg_upsilon:.1e} < Υ_eq). Hydrodynamic forcing is dominated by swash berm welding "
                f"(mean Hs={avg_hs:.2f}m, Tp={avg_tp:.1f}s) driving positive shoreline advance (mean ΔX = +{avg_dx:.2f}m)."
            )
        elif regime == "Spilling":
            parts.append(
                f"Identified {len(top_df)} spilling breaker instances characterized by dissipative hydrodynamic states "
                f"(surf similarity parameter ξ < 0.5, mean ξ={avg_xi:.2f}). These reflect steep waves "
                f"(mean Hs={avg_hs:.2f}m, Tp={avg_tp:.1f}s) dissipating energy gradually across a broad surf zone."
            )
        elif regime == "Plunging":
            parts.append(
                f"Identified {len(top_df)} plunging breaker instances occupying the intermediate surf regime "
                f"(0.5 <= ξ < 3.3, mean ξ={avg_xi:.2f}). These correspond to dynamic wave steepness conditions "
                f"(mean Hs={avg_hs:.2f}m, Tp={avg_tp:.1f}s) producing violent curling jets."
            )
        elif regime == "Surging":
            parts.append(
                f"Identified {len(top_df)} surging/collapsing breaker instances in reflective conditions "
                f"(ξ >= 3.3, mean ξ={avg_xi:.2f}). Typically occurring with long-period swell (Tp={avg_tp:.1f}s) "
                f"and low significant wave heights (Hs={avg_hs:.2f}m) where waves surge directly up the beach face."
            )
        else:
            parts.append(
                f"Retrieved {len(top_df)} nearest physical state vectors matching query intent. "
                f"Mean hydrodynamic conditions: Hs={avg_hs:.2f}m, Tp={avg_tp:.1f}s, Surge={avg_surge:.2f}m, ξ={avg_xi:.2f}."
            )

        if spec.filter_background_noise:
            parts.append(
                "Subcritical background noise (Υ < Υ_crit) was actively pruned via threshold anchor gating to eliminate context-window bloat."
            )

        if not (spec.query_recovery_timeframe or spec.phase == "Recovery"):
            if avg_dx < -0.05:
                parts.append(
                    f"Morphodynamic response is net erosive with average shoreline displacement ΔX = {avg_dx:.2f}m, "
                    f"coupled with elevated Unified Coastal Vulnerability Index Υ = {avg_upsilon:.1e}."
                )
            elif avg_dx > 0.05:
                parts.append(
                    f"Morphodynamic response shows net berm accretion / recovery with average shoreline displacement ΔX = +{avg_dx:.2f}m, "
                    f"and low baseline vulnerability Υ = {avg_upsilon:.1e}."
                )
            else:
                parts.append(f"Mean Unified Vulnerability Index Υ = {avg_upsilon:.1e}.")

        return " ".join(parts)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = -1.0,
        use_llm_parser: bool = True,
        use_llm_synthesis: bool = True,
        spec: Optional[PhysicsQuerySpec] = None,
        target_synthetic_values: Optional[Dict[str, float]] = None,
        strict_bounds: bool = True,
        mode: Optional[str] = None,
        disable_spatiotemporal_filters: bool = False,
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        """Execute a natural language query against coastal dataset & latent embeddings (Suggestion 8)."""
        is_mode_a = disable_spatiotemporal_filters or (mode is not None and ("mode a" in str(mode).lower() or "phenomenological" in str(mode).lower()))

        # 1. Parse natural language into structured physical intent
        if spec is None:
            if target_synthetic_values is not None:
                spec = PhysicsQuerySpec(
                    raw_query=query_text,
                    intent_summary=f"Target parameter query for '{query_text}'",
                    target_synthetic_values=target_synthetic_values,
                )
            elif use_llm_parser:
                parser_m = kwargs.get("parser_mode", "hybrid")
                spec = GeminiPhysicsQueryParser.parse(
                    query_text=query_text,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    parser_mode=parser_m,
                )
            else:
                spec = GeminiPhysicsQueryParser._fallback_parse(query_text)

        # 2. Build target query embedding vector
        query_vec = self._build_query_vector(spec)

        # 3. Compute vector similarities across latent embedding space (Item 5: Upsilon-weighted)
        sims = self._compute_similarity(query_vec)

        # 4. Filter candidates (including threshold anchor noise pruning & phase navigation)
        filter_mask = self._apply_hard_and_soft_filters(
            spec,
            strict_bounds=strict_bounds,
            disable_spatiotemporal_filters=is_mode_a,
        )
        candidate_indices = np.where(filter_mask)[0]

        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(self.df))

        # 5. Rank filtered candidates by latent similarity
        candidate_sims = sims[candidate_indices]
        sort_order = np.argsort(-candidate_sims)
        ranked_indices = candidate_indices[sort_order]
        ranked_sims = candidate_sims[sort_order]

        # Apply similarity threshold and top_k
        valid_mask = ranked_sims >= min_similarity
        selected_indices = ranked_indices[valid_mask][:top_k]
        selected_sims = ranked_sims[valid_mask][:top_k]

        # 6. Build result DataFrame
        top_df = self.df.iloc[selected_indices].copy()
        top_df["similarity"] = selected_sims

        # 7. Extract recovery analytics if requested
        recov_episodes = []
        if spec.query_recovery_timeframe or spec.phase == "Recovery":
            try:
                recov_episodes = extract_recovery_timeframes(self.df)
            except Exception:
                pass

        # 8. Generate physical explanations
        deterministic_expl = self._generate_explanation(spec, top_df)
        llm_expl = None
        if use_llm_synthesis:
            llm_expl = explain_with_gemini(
                query_text=query_text,
                spec=spec,
                top_df=top_df,
                model_name=self.model_name,
                api_key=self.api_key,
            )

        return PhysicsRAGResult(
            query=query_text,
            matches_df=top_df,
            similarities=selected_sims,
            explanation=deterministic_expl,
            parsed_criteria=spec,
            llm_explanation=llm_expl,
            query_vector=query_vec,
            recovery_episodes=recov_episodes,
        )

    def evaluate_query(
        self,
        query: str,
        condition_fn: Optional[Any] = None,
        target_regime: Optional[str] = None,
        k: int = 5,
        use_llm_parser: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance (Precision@k, Regime Purity@k, Recall@k, MRR) for a single query.

        Parameters
        ----------
        query : str
            Natural language query text.
        condition_fn : callable, optional
            Function condition_fn(row, df) -> bool evaluating if a dataset row meets ground truth.
        target_regime : str, optional
            Expected ground-truth breaker regime ('Spilling', 'Plunging', 'Surging').
        k : int
            Retrieval rank cutoff (default 5).
        use_llm_parser : bool
            Whether to use Gemini structured output parser.

        Returns
        -------
        metrics : dict
            Dictionary of calculated retrieval metrics.
        """
        return evaluate_retrieval_query(
            rag=self,
            query_spec_or_text=query,
            condition_fn=condition_fn,
            target_regime=target_regime,
            k=k,
            use_llm_parser=use_llm_parser,
        )

    def evaluate_benchmark(
        self,
        benchmark_queries: Optional[List[Dict[str, Any]]] = None,
        k_list: List[int] = [1, 3, 5, 10],
        use_llm_parser: bool = False,
        allow_synthetic_probes: bool = False,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run full quantitative retrieval benchmark suite across diverse physical query categories.

        Parameters
        ----------
        benchmark_queries : list[dict], optional
            Custom list of query dicts (defaults to DEFAULT_PHYSICS_BENCHMARK_QUERIES).
        k_list : list[int]
            List of k cutoffs to evaluate (e.g. [1, 3, 5, 10]).
        use_llm_parser : bool
            Whether to use Gemini structured output parser.

        Returns
        -------
        df_results : pd.DataFrame
            Detailed per-query evaluation table.
        summary_metrics : dict
            Macro-averaged benchmark summary metrics (Mean P@1, Mean P@5, Mean Regime Purity, Mean MRR).
        """
        return evaluate_rag_benchmark_suite(
            rag=self,
            benchmark_queries=benchmark_queries,
            k_list=k_list,
            use_llm_parser=use_llm_parser,
            allow_synthetic_probes=allow_synthetic_probes,
            **kwargs,
        )

    def to_hierarchical(
        self,
        parquet_path: Optional[str] = None,
        default_pre_window_hours: float = 72.0,
        default_post_window_hours: float = 24.0,
        db_conn: Optional[Any] = None,
    ) -> "HierarchicalPhysicsRAG":
        """Convert this PhysicsRAG instance into a HierarchicalPhysicsRAG engine with DuckDB sequence store."""
        return HierarchicalPhysicsRAG(
            df=self.df,
            parquet_path=parquet_path,
            embeddings=self.embeddings,
            feature_names=self.feature_names,
            scaler_meta=self.scaler_meta,
            model_name=self.model_name,
            api_key=self.api_key,
            top_k=self.top_k,
            default_pre_window_hours=default_pre_window_hours,
            default_post_window_hours=default_post_window_hours,
            db_conn=db_conn,
        )

    def query_hierarchical(
        self,
        query_text: str,
        top_k: int = 5,
        pre_window_hours: float = 72.0,
        post_window_hours: float = 24.0,
        min_similarity: float = -1.0,
        use_llm_parser: bool = True,
        use_llm_synthesis: bool = True,
        parent_columns: Optional[List[str]] = None,
    ) -> HierarchicalRAGResult:
        """Execute a hierarchical parent-child query, hydrating ±72h continuous time-series context windows."""
        hrag = self.to_hierarchical(
            default_pre_window_hours=pre_window_hours,
            default_post_window_hours=post_window_hours,
        )
        return hrag.query(
            query_text=query_text,
            top_k=top_k,
            pre_window_hours=pre_window_hours,
            post_window_hours=post_window_hours,
            min_similarity=min_similarity,
            use_llm_parser=use_llm_parser,
            use_llm_synthesis=use_llm_synthesis,
            parent_columns=parent_columns,
        )


class HybridPhysicsRAG(PhysicsRAG):
    """Proposed Architecture: Two-Tier Hybrid Cascade Physics-RAG Engine.

    Combines Tier 1 Relational SQL/DuckDB Metadata Slicing (for explicit site IDs
    and calendar date windows) with Tier 2 14D Physics Invariant Manifold Latent Ranking
    and Upsilon Disequilibrium Scoring.

    Architecture:
    1. Query Intent Parsing: Structured extraction of spatiotemporal constraints
       (site_id, date_start, date_end) and hydrodynamic parameters (breaker regime,
       wave height/period bounds, focus, phase).
    2. Tier 1 (Relational Metadata Slicing & Boundary Gate):
       - If site_id or date bounds are stated: slices candidate space strictly down
         to those matching records via high-speed columnar filtering.
       - If unstated (pure phenomenological query): candidate space remains the
         entire multi-site global dataset (N = 175,392).
    3. Tier 2 (14D Physics Invariant Latent Ranking & Disequilibrium):
       - Projects target physical intent into 14D invariant manifold vector.
       - Ranks surviving candidate pool by physical similitude (cosine proximity
         in invariant feature space) + Upsilon disequilibrium + constraint satisfaction.
    4. Tier 3 (Parent Sequence Hydration):
       - Hydrates ±72-hour continuous multi-channel time series around top-ranked events.

    Parameters
    ----------
    alpha : float
        Blending weight for cosine similarity vs constraint satisfaction (default: 0.35).
    regime_boost : float
        Multiplicative boost applied to candidates matching the target breaker regime (default: 0.10).
    """

    # Constraint dimension weights (must sum to 1.0)
    CONSTRAINT_WEIGHTS = {
        "breaker_regime": 0.20,
        "hs_range": 0.15,
        "tp_range": 0.15,
        "twl_range": 0.15,
        "phase": 0.10,
        "xi_range": 0.05,
        "morphology": 0.05,
        "upsilon_percentile": 0.05,
        "wind_range": 0.05,
        "site_regime": 0.05,
    }

    def __init__(
        self,
        *args: Any,
        alpha: float = 0.35,
        regime_boost: float = 0.10,
        rrf_k: int = 60,
        rrf_weights: Optional[Dict[str, float]] = None,
        w_cos: float = 1.0,
        w_constraints: float = 1.0,
        w_upsilon: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.regime_boost = regime_boost
        self.rrf_k = rrf_k
        self.rrf_weights = rrf_weights or {
            "cos": w_cos,
            "constraints": w_constraints,
            "upsilon": w_upsilon,
        }

    @classmethod
    def from_physics_rag(
        cls,
        rag: PhysicsRAG,
        alpha: float = 0.35,
        regime_boost: float = 0.10,
        rrf_k: int = 60,
        rrf_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> "HybridPhysicsRAG":
        """Construct a HybridPhysicsRAG from an existing PhysicsRAG instance (zero-copy)."""
        hybrid = cls.__new__(cls)
        hybrid.df = rag.df
        hybrid.embeddings = rag.embeddings
        hybrid.feature_names = rag.feature_names
        hybrid.scaler_meta = rag.scaler_meta
        hybrid.model_name = rag.model_name
        hybrid.api_key = rag.api_key
        hybrid.top_k = rag.top_k
        hybrid.similarity_metric = rag.similarity_metric
        hybrid.scaler = rag.scaler
        hybrid.centroids = rag.centroids
        hybrid.alpha = alpha
        hybrid.regime_boost = regime_boost
        hybrid.rrf_k = rrf_k
        hybrid.rrf_weights = rrf_weights or {"cos": 1.0, "constraints": 1.0, "upsilon": 1.0}
        hybrid.use_weighted_similarity = getattr(rag, "use_weighted_similarity", True)
        hybrid.feature_weights = getattr(rag, "feature_weights", DEFAULT_UPSILON_FEATURE_WEIGHTS)
        hybrid.feature_weight_vec = getattr(rag, "feature_weight_vec", np.ones(rag.embeddings.shape[1]))
        hybrid.sqrt_feature_weight_vec = getattr(rag, "sqrt_feature_weight_vec", np.ones(rag.embeddings.shape[1]))
        hybrid.weighted_embeddings = getattr(rag, "weighted_embeddings", rag.embeddings)
        return hybrid

    def _compute_constraint_satisfaction_score(
        self,
        spec: PhysicsQuerySpec,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Evaluate constraint satisfaction for each candidate row against parsed query spec."""
        df = self.df
        n = len(candidate_indices)
        if n == 0:
            return np.array([], dtype=float)

        W = self.CONSTRAINT_WEIGHTS
        active_scores = []

        # 1. Breaker regime match
        if spec.breaker_regime and "breaker_regime" in df.columns:
            regime_vals = df["breaker_regime"].values[candidate_indices]
            match = (regime_vals == spec.breaker_regime).astype(float)
            active_scores.append((W["breaker_regime"], match))

        # 2. Continuous wave height compliance (smooth, non-circular exponential loss)
        if (spec.hs_min is not None or spec.hs_max is not None) and "hs_m" in df.columns:
            hs_vals = df["hs_m"].values[candidate_indices]
            score_hs = np.ones(n, dtype=float)
            if spec.hs_min is not None:
                under = np.maximum(0.0, spec.hs_min * 0.9 - hs_vals)
                score_hs *= np.exp(-0.5 * (under / 0.5) ** 2)
            if spec.hs_max is not None:
                over = np.maximum(0.0, hs_vals - spec.hs_max * 1.1)
                score_hs *= np.exp(-0.5 * (over / 0.5) ** 2)
            active_scores.append((W["hs_range"], score_hs))

        # 3. Continuous wave period compliance (smooth, non-circular exponential loss)
        if (spec.tp_min is not None or spec.tp_max is not None) and "tp_s" in df.columns:
            tp_vals = df["tp_s"].values[candidate_indices]
            score_tp = np.ones(n, dtype=float)
            if spec.tp_min is not None:
                under = np.maximum(0.0, spec.tp_min * 0.9 - tp_vals)
                score_tp *= np.exp(-0.5 * (under / 2.0) ** 2)
            if spec.tp_max is not None:
                over = np.maximum(0.0, tp_vals - spec.tp_max * 1.1)
                score_tp *= np.exp(-0.5 * (over / 2.0) ** 2)
            active_scores.append((W["tp_range"], score_tp))

        # 4. Total water level / surge range (with 10% tolerance & exponential loss)
        if (spec.surge_min is not None or spec.surge_max is not None):
            wl_col = "total_water_level_m" if ("total_water_level_m" in df.columns and spec.surge_min is not None and spec.surge_min > 0.5) else "surge_m"
            if wl_col in df.columns:
                wl_vals = df[wl_col].values[candidate_indices]
                wl_ok = np.ones(n, dtype=float)
                if spec.surge_min is not None:
                    under = np.maximum(0.0, spec.surge_min * 0.9 - wl_vals)
                    wl_ok *= np.exp(-0.5 * (under / 0.3) ** 2)
                if spec.surge_max is not None:
                    over = np.maximum(0.0, wl_vals - spec.surge_max * 1.1)
                    wl_ok *= np.exp(-0.5 * (over / 0.3) ** 2)
                active_scores.append((W["twl_range"], wl_ok))

        # 5. Phase match (Erosion / Recovery / Noise)
        if spec.phase:
            if spec.phase == "Recovery" and "upsilon_diseq" in df.columns:
                diseq_vals = df["upsilon_diseq"].values[candidate_indices]
                phase_ok = (diseq_vals < 0).astype(float)
            elif spec.phase == "Erosion" and "upsilon_diseq" in df.columns:
                diseq_vals = df["upsilon_diseq"].values[candidate_indices]
                phase_ok = (diseq_vals > 0).astype(float)
            elif spec.phase == "Erosion" and "delta_x_m" in df.columns:
                dx_vals = df["delta_x_m"].values[candidate_indices]
                phase_ok = (dx_vals < 0).astype(float)
            elif spec.phase == "Recovery" and "delta_x_m" in df.columns:
                dx_vals = df["delta_x_m"].values[candidate_indices]
                phase_ok = (dx_vals >= 0).astype(float)
            else:
                phase_ok = np.ones(n, dtype=float) * 0.5
            active_scores.append((W["phase"], phase_ok))

        # 6. Iribarren range
        if (spec.xi_min is not None or spec.xi_max is not None) and "xi_iribarren" in df.columns:
            xi_vals = df["xi_iribarren"].values[candidate_indices]
            xi_ok = np.ones(n, dtype=float)
            if spec.xi_min is not None:
                xi_ok *= (xi_vals >= spec.xi_min).astype(float)
            if spec.xi_max is not None:
                xi_ok *= (xi_vals <= spec.xi_max).astype(float)
            active_scores.append((W["xi_range"], xi_ok))

        # 7. Morphology (delta_x direction)
        if (spec.delta_x_min is not None or spec.delta_x_max is not None) and "delta_x_m" in df.columns:
            dx_vals = df["delta_x_m"].values[candidate_indices]
            morph_ok = np.ones(n, dtype=float)
            if spec.delta_x_min is not None:
                morph_ok *= (dx_vals >= spec.delta_x_min).astype(float)
            if spec.delta_x_max is not None:
                morph_ok *= (dx_vals <= spec.delta_x_max).astype(float)
            active_scores.append((W["morphology"], morph_ok))

        # 8. Upsilon percentile
        if (spec.min_upsilon_percentile is not None or spec.max_upsilon_percentile is not None) and "upsilon" in df.columns:
            ups_vals = df["upsilon"].values[candidate_indices]
            ups_ok = np.ones(n, dtype=float)
            if spec.min_upsilon_percentile is not None:
                p_val = np.percentile(df["upsilon"].values, spec.min_upsilon_percentile)
                ups_ok *= (ups_vals >= p_val).astype(float)
            if spec.max_upsilon_percentile is not None:
                p_val = np.percentile(df["upsilon"].values, spec.max_upsilon_percentile)
                ups_ok *= (ups_vals <= p_val).astype(float)
            active_scores.append((W["upsilon_percentile"], ups_ok))

        # 9. Wind speed / normal wind range
        if spec.wind_normal_min is not None:
            w_col = "wind_speed_ms" if ("wind_speed_ms" in df.columns and spec.wind_normal_min > 6.0) else "wind_normal_ms"
            if w_col in df.columns:
                w_vals = df[w_col].values[candidate_indices]
                under = np.maximum(0.0, spec.wind_normal_min * 0.9 - w_vals)
                w_ok = np.exp(-0.5 * (under / 2.0) ** 2)
                active_scores.append((W["wind_range"], w_ok))

        # 10. Site / Geomorphic regime match
        site_col = "site_id" if "site_id" in df.columns else ("site" if "site" in df.columns else None)
        if spec.site_id and site_col:
            site_vals = df[site_col].astype(str).str.lower().values[candidate_indices]
            site_match = (site_vals == spec.site_id.lower()).astype(float)
            active_scores.append((W["site_regime"], site_match))
        elif spec.geomorphic_regime and "geomorphic_regime" in df.columns:
            reg_vals = df["geomorphic_regime"].astype(str).str.lower().values[candidate_indices]
            reg_match = (reg_vals == spec.geomorphic_regime.lower()).astype(float)
            active_scores.append((W["site_regime"], reg_match))

        # Continuous physical target proximity
        synth = _extract_target_synthetic_dict(spec.target_synthetic_values)
        prox_scores = []
        if synth.get("hs_m") is not None and "hs_m" in df.columns:
            try:
                t_hs = float(synth["hs_m"])
                vals = df["hs_m"].values[candidate_indices]
                p = np.exp(-0.5 * ((vals - t_hs) / max(0.8, t_hs * 0.3)) ** 2)
                prox_scores.append((0.25, p))
            except (TypeError, ValueError):
                pass
        if synth.get("tp_s") is not None and "tp_s" in df.columns:
            try:
                t_tp = float(synth["tp_s"])
                vals = df["tp_s"].values[candidate_indices]
                p = np.exp(-0.5 * ((vals - t_tp) / max(2.0, t_tp * 0.25)) ** 2)
                prox_scores.append((0.20, p))
            except (TypeError, ValueError):
                pass
        if synth.get("surge_m") is not None and "surge_m" in df.columns:
            try:
                t_s = float(synth["surge_m"])
                vals = df["surge_m"].values[candidate_indices]
                p = np.exp(-0.5 * ((vals - t_s) / 0.4) ** 2)
                prox_scores.append((0.15, p))
            except (TypeError, ValueError):
                pass
        if synth.get("xi_iribarren") is not None and "xi_iribarren" in df.columns:
            try:
                t_xi = float(synth["xi_iribarren"])
                vals = df["xi_iribarren"].values[candidate_indices]
                p = np.exp(-0.5 * ((vals - t_xi) / max(0.5, t_xi * 0.3)) ** 2)
                prox_scores.append((0.15, p))
            except (TypeError, ValueError):
                pass

        if not active_scores and not prox_scores:
            return np.ones(n, dtype=float)

        if active_scores:
            total_weight = sum(w for w, _ in active_scores)
            weighted_sum = np.zeros(n, dtype=float)
            for w, s in active_scores:
                weighted_sum += (w / total_weight) * s
        else:
            weighted_sum = np.ones(n, dtype=float)

        if prox_scores:
            tot_p_w = sum(w for w, _ in prox_scores)
            p_sum = sum(w * p for w, p in prox_scores) / tot_p_w
            return 0.6 * weighted_sum + 0.4 * p_sum

        return weighted_sum

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = -1.0,
        use_llm_parser: bool = True,
        use_llm_synthesis: bool = True,
        spec: Optional[PhysicsQuerySpec] = None,
        target_synthetic_values: Optional[Dict[str, float]] = None,
        mode: Optional[str] = None,
        disable_spatiotemporal_filters: bool = False,
        strict_bounds: bool = True,
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        """Execute Two-Tier Hybrid Cascade Query: Relational SQL Slicing + 14D Invariant Latent Ranking."""
        is_mode_a = disable_spatiotemporal_filters or (mode is not None and ("mode a" in str(mode).lower() or "phenomenological" in str(mode).lower()))

        # 1. Parse query intent
        if spec is None:
            nl_spec = GeminiPhysicsQueryParser._fallback_parse(query_text)
            if target_synthetic_values is not None:
                spec = PhysicsQuerySpec(
                    raw_query=query_text,
                    intent_summary=f"Hybrid target parameter query for '{query_text}'",
                    target_synthetic_values=target_synthetic_values,
                    breaker_regime=nl_spec.breaker_regime,
                    morphology=nl_spec.morphology,
                    phase=nl_spec.phase,
                    filter_background_noise=nl_spec.filter_background_noise,
                    query_recovery_timeframe=nl_spec.query_recovery_timeframe,
                    target_state=nl_spec.target_state,
                    focus=nl_spec.focus,
                    site_id=nl_spec.site_id,
                    geomorphic_regime=nl_spec.geomorphic_regime,
                    hs_min=nl_spec.hs_min,
                    hs_max=nl_spec.hs_max,
                    tp_min=nl_spec.tp_min,
                    tp_max=nl_spec.tp_max,
                    surge_min=nl_spec.surge_min,
                    surge_max=nl_spec.surge_max,
                    xi_min=nl_spec.xi_min,
                    xi_max=nl_spec.xi_max,
                    delta_x_min=nl_spec.delta_x_min,
                    delta_x_max=nl_spec.delta_x_max,
                    wind_normal_min=nl_spec.wind_normal_min,
                    wind_normal_max=nl_spec.wind_normal_max,
                    date_start=nl_spec.date_start,
                    date_end=nl_spec.date_end,
                    min_upsilon_percentile=nl_spec.min_upsilon_percentile,
                    max_upsilon_percentile=nl_spec.max_upsilon_percentile,
                )
            elif use_llm_parser:
                parser_m = str(kwargs.get("parser_mode", "hybrid")).lower().strip()
                if parser_m in ["pure_regex", "regex", "deterministic"]:
                    spec = nl_spec
                else:
                    spec = GeminiPhysicsQueryParser.parse(
                        query_text=query_text,
                        model_name=self.model_name,
                        api_key=self.api_key,
                        parser_mode=parser_m,
                    )
                    if parser_m not in ["pure_gemini", "gemini_only", "zero_shot"]:
                        if not spec.target_synthetic_values and nl_spec.target_synthetic_values:
                            spec.target_synthetic_values = nl_spec.target_synthetic_values
                        for attr in ["breaker_regime", "site_id", "geomorphic_regime", "hs_min", "tp_min", "surge_min", "wind_normal_min", "focus", "is_failure_mode", "date_start", "date_end"]:
                            if getattr(spec, attr, None) is None and getattr(nl_spec, attr, None) is not None:
                                setattr(spec, attr, getattr(nl_spec, attr))
            else:
                spec = nl_spec

        # 2. Tier 1: Relational Metadata Slicing (SQL Hard Filter on Sites & Calendar Dates)
        meta_mask = np.ones(len(self.df), dtype=bool)
        if not is_mode_a:
            site_col = "site_id" if "site_id" in self.df.columns else ("site" if "site" in self.df.columns else None)
            if spec.site_id and site_col:
                site_m = (self.df[site_col].astype(str).str.lower() == spec.site_id.lower()).values
                if np.any(site_m):
                    meta_mask = meta_mask & site_m
            elif spec.geomorphic_regime and "geomorphic_regime" in self.df.columns:
                geom_m = (self.df["geomorphic_regime"].astype(str).str.lower() == spec.geomorphic_regime.lower()).values
                if np.any(geom_m):
                    meta_mask = meta_mask & geom_m

            if (spec.date_start or spec.date_end) and "timestamp" in self.df.columns:
                try:
                    ts = pd.to_datetime(self.df["timestamp"], utc=True)
                    if spec.date_start:
                        meta_mask = meta_mask & (ts >= pd.to_datetime(spec.date_start, utc=True)).values
                    if spec.date_end:
                        meta_mask = meta_mask & (ts <= pd.to_datetime(spec.date_end, utc=True)).values
                except Exception as e:
                    logger.error(f"HybridPhysicsRAG date filtering failed: {e}")

        # Physical threshold gate (Tier 1 physical bounds)
        phys_mask = self._apply_hard_and_soft_filters(
            spec,
            strict_bounds=strict_bounds,
            disable_spatiotemporal_filters=is_mode_a,
        )
        combined_mask = meta_mask & phys_mask
        candidate_indices = np.where(combined_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.where(meta_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(self.df))

        # 3. Build target query embedding vector
        query_vec = self._build_query_vector(spec)

        # 4. Compute vector similarities across full latent embedding space
        sims = self._compute_similarity(query_vec)

        # Sub-select nearest neighbors if candidate space is vast
        M = min(1500, len(candidate_indices))
        cand_sims = sims[candidate_indices]
        if len(candidate_indices) > M:
            sub_order = np.argpartition(-cand_sims, M)[:M]
            candidate_indices = candidate_indices[sub_order]
            cand_sims = cand_sims[sub_order]

        # 5. Tier 2: 14D Invariant Latent Ranking & Universal Disequilibrium Scoring
        c_scores = self._compute_constraint_satisfaction_score(spec, candidate_indices)
        cos_min, cos_max = cand_sims.min(), cand_sims.max()
        if cos_max > cos_min:
            cos_norm = (cand_sims - cos_min) / (cos_max - cos_min)
        else:
            cos_norm = np.ones_like(cand_sims)

        # Breaker regime exact match bonus
        reg_match = np.zeros(len(candidate_indices), dtype=float)
        if spec.breaker_regime and "breaker_regime" in self.df.columns:
            reg_match = (self.df["breaker_regime"].values[candidate_indices] == spec.breaker_regime).astype(float)

        # Direction-Aware Signed Upsilon Disequilibrium Scoring (Erosion, Recovery, Equilibrium)
        q_raw = getattr(spec, "raw_query", "") or ""
        target_intent = determine_upsilon_target_intent(spec, q_raw)

        # 1. Resolve normalized relative forcing Upsilon* = Upsilon / Upsilon_ref
        if "upsilon_star" in self.df.columns:
            u_star_cands = self.df["upsilon_star"].values[candidate_indices]
        elif "upsilon" in self.df.columns and "upsilon_ref" in self.df.columns:
            u_cands = self.df["upsilon"].values[candidate_indices]
            u_ref_cands = np.maximum(self.df["upsilon_ref"].values[candidate_indices], 1e-6)
            u_star_cands = u_cands / u_ref_cands
        elif "upsilon" in self.df.columns:
            site_stats = getattr(self, "site_equilibrium_stats", None) or {}
            site_col = "site_id" if "site_id" in self.df.columns else ("site" if "site" in self.df.columns else None)
            u_cands = self.df["upsilon"].values[candidate_indices]
            u_star_cands = np.zeros(len(candidate_indices), dtype=float)
            for idx_local, c_idx in enumerate(candidate_indices):
                s_name = str(self.df[site_col].iloc[c_idx]) if (site_col and site_col in self.df.columns) else "_global"
                s_stat = site_stats.get(s_name, site_stats.get("_global", {"eq": 1500.0, "crit": 8000.0}))
                u_star_cands[idx_local] = u_cands[idx_local] / max(s_stat.get("eq", 1500.0), 1e-6)
        else:
            u_star_cands = np.ones(len(candidate_indices), dtype=float)

        # 2. Compute Direction-Aware Disequilibrium Score S_Upsilon(t)
        if target_intent == "recovery":
            # Constructive Swash / Accretion: S_Upsilon(t) = ReLU(1.0 - Upsilon*(t))
            # or ReLU((Upsilon_crit - Upsilon(t)) / Upsilon_ref)
            s_ups = np.maximum(0.0, 1.0 - u_star_cands)
            if "upsilon_crit" in self.df.columns and "upsilon" in self.df.columns and "upsilon_ref" in self.df.columns:
                u_crit_cands = self.df["upsilon_crit"].values[candidate_indices]
                u_cands = self.df["upsilon"].values[candidate_indices]
                u_ref_cands = np.maximum(self.df["upsilon_ref"].values[candidate_indices], 1e-6)
                s_crit_deficit = np.maximum(0.0, (u_crit_cands - u_cands) / u_ref_cands)
                s_ups = np.maximum(s_ups, s_crit_deficit)
        elif target_intent == "equilibrium":
            # Equilibrium Background: S_Upsilon(t) = exp(- (Upsilon*(t) - 1.0)^2 / (2 * sigma^2))
            sigma_eq = 0.35
            s_ups = np.exp(-((u_star_cands - 1.0) ** 2) / (2.0 * (sigma_eq ** 2)))
        else:
            # Erosive / Storm Queries: S_Upsilon(t) = ReLU(Upsilon*(t) - 1.0)
            s_ups = np.maximum(0.0, u_star_cands - 1.0)

        # Normalize disequilibrium score across candidate pool to [0, 1]
        ups_norm = np.zeros(len(candidate_indices), dtype=float)
        if len(s_ups) > 0 and s_ups.max() > 0:
            ups_norm = s_ups / s_ups.max()

        # Directional Pareto excess bonus for physical extremes matching query intent
        pareto_boost = np.zeros(len(candidate_indices), dtype=float)
        has_directional_extreme = False

        if target_intent == "erosion":
            effective_hs_min = spec.hs_min
            if effective_hs_min is None:
                synth_dict = _extract_target_synthetic_dict(spec.target_synthetic_values)
                if synth_dict.get("hs_m") is not None and (any(w in q_raw.lower() for w in ["exceeding", "over", "above", "greater than", ">", "storm"]) or target_intent == "erosion"):
                    try:
                        effective_hs_min = float(synth_dict["hs_m"])
                    except (TypeError, ValueError):
                        pass

            if effective_hs_min is not None and "hs_m" in self.df.columns:
                hs_cand = self.df["hs_m"].values[candidate_indices]
                hs_exc = np.maximum(0.0, hs_cand - effective_hs_min)
                if hs_exc.max() > 0:
                    pareto_boost += hs_exc / hs_exc.max()
                    has_directional_extreme = True

            if spec.surge_min is not None and "surge_m" in self.df.columns:
                s_cand = self.df["surge_m"].values[candidate_indices]
                s_exc = np.maximum(0.0, s_cand - spec.surge_min)
                if s_exc.max() > 0:
                    pareto_boost += s_exc / s_exc.max()
                    has_directional_extreme = True

            synth_vals = _extract_target_synthetic_dict(spec.target_synthetic_values)
            if any(w in q_raw.lower() for w in ["power flux", "kw", "wave power"]) or "p_flux_Wm" in synth_vals or "p_flux" in synth_vals:
                if "p_flux_Wm" in self.df.columns:
                    p_cand = self.df["p_flux_Wm"].values[candidate_indices]
                    val = synth_vals.get("p_flux_Wm", synth_vals.get("p_flux"))
                    try:
                        p_thresh = float(val) if val is not None else 50000.0
                    except (TypeError, ValueError):
                        p_thresh = 50000.0
                    p_exc = np.maximum(0.0, p_cand - p_thresh)
                    if p_exc.max() > 0:
                        pareto_boost += p_exc / p_exc.max()
                        has_directional_extreme = True

            if spec.target_state == "State_ExtremeStormPeak" or any(w in q_raw.lower() for w in ["storm peak", "catastrophic", "scarping", "overwash"]):
                has_directional_extreme = True
                pareto_boost += ups_norm
        elif target_intent == "recovery":
            # Constructive Pareto: reward calm low Hs (< 1.5m) and long-period swell (Tp > 10s)
            if "hs_m" in self.df.columns:
                hs_cand = self.df["hs_m"].values[candidate_indices]
                hs_calm = np.maximum(0.0, 1.5 - hs_cand)
                if hs_calm.max() > 0:
                    pareto_boost += hs_calm / hs_calm.max()
                    has_directional_extreme = True
            if "tp_s" in self.df.columns:
                tp_cand = self.df["tp_s"].values[candidate_indices]
                tp_swell = np.maximum(0.0, tp_cand - 10.0)
                if tp_swell.max() > 0:
                    pareto_boost += tp_swell / tp_swell.max()
                    has_directional_extreme = True
        elif target_intent == "equilibrium":
            has_directional_extreme = True
            pareto_boost += ups_norm

        if pareto_boost.max() > 0:
            pareto_boost = pareto_boost / pareto_boost.max()

        # Reciprocal Rank Fusion (RRF) across 3 peer-reviewed channels:
        # 1. Cosine similarity channel (vector embedding match)
        # 2. Constraints satisfaction channel (physical thresholds + breaker regime match)
        # 3. Direction-aware Upsilon disequilibrium channel (hydrodynamic work & energy disequilibrium)
        #
        # Formula (Cormack, Clarke, & Buettcher, SIGIR 2009):
        #   RRF_Score(d) = w_cos / (60 + rank_cos(d))
        #                + w_constraints / (60 + rank_constraints(d))
        #                + w_upsilon / (60 + rank_upsilon(d))

        # Channel 1: Cosine similarity
        cos_channel_scores = cand_sims

        # Channel 2: Physical constraints + breaker regime match bonus
        constraint_channel_scores = c_scores + 0.20 * reg_match

        # Channel 3: Direction-aware Upsilon disequilibrium + directional Pareto extreme signal
        if has_directional_extreme and pareto_boost.max() > 0:
            upsilon_channel_scores = ups_norm + pareto_boost
        else:
            upsilon_channel_scores = s_ups

        # Resolve RRF parameters (default k=60, w_cos=1.0, w_constraints=1.0, w_upsilon=1.0)
        default_weights = getattr(self, "rrf_weights", None) or {"cos": 1.0, "constraints": 1.0, "upsilon": 1.0}
        rrf_k = int(kwargs.get("rrf_k", getattr(self, "rrf_k", 60)))
        query_rrf_weights = kwargs.get("rrf_weights") or {}
        w_cos = float(kwargs.get("w_cos", query_rrf_weights.get("cos", default_weights.get("cos", 1.0))))
        w_constraints = float(kwargs.get("w_constraints", query_rrf_weights.get("constraints", default_weights.get("constraints", 1.0))))
        w_upsilon = float(kwargs.get("w_upsilon", query_rrf_weights.get("upsilon", default_weights.get("upsilon", 1.0))))

        channel_scores = {
            "cos": cos_channel_scores,
            "constraints": constraint_channel_scores,
            "upsilon": upsilon_channel_scores,
        }
        channel_weights = {
            "cos": w_cos,
            "constraints": w_constraints,
            "upsilon": w_upsilon,
        }

        rrf_scores, channel_ranks = compute_reciprocal_rank_fusion(
            scores_dict=channel_scores,
            weights=channel_weights,
            k0=rrf_k,
        )

        rank_cos = channel_ranks["cos"]
        rank_constraints = channel_ranks["constraints"]
        rank_upsilon = channel_ranks["upsilon"]

        sort_order = np.argsort(-rrf_scores)
        ranked_indices = candidate_indices[sort_order]
        ranked_sims = cand_sims[sort_order]
        ranked_rrf = rrf_scores[sort_order]
        ranked_ups_score = ups_norm[sort_order]
        ranked_rank_cos = rank_cos[sort_order]
        ranked_rank_constraints = rank_constraints[sort_order]
        ranked_rank_upsilon = rank_upsilon[sort_order]

        valid_mask = ranked_sims >= min_similarity
        selected_indices = ranked_indices[valid_mask][:top_k]
        selected_sims = ranked_sims[valid_mask][:top_k]
        selected_rrf = ranked_rrf[valid_mask][:top_k]
        selected_ups_score = ranked_ups_score[valid_mask][:top_k]
        selected_rank_cos = ranked_rank_cos[valid_mask][:top_k]
        selected_rank_constraints = ranked_rank_constraints[valid_mask][:top_k]
        selected_rank_upsilon = ranked_rank_upsilon[valid_mask][:top_k]

        top_df = self.df.iloc[selected_indices].copy()
        top_df["similarity"] = selected_sims
        top_df["rrf_score"] = selected_rrf
        top_df["hybrid_score"] = selected_rrf  # Maintained for full backwards compatibility
        top_df["upsilon_diseq_score"] = selected_ups_score
        top_df["target_intent"] = target_intent
        top_df["rank_cos"] = selected_rank_cos
        top_df["rank_constraints"] = selected_rank_constraints
        top_df["rank_upsilon"] = selected_rank_upsilon

        # 6. Extract recovery analytics if requested
        recov_episodes = []
        if spec.query_recovery_timeframe or spec.phase == "Recovery":
            try:
                recov_episodes = extract_recovery_timeframes(self.df)
            except Exception:
                pass

        # 7. Generate physical explanations
        deterministic_expl = self._generate_explanation(spec, top_df)
        llm_expl = None
        if use_llm_synthesis:
            llm_expl = explain_with_gemini(
                query_text=query_text,
                spec=spec,
                top_df=top_df,
                model_name=self.model_name,
                api_key=self.api_key,
            )

        return PhysicsRAGResult(
            query=query_text,
            matches_df=top_df,
            similarities=selected_sims,
            explanation=deterministic_expl,
            parsed_criteria=spec,
            llm_explanation=llm_expl,
            query_vector=query_vec,
            recovery_episodes=recov_episodes,
        )


class HierarchicalPhysicsRAG(PhysicsRAG):
    """Hierarchical Parent-Child Physics-RAG Engine using embedded DuckDB & Scikit-Learn.

    Architecture:
    - Child Layer: In-memory Scikit-Learn NearestNeighbors / Cosine indexing over normalized physical scalars & regime features.
    - Parent Layer: Local columnar multi-channel sequence storage in DuckDB / Apache Parquet.
    - Hydration Engine: Slices surrounding continuous temporal context windows (e.g. -72h to +24h) for top child matches.
    - LLM Engine: Gemini Flash structured query parsing and multi-scale diagnostic narrative synthesis.
    """

    DEFAULT_CHILD_FEATURE_COLS = [
        "hs_m", "tp_s", "surge_m", "wind_normal_ms",
        "xi_iribarren", "upsilon", "dean_number", "delta_p_dissip",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        parquet_path: Optional[str] = None,
        child_feature_cols: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        scaler_meta: Optional[Dict[str, Any]] = None,
        scaler_type: str = "robust",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        top_k: int = 5,
        default_pre_window_hours: float = 72.0,
        default_post_window_hours: float = 24.0,
        db_conn: Optional[Any] = None,
        survey_aligned_epochs: Optional[Dict[str, pd.DataFrame]] = None,
        use_reranker: bool = True,
        rerank_candidate_pool: int = 20,
        **kwargs: Any,
    ):
        super().__init__(
            df=df,
            embeddings=embeddings,
            feature_names=feature_names,
            scaler_meta=scaler_meta,
            model_name=model_name,
            api_key=api_key,
            top_k=top_k,
            **kwargs,
        )
        self.parquet_path = parquet_path
        self.default_pre_window_hours = default_pre_window_hours
        self.default_post_window_hours = default_post_window_hours
        self.scaler_type = scaler_type
        self.survey_aligned_epochs = survey_aligned_epochs
        self.use_reranker = use_reranker
        self.rerank_candidate_pool = rerank_candidate_pool

        # Ensure timestamp column is datetime
        if "timestamp" in self.df.columns:
            try:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            except Exception:
                pass
        else:
            self.df["timestamp"] = pd.date_range("2022-01-01", periods=len(self.df), freq="1h")

        # Child feature columns
        self.child_feature_cols = child_feature_cols or [c for c in self.DEFAULT_CHILD_FEATURE_COLS if c in self.df.columns]

        # In-memory Scikit-Learn NearestNeighbors index
        n_neighbors = min(max(top_k * 4, 10), len(self.df))
        self.knn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(self.embeddings)

        # DuckDB Parent sequence storage
        self.con = db_conn or (duckdb.connect(":memory:") if duckdb is not None else None)
        self._init_duckdb_store()

    def _init_duckdb_store(self) -> None:
        """Register the parent time-series DataFrame or Parquet file in DuckDB."""
        if self.con is None or duckdb is None:
            return

        try:
            if self.parquet_path and os.path.exists(self.parquet_path):
                self.con.execute(f"CREATE OR REPLACE VIEW parent_sensors AS SELECT * FROM '{self.parquet_path}'")
            else:
                self.con.register("parent_sensors", self.df)
        except Exception:
            pass

    def export_parent_parquet(self, parquet_path: str) -> str:
        """Save continuous multi-channel dataset to local Apache Parquet columnar storage using DuckDB."""
        parent_dir = os.path.dirname(os.path.abspath(parquet_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if self.con is not None and duckdb is not None:
            try:
                self.con.register("parent_export_df", self.df)
                self.con.execute(f"COPY parent_export_df TO '{parquet_path}' (FORMAT PARQUET)")
                self.parquet_path = parquet_path
                self.con.execute(f"CREATE OR REPLACE VIEW parent_sensors AS SELECT * FROM '{parquet_path}'")
                return parquet_path
            except Exception:
                pass

        try:
            self.df.to_parquet(parquet_path, index=False)
            self.parquet_path = parquet_path
            if self.con is not None and duckdb is not None:
                self.con.execute(f"CREATE OR REPLACE VIEW parent_sensors AS SELECT * FROM '{parquet_path}'")
        except Exception:
            # If disk write fails or pyarrow is incompatible, retain registered in-memory table
            if self.con is not None and duckdb is not None:
                self.con.register("parent_sensors", self.df)
        return parquet_path

    def export_partitioned_parent_parquet(self, output_dir: str = "data") -> Dict[str, str]:
        """Save site-partitioned multi-channel parent time-series datasets to local Apache Parquet files.

        Saves files as data/{site}_sensors.parquet (e.g., data/duck_nc_sensors.parquet,
        data/truc_vert_france_sensors.parquet) and registers site views in DuckDB.
        """
        os.makedirs(output_dir, exist_ok=True)
        partition_paths: Dict[str, str] = {}

        if "site" in self.df.columns:
            sites = self.df["site"].unique()
        else:
            sites = ["duck_nc"]

        for site_key in sites:
            s_key = str(site_key).lower()
            target_path = os.path.join(output_dir, f"{s_key}_sensors.parquet")
            site_sub_df = self.df[self.df["site"] == site_key] if "site" in self.df.columns else self.df

            try:
                if self.con is not None and duckdb is not None:
                    self.con.register(f"sub_df_{s_key}", site_sub_df)
                    self.con.execute(f"COPY sub_df_{s_key} TO '{target_path}' (FORMAT PARQUET)")
                    self.con.execute(f"CREATE OR REPLACE VIEW {s_key}_sensors AS SELECT * FROM '{target_path}'")
                else:
                    site_sub_df.to_parquet(target_path, index=False)
            except Exception:
                try:
                    site_sub_df.to_parquet(target_path, index=False)
                except Exception:
                    pass
            partition_paths[s_key] = target_path

        return partition_paths

    @classmethod
    def from_dataset(
        cls,
        df: pd.DataFrame,
        parquet_path: Optional[str] = None,
        child_feature_cols: Optional[List[str]] = None,
        scaler_type: str = "robust",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "HierarchicalPhysicsRAG":
        """Factory method to construct HierarchicalPhysicsRAG directly from a DataFrame."""
        embeddings, feature_names, meta = compute_latent_embeddings(df, scaler_type=scaler_type)
        return cls(
            df=df,
            parquet_path=parquet_path,
            child_feature_cols=child_feature_cols,
            embeddings=embeddings,
            feature_names=feature_names,
            scaler_meta=meta,
            scaler_type=scaler_type,
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )

    @classmethod
    def from_rag(
        cls,
        rag: PhysicsRAG,
        parquet_path: Optional[str] = None,
        child_feature_cols: Optional[List[str]] = None,
        db_conn: Optional[Any] = None,
        use_reranker: bool = True,
        rerank_candidate_pool: int = 20,
        **kwargs: Any,
    ) -> "HierarchicalPhysicsRAG":
        """Factory method to construct HierarchicalPhysicsRAG directly from an existing PhysicsRAG instance."""
        model_name = kwargs.pop("model_name", getattr(rag, "model_name", "gemini-2.5-flash"))
        api_key = kwargs.pop("api_key", getattr(rag, "api_key", None))
        top_k = kwargs.pop("top_k", getattr(rag, "top_k", 5))
        return cls(
            df=rag.df,
            parquet_path=parquet_path,
            child_feature_cols=child_feature_cols,
            embeddings=rag.embeddings,
            feature_names=rag.feature_names,
            scaler_meta=rag.scaler_meta,
            scaler=getattr(rag, "scaler", None),
            model_name=model_name,
            api_key=api_key,
            top_k=top_k,
            db_conn=db_conn,
            use_weighted_similarity=getattr(rag, "use_weighted_similarity", True),
            feature_weights=getattr(rag, "feature_weights", None),
            use_reranker=use_reranker,
            rerank_candidate_pool=rerank_candidate_pool,
            **kwargs,
        )

    def hydrate_parent_window(
        self,
        event_timestamp: Union[str, pd.Timestamp],
        match_index: int = 0,
        similarity: float = 1.0,
        pre_window_hours: Optional[float] = None,
        post_window_hours: Optional[float] = None,
        columns: Optional[List[str]] = None,
        site_id: Optional[str] = None,
    ) -> ParentWindowContext:
        """Slice and hydrate surrounding continuous parent time-series context window using DuckDB."""
        pre_h = pre_window_hours if pre_window_hours is not None else self.default_pre_window_hours
        post_h = post_window_hours if post_window_hours is not None else self.default_post_window_hours
        ts_str = pd.to_datetime(event_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cols_sql = ", ".join(columns) if columns else "*"

        # Determine site-partitioned source view or parquet file path
        table_ref = "parent_sensors"
        target_site = site_id
        if not target_site and match_index < len(self.df):
            row_match = self.df.iloc[match_index]
            target_site = str(row_match.get("site_id", row_match.get("site", ""))).lower()

        if target_site:
            part_path = os.path.join("data", "processed", f"{target_site.lower()}_2012_2016_sensors.parquet")
            if not os.path.exists(part_path):
                part_path = os.path.join("data", f"{target_site.lower()}_sensors.parquet")
            if os.path.exists(part_path):
                table_ref = f"'{os.path.abspath(part_path)}'"
            else:
                table_ref = f"{target_site.lower()}_sensors"

        parent_df = pd.DataFrame()
        if self.con is not None and duckdb is not None:
            try:
                query = f"""
                    SELECT {cols_sql}
                    FROM {table_ref}
                    WHERE CAST(timestamp AS TIMESTAMP) BETWEEN CAST('{ts_str}' AS TIMESTAMP) - INTERVAL '{pre_h} HOUR'
                                                          AND CAST('{ts_str}' AS TIMESTAMP) + INTERVAL '{post_h} HOUR'
                    ORDER BY timestamp ASC
                """
                parent_df = self.con.execute(query).df()
            except Exception:
                try:
                    # Fallback to main parent_sensors view
                    query = f"""
                        SELECT {cols_sql}
                        FROM parent_sensors
                        WHERE CAST(timestamp AS TIMESTAMP) BETWEEN CAST('{ts_str}' AS TIMESTAMP) - INTERVAL '{pre_h} HOUR'
                                                              AND CAST('{ts_str}' AS TIMESTAMP) + INTERVAL '{post_h} HOUR'
                        ORDER BY timestamp ASC
                    """
                    parent_df = self.con.execute(query).df()
                except Exception:
                    parent_df = pd.DataFrame()

        if parent_df.empty:
            # Fallback fast-pandas slicing
            try:
                ts = pd.to_datetime(event_timestamp)
                t_start = ts - pd.Timedelta(hours=pre_h)
                t_end = ts + pd.Timedelta(hours=post_h)
                source_df = self.df
                if target_site:
                    if "site_id" in self.df.columns:
                        source_df = self.df[self.df["site_id"].astype(str).str.lower() == target_site.lower()]
                    elif "site" in self.df.columns:
                        source_df = self.df[self.df["site"].astype(str).str.lower() == target_site.lower()]
                if pd.api.types.is_datetime64_any_dtype(source_df["timestamp"]):
                    ts_col = source_df["timestamp"]
                else:
                    ts_col = pd.to_datetime(source_df["timestamp"])
                if ts_col.dt.tz is not None and t_start.tz is None:
                    t_start = t_start.tz_localize("UTC")
                    t_end = t_end.tz_localize("UTC")
                elif ts_col.dt.tz is None and t_start.tz is not None:
                    t_start = t_start.tz_convert(None)
                    t_end = t_end.tz_convert(None)
                mask = (ts_col >= t_start) & (ts_col <= t_end)
                parent_df = source_df[mask].sort_values("timestamp").copy()
                if columns:
                    valid_cols = [c for c in columns if c in parent_df.columns]
                    parent_df = parent_df[valid_cols]
            except Exception:
                parent_df = pd.DataFrame()

        # Compute temporal evolution statistics across the hydrated window
        stats = self._compute_window_statistics(parent_df, event_timestamp)

        w_start = parent_df["timestamp"].min() if not parent_df.empty else (pd.to_datetime(event_timestamp) - pd.Timedelta(hours=pre_h))
        w_end = parent_df["timestamp"].max() if not parent_df.empty else (pd.to_datetime(event_timestamp) + pd.Timedelta(hours=post_h))

        return ParentWindowContext(
            match_index=match_index,
            event_timestamp=event_timestamp,
            similarity=similarity,
            window_start=w_start,
            window_end=w_end,
            pre_window_hours=pre_h,
            post_window_hours=post_h,
            parent_df=parent_df,
            summary_stats=stats,
        )

    def _compute_window_statistics(self, parent_df: pd.DataFrame, event_ts: Any) -> Dict[str, Any]:
        """Compute hydrodynamic, morphodynamic, and wave energy evolution metrics across a parent window."""
        stats: Dict[str, Any] = {}
        if parent_df.empty:
            return stats

        try:
            # Wave height dynamics
            if "hs_m" in parent_df.columns:
                stats["peak_hs_m"] = float(parent_df["hs_m"].max())
                stats["mean_hs_m"] = float(parent_df["hs_m"].mean())
                hs_start = float(parent_df["hs_m"].iloc[0])
                stats["hs_start_m"] = hs_start
                peak_idx = int(parent_df["hs_m"].argmax())
                peak_time = parent_df["timestamp"].iloc[peak_idx]
                dt_hours = max(1.0, (pd.to_datetime(peak_time) - pd.to_datetime(parent_df["timestamp"].iloc[0])).total_seconds() / 3600.0)
                stats["hs_rise_rate_m_per_hr"] = float((stats["peak_hs_m"] - hs_start) / dt_hours)

            # Wave period dynamics
            if "tp_s" in parent_df.columns:
                stats["peak_tp_s"] = float(parent_df["tp_s"].max())
                stats["mean_tp_s"] = float(parent_df["tp_s"].mean())

            # Surge residual dynamics
            if "surge_m" in parent_df.columns:
                stats["peak_surge_m"] = float(parent_df["surge_m"].max())
                stats["mean_surge_m"] = float(parent_df["surge_m"].mean())
                stats["surge_duration_hours"] = float((parent_df["surge_m"] > 0.3).sum())

            # Unified Index dynamics
            if "upsilon" in parent_df.columns:
                stats["peak_upsilon"] = float(parent_df["upsilon"].max())
                stats["mean_upsilon"] = float(parent_df["upsilon"].mean())

            # Cumulative Wave Energy Impulse (integral of wave power flux P dt in MJ/m)
            if "p_flux_Wm" in parent_df.columns:
                dt_sec = 3600.0
                if len(parent_df) > 1:
                    dt_sec = (pd.to_datetime(parent_df["timestamp"].iloc[-1]) - pd.to_datetime(parent_df["timestamp"].iloc[0])).total_seconds() / max(1, len(parent_df) - 1)
                total_joules = np.sum(parent_df["p_flux_Wm"].values) * dt_sec
                stats["cumulative_power_impulse_MJ_per_m"] = float(total_joules / 1e6)

            # Shoreline displacement dynamics
            if "shoreline_pos_obs_m" in parent_df.columns:
                x_start = float(parent_df["shoreline_pos_obs_m"].iloc[0])
                x_end = float(parent_df["shoreline_pos_obs_m"].iloc[-1])
                stats["delta_x_start_m"] = x_start
                stats["delta_x_end_m"] = x_end
                stats["net_delta_x_m"] = float(x_end - x_start)
            elif "delta_x_m" in parent_df.columns:
                stats["net_delta_x_m"] = float(np.sum(parent_df["delta_x_m"].values))
                stats["delta_x_start_m"] = float(parent_df["delta_x_m"].iloc[0])
                stats["delta_x_end_m"] = float(parent_df["delta_x_m"].iloc[-1])

            # Disequilibrium dynamics
            if "upsilon_diseq" in parent_df.columns:
                stats["mean_upsilon_diseq"] = float(parent_df["upsilon_diseq"].mean())
                stats["recovery_phase_fraction"] = float((parent_df["upsilon_diseq"] < 0).mean())
        except Exception:
            pass

        return stats

    def _score_candidates_with_gemini(
        self,
        query_text: str,
        spec: PhysicsQuerySpec,
        candidates_df: pd.DataFrame,
        parent_windows: List[ParentWindowContext],
    ) -> Optional[List[float]]:
        """Score candidate physical episodes using Gemini structured causal reasoning over DuckDB sequences."""
        n_cand = len(candidates_df)
        if n_cand == 0:
            return []

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()

            cand_texts = []
            for i in range(min(n_cand, 20)):
                row = candidates_df.iloc[i]
                pw = parent_windows[i] if i < len(parent_windows) else None
                st = pw.summary_stats if pw else {}
                reg = str(row.get("breaker_regime", "N/A"))
                hs = float(row.get("hs_m", 0.0))
                tp = float(row.get("tp_s", 0.0))
                surge = float(row.get("surge_m", row.get("total_water_level_m", 0.0)))
                dx = float(row.get("delta_x_m", 0.0))

                cand_text = (
                    f"Candidate #{i}: Site={row.get('site', 'N/A')}, Timestamp={row.get('timestamp', 'N/A')}, "
                    f"Hs={hs:.2f}m, Tp={tp:.1f}s, Surge={surge:.2f}m, Regime={reg}, DeltaX={dx:+.2f}m, "
                    f"Antecedent Hs Rise Rate={st.get('hs_rise_rate_m_per_hr', 0.0):.3f} m/hr, "
                    f"Peak Surge={st.get('peak_surge_m', 0.0):.2f}m, "
                    f"Cumulative Wave Energy Impulse={st.get('cumulative_power_impulse_MJ_per_m', 0.0):.1f} MJ/m, "
                    f"Net Window DeltaX={st.get('net_delta_x_m', 0.0):+.2f}m, "
                    f"Recovery Phase Fraction={st.get('recovery_phase_fraction', 0.0)*100:.0f}%"
                )
                cand_texts.append(cand_text)

            prompt = (
                f"You are a coastal oceanography expert reranking hydrodynamic retrieval candidates for the query:\n"
                f"Query: '{query_text}'\n"
                f"Physical Intent: {spec.intent_summary or spec.raw_query}\n"
                f"Target Regime/Phase: {spec.breaker_regime or spec.phase or 'General'}\n\n"
                f"Candidates with Hydrated Context Sequences:\n" + "\n".join(cand_texts) + "\n\n"
                f"Score each candidate from 0.0 to 1.0 based on how well its instantaneous physical state AND continuous "
                f"antecedent/post-event sequence dynamics causally fulfill the query intent. "
                f"Return ONLY a JSON array of float scores corresponding to candidate indices in order, e.g. [0.95, 0.82, ...]."
            )

            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            if resp and resp.text:
                import json
                parsed_scores = json.loads(resp.text)
                if isinstance(parsed_scores, list) and len(parsed_scores) == min(n_cand, 20):
                    scores = [float(s) for s in parsed_scores]
                    if len(scores) < n_cand:
                        scores.extend([0.5] * (n_cand - len(scores)))
                    return scores
        except Exception as e:
            logger.error(f"Gemini candidate scoring failed for '{query_text}': {e}")
        return None

    def _score_candidates_deterministic(
        self,
        query_text: str,
        spec: PhysicsQuerySpec,
        candidates_df: pd.DataFrame,
        parent_windows: List[ParentWindowContext],
    ) -> List[float]:
        """Deterministic physical causality scoring evaluating sequence dynamics and conservation laws."""
        scores: List[float] = []
        q_lower = query_text.lower()
        target_intent = determine_upsilon_target_intent(spec, query_text)
        is_storm = (target_intent == "erosion") or any(w in q_lower for w in ["storm", "scarping", "high energy", "extreme", "severe", "over 3m", "over 4m"]) or bool(getattr(spec, "query_storm_evolution", False)) or getattr(spec, "phase", "") == "Erosion" or getattr(spec, "morphology", "") == "Erosive"
        is_recovery = (target_intent == "recovery") or any(w in q_lower for w in ["accretion", "recovery", "swash", "healing", "calm", "berm"]) or bool(getattr(spec, "query_recovery_timeframe", False)) or getattr(spec, "phase", "") == "Recovery" or getattr(spec, "morphology", "") == "Accretive"
        is_reflective = getattr(spec, "breaker_regime", None) == "Surging" or any(w in q_lower for w in ["surging", "collapsing", "reflective"])
        is_dissipative = getattr(spec, "breaker_regime", None) == "Spilling" or any(w in q_lower for w in ["spilling", "dissipative"])
        is_compound = "compound" in q_lower

        for i in range(len(candidates_df)):
            row = candidates_df.iloc[i]
            pw = parent_windows[i] if i < len(parent_windows) else None
            st = pw.summary_stats if pw else {}

            score = 0.5  # Base neutral score

            # 1. Breaker regime match
            target_regime = getattr(spec, "breaker_regime", None)
            if target_regime:
                if str(row.get("breaker_regime")) == target_regime:
                    score += 0.35
                elif target_regime == "Surging" and float(row.get("xi_iribarren", 0.0)) >= 1.0:
                    score += 0.35
                elif target_regime == "Plunging" and 0.5 <= float(row.get("xi_iribarren", 0.0)) < 1.0:
                    score += 0.35
                elif target_regime == "Spilling" and float(row.get("xi_iribarren", 1.0)) < 0.5:
                    score += 0.35
                else:
                    score -= 0.35
            elif is_reflective:
                if str(row.get("breaker_regime")) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 1.0:
                    score += 0.35
            elif is_dissipative:
                if str(row.get("breaker_regime")) == "Spilling" or float(row.get("xi_iribarren", 1.0)) < 0.5:
                    score += 0.35

            # 2. Storm dynamics: antecedent rise rate + cumulative wave impulse + net shoreline erosion
            if is_storm:
                if st.get("hs_rise_rate_m_per_hr", 0.0) > 0.03:
                    score += 0.15
                if st.get("cumulative_power_impulse_MJ_per_m", 0.0) > 30.0:
                    score += 0.15
                if st.get("net_delta_x_m", 0.0) < -0.05 or float(row.get("delta_x_m", 0.0)) < -0.05:
                    score += 0.1

            # 3. Recovery dynamics: constructive swash fraction + net berm accretion
            if is_recovery:
                if st.get("recovery_phase_fraction", 0.0) > 0.4:
                    score += 0.15
                if st.get("net_delta_x_m", 0.0) > 0.05 or float(row.get("delta_x_m", 0.0)) > 0.05:
                    score += 0.2
                if float(row.get("upsilon_diseq", 0.0)) < 0.0:
                    score += 0.15

            # 4. Compound wave-surge
            if is_compound:
                hs_val = float(row.get("hs_m", 0.0))
                s_val = float(row.get("surge_m", row.get("total_water_level_m", 0.0)))
                if hs_val >= 2.0 and s_val >= 0.3:
                    score += 0.25
                if st.get("peak_surge_m", 0.0) >= 0.35:
                    score += 0.15

            # 5. Numerical threshold compliance
            hs_min_val = getattr(spec, "hs_min", None)
            hs_max_val = getattr(spec, "hs_max", None)
            tp_min_val = getattr(spec, "tp_min", None)
            tp_max_val = getattr(spec, "tp_max", None)
            if hs_min_val is not None and float(row.get("hs_m", 0.0)) < hs_min_val:
                score -= 0.3
            if hs_max_val is not None and float(row.get("hs_m", 0.0)) > hs_max_val:
                score -= 0.3
            if tp_min_val is not None and float(row.get("tp_s", 0.0)) < tp_min_val:
                score -= 0.3
            if tp_max_val is not None and float(row.get("tp_s", 0.0)) > tp_max_val:
                score -= 0.3

            scores.append(float(np.clip(score, 0.0, 1.0)))

        return scores

    def _score_candidates(
        self,
        query_text: str,
        spec: PhysicsQuerySpec,
        candidates_df: pd.DataFrame,
        parent_windows: List[ParentWindowContext],
    ) -> List[float]:
        """Score candidates using live Gemini reasoning over DuckDB sequences with deterministic fallback."""
        scores: Optional[List[float]] = None
        if self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            scores = self._score_candidates_with_gemini(query_text, spec, candidates_df, parent_windows)
        if scores is None:
            scores = self._score_candidates_deterministic(query_text, spec, candidates_df, parent_windows)
        return scores

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        pre_window_hours: Optional[float] = None,
        post_window_hours: Optional[float] = None,
        min_similarity: float = -1.0,
        use_llm_parser: bool = True,
        use_llm_synthesis: bool = True,
        parent_columns: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
        rerank_candidate_pool: Optional[int] = None,
        spec: Optional[PhysicsQuerySpec] = None,
        target_synthetic_values: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> HierarchicalRAGResult:
        """Execute Two-Stage Hierarchical Parent-Child query with DuckDB hydration and Gemini causal reranking."""
        do_rerank = self.use_reranker if use_reranker is None else use_reranker
        pool_size = rerank_candidate_pool or self.rerank_candidate_pool
        candidate_k = max(top_k * 4, pool_size) if do_rerank else top_k

        # 1. Base Child Vector Retrieval (Stage 1)
        base_res = super().query(
            query_text=query_text,
            top_k=candidate_k,
            min_similarity=min_similarity,
            use_llm_parser=use_llm_parser,
            use_llm_synthesis=False,
            spec=spec,
            target_synthetic_values=target_synthetic_values,
            **kwargs,
        )

        candidate_df = base_res.matches_df
        candidate_sims = base_res.similarities
        active_spec = (
            base_res.parsed_criteria
            if isinstance(base_res.parsed_criteria, PhysicsQuerySpec)
            else (spec or GeminiPhysicsQueryParser._fallback_parse(query_text))
        )

        # 2. Hydrate Parent Context Windows via DuckDB (Stage 2a)
        parent_windows: List[ParentWindowContext] = []
        if do_rerank:
            for idx in range(len(candidate_df)):
                row = candidate_df.iloc[idx]
                match_ts = row["timestamp"] if "timestamp" in row else self.df.iloc[idx]["timestamp"]
                sim = float(candidate_sims[idx]) if idx < len(candidate_sims) else 1.0

                pw = self.hydrate_parent_window(
                    event_timestamp=match_ts,
                    match_index=idx,
                    similarity=sim,
                    pre_window_hours=pre_window_hours,
                    post_window_hours=post_window_hours,
                    columns=parent_columns,
                )
                parent_windows.append(pw)

        # 3. Two-Stage Reranking (Stage 2b)
        if do_rerank and len(candidate_df) > 1:
            causal_scores = self._score_candidates(
                query_text=query_text,
                spec=active_spec,
                candidates_df=candidate_df,
                parent_windows=parent_windows,
            )
            rerank_scores = []
            for i in range(len(candidate_df)):
                s_lat = float(candidate_sims[i]) if i < len(candidate_sims) else 0.5
                s_cau = float(causal_scores[i]) if i < len(causal_scores) else 0.5
                # Weighted combination: 50% manifold similarity + 50% temporal sequence causality
                rerank_scores.append(0.5 * s_lat + 0.5 * s_cau)

            sorted_order = np.argsort(-np.array(rerank_scores))[:top_k]
            child_df = candidate_df.iloc[sorted_order].copy().reset_index(drop=True)
            selected_sims = np.array(rerank_scores)[sorted_order]
            selected_windows = [parent_windows[i] for i in sorted_order if i < len(parent_windows)]
        else:
            child_df = candidate_df.head(top_k).copy().reset_index(drop=True)
            selected_sims = candidate_sims[:top_k]
            selected_windows = parent_windows[:top_k]

        # 4. Generate Multi-Scale Explanations
        deterministic_expl = self._generate_hierarchical_explanation(active_spec, child_df, selected_windows)
        llm_expl = None
        if use_llm_synthesis:
            llm_expl = explain_hierarchical_with_gemini(
                query_text=query_text,
                spec=active_spec,
                child_df=child_df,
                parent_windows=selected_windows,
                model_name=self.model_name,
                api_key=self.api_key,
            )

        return HierarchicalRAGResult(
            query=query_text,
            child_matches_df=child_df,
            parent_windows=selected_windows,
            similarities=selected_sims,
            explanation=deterministic_expl,
            parsed_criteria=active_spec,
            llm_explanation=llm_expl,
            query_vector=base_res.query_vector,
        )

    def _generate_hierarchical_explanation(
        self,
        spec: PhysicsQuerySpec,
        child_df: pd.DataFrame,
        parent_windows: List[ParentWindowContext],
    ) -> str:
        """Synthesize deterministic multi-scale explanation connecting child hits with parent sequence dynamics."""
        base_expl = self._generate_explanation(spec, child_df)
        if not parent_windows:
            return base_expl

        top_pw = parent_windows[0]
        st = top_pw.summary_stats
        if not st:
            return base_expl

        parts = [base_expl, "\n[Parent Temporal Sequence Evolution]:"]
        parts.append(
            f"Hydrated ±{top_pw.pre_window_hours:.0f}h context window around event {top_pw.event_timestamp}. "
            f"Antecedent wave energy built up with Hs rising at {st.get('hs_rise_rate_m_per_hr', 0.0):.3f} m/hr to peak Hs = {st.get('peak_hs_m', 0.0):.2f}m. "
            f"Coincident peak storm surge reached {st.get('peak_surge_m', 0.0):.2f}m (elevated duration: {st.get('surge_duration_hours', 0.0):.1f}h), "
            f"delivering cumulative wave energy impulse of {st.get('cumulative_power_impulse_MJ_per_m', 0.0):.1f} MJ/m. "
            f"Observed net shoreline displacement across window is ΔX = {st.get('net_delta_x_m', 0.0):+.2f}m."
        )
        return " ".join(parts)

    def query_failure_modes(
        self,
        query_text: str = "When does Upsilon under-predict erosion?",
        top_k_per_site: int = 2,
        pre_window_hours: float = 72.0,
        post_window_hours: float = 24.0,
        sites: Optional[List[str]] = None,
        aligned_surveys: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Automatically query and diagnose hydrodynamic failure modes and model under-prediction across all sites.

        Scans either authentic survey epochs (when aligned_surveys is provided) or the continuous
        parent storage layer to identify episodes where severe observed shoreline erosion
        coincided with low-to-moderate baseline Upsilon forcing.
        Hydrates continuous parent sequences from DuckDB and synthesizes an automated diagnostic root-cause attribution.

        Parameters
        ----------
        query_text : str
            The failure mode natural language query.
        top_k_per_site : int
            Number of peak discrepancy episodes to retrieve per site archetype (default 2).
        pre_window_hours : float
            Hydration context hours prior to event (default 72h).
        post_window_hours : float
            Hydration context hours after event (default 24h).
        sites : list of str, optional
            List of site keys to audit (default: all sites in df).
        aligned_surveys : dict of str -> pd.DataFrame, optional
            Dictionary mapping site keys to aligned in-situ survey epoch DataFrames.
            If None, checks self.survey_aligned_epochs, otherwise falls back to hourly scan.

        Returns
        -------
        result : dict
            Dictionary containing 'query', 'failure_episodes_df', 'parent_windows', and 'diagnostic_summary'.
        """
        try:
            from src.physics import COASTAL_SITE_PROFILES
        except ImportError:
            from physics import COASTAL_SITE_PROFILES  # type: ignore

        if sites is None:
            if "site" in self.df.columns:
                sites = list(self.df["site"].unique())
            else:
                sites = ["duck_nc"]

        if aligned_surveys is None:
            aligned_surveys = getattr(self, "survey_aligned_epochs", None)

        records: List[Dict[str, Any]] = []
        hydrated_windows: Dict[str, List[ParentWindowContext]] = {}

        if aligned_surveys is not None:
            # === AUTHENTIC SURVEY-EPOCH FAILURE MODE AUDIT ===
            for site_key in sites:
                if site_key not in aligned_surveys:
                    continue
                df_a = aligned_surveys[site_key].copy()
                if len(df_a) < 2:
                    continue

                site_mask = self.df["site"] == site_key if "site" in self.df.columns else np.ones(len(self.df), dtype=bool)
                sub_df = self.df[site_mask].copy().reset_index(drop=True)

                prof = COASTAL_SITE_PROFILES.get(site_key, {
                    "name": site_key, "tan_beta": 0.08, "is_reef": False, "is_embayment": False, "z_crest_m": 2.5
                })

                # Extract observed shoreline displacement and antecedent Upsilon forcing
                if "surveyed_delta_x_m" in df_a.columns:
                    dx = df_a["surveyed_delta_x_m"].values.astype(float)
                elif "cross_shore_change_m" in df_a.columns:
                    dx = df_a["cross_shore_change_m"].values.astype(float)
                else:
                    continue

                if "antecedent_peak_upsilon" in df_a.columns:
                    u = df_a["antecedent_peak_upsilon"].values.astype(float)
                elif "peak_upsilon" in df_a.columns:
                    u = df_a["peak_upsilon"].values.astype(float)
                elif "upsilon" in df_a.columns:
                    u = df_a["upsilon"].values.astype(float)
                else:
                    u = np.zeros(len(df_a))

                # Standardized discrepancy: severe surveyed erosion (negative dx) with modest Upsilon
                z_dx_erosion = -(dx - np.nanmean(dx)) / max(float(np.nanstd(dx)), 1e-4)
                z_u = (u - np.nanmean(u)) / max(float(np.nanstd(u)), 1e-4)
                discrepancy = z_dx_erosion - z_u

                top_indices = np.argsort(discrepancy)[-top_k_per_site:][::-1]
                hydrated_windows[site_key] = []

                for rank, idx in enumerate(top_indices):
                    row_a = df_a.iloc[idx]
                    surv_ts = df_a.index[idx] if isinstance(df_a.index, pd.DatetimeIndex) else row_a.get("survey_date", f"Epoch_{idx}")
                    prev_ts = df_a.index[idx - 1] if (idx > 0 and isinstance(df_a.index, pd.DatetimeIndex)) else (pd.to_datetime(surv_ts) - pd.Timedelta(hours=168))

                    # Locate peak hydrodynamic forcing hour in continuous series during this epoch
                    storm_row = row_a
                    peak_ts = surv_ts
                    if "timestamp" in sub_df.columns:
                        sub_df_ts = pd.to_datetime(sub_df["timestamp"])
                        surv_ts_dt = pd.to_datetime(surv_ts)
                        prev_ts_dt = pd.to_datetime(prev_ts)
                        if sub_df_ts.dt.tz is not None and surv_ts_dt.tz is None:
                            surv_ts_dt = surv_ts_dt.tz_localize("UTC")
                            prev_ts_dt = prev_ts_dt.tz_localize("UTC")
                        elif sub_df_ts.dt.tz is None and surv_ts_dt.tz is not None:
                            surv_ts_dt = surv_ts_dt.tz_convert(None)
                            prev_ts_dt = prev_ts_dt.tz_convert(None)

                        mask_epoch = (sub_df_ts >= prev_ts_dt) & (sub_df_ts <= surv_ts_dt)
                        sub_storm = sub_df[mask_epoch]
                        if sub_storm.empty:
                            sub_storm = sub_df.iloc[max(0, idx - 72) : min(len(sub_df), idx + 24)]

                        if not sub_storm.empty and "hs_m" in sub_storm.columns:
                            p_idx = int(sub_storm["hs_m"].argmax())
                            storm_row = sub_storm.iloc[p_idx]
                            peak_ts = storm_row["timestamp"]

                    hs_val = float(storm_row.get("hs_m", row_a.get("peak_hs_m", row_a.get("antecedent_max_hs_m", 0.0))))
                    tp_val = float(storm_row.get("tp_s", row_a.get("peak_tp_s", row_a.get("antecedent_max_tp_s", 0.0))))
                    surge_val = float(storm_row.get("surge_m", row_a.get("peak_surge_m", row_a.get("antecedent_max_surge_m", 0.0))))
                    tide_val = float(storm_row.get("tide_m", 0.0))
                    twl_val = float(storm_row.get("total_water_level_m", surge_val + tide_val + hs_val * 0.2))
                    dx_val = float(dx[idx])
                    u_val = float(u[idx])

                    # Physical root cause attribution
                    if prof.get("is_reef") and (surge_val > 0.3 or tide_val > 0.2):
                        cause = "Reef Depth-Modulated Overtopping (Tidal surge reduces reef flat friction, transmitting energetic bores to shore)"
                    elif prof.get("is_embayment") and abs(storm_row.get("wave_dir_deg", 0.0) - prof.get("shore_normal_deg", 0.0)) > 20.0:
                        cause = "Embayed Rotation & Alongshore Rip Scour (High wave obliquity drives localized rip cell sand evacuation)"
                    elif tp_val >= 13.0 and prof.get("tan_beta", 0.08) <= 0.07:
                        cause = "Infragravity Swash & Surf-Beat Resonance (Long-period swell excites bound infragravity waves over dissipative bars)"
                    elif surge_val > 0.35 or twl_val > prof.get("z_crest_m", 2.5) * 0.8:
                        cause = "Storm Surge & Elevated TWL Scarping (High water level elevates wave attack above baseline equilibrium berm)"
                    else:
                        cause = "Antecedent Beach Depletion (Successive wave pulses erode a previously scarped profile)"

                    records.append({
                        "Site": prof.get("name", site_key).split(",")[0],
                        "Site Key": site_key,
                        "Timestamp": str(peak_ts)[:16],
                        "Survey Date": str(surv_ts)[:10],
                        "Storm Peak": str(peak_ts)[:16],
                        "Surveyed Δx (m)": f"{dx_val:+.2f}",
                        "Observed Δx/Δt (m/h)": f"{dx_val:+.2f} m (survey)",
                        "Hs (m)": f"{hs_val:.2f}",
                        "Tp (s)": f"{tp_val:.1f}",
                        "Surge (m)": f"{surge_val:+.2f}",
                        "Upsilon [kg/s]": f"{u_val:.1f}",
                        "Discrepancy Score": f"{discrepancy[idx]:.1f}",
                        "Physical Root Cause": cause,
                    })

                    # Hydrate continuous parent window around the peak storm event via DuckDB
                    pw = self.hydrate_parent_window(
                        event_timestamp=peak_ts,
                        site_id=site_key,
                        similarity=float(discrepancy[idx]),
                        pre_window_hours=pre_window_hours,
                        post_window_hours=post_window_hours,
                    )
                    hydrated_windows[site_key].append(pw)

        else:
            # === LEGACY HOURLY DISCREPANCY AUDIT ===
            for site_key in sites:
                site_mask = self.df["site"] == site_key if "site" in self.df.columns else np.ones(len(self.df), dtype=bool)
                sub_df = self.df[site_mask].copy().reset_index(drop=True)
                if len(sub_df) < 10:
                    continue

                prof = COASTAL_SITE_PROFILES.get(site_key, {
                    "name": site_key, "tan_beta": 0.08, "is_reef": False, "is_embayment": False, "z_crest_m": 2.5
                })

                # Calculate observed shoreline change rate and Upsilon
                if "shoreline_change_rate_mh" in sub_df.columns:
                    dx = sub_df["shoreline_change_rate_mh"].values
                elif "shoreline_pos_obs_m" in sub_df.columns:
                    dx = np.diff(sub_df["shoreline_pos_obs_m"].values, prepend=sub_df["shoreline_pos_obs_m"].values[0])
                else:
                    dx = np.zeros(len(sub_df))

                u = sub_df["upsilon"].values if "upsilon" in sub_df.columns else np.zeros(len(sub_df))

                # Standardized discrepancy: severe erosion (negative dx) with low/moderate Upsilon
                z_dx_erosion = -(dx - np.mean(dx)) / max(float(np.std(dx)), 1e-4)
                z_u = (u - np.mean(u)) / max(float(np.std(u)), 1e-4)
                discrepancy = z_dx_erosion - z_u

                top_indices = np.argsort(discrepancy)[-top_k_per_site:][::-1]
                hydrated_windows[site_key] = []

                for rank, idx in enumerate(top_indices):
                    row = sub_df.iloc[idx]
                    hs_val = float(row.get("hs_m", 0.0))
                    tp_val = float(row.get("tp_s", 0.0))
                    surge_val = float(row.get("surge_m", 0.0))
                    tide_val = float(row.get("tide_m", 0.0))
                    twl_val = float(row.get("total_water_level_m", surge_val + tide_val + hs_val * 0.2))
                    dx_val = float(dx[idx])
                    u_val = float(u[idx])
                    ts = row["timestamp"] if "timestamp" in row else f"Step_{idx}"

                    # Physical root cause attribution
                    if prof.get("is_reef") and (surge_val > 0.3 or tide_val > 0.2):
                        cause = "Reef Depth-Modulated Overtopping (Tidal surge reduces reef flat friction, transmitting energetic bores to shore)"
                    elif prof.get("is_embayment") and abs(row.get("wave_dir_deg", 0.0) - prof.get("shore_normal_deg", 0.0)) > 20.0:
                        cause = "Embayed Rotation & Alongshore Rip Scour (High wave obliquity drives localized rip cell sand evacuation)"
                    elif tp_val >= 13.0 and prof.get("tan_beta", 0.08) <= 0.07:
                        cause = "Infragravity Swash & Surf-Beat Resonance (Long-period swell excites bound infragravity waves over dissipative bars)"
                    elif surge_val > 0.35 or twl_val > prof.get("z_crest_m", 2.5) * 0.8:
                        cause = "Storm Surge & Elevated TWL Scarping (High water level elevates wave attack above baseline equilibrium berm)"
                    else:
                        cause = "Antecedent Beach Depletion (Successive wave pulses erode a previously scarped profile)"

                    records.append({
                        "Site": prof.get("name", site_key).split(",")[0],
                        "Site Key": site_key,
                        "Timestamp": str(ts)[:16],
                        "Survey Date": str(ts)[:10],
                        "Storm Peak": str(ts)[:16],
                        "Surveyed Δx (m)": f"{dx_val:+.3f}",
                        "Observed Δx/Δt (m/h)": f"{dx_val:+.3f}",
                        "Hs (m)": f"{hs_val:.2f}",
                        "Tp (s)": f"{tp_val:.1f}",
                        "Surge (m)": f"{surge_val:+.2f}",
                        "Upsilon [kg/s]": f"{u_val:.1f}",
                        "Discrepancy Score": f"{discrepancy[idx]:.1f}",
                        "Physical Root Cause": cause,
                    })

                    # Hydrate continuous parent window via DuckDB
                    pw = self.hydrate_parent_window(
                        event_timestamp=ts,
                        match_index=idx,
                        similarity=float(discrepancy[idx]),
                        pre_window_hours=pre_window_hours,
                        post_window_hours=post_window_hours,
                    )
                    hydrated_windows[site_key].append(pw)

        df_episodes = pd.DataFrame(records)

        # Generate summary synthesis
        summary_text = (
            f"=== Physics-RAG Automated Failure Mode Audit ({query_text}) ===\n"
            f"Analyzed {len(sites)} coastal site archetypes. Identified {len(df_episodes)} primary failure mode episodes.\n"
            "Key Oceanographic Failure Mechanisms:\n"
            "1. Storm Surge & Elevated Water Levels: Extreme surge and spring tides enable modest waves to directly attack sub-aerial dunes.\n"
            "2. Infragravity Swash Resonance: Narrow long-period swell (Tp > 13s) amplifies unmodeled surf beat at dissipative beaches.\n"
            "3. Oblique Swell & Embayment Rotation: High wave obliquity induces alongshore rip cells and embayed rotation.\n"
            "4. Reef-Flat Depth Modulation: Elevated water levels reduce coral friction, allowing energetic wave bores to penetrate to the shoreline.\n"
        )

        return {
            "query": query_text,
            "failure_episodes_df": df_episodes,
            "parent_windows": hydrated_windows,
            "diagnostic_summary": summary_text,
        }


class Baseline4HierarchicalRerankerRAG(HierarchicalPhysicsRAG):
    """Baseline 4: Two-Stage Hierarchical Reranker with Live Gemini Reasoning.

    Architecture:
    - Stage 1 (Fast Latent Candidate Generation): 14D Invariant Manifold + Upsilon Metric Tensor (w_Upsilon)
      retrieves top candidate episodes (<5ms).
    - Stage 2a (Continuous Sequence Hydration): Slices ±72h continuous multi-channel sensor sequences
      from embedded DuckDB columnar storage.
    - Stage 2b (Gemini Physical Reasoning Reranker): Evaluates cumulative wave energy impulse, antecedent
      energy rise rate, surge duration, and net shoreline displacement to rerank candidates with deep causal fidelity.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("use_reranker", True)
        super().__init__(*args, **kwargs)


# Canonical aliases
TwoStageHierarchicalRerankerRAG = Baseline4HierarchicalRerankerRAG


# ==============================================================================
# Standard Physics-RAG Evaluation Suite & Metrics Engine
# ==============================================================================

def _get_site_series(df: pd.DataFrame) -> pd.Series:
    """Extract site identifier series handling site_id vs site aliasing."""
    if "site_id" in df.columns:
        return df["site_id"]
    elif "site" in df.columns:
        return df["site"]
    return pd.Series([""] * len(df), index=df.index)


def _get_row_site(row: Any) -> str:
    """Extract site identifier string from row dict or Series."""
    if hasattr(row, "get"):
        return str(row.get("site_id", row.get("site", "")))
    return ""


def _get_row_ts(row: Any) -> pd.Timestamp:
    """Extract UTC timestamp from row dict or Series."""
    ts = row.get("timestamp") if hasattr(row, "get") else getattr(row, "timestamp", None)
    if isinstance(ts, str):
        return pd.to_datetime(ts, utc=True)
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize("UTC") if ts.tz is None else ts
    return pd.Timestamp("1970-01-01", tz="UTC")


def _make_graded_fn(exact_cond: Any, partial_cond: Any, peak_work_cond: Optional[Any] = None) -> Any:
    """Helper creating 4-tier peak hydrodynamic work graded relevance function:
    - Grade 3: Peak Hydrodynamic Work & Disequilibrium (peak storm impact / maximum Hs & Upsilon)
    - Grade 2: Active Storm Phase (exceeding storm erosion threshold within event window)
    - Grade 1: Pre-storm onset / post-storm relaxation within calendar window
    - Grade 0: Irrelevant (wrong site or outside event window)
    """
    if peak_work_cond is not None:
        return lambda row, df: 3 if peak_work_cond(row, df) else (2 if exact_cond(row, df) else (1 if partial_cond(row, df) else 0))
    return lambda row, df: 2 if exact_cond(row, df) else (1 if partial_cond(row, df) else 0)


def _get_breaker_series(df: pd.DataFrame) -> pd.Series:
    """Extract or compute breaker regime series ('Spilling', 'Plunging', 'Surging')."""
    if "breaker_regime" in df.columns:
        return df["breaker_regime"]
    elif "xi_iribarren" in df.columns:
        xi = df["xi_iribarren"].values
        return pd.Series(
            np.where(xi < 0.5, "Spilling", np.where(xi < 1.0, "Plunging", "Surging")),
            index=df.index,
        )
    return pd.Series([""] * len(df), index=df.index)


def _get_breaker(row: Any) -> str:
    """Extract or compute breaker regime string for a single row."""
    if hasattr(row, "get"):
        reg = row.get("breaker_regime")
        if reg:
            return str(reg)
        xi = float(row.get("xi_iribarren", 0.0))
        if xi > 0:
            return "Spilling" if xi < 0.5 else ("Plunging" if xi < 1.0 else "Surging")
    return ""


def _get_steepness_series(df: pd.DataFrame) -> pd.Series:
    """Extract or dynamically compute deepwater wave steepness H0/L0 series."""
    if "wave_steepness" in df.columns:
        return df["wave_steepness"]
    elif "steepness" in df.columns:
        return df["steepness"]
    elif "s0" in df.columns:
        return df["s0"]
    elif "hs_m" in df.columns and "l_0_m" in df.columns:
        l0 = df["l_0_m"].replace(0, np.nan)
        return (df["hs_m"] / l0).fillna(0.0)
    elif "hs_m" in df.columns and "tp_s" in df.columns:
        l0 = 1.5613 * (df["tp_s"] ** 2)
        l0 = l0.replace(0, np.nan)
        return (df["hs_m"] / l0).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _get_row_steepness(row: Any) -> float:
    """Extract or compute deepwater wave steepness H0/L0 for a single row."""
    if hasattr(row, "get"):
        for k in ["wave_steepness", "steepness", "s0"]:
            val = row.get(k)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        hs = float(row.get("hs_m", 0.0))
        l0 = float(row.get("l_0_m", 0.0))
        if l0 > 0:
            return hs / l0
        tp = float(row.get("tp_s", 0.0))
        if tp > 0:
            return hs / (1.5613 * (tp ** 2))
    return 0.0


def _get_runup_ig_series(df: pd.DataFrame) -> pd.Series:
    """Extract or dynamically compute infragravity swash runup excursion R_ig series."""
    if "runup_r_ig_m" in df.columns:
        return df["runup_r_ig_m"]
    elif "r_ig_m" in df.columns:
        return df["r_ig_m"]
    elif "r_ig" in df.columns:
        return df["r_ig"]
    elif "hs_m" in df.columns and "tp_s" in df.columns:
        # Stockdon et al. (2006): R_ig = 0.06 * sqrt(H0 * L0) = 0.075 * Tp * sqrt(Hs)
        hs = df["hs_m"].clip(lower=0.0)
        tp = df["tp_s"].clip(lower=0.0)
        return 0.075 * tp * np.sqrt(hs)
    elif "runup_r2_m" in df.columns:
        return df["runup_r2_m"] * 0.40
    return pd.Series(0.0, index=df.index)


def _get_row_runup_ig(row: Any) -> float:
    """Extract or compute infragravity swash runup excursion R_ig for a single row."""
    if hasattr(row, "get"):
        for k in ["runup_r_ig_m", "r_ig_m", "r_ig"]:
            val = row.get(k)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        hs = max(0.0, float(row.get("hs_m", 0.0)))
        tp = max(0.0, float(row.get("tp_s", 0.0)))
        if hs > 0 and tp > 0:
            return 0.075 * tp * np.sqrt(hs)
        r2 = float(row.get("runup_r2_m", 0.0))
        if r2 > 0:
            return r2 * 0.40
    return 0.0


ALL_PHYSICS_BENCHMARK_QUERIES_60: List[Dict[str, Any]] = [
    # === CATEGORY 1: HISTORICAL STORM EVENTS (6 QUERIES - ARCHIVED MODE B) ===
    {
        "query": "June 2016 East Coast Low catastrophic erosion event at Narrabeen",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "tp_s": 12.0, "tan_beta": 0.05},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2016-06-03") & (df["timestamp"] <= "2016-06-08") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2016-05-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Narrabeen June 2016 East Coast Low (ECL) storm peak with Hs >= 3.0m and extreme dune scarping",
    },
    {
        "query": "Hurricane Matthew storm peak and overwash at Duck in October 2016",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.8, "tp_s": 12.5, "tan_beta": 0.08},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2016-10-07") & (df["timestamp"] <= "2016-10-11") & (df["hs_m"] >= 3.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-11", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-08", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-11", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2016-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-18", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-08", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Duck Hurricane Matthew landfall sequence (Oct 7-11, 2016) with extreme waves Hs >= 3.5m",
    },
    {
        "query": "Megastorms Hercules and Christine early 2014 dune erosion at Truc Vert",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 7.0, "tp_s": 15.0, "tan_beta": 0.04},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2014-01-01") & (df["timestamp"] <= "2014-03-10") & (df["hs_m"] >= 5.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and ((pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 3.5),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.5,
        ),
        "description": "Truc Vert megastorm cluster (Hercules/Christine, winter 2013-2014) with extreme North Atlantic swell Hs >= 5.0m",
    },
    {
        "query": "2015-2016 El Niño severe winter storm waves at Torrey Pines",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.2, "tp_s": 14.0, "tan_beta": 0.03},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-12-01") & (df["timestamp"] <= "2016-03-01") & (df["hs_m"] >= 2.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-03-01", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-05", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-03-01", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and ((pd.Timestamp("2015-11-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-03-31", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 1.8),
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-05", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        ),
        "description": "Torrey Pines energetic Pacific winter storm waves during peak 2015-2016 El Niño with Hs >= 2.5m",
    },
    {
        "query": "April 2015 cyclonic East Coast Low wave impact at Narrabeen",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "tp_s": 11.5, "tan_beta": 0.05},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2015-04-20") & (df["timestamp"] <= "2015-04-25") & (df["hs_m"] >= 3.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-21", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-23", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2015-04-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-30", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-21", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-23", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Narrabeen April 2015 East Coast Low cyclonic storm peak reaching Hs >= 3.5m",
    },
    {
        "query": "Hurricane Sandy late October 2012 storm surge and wave attack at Duck",
        "category": "1. Historical Storm Events",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "tp_s": 13.0, "tan_beta": 0.08},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2012-10-27") & (df["timestamp"] <= "2012-10-31") & (df["hs_m"] >= 3.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-27", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-30", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-27", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2012-10-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-08", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-30", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Duck Hurricane Sandy severe barrier attack (Oct 27-31, 2012) with Hs >= 3.5m",
    },

    # === CATEGORY 2: SURVEY-VALIDATED PROFILE CHANGES (6 QUERIES) ===
    {
        "query": "Narrabeen June 2016 severe subaerial beach cut and shoreline retreat",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.0, "tp_s": 12.0, "tan_beta": 0.05},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2016-05-30") & (df["timestamp"] <= "2016-06-15") & (df["hs_m"] >= 2.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-05-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-05-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2016-05-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-30", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.2),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Survey-anchored beach cut window at Narrabeen corresponding to documented ΔX = -42.7m subaerial shoreline retreat",
    },
    {
        "query": "Truc Vert catastrophic dune scarping and volume loss during winter 2013-2014",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 6.5, "tp_s": 14.5, "tan_beta": 0.04},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2013-12-15") & (df["timestamp"] <= "2014-03-15") & (df["hs_m"] >= 4.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.5,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and ((pd.Timestamp("2013-11-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-15", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 3.0),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.5,
        ),
        "description": "Field survey documented dune volume cut at Truc Vert during the extreme 2013-2014 winter storm sequence (Hs >= 4.5m)",
    },
    {
        "query": "Duck Hurricane Sandy barrier beach subaerial volume loss October 2012",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.2, "tp_s": 12.5, "tan_beta": 0.08},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2012-10-25") & (df["timestamp"] <= "2012-11-05") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-30", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2012-10-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-15", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.2),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-10-30", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Survey-observed barrier island erosion window at Duck NC following Hurricane Sandy (Hs >= 3.0m)",
    },
    {
        "query": "Torrey Pines constructive berm welding during December 2015 long period swell",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.1, "tp_s": 13.5, "tan_beta": 0.03},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-12-01") & (df["timestamp"] <= "2015-12-31") & (df["tp_s"] >= 12.0) & (df["hs_m"] <= 1.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-12-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-12-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.5,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and (float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("hs_m", 0.0)) <= 1.8),
        ),
        "description": "Documented accretion survey epoch at Torrey Pines (+29.7m advance) driven by long-period constructive swell (Tp >= 12s, Hs <= 1.5m)",
    },
    {
        "query": "Torrey Pines summer berm narrowing under southern hemisphere swell",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.0, "tp_s": 13.0, "tan_beta": 0.03},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (pd.to_datetime(df["timestamp"]).dt.month == 7) & (df["tp_s"] >= 12.0) & (df["hs_m"] <= 1.2),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and _get_row_ts(row).month == 7 and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and _get_row_ts(row).month == 7 and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.2,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and (_get_row_ts(row).month in [6, 7, 8] and float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("hs_m", 0.0)) <= 1.5),
        ),
        "description": "Mid-summer low-amplitude southern swell at Torrey Pines (July, Tp >= 12s, Hs <= 1.2m)",
    },
    {
        "query": "Truc Vert post-storm ridge and runnel welding summer 2016",
        "category": "2. Survey-Validated Profile Changes",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.9, "tp_s": 10.5, "tan_beta": 0.04},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2016-06-01") & (df["timestamp"] <= "2016-08-31") & (df["hs_m"] <= 1.2) & (df["tp_s"] >= 9.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2016-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.2 and float(row.get("tp_s", 0.0)) >= 9.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2016-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.2 and float(row.get("tp_s", 0.0)) >= 9.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and (float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("tp_s", 0.0)) >= 8.0),
        ),
        "description": "Documented summer 2016 intertidal recovery survey epoch at Truc Vert (+25.7m advance, Hs <= 1.2m, Tp >= 9s)",
    },

    # === CATEGORY 3: PHYSICAL TELEMETRY THRESHOLDS (6 QUERIES) ===
    {
        "query": "Extreme wave height Hs over 4.5m with long period swell",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 5.0, "tp_s": 13.0},
        "mask_fn": lambda df: (df["hs_m"] >= 4.5) & (df["tp_s"] >= 12.0),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 4.5 and float(row.get("tp_s", 0.0)) >= 12.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) >= 4.5 and float(row.get("tp_s", 0.0)) >= 12.0,
            lambda row, df: float(row.get("hs_m", 0.0)) >= 3.5 and float(row.get("tp_s", 0.0)) >= 10.0,
        ),
        "description": "Pure physical telemetry: extreme significant wave height Hs >= 4.5m with long-period swell Tp >= 12s",
    },
    {
        "query": "Severe 2% wave runup excursion R2 exceeding 2.0m",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 12.0, "total_water_level_m": 2.2},
        "mask_fn": lambda df: df["runup_r2_m"] >= 2.0,
        "condition_fn": lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 2.0,
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 1.5,
        ),
        "description": "Pure physical telemetry: extreme 2% wave runup excursion R2 >= 2.0m threatening the dune toe",
    },
    {
        "query": "Elevated total water level TWL over 2.0m during astronomical high tide",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 11.0, "total_water_level_m": 2.2},
        "mask_fn": lambda df: df["total_water_level_m"] >= 2.0,
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 2.0,
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.5,
        ),
        "description": "Pure physical telemetry: extreme total water level elevation TWL >= 2.0m",
    },
    {
        "query": "Compound extreme wave height over 3.0m and elevated total water level over 1.8m",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 11.5, "total_water_level_m": 2.0},
        "mask_fn": lambda df: (df["hs_m"] >= 3.0) & (df["total_water_level_m"] >= 1.8),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 3.0 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) >= 3.0 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
            lambda row, df: float(row.get("hs_m", 0.0)) >= 2.4 and float(row.get("total_water_level_m", 0.0)) >= 1.4,
        ),
        "description": "Coupled physical hazard: concurrent severe wave energy (Hs >= 3.0m) and high water level (TWL >= 1.8m)",
    },
    {
        "query": "Find extreme wave dissipation during active dune scarping with power flux exceeding 60 kW/m",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.0, "tp_s": 12.0},
        "mask_fn": lambda df: df["p_flux_Wm"] >= 60000.0,
        "condition_fn": lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 60000.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 60000.0,
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 35000.0,
        ),
        "description": "Energy flux threshold: un-attenuated offshore wave power flux P_flux >= 60,000 W/m during active dune scarping",
    },
    {
        "query": "Spilling dissipative breakers with surf similarity under 0.4",
        "category": "3. Physical Telemetry Thresholds",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Spilling",
        "target_synthetic_values": {"hs_m": 3.0, "tp_s": 7.0, "tan_beta": 0.04},
        "mask_fn": lambda df: (df["xi_iribarren"] < 0.4) | (df.get("breaker_regime", pd.Series([""] * len(df), index=df.index)) == "Spilling"),
        "condition_fn": lambda row, df: float(row.get("xi_iribarren", 1.0)) < 0.4 or str(row.get("breaker_regime", "")) == "Spilling",
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("xi_iribarren", 1.0)) < 0.4 or str(row.get("breaker_regime", "")) == "Spilling",
            lambda row, df: float(row.get("xi_iribarren", 1.0)) < 0.6,
        ),
        "description": "Breaker mechanics: dissipative spilling breakers with surf similarity parameter ξ < 0.4 (Battjes 1974)",
    },

    # === CATEGORY 4: COMPLEX PHYSICAL REGIMES (6 QUERIES) ===
    {
        "query": "Plunging curling breakers with intermediate Iribarren between 0.5 and 1.2",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Plunging",
        "target_synthetic_values": {"hs_m": 2.0, "tp_s": 10.0, "tan_beta": 0.06},
        "mask_fn": lambda df: (df["xi_iribarren"] >= 0.5) & (df["xi_iribarren"] <= 1.2),
        "condition_fn": lambda row, df: 0.5 <= float(row.get("xi_iribarren", 0.0)) <= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: 0.5 <= float(row.get("xi_iribarren", 0.0)) <= 1.2,
            lambda row, df: 0.35 <= float(row.get("xi_iribarren", 0.0)) <= 1.5,
        ),
        "description": "Breaker mechanics: intermediate plunging breakers with surf similarity 0.5 <= ξ <= 1.2",
    },
    {
        "query": "Surging reflective shorebreak with surf similarity exceeding 1.2",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Surging",
        "target_synthetic_values": {"hs_m": 0.6, "tp_s": 14.0, "tan_beta": 0.09},
        "mask_fn": lambda df: df["xi_iribarren"] >= 1.2,
        "condition_fn": lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 1.2,
            lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 0.9,
        ),
        "description": "Breaker mechanics: reflective surging shorebreak with ξ >= 1.2 on steep beach faces",
    },
    {
        "query": "High energy reflective collision regime with elevated TWL and steep slope",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.5, "tp_s": 11.0, "tan_beta": 0.08, "total_water_level_m": 1.8},
        "mask_fn": lambda df: (df["total_water_level_m"] >= 1.6) & (df["tan_beta"] >= 0.07) & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.6 and float(row.get("tan_beta", 0.0)) >= 0.07 and float(row.get("hs_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.6 and float(row.get("tan_beta", 0.0)) >= 0.07 and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.3 and float(row.get("tan_beta", 0.0)) >= 0.05 and float(row.get("hs_m", 0.0)) >= 1.5,
        ),
        "description": "Sallenger (2000) collision regime: high water levels striking steep reflective foreshore (TWL >= 1.6m, tanβ >= 0.07, Hs >= 2.0m)",
    },
    {
        "query": "Macrotidal energy dissipation where tidal range buffers dune attack",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 11.0, "total_water_level_m": 1.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["hs_m"] >= 3.0) & (df["total_water_level_m"] < 1.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 3.0 and float(row.get("total_water_level_m", 2.0)) < 1.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 3.0 and float(row.get("total_water_level_m", 2.0)) < 1.5,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 2.2 and float(row.get("total_water_level_m", 2.0)) < 1.8,
        ),
        "description": "Macrotidal tidal smearing: energetic waves (Hs >= 3.0m) breaking during low/mid tide at Truc Vert (TWL < 1.5m), buffering dunes",
    },
    {
        "query": "Low-steepness constructive swell with wave steepness below 0.015",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.9, "tp_s": 13.0},
        "mask_fn": lambda df: (df["hs_m"] / (1.56 * (df["tp_s"] ** 2) + 1e-4) < 0.015) & (df["tp_s"] >= 10.0),
        "condition_fn": lambda row, df: (float(row.get("hs_m", 0.0)) / (1.56 * (float(row.get("tp_s", 1.0)) ** 2) + 1e-4) < 0.015) and float(row.get("tp_s", 0.0)) >= 10.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: (float(row.get("hs_m", 0.0)) / (1.56 * (float(row.get("tp_s", 1.0)) ** 2) + 1e-4) < 0.015) and float(row.get("tp_s", 0.0)) >= 10.0,
            lambda row, df: (float(row.get("hs_m", 0.0)) / (1.56 * (float(row.get("tp_s", 1.0)) ** 2) + 1e-4) < 0.022) and float(row.get("tp_s", 0.0)) >= 8.5,
        ),
        "description": "Constructive swash physics: low wave steepness H0/L0 < 0.015 driving net onshore sediment transport",
    },
    {
        "query": "Infragravity wave resonance with peak wave period exceeding 14 seconds",
        "category": "4. Complex Physical Regimes",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.2, "tp_s": 15.0},
        "mask_fn": lambda df: (df["tp_s"] >= 14.0) & (df["hs_m"] >= 1.5),
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 14.0 and float(row.get("hs_m", 0.0)) >= 1.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("tp_s", 0.0)) >= 14.0 and float(row.get("hs_m", 0.0)) >= 1.5,
            lambda row, df: float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) >= 1.2,
        ),
        "description": "Infragravity dynamics: narrow long-period swell (Tp >= 14s, Hs >= 1.5m) exciting low-frequency surf beat",
    },

    # === CATEGORY 5: MULTI-CRITERIA & CROSS-SITE (6 QUERIES) ===
    {
        "query": "Severe East Coast Low swell from east-southeast attacking Narrabeen",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 12.0, "wave_dir_deg": 110.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["hs_m"] >= 3.0) & (df["wave_dir_deg"] >= 90) & (df["wave_dir_deg"] <= 150),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("hs_m", 0.0)) >= 3.0 and 90.0 <= float(row.get("wave_dir_deg", 0.0)) <= 150.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("hs_m", 0.0)) >= 3.0 and 90.0 <= float(row.get("wave_dir_deg", 0.0)) <= 150.0,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and (float(row.get("hs_m", 0.0)) >= 2.2 or (70.0 <= float(row.get("wave_dir_deg", 0.0)) <= 170.0)),
        ),
        "description": "Directional wave forcing: severe easterly/east-southeasterly storm swell (Hs >= 3.0m, 90 deg <= Dir <= 150 deg) attacking Narrabeen",
    },
    {
        "query": "Duck nor'easter storm waves over 3.0m with oblique northeast wave approach",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 10.0, "wave_dir_deg": 75.0},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["hs_m"] >= 3.0) & (df["wave_dir_deg"] >= 45) & (df["wave_dir_deg"] <= 110),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 3.0 and 45.0 <= float(row.get("wave_dir_deg", 0.0)) <= 110.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 3.0 and 45.0 <= float(row.get("wave_dir_deg", 0.0)) <= 110.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("hs_m", 0.0)) >= 2.2 or (30.0 <= float(row.get("wave_dir_deg", 0.0)) <= 130.0)),
        ),
        "description": "Directional storm forcing: Duck nor'easter storms with Hs >= 3.0m and oblique northeasterly wave angles (45 deg <= Dir <= 110 deg)",
    },
    {
        "query": "Biscay megaswell with significant wave height over 6.0m at Truc Vert",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 6.8, "tp_s": 15.5},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["hs_m"] >= 6.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 6.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 6.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 4.5,
        ),
        "description": "Oceanic megaswell extreme: deep-water North Atlantic storm swells striking Truc Vert with Hs >= 6.0m",
    },
    {
        "query": "Low-energy summer swell under 1.0m on steep reflective beach",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.8, "tp_s": 12.0, "tan_beta": 0.07},
        "mask_fn": lambda df: (df["hs_m"] <= 1.0) & (df["tp_s"] >= 10.0) & (df["tan_beta"] >= 0.06),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) <= 1.0 and float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("tan_beta", 0.0)) >= 0.06,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) <= 1.0 and float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("tan_beta", 0.0)) >= 0.06,
            lambda row, df: float(row.get("hs_m", 0.0)) <= 1.4 and float(row.get("tp_s", 0.0)) >= 8.5 and float(row.get("tan_beta", 0.0)) >= 0.04,
        ),
        "description": "Reflective summer state: calm constructive swell (Hs <= 1.0m, Tp >= 10s) on steep foreshore slopes (tanβ >= 0.06)",
    },
    {
        "query": "High water level surge without extreme waves causing toe saturation",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.4, "tp_s": 9.0, "total_water_level_m": 1.8},
        "mask_fn": lambda df: (df["total_water_level_m"] >= 1.6) & (df["hs_m"] <= 1.8),
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.6 and float(row.get("hs_m", 0.0)) <= 1.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.6 and float(row.get("hs_m", 0.0)) <= 1.8,
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.3 and float(row.get("hs_m", 0.0)) <= 2.2,
        ),
        "description": "Decoupled hazard: elevated total water level (TWL >= 1.6m) during modest wave energy (Hs <= 1.8m), testing multi-variable discrimination",
    },
    {
        "query": "Equilibrium beach recovery with long period swell over 12s and waves under 1.2m",
        "category": "5. Multi-Criteria & Cross-Site",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.9, "tp_s": 13.5},
        "mask_fn": lambda df: (df["tp_s"] >= 12.0) & (df["hs_m"] <= 1.2),
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.2,
            lambda row, df: float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("hs_m", 0.0)) <= 1.6,
        ),
        "description": "Equilibrium recovery state: sustained long-period constructive wave action (Tp >= 12s, Hs <= 1.2m) driving profile rebuilding",
    },

    # === CATEGORY 6: ADVANCED MORPHODYNAMIC & ENERGETIC REGIMES (15 QUERIES - MODE A) ===
    {
        "query": "Extreme wave power flux exceeding 80 kW/m during deepwater wave shoaling",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "tp_s": 13.0, "p_flux_Wm": 85000.0},
        "mask_fn": lambda df: df["p_flux_Wm"] >= 80000.0,
        "condition_fn": lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 80000.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 80000.0,
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 60000.0,
        ),
        "description": "Extreme wave energy flux P >= 80 kW/m driving rapid cross-shore boundary layer transport",
    },
    {
        "query": "Infragravity surf beat dominant over incident wave energy in dissipative surf zone",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"tp_s": 15.0, "runup_r_ig_m": 1.2},
        "mask_fn": lambda df: (_get_runup_ig_series(df) >= 1.0) & (df["tp_s"] >= 14.0),
        "condition_fn": lambda row, df: _get_row_runup_ig(row) >= 1.0 and float(row.get("tp_s", 0.0)) >= 14.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_runup_ig(row) >= 1.0 and float(row.get("tp_s", 0.0)) >= 14.0,
            lambda row, df: _get_row_runup_ig(row) >= 0.7 and float(row.get("tp_s", 0.0)) >= 12.0,
        ),
        "description": "Infragravity wave resonance and surf beat excursion R_ig >= 1.0m under long swell Tp >= 14s",
    },
    {
        "query": "Collapsing and surging breaker transition on steep reflective foreshore",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Surging",
        "target_synthetic_values": {"tan_beta": 0.08, "xi_iribarren": 1.3, "breaker_regime": "Surging"},
        "mask_fn": lambda df: (_get_breaker_series(df) == "Surging") | (df["xi_iribarren"] >= 1.2),
        "condition_fn": lambda row, df: _get_breaker(row) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_breaker(row) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 1.2,
            lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 0.9,
        ),
        "description": "Surging and collapsing breaker transition with high surf similarity parameter xi >= 1.2",
    },
    {
        "query": "Severe dune face scarping under combined high spring tide and plunging breakers",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"total_water_level_m": 2.0, "hs_m": 2.8, "breaker_regime": "Plunging"},
        "mask_fn": lambda df: (_get_breaker_series(df) == "Plunging") & (df["total_water_level_m"] >= 1.8) & (df["hs_m"] >= 2.5),
        "condition_fn": lambda row, df: _get_breaker(row) == "Plunging" and float(row.get("total_water_level_m", 0.0)) >= 1.8 and float(row.get("hs_m", 0.0)) >= 2.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_breaker(row) == "Plunging" and float(row.get("total_water_level_m", 0.0)) >= 1.8 and float(row.get("hs_m", 0.0)) >= 2.5,
            lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.5 and float(row.get("hs_m", 0.0)) >= 2.0,
        ),
        "description": "Compound collision regime: elevated TWL >= 1.8m and plunging curling breakers Hs >= 2.5m directly attacking dune toe",
    },
    {
        "query": "Post-storm subaerial berm accretion under micro-energy swell with Hs under 0.8m",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.7, "tp_s": 11.0, "dean_number": 1.2},
        "mask_fn": lambda df: (df["hs_m"] <= 0.8) & (df["tp_s"] >= 10.0),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) <= 0.8 and float(row.get("tp_s", 0.0)) >= 10.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) <= 0.8 and float(row.get("tp_s", 0.0)) >= 10.0,
            lambda row, df: float(row.get("hs_m", 0.0)) <= 1.1 and float(row.get("tp_s", 0.0)) >= 8.5,
        ),
        "description": "Micro-energy constructive wave recovery: low wave heights Hs <= 0.8m and swell periods Tp >= 10s welding berm",
    },
    {
        "query": "High Iribarren reflective collision regime with steep beach slope and narrow surf zone",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"xi_iribarren": 1.1, "tan_beta": 0.08, "hs_m": 1.8},
        "mask_fn": lambda df: (df["xi_iribarren"] >= 0.9) & (df["tan_beta"] >= 0.06) & (df["hs_m"] >= 1.5),
        "condition_fn": lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 0.9 and float(row.get("tan_beta", 0.0)) >= 0.06 and float(row.get("hs_m", 0.0)) >= 1.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 0.9 and float(row.get("tan_beta", 0.0)) >= 0.06 and float(row.get("hs_m", 0.0)) >= 1.5,
            lambda row, df: float(row.get("xi_iribarren", 0.0)) >= 0.7 and float(row.get("hs_m", 0.0)) >= 1.2,
        ),
        "description": "Reflective steep-slope wave attack with surf similarity xi >= 0.9 and tan beta >= 0.06",
    },
    {
        "query": "Dissipative spilling surf zone with wide wave energy attenuation buffer",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Spilling",
        "target_synthetic_values": {"dean_number": 6.0, "xi_iribarren": 0.35, "breaker_regime": "Spilling"},
        "mask_fn": lambda df: (_get_breaker_series(df) == "Spilling") & (df["dean_number"] >= 4.5) & (df["xi_iribarren"] <= 0.45),
        "condition_fn": lambda row, df: _get_breaker(row) == "Spilling" and float(row.get("dean_number", 0.0)) >= 4.5 and float(row.get("xi_iribarren", 1.0)) <= 0.45,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_breaker(row) == "Spilling" and float(row.get("dean_number", 0.0)) >= 4.5 and float(row.get("xi_iribarren", 1.0)) <= 0.45,
            lambda row, df: _get_breaker(row) == "Spilling" and float(row.get("dean_number", 0.0)) >= 3.5,
        ),
        "description": "Dissipative spilling equilibrium: high Dean parameter Omega >= 4.5 and low Iribarren xi <= 0.45 buffering wave impact",
    },
    {
        "query": "Prolonged storm duration with cumulative wave energy causing deep beach cut",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "upsilon": 30000.0},
        "mask_fn": lambda df: (df["hs_m"] >= 3.2) & (df["upsilon"] >= 20000.0),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 3.2 and float(row.get("upsilon", 0.0)) >= 20000.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) >= 3.2 and float(row.get("upsilon", 0.0)) >= 20000.0,
            lambda row, df: float(row.get("hs_m", 0.0)) >= 2.5 and float(row.get("upsilon", 0.0)) >= 12000.0,
        ),
        "description": "Cumulative storm impact: sustained high waves Hs >= 3.2m with extreme erosion potential Upsilon >= 20,000 kg/s",
    },
    {
        "query": "Compound high surge exceeding 0.8m with moderate wave height causing toe overwash",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"surge_m": 0.8, "hs_m": 2.2},
        "mask_fn": lambda df: (df["surge_m"] >= 0.4) & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: float(row.get("surge_m", 0.0)) >= 0.4 and float(row.get("hs_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("surge_m", 0.0)) >= 0.4 and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: float(row.get("surge_m", 0.0)) >= 0.25 and float(row.get("hs_m", 0.0)) >= 1.5,
        ),
        "description": "Surge-dominated overwash hazard: positive storm surge residual >= 0.4m coupled with waves Hs >= 2.0m",
    },
    {
        "query": "Short-period wind sea chop with steepness exceeding 0.035 causing rapid surface scour",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.2, "tp_s": 6.5, "wave_steepness": 0.04},
        "mask_fn": lambda df: (_get_steepness_series(df) >= 0.030) & (df["hs_m"] >= 2.0) & (df["tp_s"] <= 8.0),
        "condition_fn": lambda row, df: _get_row_steepness(row) >= 0.030 and float(row.get("hs_m", 0.0)) >= 2.0 and float(row.get("tp_s", 0.0)) <= 8.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_steepness(row) >= 0.030 and float(row.get("hs_m", 0.0)) >= 2.0 and float(row.get("tp_s", 0.0)) <= 8.0,
            lambda row, df: _get_row_steepness(row) >= 0.022 and float(row.get("hs_m", 0.0)) >= 1.5,
        ),
        "description": "High-steepness erosive sea chop: H0/L0 >= 0.030 with short wave periods Tp <= 8s driving surface grain shearing",
    },
    {
        "query": "Macrotidal low-tide surf zone shoaling where wide intertidal bar dissipates energy",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "total_water_level_m": 0.8},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["total_water_level_m"] <= 1.2) & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("total_water_level_m", 0.0)) <= 1.2 and float(row.get("hs_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("total_water_level_m", 0.0)) <= 1.2 and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("total_water_level_m", 0.0)) <= 1.5,
        ),
        "description": "Intertidal bar dissipation: macrotidal energetic breaking during low water levels (TWL <= 1.2m, Hs >= 2.0m)",
    },
    {
        "query": "Intermediate morphodynamic state with transverse bar and rip current circulation",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": "Plunging",
        "target_synthetic_values": {"dean_number": 3.0, "breaker_regime": "Plunging"},
        "mask_fn": lambda df: (df["dean_number"] >= 2.0) & (df["dean_number"] <= 4.0) & (_get_breaker_series(df) == "Plunging"),
        "condition_fn": lambda row, df: 2.0 <= float(row.get("dean_number", 0.0)) <= 4.0 and _get_breaker(row) == "Plunging",
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: 2.0 <= float(row.get("dean_number", 0.0)) <= 4.0 and _get_breaker(row) == "Plunging",
            lambda row, df: 1.5 <= float(row.get("dean_number", 0.0)) <= 4.8,
        ),
        "description": "Wright & Short (1984) intermediate state (2.0 <= Omega <= 4.0) with plunging waves and rhythmic rip bar cells",
    },
    {
        "query": "Extreme wave runup R2 exceeding 2.5m overtopping subaerial dune crest",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"runup_r2_m": 2.5, "hs_m": 3.8},
        "mask_fn": lambda df: df["runup_r2_m"] >= 2.2,
        "condition_fn": lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 2.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 2.2,
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 1.7,
        ),
        "description": "Overtopping regime: extreme 2% wave runup limit R2 >= 2.2m penetrating subaerial dune elevations",
    },
    {
        "query": "Calm constructive swash welding with long-period swell Tp over 15s",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"tp_s": 16.0, "hs_m": 1.1},
        "mask_fn": lambda df: (df["tp_s"] >= 15.0) & (df["hs_m"] <= 1.5),
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 15.0 and float(row.get("hs_m", 0.0)) <= 1.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("tp_s", 0.0)) >= 15.0 and float(row.get("hs_m", 0.0)) <= 1.5,
            lambda row, df: float(row.get("tp_s", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) <= 1.8,
        ),
        "description": "Constructive swell welding: ultra-long wave period Tp >= 15s with gentle wave heights Hs <= 1.5m",
    },
    {
        "query": "High-steepness erosive wave attack with Hs over 3.5m and steepness exceeding 0.025",
        "category": "6. Advanced Morphodynamics",
        "mode": "Mode A: Pure Phenomenological",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "wave_steepness": 0.03},
        "mask_fn": lambda df: (df["hs_m"] >= 3.5) & (_get_steepness_series(df) >= 0.025),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 3.5 and _get_row_steepness(row) >= 0.025,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) >= 3.5 and _get_row_steepness(row) >= 0.025,
            lambda row, df: float(row.get("hs_m", 0.0)) >= 2.8 and _get_row_steepness(row) >= 0.018,
        ),
        "description": "Severe erosive wave attack: concurrent extreme wave height Hs >= 3.5m and steep wave slope H0/L0 >= 0.025",
    },

    # === CATEGORY 7: EXTENDED MULTI-SITE HISTORICAL STORM EVENTS (15 QUERIES - MODE B) ===
    {
        "query": "Duck January 2016 Winter Storm Jonas severe nor'easter waves and surge",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "tp_s": 11.5, "surge_m": 0.4},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2016-01-22") & (df["timestamp"] <= "2016-01-26") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2016-01-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-30", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.2),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        ),
        "description": "Duck Winter Storm Jonas nor'easter peak (Jan 22-26, 2016) with severe wave action Hs >= 3.0m",
    },
    {
        "query": "Narrabeen August 2014 East Coast Low intense swell attack",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.0, "tp_s": 12.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2014-08-15") & (df["timestamp"] <= "2014-08-22") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.3,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2014-08-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-28", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.2),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.3,
        ),
        "description": "Narrabeen intense winter East Coast Low (Aug 15-22, 2014) with significant wave height Hs >= 3.0m",
    },
    {
        "query": "Truc Vert February 2014 Megastorm Ulla extreme Biscay wave impact",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 6.0, "tp_s": 14.5},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2014-02-12") & (df["timestamp"] <= "2014-02-17") & (df["hs_m"] >= 5.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-17", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-17", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-17", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and ((pd.Timestamp("2014-02-08", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-22", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 3.5),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-17", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        ),
        "description": "Truc Vert Megastorm Ulla peak (Feb 12-17, 2014) with catastrophic North Atlantic swell Hs >= 5.0m",
    },
    {
        "query": "Torrey Pines January 2016 peak El Niño energetic northwest Pacific swell",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.5, "tp_s": 14.0},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2016-01-15") & (df["timestamp"] <= "2016-01-25") & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 1.9,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and ((pd.Timestamp("2016-01-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-31", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 1.5),
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-25", tz="UTC") and float(row.get("hs_m", 0.0)) >= 1.9,
        ),
        "description": "Torrey Pines energetic Pacific winter swell during peak 2015-2016 El Nino (Jan 15-25, 2016, Hs >= 2.0m)",
    },
    {
        "query": "Duck March 2014 nor'easter storm waves with elevated total water level",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.2, "tp_s": 10.5},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2014-03-25") & (df["timestamp"] <= "2014-03-31") & (df["hs_m"] >= 2.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2014-03-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-05", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 1.8),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        ),
        "description": "Duck late-winter nor'easter sequence (Mar 25-31, 2014) with significant wave height Hs >= 2.5m",
    },
    {
        "query": "Narrabeen June 2012 severe cyclonic ECL storm and dune retreat",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.2, "tp_s": 12.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2012-06-03") & (df["timestamp"] <= "2012-06-12") & (df["hs_m"] >= 3.2),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2012-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-06-12", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.2,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2012-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-06-12", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2012-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-06-12", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.2,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2012-05-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-06-20", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.2),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2012-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-06-12", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Narrabeen June 2012 East Coast Low storm event with extreme erosion (Hs >= 3.2m)",
    },
    {
        "query": "Truc Vert December 2013 Megastorm Dirk preliminary winter dune erosion",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 5.2, "tp_s": 14.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2013-12-22") & (df["timestamp"] <= "2013-12-28") & (df["hs_m"] >= 4.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-12-28", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-12-28", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-12-28", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and ((pd.Timestamp("2013-12-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-01-05", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 3.0),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-22", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-12-28", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        ),
        "description": "Truc Vert preliminary winter megastorm Dirk (Dec 22-28, 2013) with energetic swell Hs >= 4.0m",
    },
    {
        "query": "Torrey Pines February 2016 prolonged Pacific swell sequence",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.0, "tp_s": 13.5},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2016-01-30") & (df["timestamp"] <= "2016-02-10") & (df["hs_m"] >= 1.6),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 1.6,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 1.6,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and ((pd.Timestamp("2016-01-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-15", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 1.2),
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2016-01-30", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        ),
        "description": "Torrey Pines sustained Pacific winter storm sequence (Jan 30 - Feb 10, 2016) with Hs >= 1.6m",
    },
    {
        "query": "Duck October 2015 Hurricane Joaquin distal swell and nor'easter surge",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "tp_s": 12.0, "surge_m": 0.5},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2015-10-01") & (df["timestamp"] <= "2015-10-06") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2015-09-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-10", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.0),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        ),
        "description": "Duck severe October 2015 Joaquin swell and coastal nor'easter surge window (Oct 1-6, 2015, Hs >= 3.0m)",
    },
    {
        "query": "Narrabeen November 2013 late spring moderate ECL embayed rotation",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 11.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2013-11-10") & (df["timestamp"] <= "2013-11-20") & (df["hs_m"] >= 2.2),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.2,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.6,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.2,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2013-11-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-25", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 1.6),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.6,
        ),
        "description": "Narrabeen moderate spring East Coast Low event (Nov 10-20, 2013) with Hs >= 2.2m",
    },
    {
        "query": "Truc Vert January 2015 high-energy winter storm wave attack",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "tp_s": 13.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2015-01-10") & (df["timestamp"] <= "2015-01-20") & (df["hs_m"] >= 3.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and ((pd.Timestamp("2015-01-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-25", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "Truc Vert energetic winter storm pulse (Jan 10-20, 2015) reaching Hs >= 3.5m",
    },
    {
        "query": "Torrey Pines July 2014 southern swell intertidal berm narrowing",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.0, "tp_s": 13.0},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2014-07-01") & (df["timestamp"] <= "2014-07-31") & (df["tp_s"] >= 12.0) & (df["hs_m"] <= 1.3),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2014-07-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-07-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.3,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2014-07-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-07-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) <= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2014-07-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-07-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("hs_m", 0.0)) <= 1.3,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and (float(row.get("tp_s", 0.0)) >= 10.0 and float(row.get("hs_m", 0.0)) <= 1.6),
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2014-07-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-07-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) <= 1.2,
        ),
        "description": "Torrey Pines southern hemisphere long swell window (July 2014, Tp >= 12s, Hs <= 1.3m)",
    },
    {
        "query": "Duck September 2014 post-summer barrier profile baseline survey epoch",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.1, "tp_s": 9.0},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2014-09-01") & (df["timestamp"] <= "2014-09-30") & (df["hs_m"] <= 1.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-30", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-30", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-30", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) <= 1.8,
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-30", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.0,
        ),
        "description": "Duck calm September 2014 late summer baseline profile conditions (Hs <= 1.5m)",
    },
    {
        "query": "Narrabeen April 2014 severe cyclonic swell impact and beach rotation",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.6, "tp_s": 11.5},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2014-04-10") & (df["timestamp"] <= "2014-04-20") & (df["hs_m"] >= 2.6),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-04-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.6,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-04-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-04-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.6,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2014-04-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-25", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.0),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-04-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-04-20", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        ),
        "description": "Narrabeen April 2014 cyclonic storm impact (Apr 10-20, 2014) with Hs >= 2.6m",
    },
    {
        "query": "Truc Vert summer 2015 constructive intertidal bar welding survey window",
        "category": "7. Extended Historical Storms",
        "mode": "Mode B: Full Operational",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.0, "tp_s": 10.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2015-06-01") & (df["timestamp"] <= "2015-08-31") & (df["hs_m"] <= 1.4) & (df["tp_s"] >= 8.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.4 and float(row.get("tp_s", 0.0)) >= 8.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.1 and float(row.get("tp_s", 0.0)) >= 9.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.4 and float(row.get("tp_s", 0.0)) >= 8.5,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and (float(row.get("hs_m", 0.0)) <= 1.7 and float(row.get("tp_s", 0.0)) >= 7.5),
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-08-31", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.1 and float(row.get("tp_s", 0.0)) >= 9.0,
        ),
        "description": "Truc Vert summer 2015 ridge and runnel berm welding epoch (Hs <= 1.4m, Tp >= 8.5s)",
    },

]

# Mode A queries: Pure Phenomenological Search across all 175,392 global multi-site hours.
# This is the primary evaluation regime for latent physical manifolds: testing whether
# 14D invariant manifolds (surf similarity xi, Dean ratio, Upsilon*, wave power flux)
# reliably capture hydrodynamic and morphodynamic states across disparate global
# observatories without calendar/site metadata crutches.
DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A: List[Dict[str, Any]] = [
    q for q in ALL_PHYSICS_BENCHMARK_QUERIES_60 if "Mode A" in q.get("mode", "")
]

# Mode B queries: Full Operational / Historical Named Storms with Calendar Dates.
# Archived: Mode B tests relational date/site string filtering rather than latent physical manifold similitude.
ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B: List[Dict[str, Any]] = [
    q for q in ALL_PHYSICS_BENCHMARK_QUERIES_60 if "Mode B" in q.get("mode", "")
]
DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_B = ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B

# Primary default benchmark query suite: 30 pure phenomenological queries (Mode A)
DEFAULT_PHYSICS_BENCHMARK_QUERIES: List[Dict[str, Any]] = DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A


NEW_PHYSICS_BENCHMARK_QUERIES_40: List[Dict[str, Any]] = [
    # =========================================================================
    # === CATEGORY 8: DELIBERATE AMBIGUITY & INTENT STRESS TESTS (20 QUERIES) ===
    # =========================================================================
    # These queries deliberately blend conflicting temporal and physical bounds,
    # antagonistic hydrodynamic regimes, or counter-intuitive phase descriptors
    # to rigorously stress-test the Hybrid Query Parser's zero-shot intent mapping.
    {
        "query": "Calm summer low-steepness swell welding the subaerial berm during the October 2012 Hurricane Sandy window at Duck",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.1, "tp_s": 11.0, "wave_steepness": 0.008},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2012-10-25") & (df["timestamp"] <= "2012-11-05") & (df["hs_m"] <= 1.5) & (df["tp_s"] >= 9.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.5 and float(row.get("tp_s", 0.0)) >= 9.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.2 and float(row.get("tp_s", 0.0)) >= 10.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.5 and float(row.get("tp_s", 0.0)) >= 9.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("hs_m", 0.0)) <= 1.8 and float(row.get("tp_s", 0.0)) >= 8.0),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2012-10-25", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2012-11-05", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.2 and float(row.get("tp_s", 0.0)) >= 10.0,
        ),
        "description": "Temporal-physical conflict: Bounded within Hurricane Sandy window (Oct-Nov 2012) but filtering for quiescent post-storm recovery swell (Hs <= 1.5m, Tp >= 9s)",
    },
    {
        "query": "Severe subaerial dune scarping with negative delta X under low wave heights Hs < 1.2m and dissipative Dean number Omega > 5.5",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": "Spilling",
        "target_synthetic_values": {"hs_m": 1.1, "dean_number": 6.0, "breaker_regime": "Spilling"},
        "mask_fn": lambda df: (df["hs_m"] <= 1.3) & (df["dean_number"] >= 5.0) & (_get_breaker_series(df) == "Spilling"),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) <= 1.3 and float(row.get("dean_number", 0.0)) >= 5.0 and _get_breaker(row) == "Spilling",
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) <= 1.3 and float(row.get("dean_number", 0.0)) >= 5.0 and _get_breaker(row) == "Spilling",
            lambda row, df: float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("dean_number", 0.0)) >= 4.0,
        ),
        "description": "Phase conflict: Explicit erosive scarping intent paired with calm wave height (Hs < 1.2m) and high Dean dissipative regime",
    },
    {
        "query": "Negative storm surge depression below -0.3m during peak extratropical cyclone swell Hs > 4.2m at Duck",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "surge_m": -0.35},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["hs_m"] >= 4.0) & (df["surge_m"] <= -0.20),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 4.0 and float(row.get("surge_m", 0.0)) <= -0.20,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 4.5 and float(row.get("surge_m", 0.0)) <= -0.30,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 4.0 and float(row.get("surge_m", 0.0)) <= -0.20,
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 3.5 and float(row.get("surge_m", 0.0)) <= -0.10,
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 4.5 and float(row.get("surge_m", 0.0)) <= -0.30,
        ),
        "description": "Meteorological conflict: Extreme storm wave height (Hs > 4.2m) combined with negative storm surge residual (surge <= -0.20m)",
    },
    {
        "query": "Winter storm Jonas epoch with accretionary Dean number Omega < 2.0 and low wave energy",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.2, "dean_number": 1.8},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2016-01-20") & (df["timestamp"] <= "2016-01-28") & (df["hs_m"] <= 1.6) & (df["dean_number"] <= 2.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-28", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("dean_number", 0.0)) <= 2.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-01-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-01-28", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("dean_number", 0.0)) <= 2.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("hs_m", 0.0)) <= 1.8 and float(row.get("dean_number", 0.0)) <= 3.0),
        ),
        "description": "Temporal-dynamic mismatch: Named winter storm window (Jonas, Jan 2016) filtering for uncharacteristic low-energy accretionary conditions",
    },
    {
        "query": "High storm surge residual exceeding 0.7m during calm micro-energy waves Hs < 1.0m",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"surge_m": 0.75, "hs_m": 0.8},
        "mask_fn": lambda df: (df["surge_m"] >= 0.50) & (df["hs_m"] <= 1.2),
        "condition_fn": lambda row, df: float(row.get("surge_m", 0.0)) >= 0.50 and float(row.get("hs_m", 0.0)) <= 1.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("surge_m", 0.0)) >= 0.50 and float(row.get("hs_m", 0.0)) <= 1.2,
            lambda row, df: float(row.get("surge_m", 0.0)) >= 0.35 and float(row.get("hs_m", 0.0)) <= 1.5,
        ),
        "description": "Hydrodynamic decoupling: Large atmospheric surge residual (>= 0.5m) occurring under calm wave action (Hs <= 1.2m)",
    },
    {
        "query": "Reflective surging breakers with surf similarity xi > 1.2 in macrotidal Truc Vert during astronomical high water",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": "Surging",
        "target_synthetic_values": {"xi_iribarren": 1.3, "total_water_level_m": 2.2, "breaker_regime": "Surging"},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["xi_iribarren"] >= 1.0) & (df["total_water_level_m"] >= 1.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("xi_iribarren", 0.0)) >= 1.0 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("xi_iribarren", 0.0)) >= 1.0 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and (float(row.get("xi_iribarren", 0.0)) >= 0.8 and float(row.get("total_water_level_m", 0.0)) >= 1.4),
        ),
        "description": "Site archetype inversion: Reflective surging breaker regime forced onto an otherwise dissipative/intermediate macrotidal coast",
    },
    {
        "query": "Prolonged high wave power flux over 40 kW/m without significant storm surge residual under 0.1m",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"p_flux_Wm": 45000.0, "surge_m": 0.05},
        "mask_fn": lambda df: (df["p_flux_Wm"] >= 35000.0) & (df["surge_m"].abs() <= 0.15),
        "condition_fn": lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 35000.0 and abs(float(row.get("surge_m", 0.0))) <= 0.15,
        "peak_work_cond": lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 45000.0 and abs(float(row.get("surge_m", 0.0))) <= 0.10,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 35000.0 and abs(float(row.get("surge_m", 0.0))) <= 0.15,
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 25000.0 and abs(float(row.get("surge_m", 0.0))) <= 0.25,
            lambda row, df: float(row.get("p_flux_Wm", 0.0)) >= 45000.0 and abs(float(row.get("surge_m", 0.0))) <= 0.10,
        ),
        "description": "Cross-channel decoupling: Severe deepwater wave energy flux (P >= 35 kW/m) completely unaccompanied by barometric surge",
    },
    {
        "query": "Mild constructive swash welding during active East Coast Low storm window at Narrabeen",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.2, "tp_s": 11.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2016-06-01") & (df["timestamp"] <= "2016-06-15") & (df["hs_m"] <= 1.6) & (df["tp_s"] >= 9.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("tp_s", 0.0)) >= 9.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6 and float(row.get("tp_s", 0.0)) >= 9.5,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and (float(row.get("hs_m", 0.0)) <= 2.0 and float(row.get("tp_s", 0.0)) >= 8.5),
        ),
        "description": "Event-reversal stress: Named ECL window at Narrabeen but filtering for post-storm constructive relaxation hours",
    },
    {
        "query": "Hurricane Matthew landfall sequence with short-period wind sea chop Tp < 7.0s at Duck",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 6.5},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2016-10-07") & (df["timestamp"] <= "2016-10-12") & (df["tp_s"] <= 7.5) & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-12", tz="UTC") and float(row.get("tp_s", 0.0)) <= 7.5 and float(row.get("hs_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-12", tz="UTC") and float(row.get("tp_s", 0.0)) <= 7.5 and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("tp_s", 0.0)) <= 8.5 and float(row.get("hs_m", 0.0)) >= 1.5),
        ),
        "description": "Spectral dispersion stress: Tropical storm landfall targeting short-period steep chop rather than peak ocean swell",
    },
    {
        "query": "Extreme wave runup R2 > 2.0m occurring during microtidal low water level TWL < 0.6m",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"runup_r2_m": 2.2, "total_water_level_m": 0.5},
        "mask_fn": lambda df: (df["runup_r2_m"] >= 1.8) & (df["total_water_level_m"] <= 0.8),
        "condition_fn": lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 1.8 and float(row.get("total_water_level_m", 0.0)) <= 0.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 1.8 and float(row.get("total_water_level_m", 0.0)) <= 0.8,
            lambda row, df: float(row.get("runup_r2_m", 0.0)) >= 1.4 and float(row.get("total_water_level_m", 0.0)) <= 1.2,
        ),
        "description": "Boundary elevation conflict: High swash dynamic excursion (R2 >= 1.8m) decoupled from low absolute total water level (TWL <= 0.8m)",
    },
    {
        "query": "High Iribarren reflective collision regime during El Niño winter window at Torrey Pines",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": "Surging",
        "target_synthetic_values": {"xi_iribarren": 0.9, "hs_m": 1.8, "tan_beta": 0.055},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-12-01") & (df["timestamp"] <= "2016-03-01") & (df["xi_iribarren"] >= 0.75),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-03-01", tz="UTC") and float(row.get("xi_iribarren", 0.0)) >= 0.75,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-03-01", tz="UTC") and float(row.get("xi_iribarren", 0.0)) >= 0.75,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("xi_iribarren", 0.0)) >= 0.60,
        ),
        "description": "Macro-climate anomaly stress: El Niño winter targeting steep reflective swash regimes at Torrey Pines",
    },
    {
        "query": "Dissipative spilling wave conditions with Dean number Omega > 6.0 during calm summer months",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": "Spilling",
        "target_synthetic_values": {"dean_number": 6.5, "hs_m": 1.5},
        "mask_fn": lambda df: (df["timestamp"].dt.month.isin([6, 7, 8])) & (df["dean_number"] >= 5.5) & (_get_breaker_series(df) == "Spilling"),
        "condition_fn": lambda row, df: _get_row_ts(row).month in [6, 7, 8] and float(row.get("dean_number", 0.0)) >= 5.5 and _get_breaker(row) == "Spilling",
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_ts(row).month in [6, 7, 8] and float(row.get("dean_number", 0.0)) >= 5.5 and _get_breaker(row) == "Spilling",
            lambda row, df: _get_row_ts(row).month in [6, 7, 8] and float(row.get("dean_number", 0.0)) >= 4.0,
        ),
        "description": "Seasonal inversion: Highly dissipative conditions (Omega >= 5.5) occurring during typically accretive summer months",
    },
    {
        "query": "Rapid shoreline retreat under moderate wave height Hs < 2.0m with steep wave slope exceeding 0.03",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.8, "wave_steepness": 0.032},
        "mask_fn": lambda df: (df["hs_m"] <= 2.2) & (_get_steepness_series(df) >= 0.028),
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) <= 2.2 and _get_row_steepness(row) >= 0.028,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("hs_m", 0.0)) <= 2.2 and _get_row_steepness(row) >= 0.028,
            lambda row, df: float(row.get("hs_m", 0.0)) <= 2.5 and _get_row_steepness(row) >= 0.022,
        ),
        "description": "Steepness-driven erosive attack: Moderate wave amplitude coupled with destructive steepness (s0 >= 0.028)",
    },
    {
        "query": "Peak astronomical spring tide high water TWL > 2.2m under micro-energy waves Hs < 0.8m at Truc Vert",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"total_water_level_m": 2.4, "hs_m": 0.7},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["total_water_level_m"] >= 2.0) & (df["hs_m"] <= 0.9),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("total_water_level_m", 0.0)) >= 2.0 and float(row.get("hs_m", 0.0)) <= 0.9,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("total_water_level_m", 0.0)) >= 2.0 and float(row.get("hs_m", 0.0)) <= 0.9,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and (float(row.get("total_water_level_m", 0.0)) >= 1.6 and float(row.get("hs_m", 0.0)) <= 1.2),
        ),
        "description": "Tide-wave imbalance: Macro-tidal inundation elevation (TWL >= 2.0m) paired with glassy wave calm (Hs <= 0.9m)",
    },
    {
        "query": "Subaerial berm recovery and swell welding during February 2014 Megastorm Ulla window at Truc Vert",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.5, "tp_s": 12.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2014-02-10") & (df["timestamp"] <= "2014-02-25") & (df["hs_m"] <= 2.2) & (df["tp_s"] >= 10.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-25", tz="UTC") and float(row.get("hs_m", 0.0)) <= 2.2 and float(row.get("tp_s", 0.0)) >= 10.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-02-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-02-25", tz="UTC") and float(row.get("hs_m", 0.0)) <= 2.2 and float(row.get("tp_s", 0.0)) >= 10.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and (float(row.get("hs_m", 0.0)) <= 2.8 and float(row.get("tp_s", 0.0)) >= 9.0),
        ),
        "description": "Megastorm inter-pulse lull: Filtering for quiescent swell recovery during the historic February 2014 winter window",
    },
    {
        "query": "Sustained high wave period Tp > 15s with negative disequilibrium accretion potential",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"tp_s": 16.0, "hs_m": 1.2, "upsilon": 5000.0},
        "mask_fn": lambda df: (df["tp_s"] >= 14.5) & (df["hs_m"] <= 1.4),
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 14.5 and float(row.get("hs_m", 0.0)) <= 1.4,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: float(row.get("tp_s", 0.0)) >= 14.5 and float(row.get("hs_m", 0.0)) <= 1.4,
            lambda row, df: float(row.get("tp_s", 0.0)) >= 12.5 and float(row.get("hs_m", 0.0)) <= 1.7,
        ),
        "description": "Disequilibrium state: Long period constructive swell producing negative erosion disequilibrium",
    },
    {
        "query": "Strong onshore gale winds > 15 m/s with gentle wave heights Hs < 1.5m at Duck",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"wind_speed_ms": 16.0, "hs_m": 1.2},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["wind_speed_ms"] >= 13.0) & (df["hs_m"] <= 1.6),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("wind_speed_ms", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) <= 1.6,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("wind_speed_ms", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) <= 1.6,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("wind_speed_ms", 0.0)) >= 10.0 and float(row.get("hs_m", 0.0)) <= 2.0),
        ),
        "description": "Atmospheric-wave decoupling: Gale force winds (>= 13 m/s) with limited fetch producing modest wave height (Hs <= 1.6m)",
    },
    {
        "query": "Plunging breaker collision during calm astronomical neap tide at Narrabeen",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": "Plunging",
        "target_synthetic_values": {"hs_m": 2.2, "total_water_level_m": 0.8, "breaker_regime": "Plunging"},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (_get_breaker_series(df) == "Plunging") & (df["total_water_level_m"] <= 0.9),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and _get_breaker(row) == "Plunging" and float(row.get("total_water_level_m", 0.0)) <= 0.9,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and _get_breaker(row) == "Plunging" and float(row.get("total_water_level_m", 0.0)) <= 0.9,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and (_get_breaker(row) == "Plunging" or float(row.get("total_water_level_m", 0.0)) <= 1.2),
        ),
        "description": "Hydrodynamic phase stress: Plunging breaker dynamics during suppressed neap tidal water levels",
    },
    {
        "query": "Low wave steepness swell s0 < 0.010 with dune collision regime at Torrey Pines",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"wave_steepness": 0.007, "runup_r2_m": 2.0},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (_get_steepness_series(df) <= 0.009) & (df["runup_r2_m"] >= 1.6),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and _get_row_steepness(row) <= 0.009 and float(row.get("runup_r2_m", 0.0)) >= 1.6,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and _get_row_steepness(row) <= 0.009 and float(row.get("runup_r2_m", 0.0)) >= 1.6,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and (_get_row_steepness(row) <= 0.012 and float(row.get("runup_r2_m", 0.0)) >= 1.3),
        ),
        "description": "Parametric contradiction: Very flat swell (s0 <= 0.009) yet penetrating the subaerial dune via infragravity runup",
    },
    {
        "query": "High-energy wave power flux > 50 kW/m during late summer baseline survey epoch at Duck",
        "category": "8. Deliberate Ambiguity & Intent Stress Tests",
        "mode": "Mode C: Deliberate Ambiguity",
        "target_regime": None,
        "target_synthetic_values": {"p_flux_Wm": 55000.0, "hs_m": 3.8},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"].dt.month == 9) & (df["p_flux_Wm"] >= 40000.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and _get_row_ts(row).month == 9 and float(row.get("p_flux_Wm", 0.0)) >= 40000.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and _get_row_ts(row).month == 9 and float(row.get("p_flux_Wm", 0.0)) >= 50000.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and _get_row_ts(row).month == 9 and float(row.get("p_flux_Wm", 0.0)) >= 40000.0,
            lambda row, df: _get_row_site(row) == "duck_nc" and (_get_row_ts(row).month == 9 and float(row.get("p_flux_Wm", 0.0)) >= 25000.0),
            lambda row, df: _get_row_site(row) == "duck_nc" and _get_row_ts(row).month == 9 and float(row.get("p_flux_Wm", 0.0)) >= 50000.0,
        ),
        "description": "Baseline epoch anomaly: September post-summer baseline survey window experiencing an extreme early-season cyclone wave power spike",
    },

    # =========================================================================
    # === CATEGORY 9: 3D MORPHODYNAMIC PATHOLOGIES & 1D FAILURE MODES (20 QUERIES) ===
    # =========================================================================
    # These queries heavily target authentic field survey failure modes where
    # 1D equilibrium indices falsely predict erosion or accretion due to decoupled
    # 3D processes: 2D embayed rotation, infragravity resonance, antecedent scarp
    # unbuffering, and multi-year offshore sandbar sequestration.
    {
        "query": "Narrabeen June 2016 East Coast Low extreme wave obliquity and alongshore radiation stress Sxy divergence",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.5, "rad_stress_sxy": 130.0, "wave_dir_deg": 105.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2016-06-03") & (df["timestamp"] <= "2016-06-08") & (df["hs_m"] >= 3.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-03", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.5,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and ((pd.Timestamp("2016-05-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-15", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.5),
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2016-06-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-06-07", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.0,
        ),
        "description": "3D Pathology: 2D embayment rotation at Narrabeen (PF1 -16.11m retreat) driven by oblique wave radiation stress Sxy > 120 N/m where 1D models fail",
    },
    {
        "query": "Torrey Pines June 2015 ultra-long period swell infragravity surf-beat resonance stripping subaerial sand",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.48, "tp_s": 16.0, "wave_steepness": 0.0052},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-06-12") & (df["timestamp"] <= "2015-06-20") & (df["tp_s"] >= 14.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-06-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-06-20", tz="UTC") and float(row.get("tp_s", 0.0)) >= 14.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-06-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-06-18", tz="UTC") and float(row.get("tp_s", 0.0)) >= 15.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-06-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-06-20", tz="UTC") and float(row.get("tp_s", 0.0)) >= 14.5,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("tp_s", 0.0)) >= 13.0,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-06-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-06-18", tz="UTC") and float(row.get("tp_s", 0.0)) >= 15.5,
        ),
        "description": "3D Pathology: Infragravity resonance cut at Torrey Pines (-9.8m survey loss) under modest wave height (Hs=1.48m) where 1D ODE predicted +0.42m accretion",
    },
    {
        "query": "Duck October 2015 antecedent storm sequence unbuffering subaerial berm prior to Joaquin impact",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.7, "tp_s": 10.5, "p_flux_Wm": 35000.0},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2015-09-28") & (df["timestamp"] <= "2015-10-06") & (df["hs_m"] >= 2.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-09-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-04", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-09-28", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-06", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2015-09-24", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-10", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.0),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-10-04", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        ),
        "description": "3D Pathology: Clustered storm impulse at Duck (-6.62m erosion) where prior berm stripping left dune toe unbuffered (1D model predicted only -1.15m)",
    },
    {
        "query": "Truc Vert August 2014 persistent profile deficit from offshore sandbar sediment sequestration",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.2, "tp_s": 10.5, "dean_number": 2.2},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2014-08-15") & (df["timestamp"] <= "2014-08-25") & (df["hs_m"] <= 1.6),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-25", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-08-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-23", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.3,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-08-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-25", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.6,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) <= 1.8,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-08-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-23", tz="UTC") and float(row.get("hs_m", 0.0)) <= 1.3,
        ),
        "description": "3D Pathology: Offshore bar sand sequestration at Truc Vert (-8.45m deficit in Aug 2014) where 1D ODE predicted +1.10m fair-weather accretion",
    },
    {
        "query": "Truc Vert macrotidal spring low-tide surf zone shoaling where intertidal bars dissipate storm energy",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "total_water_level_m": 0.5},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["hs_m"] >= 3.2) & (df["total_water_level_m"] <= 0.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 3.2 and float(row.get("total_water_level_m", 0.0)) <= 0.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 3.2 and float(row.get("total_water_level_m", 0.0)) <= 0.8,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 2.5 and float(row.get("total_water_level_m", 0.0)) <= 1.2,
        ),
        "description": "3D Pathology: Dissipative sandflat buffering where extreme wave energy breaking at low tide produces zero dune face erosion",
    },
    {
        "query": "Narrabeen April 2015 cyclonic swell with sustained south-southeast wave obliquity driving northward drift",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "tp_s": 12.0, "rad_stress_sxy": 90.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2015-04-18") & (df["timestamp"] <= "2015-04-26") & (df["hs_m"] >= 2.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-24", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-18", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-26", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("hs_m", 0.0)) >= 2.2,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2015-04-20", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-04-24", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.2,
        ),
        "description": "3D Pathology: Oblique storm waves generating sustained alongshore sediment transport divergence in an embayed boundary",
    },
    {
        "query": "Torrey Pines winter El Niño long-period swell swash resonance with runup R2 exceeding dune toe",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.2, "tp_s": 15.5, "runup_r2_m": 2.4},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-12-15") & (df["timestamp"] <= "2016-02-15") & (df["tp_s"] >= 14.0) & (df["runup_r2_m"] >= 1.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-15", tz="UTC") and float(row.get("tp_s", 0.0)) >= 14.0 and float(row.get("runup_r2_m", 0.0)) >= 1.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-02-15", tz="UTC") and float(row.get("tp_s", 0.0)) >= 14.0 and float(row.get("runup_r2_m", 0.0)) >= 1.8,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("tp_s", 0.0)) >= 12.0 and float(row.get("runup_r2_m", 0.0)) >= 1.4,
        ),
        "description": "3D Pathology: Infragravity swash resonance penetrating the subaerial cliff toe during El Niño winter",
    },
    {
        "query": "Duck March 2014 bimodal sea state where antecedent scarping leaves dune face vulnerable to overwash",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.2, "tp_s": 10.0, "surge_m": 0.4},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2014-03-24") & (df["timestamp"] <= "2014-03-31") & (df["hs_m"] >= 2.5),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-24", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-03-24", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-31", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.5,
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 2.0,
        ),
        "description": "3D Pathology: Late winter nor'easter scarping where antecedent beach lowered profile accelerates overwash",
    },
    {
        "query": "Truc Vert Megastorm Hercules outer bar rip current excavation and seaward sediment jet",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 6.5, "tp_s": 15.0, "p_flux_Wm": 90000.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2014-01-04") & (df["timestamp"] <= "2014-01-09") & (df["hs_m"] >= 5.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-01-09", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-01-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-04", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-01-09", tz="UTC") and float(row.get("hs_m", 0.0)) >= 5.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 3.8,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2014-01-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-01-08", tz="UTC") and float(row.get("hs_m", 0.0)) >= 6.0,
        ),
        "description": "3D Pathology: Megastorm Hercules ripping deep crescentic bar channels and jetting sediment beyond cross-shore closure depth",
    },
    {
        "query": "Narrabeen August 2014 embayment rotation with severe alongshore gradient in wave breaking height",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "tp_s": 12.5},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2014-08-16") & (df["timestamp"] <= "2014-08-22") & (df["hs_m"] >= 2.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-16", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2014-08-16", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-08-22", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("hs_m", 0.0)) >= 2.2,
        ),
        "description": "3D Pathology: August 2014 winter East Coast Low driving embayment rotation across Narrabeen-Collaroy",
    },
    {
        "query": "Torrey Pines southern swell long-period infragravity swash collision during astronomical high tide",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"tp_s": 15.0, "total_water_level_m": 2.0, "hs_m": 1.4},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["tp_s"] >= 13.5) & (df["total_water_level_m"] >= 1.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("tp_s", 0.0)) >= 13.5 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("tp_s", 0.0)) >= 13.5 and float(row.get("total_water_level_m", 0.0)) >= 1.8,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("tp_s", 0.0)) >= 11.5 and float(row.get("total_water_level_m", 0.0)) >= 1.5,
        ),
        "description": "3D Pathology: Southern hemisphere long swell colliding with cliff base under combined astronomical high tide and infragravity runup",
    },
    {
        "query": "Duck September 2014 post-storm berm deficit failing to recover under short-period swell",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.1, "tp_s": 7.5},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2014-09-01") & (df["timestamp"] <= "2014-09-25") & (df["tp_s"] <= 8.5) & (df["hs_m"] <= 1.4),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-25", tz="UTC") and float(row.get("tp_s", 0.0)) <= 8.5 and float(row.get("hs_m", 0.0)) <= 1.4,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2014-09-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-09-25", tz="UTC") and float(row.get("tp_s", 0.0)) <= 8.5 and float(row.get("hs_m", 0.0)) <= 1.4,
            lambda row, df: _get_row_site(row) == "duck_nc" and (float(row.get("tp_s", 0.0)) <= 9.5 and float(row.get("hs_m", 0.0)) <= 1.8),
        ),
        "description": "3D Pathology: Post-summer profile recovery failure where short wave periods lack the onshore mass transport required to weld berm sand",
    },
    {
        "query": "Truc Vert winter 2013-2014 extreme cumulative wave power flux exhausting intertidal sediment supply",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"p_flux_Wm": 70000.0, "hs_m": 5.2},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2013-12-15") & (df["timestamp"] <= "2014-03-01") & (df["p_flux_Wm"] >= 50000.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-01", tz="UTC") and float(row.get("p_flux_Wm", 0.0)) >= 50000.0,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-01", tz="UTC") and float(row.get("p_flux_Wm", 0.0)) >= 70000.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-01", tz="UTC") and float(row.get("p_flux_Wm", 0.0)) >= 50000.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("p_flux_Wm", 0.0)) >= 35000.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2013-12-15", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2014-03-01", tz="UTC") and float(row.get("p_flux_Wm", 0.0)) >= 70000.0,
        ),
        "description": "3D Pathology: Sustained mega-cyclone clustering that permanently strips the subaerial reservoir beyond 1D recovery thresholds",
    },
    {
        "query": "Narrabeen Collaroy embayed headland wave refraction shadow with asymmetric alongshore transport",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"wave_dir_deg": 140.0, "hs_m": 2.5, "rad_stress_sxy": 80.0},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["wave_dir_deg"] >= 130.0) & (df["hs_m"] >= 2.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("wave_dir_deg", 0.0)) >= 130.0 and float(row.get("hs_m", 0.0)) >= 2.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("wave_dir_deg", 0.0)) >= 130.0 and float(row.get("hs_m", 0.0)) >= 2.0,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("wave_dir_deg", 0.0)) >= 115.0 and float(row.get("hs_m", 0.0)) >= 1.6,
        ),
        "description": "3D Pathology: Southern wave refraction behind headland shelter setting up strong alongshore energy gradients",
    },
    {
        "query": "Torrey Pines low-steepness infragravity swash overtopping cobblestone berm during El Niño",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"runup_r2_m": 2.2, "wave_steepness": 0.007},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["runup_r2_m"] >= 1.9) & (_get_steepness_series(df) <= 0.008),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("runup_r2_m", 0.0)) >= 1.9 and _get_row_steepness(row) <= 0.008,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("runup_r2_m", 0.0)) >= 1.9 and _get_row_steepness(row) <= 0.008,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and float(row.get("runup_r2_m", 0.0)) >= 1.5 and _get_row_steepness(row) <= 0.011,
        ),
        "description": "3D Pathology: Swash resonance overtopping coarse sediment berms without offshore wave breaking dissipation",
    },
    {
        "query": "Duck November 2015 nor'easter antecedent clustering where second storm peak drives accelerated dune toe scarping",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.4, "tp_s": 11.0, "surge_m": 0.35},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2015-11-05") & (df["timestamp"] <= "2015-11-15") & (df["hs_m"] >= 2.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-11-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-11-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2015-11-05", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-11-15", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.8,
            lambda row, df: _get_row_site(row) == "duck_nc" and float(row.get("hs_m", 0.0)) >= 2.2,
        ),
        "description": "3D Pathology: Clustered storm sequence where antecedent storm memory induces accelerated non-linear dune retreat",
    },
    {
        "query": "Truc Vert January 2015 outer bar channel rip drainage during falling macrotidal ebb tide",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "total_water_level_m": 1.0},
        "mask_fn": lambda df: (_get_site_series(df) == "truc_vert_france") & (df["timestamp"] >= "2015-01-12") & (df["timestamp"] <= "2015-01-18") & (df["hs_m"] >= 3.0),
        "condition_fn": lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-18", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "truc_vert_france" and pd.Timestamp("2015-01-12", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-01-18", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.0,
            lambda row, df: _get_row_site(row) == "truc_vert_france" and float(row.get("hs_m", 0.0)) >= 2.2,
        ),
        "description": "3D Pathology: Storm rip drainage across crescentic bar neck during falling macrotidal stage",
    },
    {
        "query": "Narrabeen November 2013 spring East Coast Low oblique swell inducing profile rotational skew",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 10.5},
        "mask_fn": lambda df: (_get_site_series(df) == "narrabeen_aus") & (df["timestamp"] >= "2013-11-10") & (df["timestamp"] <= "2013-11-18") & (df["hs_m"] >= 2.4),
        "condition_fn": lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-18", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.4,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and pd.Timestamp("2013-11-10", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2013-11-18", tz="UTC") and float(row.get("hs_m", 0.0)) >= 2.4,
            lambda row, df: _get_row_site(row) == "narrabeen_aus" and float(row.get("hs_m", 0.0)) >= 1.8,
        ),
        "description": "3D Pathology: Rotational profile response under moderately oblique spring storm forcing",
    },
    {
        "query": "Torrey Pines December 2015 El Niño infragravity surf-beat resonance under deepwater swell steepness under 0.006",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"tp_s": 16.5, "wave_steepness": 0.0055, "hs_m": 1.8},
        "mask_fn": lambda df: (_get_site_series(df) == "torrey_pines_ca") & (df["timestamp"] >= "2015-12-01") & (df["timestamp"] <= "2015-12-31") & (df["tp_s"] >= 15.0) & (_get_steepness_series(df) <= 0.007),
        "condition_fn": lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-12-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 15.0 and _get_row_steepness(row) <= 0.007,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and pd.Timestamp("2015-12-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2015-12-31", tz="UTC") and float(row.get("tp_s", 0.0)) >= 15.0 and _get_row_steepness(row) <= 0.007,
            lambda row, df: _get_row_site(row) == "torrey_pines_ca" and (float(row.get("tp_s", 0.0)) >= 13.0 and _get_row_steepness(row) <= 0.010),
        ),
        "description": "3D Pathology: Extreme low steepness Pacific swell driving infragravity wave amplification against the cliffs",
    },
    {
        "query": "Duck October 2016 Hurricane Matthew compound surge-wave overwash stripping subaerial profile beyond 1D equilibrium recovery",
        "category": "9. 3D Morphodynamic Pathologies & 1D Failure Modes",
        "mode": "Mode D: 3D Physical Pathology",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 4.8, "surge_m": 0.85, "tp_s": 13.0},
        "mask_fn": lambda df: (_get_site_series(df) == "duck_nc") & (df["timestamp"] >= "2016-10-07") & (df["timestamp"] <= "2016-10-11") & (df["hs_m"] >= 3.8),
        "condition_fn": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-11", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.8,
        "peak_work_cond": lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-08", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.2,
        "graded_relevance_fn": _make_graded_fn(
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-07", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-11", tz="UTC") and float(row.get("hs_m", 0.0)) >= 3.8,
            lambda row, df: _get_row_site(row) == "duck_nc" and ((pd.Timestamp("2016-10-01", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-18", tz="UTC")) or float(row.get("hs_m", 0.0)) >= 2.8),
            lambda row, df: _get_row_site(row) == "duck_nc" and pd.Timestamp("2016-10-08", tz="UTC") <= _get_row_ts(row) <= pd.Timestamp("2016-10-10", tz="UTC") and float(row.get("hs_m", 0.0)) >= 4.2,
        ),
        "description": "3D Pathology: Catastrophic 3D barrier island overwash stripping subaerial dunes past morphological tipping point",
    },
]

ALL_PHYSICS_BENCHMARK_QUERIES_100: List[Dict[str, Any]] = (
    ALL_PHYSICS_BENCHMARK_QUERIES_60 + NEW_PHYSICS_BENCHMARK_QUERIES_40
)

BENCHMARK_QUERIES_DELIBERATE_AMBIGUITY: List[Dict[str, Any]] = [
    q for q in NEW_PHYSICS_BENCHMARK_QUERIES_40 if "Ambiguity" in q.get("category", "")
]
BENCHMARK_QUERIES_PHYSICAL_PATHOLOGIES: List[Dict[str, Any]] = [
    q for q in NEW_PHYSICS_BENCHMARK_QUERIES_40 if "Patholog" in q.get("category", "")
]


_RAW_DATA_LLM_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "raw_llm_query_cache.json"
)
_RAW_DATA_LLM_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_raw_llm_cache() -> None:
    global _RAW_DATA_LLM_CACHE
    if not _RAW_DATA_LLM_CACHE and os.path.exists(_RAW_DATA_LLM_CACHE_FILE):
        try:
            with open(_RAW_DATA_LLM_CACHE_FILE, "r") as f:
                loaded = json.load(f)
                _RAW_DATA_LLM_CACHE.clear()
                _RAW_DATA_LLM_CACHE.update(loaded)
        except Exception:
            pass


def _save_raw_llm_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_RAW_DATA_LLM_CACHE_FILE), exist_ok=True)
        tmp_file = _RAW_DATA_LLM_CACHE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(_RAW_DATA_LLM_CACHE, f, indent=2)
        os.replace(tmp_file, _RAW_DATA_LLM_CACHE_FILE)
    except Exception as e:
        logger.warning(f"Failed to save RawDataLLM cache: {e}")


class RawDataLLMRAG:
    """Baseline 1: LLM reasoning agent operating directly on raw telemetry columns and schema.

    Feeds raw data column names and preview records directly to the LLM (or deterministic
    schema-level reasoning parser when offline) to test whether an LLM can reason over
    raw telemetry without higher-order physics feature engineering.
    """

    SYSTEM_PROMPT = """You are an oceanographic data retrieval agent reasoning directly over raw sensor telemetry columns and site metadata:
- site_id: Target site identifier string (e.g. 'duck_nc', 'narrabeen_aus', 'torrey_pines_ca', 'truc_vert_france')
- timestamp: Date/time of observation (ISO format YYYY-MM-DD HH:MM:SS)
- hs_m: Significant wave height (meters)
- tp_s: Peak wave period (seconds)
- total_water_level_m: Total water level elevation (meters)
- surge_m: Storm surge residual (meters)
- wind_speed_ms: Wind speed (m/s)
- tan_beta: Beach foreshore slope

Your task is to parse a natural language query into raw column and temporal/site filtering criteria.
Return a strictly typed JSON object with:
{
  "site_id": string|null,
  "date_start": string|null,
  "date_end": string|null,
  "hs_min": float|null,
  "hs_max": float|null,
  "tp_min": float|null,
  "tp_max": float|null,
  "twl_min": float|null,
  "twl_max": float|null,
  "surge_min": float|null,
  "surge_max": float|null,
  "wind_min": float|null,
  "wind_max": float|null
}
Dates must be formatted as YYYY-MM-DD. If a specific month or year is requested (e.g., 'June 2016'), set date_start and date_end to encompass that entire calendar window.
"""

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        use_live_api: bool = False,
        **kwargs: Any,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.model_name = model_name
        self.api_key = api_key
        self.use_live_api = use_live_api
        raw_candidates = ["site_id", "site", "timestamp", "hs_m", "tp_s", "total_water_level_m", "surge_m", "wind_speed_ms", "wind_normal_ms", "tan_beta"]
        self.raw_cols = [c for c in raw_candidates if c in self.df.columns]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        disable_spatiotemporal_filters: bool = False,
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        is_mode_a = disable_spatiotemporal_filters or (mode is not None and ("mode a" in str(mode).lower() or "phenomenological" in str(mode).lower()))
        criteria: Dict[str, Any] = {}
        if self.use_live_api:
            raw_cache_key = f"{self.model_name}:{query_text.strip()}"
            _load_raw_llm_cache()
            if raw_cache_key in _RAW_DATA_LLM_CACHE:
                criteria = _RAW_DATA_LLM_CACHE[raw_cache_key]
            else:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.api_key, http_options={"timeout": 45000}) if self.api_key else genai.Client(http_options={"timeout": 45000})
                    sample_df = self.df[self.raw_cols].dropna().head(5).copy()
                    if "timestamp" in sample_df.columns:
                        sample_df["timestamp"] = sample_df["timestamp"].astype(str)
                    sample_rows = sample_df.to_dict(orient="records")
                    prompt = (
                        f"Available raw telemetry columns: {self.raw_cols}\n"
                        f"Sample rows from telemetry dataset: {sample_rows}\n"
                        f"User natural language query: '{query_text}'\n"
                        f"Extract JSON filtering criteria over these raw variables and metadata."
                    )
                    response = _call_gemini_with_retry(
                        client=client,
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "system_instruction": self.SYSTEM_PROMPT,
                            "response_mime_type": "application/json",
                            "temperature": 0.0,
                            "http_options": {"timeout": 45000},
                        },
                        max_retries=3,
                        backoff=2.0,
                    )
                    if response and response.text:
                        import json
                        criteria = json.loads(response.text)
                        _RAW_DATA_LLM_CACHE[raw_cache_key] = criteria
                        _save_raw_llm_cache()
                except Exception as e:
                    logger.error(f"RawDataLLMRAG Gemini call failed for '{query_text}': {e}")
                    criteria = self._offline_raw_reasoning(query_text)
        else:
            criteria = self._offline_raw_reasoning(query_text)

        sub = self.df.copy()

        # 1. Site filtering (bypassed in Mode A: Pure Phenomenological Search)
        site_val = criteria.get("site_id")
        if not is_mode_a and site_val:
            site_col = "site_id" if "site_id" in sub.columns else ("site" if "site" in sub.columns else None)
            if site_col:
                s_mask = sub[site_col].astype(str).str.lower() == str(site_val).lower()
                if s_mask.any():
                    sub = sub[s_mask]

        # 2. Date window filtering (bypassed in Mode A: Pure Phenomenological Search)
        d_start = criteria.get("date_start")
        d_end = criteria.get("date_end")
        if not is_mode_a and (d_start or d_end) and "timestamp" in sub.columns and len(sub) > 0:
            try:
                ts = pd.to_datetime(sub["timestamp"], utc=True)
                mask = pd.Series(True, index=sub.index)
                if d_start:
                    mask &= (ts >= pd.to_datetime(d_start, utc=True))
                if d_end:
                    mask &= (ts <= pd.to_datetime(d_end, utc=True))
                sub = sub[mask]
            except Exception as e:
                logger.error(f"RawDataLLMRAG date filtering error: {e}")

        # 3. Sensor threshold filtering
        if criteria.get("hs_min") is not None and "hs_m" in sub.columns:
            sub = sub[sub["hs_m"] >= criteria["hs_min"]]
        if criteria.get("hs_max") is not None and "hs_m" in sub.columns:
            sub = sub[sub["hs_m"] <= criteria["hs_max"]]
        if criteria.get("tp_min") is not None and "tp_s" in sub.columns:
            sub = sub[sub["tp_s"] >= criteria["tp_min"]]
        if criteria.get("tp_max") is not None and "tp_s" in sub.columns:
            sub = sub[sub["tp_s"] <= criteria["tp_max"]]
        if criteria.get("twl_min") is not None and "total_water_level_m" in sub.columns:
            sub = sub[sub["total_water_level_m"] >= criteria["twl_min"]]
        if criteria.get("surge_min") is not None and "surge_m" in sub.columns:
            sub = sub[sub["surge_m"] >= criteria["surge_min"]]
        if criteria.get("wind_min") is not None and "wind_speed_ms" in sub.columns:
            sub = sub[sub["wind_speed_ms"] >= criteria["wind_min"]]

        if len(sub) == 0:
            matches = self.df.head(top_k).copy()
            sims = np.zeros(len(matches))
        else:
            sort_col = "hs_m" if "hs_m" in sub.columns else ("total_water_level_m" if "total_water_level_m" in sub.columns else self.raw_cols[0])
            descending = True
            if criteria.get("hs_max") is not None or "calm" in query_text.lower() or "recovery" in query_text.lower():
                descending = False
            matches = sub.sort_values(sort_col, ascending=not descending).head(top_k).copy()
            sims = np.linspace(0.85, 0.70, len(matches))

        return PhysicsRAGResult(
            query=query_text,
            matches_df=matches,
            similarities=np.array(sims),
            explanation="RawDataLLMRAG: LLM raw schema reasoning over sensor telemetry",
            parsed_criteria=criteria,
        )

    def _offline_raw_reasoning(self, query_text: str) -> Dict[str, Any]:
        """Offline schema-aware reasoning mapping natural language to raw sensor column criteria."""
        q_lower = query_text.lower()
        crit: Dict[str, Any] = {}

        # Site extraction
        if any(w in q_lower for w in ["narrabeen", "sydney", "australia"]):
            crit["site_id"] = "narrabeen_aus"
        elif any(w in q_lower for w in ["duck", "north carolina"]):
            crit["site_id"] = "duck_nc"
        elif any(w in q_lower for w in ["torrey", "torrey pines", "california"]):
            crit["site_id"] = "torrey_pines_ca"
        elif any(w in q_lower for w in ["truc vert", "france", "biscay"]):
            crit["site_id"] = "truc_vert_france"
        elif any(w in q_lower for w in ["oahu", "hawaii"]):
            crit["site_id"] = "oahu_north_shore_hi"

        # Date extraction
        if "june 2016" in q_lower:
            crit["date_start"] = "2016-06-01"
            crit["date_end"] = "2016-06-30"
        elif "october 2016" in q_lower:
            crit["date_start"] = "2016-10-01"
            crit["date_end"] = "2016-10-31"
        elif "october 2012" in q_lower:
            crit["date_start"] = "2012-10-01"
            crit["date_end"] = "2012-10-31"
        elif "april 2015" in q_lower:
            crit["date_start"] = "2015-04-01"
            crit["date_end"] = "2015-04-30"
        elif "december 2015" in q_lower:
            crit["date_start"] = "2015-12-01"
            crit["date_end"] = "2015-12-31"
        elif "january 2016" in q_lower or "jonas" in q_lower:
            crit["date_start"] = "2016-01-01"
            crit["date_end"] = "2016-01-31"
        elif "august 2014" in q_lower:
            crit["date_start"] = "2014-08-01"
            crit["date_end"] = "2014-08-31"
        elif "february 2014" in q_lower or "ulla" in q_lower:
            crit["date_start"] = "2014-02-01"
            crit["date_end"] = "2014-02-28"
        elif "march 2014" in q_lower:
            crit["date_start"] = "2014-03-01"
            crit["date_end"] = "2014-03-31"
        elif "june 2012" in q_lower:
            crit["date_start"] = "2012-06-01"
            crit["date_end"] = "2012-06-30"
        elif "december 2013" in q_lower or "dirk" in q_lower:
            crit["date_start"] = "2013-12-01"
            crit["date_end"] = "2013-12-31"
        elif "february 2016" in q_lower:
            crit["date_start"] = "2016-02-01"
            crit["date_end"] = "2016-02-29"
        elif "october 2015" in q_lower or "joaquin" in q_lower:
            crit["date_start"] = "2015-10-01"
            crit["date_end"] = "2015-10-31"
        elif "november 2013" in q_lower:
            crit["date_start"] = "2013-11-01"
            crit["date_end"] = "2013-11-30"
        elif "january 2015" in q_lower:
            crit["date_start"] = "2015-01-01"
            crit["date_end"] = "2015-01-31"
        elif "july 2014" in q_lower:
            crit["date_start"] = "2014-07-01"
            crit["date_end"] = "2014-07-31"
        elif "september 2014" in q_lower:
            crit["date_start"] = "2014-09-01"
            crit["date_end"] = "2014-09-30"
        elif "april 2014" in q_lower:
            crit["date_start"] = "2014-04-01"
            crit["date_end"] = "2014-04-30"
        elif "summer 2015" in q_lower:
            crit["date_start"] = "2015-06-01"
            crit["date_end"] = "2015-08-31"
        elif "winter 2013-2014" in q_lower or "winter 2013" in q_lower:
            crit["date_start"] = "2013-12-01"
            crit["date_end"] = "2014-03-31"
        elif "summer 2016" in q_lower:
            crit["date_start"] = "2016-06-01"
            crit["date_end"] = "2016-08-31"
        elif "2015-2016" in q_lower:
            crit["date_start"] = "2015-12-01"
            crit["date_end"] = "2016-03-31"

        if "over 3m" in q_lower or "wave height over" in q_lower:
            crit["hs_min"] = 2.7
        elif "over 4m" in q_lower:
            crit["hs_min"] = 3.8
        elif any(w in q_lower for w in ["storm", "high energy", "severe", "scarping", "extreme"]):
            crit["hs_min"] = 2.2
        elif any(w in q_lower for w in ["calm", "fair weather", "mild", "recovery"]):
            crit["hs_max"] = 1.3

        if any(w in q_lower for w in ["long-period", "long period", "swell"]):
            crit["tp_min"] = 10.5
        elif any(w in q_lower for w in ["short period", "wind sea", "choppy"]):
            crit["tp_max"] = 8.0

        if any(w in q_lower for w in ["total water level", "water level", "dune toe"]):
            crit["twl_min"] = 1.6
        elif "surge" in q_lower:
            crit["surge_min"] = 0.35

        if any(w in q_lower for w in ["gale", "wind"]):
            crit["wind_min"] = 8.0

        return crit


# Alias TextRAG to RawDataLLMRAG for backwards compatibility
TextRAG = RawDataLLMRAG


class RawDataRAG:
    """(Deprecated) Baseline: Nearest-neighbor search directly on standardized raw sensor channels.

    .. deprecated::
        Retained for backward compatibility. Raw sensor nearest-neighbor search lacks physical
        manifold structure and cannot resolve phenomenological domain queries.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        raw_candidates = ["hs_m", "tp_s", "total_water_level_m", "surge_m", "wind_speed_ms", "wind_normal_ms"]
        self.raw_cols = [c for c in raw_candidates if c in self.df.columns]
        if not self.raw_cols:
            self.raw_cols = ["hs_m", "tp_s"]
        X_raw = self.df[self.raw_cols].fillna(0.0).values
        self.scaler = RobustScaler().fit(X_raw)
        self.embeddings = self.scaler.transform(X_raw)
        self.nn = NearestNeighbors(metric="cosine").fit(self.embeddings)

    def query(self, query_text: str, top_k: int = 5, target_synthetic_values: Optional[Dict[str, Any]] = None, **kwargs: Any) -> PhysicsRAGResult:
        vals = target_synthetic_values or {}
        if not vals:
            # Extract raw numeric constraints from query_text
            extracted: Dict[str, float] = {}
            import re
            hs_m = re.search(r'(?:hs|wave height|waves?)\s*(?:over|>|>=|of|around)?\s*(\d+\.?\d*)\s*m?', query_text, re.I)
            if hs_m:
                extracted["hs_m"] = float(hs_m.group(1))
            tp_m = re.search(r'(?:tp|period|swell)\s*(?:over|>|>=|of|around)?\s*(\d+\.?\d*)\s*s', query_text, re.I)
            if tp_m:
                extracted["tp_s"] = float(tp_m.group(1))
            twl_m = re.search(r'(?:twl|water level)\s*(?:over|>|>=|of|around)?\s*(\d+\.?\d*)\s*m?', query_text, re.I)
            if twl_m:
                extracted["total_water_level_m"] = float(twl_m.group(1))
            surge_m = re.search(r'(?:surge)\s*(?:over|>|>=|of|around)?\s*(\d+\.?\d*)\s*m?', query_text, re.I)
            if surge_m:
                extracted["surge_m"] = float(surge_m.group(1))
            wind_m = re.search(r'(?:wind)\s*(?:over|>|>=|of|around)?\s*(\d+\.?\d*)\s*(?:m/s|mps)?', query_text, re.I)
            if wind_m:
                extracted["wind_speed_ms"] = float(wind_m.group(1))
            vals = extracted

        # If target values omitted and no numeric constraints present in query text:
        if not vals:
            # Methodological rigor: do NOT default to dataset column means!
            return PhysicsRAGResult(
                query=query_text,
                matches_df=pd.DataFrame(),
                similarities=np.array([]),
                explanation="Raw-Data RAG: No explicit numerical sensor measurements provided in query text or probe targets. Raw telemetry search cannot interpret phenomenological text without values.",
                parsed_criteria={},
            )

        q_list = []
        for col in self.raw_cols:
            if col in vals:
                q_list.append(float(vals[col]))
            elif col == "total_water_level_m" and "surge_m" in vals:
                q_list.append(float(vals["surge_m"]))
            elif col == "surge_m" and "total_water_level_m" in vals:
                q_list.append(float(vals["total_water_level_m"]))
            elif col == "wind_speed_ms" and "wind_normal_ms" in vals:
                q_list.append(float(vals["wind_normal_ms"]))
            else:
                q_list.append(0.0)
        q_vec = np.array([q_list])
        q_scaled = self.scaler.transform(q_vec)
        distances, indices = self.nn.kneighbors(q_scaled, n_neighbors=min(top_k, len(self.df)))
        sims = 1.0 - distances[0]
        matches = self.df.iloc[indices[0]].copy()
        return PhysicsRAGResult(
            query=query_text,
            matches_df=matches,
            similarities=np.array(sims),
            explanation="Raw-Data RAG nearest-neighbor result",
            parsed_criteria=vals,
        )


class RuleBasedSQLRAG:
    """Baseline 2: Rule-based keyword and threshold filtering (SQL filter simulation)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        # Improvement 4: Sandy Coast Breaker Regime Harmonization (Wright & Short 1984, Battjes 1974)
        if "xi_iribarren" in self.df.columns and ("breaker_regime" not in self.df.columns or (self.df["breaker_regime"] == "Surging").sum() == 0):
            xi = self.df["xi_iribarren"].values
            self.df["breaker_regime"] = np.where(xi < 0.5, "Spilling", np.where(xi < 1.0, "Plunging", "Surging"))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        disable_spatiotemporal_filters: bool = False,
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        q_lower = query_text.lower()
        sub = self.df.copy()
        is_mode_a = disable_spatiotemporal_filters or (mode is not None and ("mode a" in str(mode).lower() or "phenomenological" in str(mode).lower()))

        # Rule 0: Site filtering (only in Mode B / full operational mode)
        site_col = "site_id" if "site_id" in sub.columns else ("site" if "site" in sub.columns else None)
        if not is_mode_a and site_col:
            if "narrabeen" in q_lower:
                sub = sub[sub[site_col] == "narrabeen_aus"]
            elif "duck" in q_lower:
                sub = sub[sub[site_col] == "duck_nc"]
            elif "torrey" in q_lower or "torrey pines" in q_lower:
                sub = sub[sub[site_col] == "torrey_pines_ca"]
            elif "truc vert" in q_lower or "biscay" in q_lower:
                sub = sub[sub[site_col] == "truc_vert_france"]

        # Rule 0b: Timestamp / epoch filtering (only in Mode B / full operational mode)
        if not is_mode_a and "timestamp" in sub.columns and len(sub) > 0:
            ts = pd.to_datetime(sub["timestamp"], utc=True)
            if "june 2016" in q_lower:
                sub = sub[(ts >= "2016-06-01") & (ts <= "2016-06-15")]
            elif "matthew" in q_lower or "october 2016" in q_lower:
                sub = sub[(ts >= "2016-10-01") & (ts <= "2016-10-15")]
            elif "sandy" in q_lower or "october 2012" in q_lower:
                sub = sub[(ts >= "2012-10-25") & (ts <= "2012-11-05")]
            elif "april 2015" in q_lower:
                sub = sub[(ts >= "2015-04-20") & (ts <= "2015-04-25")]
            elif any(w in q_lower for w in ["hercules", "christine", "winter 2013"]):
                sub = sub[(ts >= "2013-12-15") & (ts <= "2014-03-15")]
            elif "el nino" in q_lower or "el niño" in q_lower:
                sub = sub[(ts >= "2015-12-01") & (ts <= "2016-03-01")]
            elif "december 2015" in q_lower:
                sub = sub[(ts >= "2015-12-01") & (ts <= "2015-12-31")]
            elif "summer 2016" in q_lower:
                sub = sub[(ts >= "2016-06-01") & (ts <= "2016-08-31")]
            elif "summer" in q_lower and "berm" in q_lower:
                sub = sub[ts.dt.month == 7]
            elif "january 2016" in q_lower or "jonas" in q_lower:
                sub = sub[(ts >= "2016-01-15") & (ts <= "2016-01-31")]
            elif "august 2014" in q_lower:
                sub = sub[(ts >= "2014-08-10") & (ts <= "2014-08-25")]
            elif "february 2014" in q_lower or "ulla" in q_lower:
                sub = sub[(ts >= "2014-02-10") & (ts <= "2014-02-20")]
            elif "march 2014" in q_lower:
                sub = sub[(ts >= "2014-03-20") & (ts <= "2014-03-31")]
            elif "june 2012" in q_lower:
                sub = sub[(ts >= "2012-06-01") & (ts <= "2012-06-15")]
            elif "december 2013" in q_lower or "dirk" in q_lower:
                sub = sub[(ts >= "2013-12-20") & (ts <= "2013-12-31")]
            elif "february 2016" in q_lower:
                sub = sub[(ts >= "2016-01-25") & (ts <= "2016-02-15")]
            elif "october 2015" in q_lower or "joaquin" in q_lower:
                sub = sub[(ts >= "2015-10-01") & (ts <= "2015-10-10")]
            elif "november 2013" in q_lower:
                sub = sub[(ts >= "2013-11-05") & (ts <= "2013-11-25")]
            elif "january 2015" in q_lower:
                sub = sub[(ts >= "2015-01-05") & (ts <= "2015-01-25")]
            elif "july 2014" in q_lower:
                sub = sub[(ts >= "2014-07-01") & (ts <= "2014-07-31")]
            elif "september 2014" in q_lower:
                sub = sub[(ts >= "2014-09-01") & (ts <= "2014-09-30")]
            elif "april 2014" in q_lower:
                sub = sub[(ts >= "2014-04-05") & (ts <= "2014-04-25")]
            elif "summer 2015" in q_lower:
                sub = sub[(ts >= "2015-06-01") & (ts <= "2015-08-31")]

        # Rule 1: Breaker regime keywords
        if "spilling" in q_lower:
            sub = sub[sub["breaker_regime"] == "Spilling"]
        elif "plunging" in q_lower:
            sub = sub[sub["breaker_regime"] == "Plunging"]
        elif "surging" in q_lower:
            sub = sub[sub["breaker_regime"] == "Surging"]

        # Rule 2: Wave height / storm extremes
        if "over 6.0m" in q_lower or "over 6m" in q_lower:
            if "hs_m" in sub.columns:
                sub = sub[sub["hs_m"] >= 5.5]
        elif "over 4.5m" in q_lower:
            if "hs_m" in sub.columns:
                sub = sub[sub["hs_m"] >= 4.0]
        elif "over 3m" in q_lower or "over 3.0m" in q_lower or "wave height over" in q_lower or "high energy storm" in q_lower:
            if "hs_m" in sub.columns:
                sub = sub[sub["hs_m"] >= 2.7]
        elif "under 1.0m" in q_lower or "under 1.2m" in q_lower:
            if "hs_m" in sub.columns:
                sub = sub[sub["hs_m"] <= (1.0 if "1.0" in q_lower else 1.2)]

        # Rule 3: Water level / surge / runup extremes
        if "r2" in q_lower or "runup" in q_lower:
            if "runup_r2_m" in sub.columns:
                sub = sub[sub["runup_r2_m"] >= 1.8]
            elif "total_water_level_m" in sub.columns:
                sub = sub[sub["total_water_level_m"] >= 1.8]
        elif "twl over 2.0m" in q_lower or "over 2.0m" in q_lower or "total water level" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[sub["total_water_level_m"] >= 1.6]
            elif "surge_m" in sub.columns:
                sub = sub[sub["surge_m"] >= 0.04]
        elif "surge above" in q_lower or "surge over" in q_lower:
            if "surge_m" in sub.columns:
                sub = sub[sub["surge_m"] >= 0.04]

        # Rule 4: Compound wave-surge
        if "compound" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["hs_m"] >= 2.5) & (sub["total_water_level_m"] >= 1.6)]
            else:
                sub = sub[(sub["hs_m"] >= 2.5) & (sub["surge_m"] >= 0.03)]

        # Rule 5: Gale wind / power flux / wave dissipation
        if "power flux" in q_lower or "60 kw" in q_lower or "dune scarping" in q_lower or "wave dissipation" in q_lower:
            if "p_flux_Wm" in sub.columns:
                sub = sub[sub["p_flux_Wm"] >= 50000.0]
        elif "gale" in q_lower or "wind forcing" in q_lower or "wind events" in q_lower:
            if "wind_speed_ms" in sub.columns:
                sub = sub[sub["wind_speed_ms"] >= 7.0]
            elif "wind_normal_ms" in sub.columns:
                sub = sub[sub["wind_normal_ms"] >= 6.0]

        # Rule 6: Berm accretion / recovery
        if "accretion" in q_lower or "recovery" in q_lower or "swash recovery" in q_lower or "healing" in q_lower or "welding" in q_lower:
            if "upsilon_diseq" in sub.columns:
                sub = sub[sub["upsilon_diseq"] < 0.0]
            elif "accretion_potential" in sub.columns:
                sub = sub[sub["accretion_potential"] >= 2.0]
            elif "delta_x_m" in sub.columns:
                sub = sub[sub["delta_x_m"] >= 0.0]

        # Rule 7: Summer swell
        if "summer swell" in q_lower:
            sub = sub[(sub["tp_s"] >= 11.0) & (sub["hs_m"] < 1.3)]

        # Rule 8: Tide / Macrotidal
        if "macrotidal" in q_lower or "tidal smearing" in q_lower:
            if "psi_tide" in sub.columns:
                sub = sub[sub["psi_tide"] < 0.45]
            elif site_col and site_col in sub.columns:
                sub = sub[sub[site_col] == "truc_vert_france"]

        # Rule 9: Sheltered / protected embayments
        if "sheltered" in q_lower or "protected embayments" in q_lower:
            if site_col and site_col in sub.columns:
                sub = sub[sub[site_col].isin(["torrey_pines_ca", "narrabeen_aus"]) & (sub["hs_m"] < 1.1)]

        # Rule 10: Oblique waves / headland / nor'easter
        if "oblique" in q_lower or "headland" in q_lower or "nor'easter" in q_lower:
            if "wave_dir_deg" in sub.columns:
                sub = sub[(sub["wave_dir_deg"] >= 45.0) & (sub["wave_dir_deg"] <= 150.0)]
            elif "phi_longshore" in sub.columns:
                sub = sub[sub["phi_longshore"].abs() > 0.2]

        # Rule 11: Sandbar dissipation / bar-breach
        if "sandbar dissipation" in q_lower or "buffering" in q_lower or "bar-breach" in q_lower:
            if "dean_number" in sub.columns:
                sub = sub[(sub["dean_number"] >= 4.0) & (sub["hs_m"] >= 2.2)]
            elif "delta_p_dissip" in sub.columns:
                sub = sub[sub["delta_p_dissip"] > 0.35]

        # Rule 12: Infragravity swash
        if "infragravity" in q_lower:
            sub = sub[(sub["tp_s"] >= 13.0) & (sub["hs_m"] >= 1.5)]

        # Rule 13: High tide gating / collision
        if "collision" in q_lower or "reflective collision" in q_lower:
            if "total_water_level_m" in sub.columns and "tan_beta" in sub.columns:
                sub = sub[(sub["total_water_level_m"] >= 1.5) & (sub["tan_beta"] >= 0.06)]
        elif "tide-gated" in q_lower or "high tide" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["total_water_level_m"] >= 1.5) & (sub["hs_m"] >= 2.0)]

        # Rule 14: Steepness below
        if "steepness below" in q_lower or "low-steepness" in q_lower:
            if "hs_m" in sub.columns and "tp_s" in sub.columns:
                steep = sub["hs_m"] / (1.56 * (sub["tp_s"] ** 2) + 1e-4)
                sub = sub[steep < 0.020]

        # Fallback if no matches found: rank by most relevant variable while respecting site filters
        if len(sub) == 0:
            base_df = self.df.copy()
            if not is_mode_a and site_col:
                if "narrabeen" in q_lower:
                    base_df = base_df[base_df[site_col] == "narrabeen_aus"]
                elif "duck" in q_lower:
                    base_df = base_df[base_df[site_col] == "duck_nc"]
                elif "torrey" in q_lower or "torrey pines" in q_lower:
                    base_df = base_df[base_df[site_col] == "torrey_pines_ca"]
                elif "truc vert" in q_lower or "biscay" in q_lower:
                    base_df = base_df[base_df[site_col] == "truc_vert_france"]
            if len(base_df) == 0:
                base_df = self.df.copy()

            if "storm" in q_lower or "high" in q_lower or "wave" in q_lower:
                sub = base_df.sort_values("hs_m", ascending=False) if "hs_m" in base_df.columns else base_df
            elif "accretion" in q_lower or "recovery" in q_lower:
                if "delta_x_m" in base_df.columns:
                    sub = base_df.sort_values("delta_x_m", ascending=False)
                elif "accretion_potential" in base_df.columns:
                    sub = base_df.sort_values("accretion_potential", ascending=False)
                else:
                    sub = base_df
            else:
                sub = base_df
        else:
            # Rank matching rows by storm severity or accretion
            if "storm" in q_lower or "high" in q_lower or "wave" in q_lower:
                if "hs_m" in sub.columns:
                    sub = sub.sort_values("hs_m", ascending=False)
            elif "accretion" in q_lower or "recovery" in q_lower:
                if "delta_x_m" in sub.columns:
                    sub = sub.sort_values("delta_x_m", ascending=False)
                elif "accretion_potential" in sub.columns:
                    sub = sub.sort_values("accretion_potential", ascending=False)

        matches = sub.head(top_k).copy()
        sims = [1.0 - (i * 0.05) for i in range(len(matches))]
        return PhysicsRAGResult(
            query=query_text,
            matches_df=matches,
            similarities=np.array(sims),
            explanation="Rule-based SQL filter result",
            parsed_criteria={},
        )


def evaluate_retrieval_query(
    rag: Any,
    query_spec_or_text: Union[str, Dict[str, Any]],
    condition_fn: Optional[Any] = None,
    target_regime: Optional[str] = None,
    k: int = 5,
    use_llm_parser: bool = False,
    allow_synthetic_probes: bool = False,
    precomputed_rag_res: Optional[Any] = None,
    mode: Optional[str] = None,
    disable_spatiotemporal_filters: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate Precision@k, Regime Purity@k, Recall@k, Reciprocal Rank (MRR), and NDCG@k for a query.

    Parameters
    ----------
    rag : PhysicsRAG
        Instantiated PhysicsRAG engine or baseline.
    query_spec_or_text : str or dict
        Query string or query specification dictionary.
    condition_fn : callable, optional
        Function condition_fn(row, df) -> bool evaluating if a dataset row meets ground truth.
    target_regime : str, optional
        Expected ground-truth breaker regime ('Spilling', 'Plunging', 'Surging').
    k : int
        Retrieval rank cutoff (default 5).
    use_llm_parser : bool
        Whether to use Gemini structured output parser.
    allow_synthetic_probes : bool
        Whether to permit target synthetic probe vectors.
    precomputed_rag_res : PhysicsRAGResult, optional
        Cached result from a prior engine query to avoid re-computation.
    mode : str, optional
        Evaluation mode ('Mode A: Pure Phenomenological' or 'Mode B: Full Operational').
    disable_spatiotemporal_filters : bool, default False
        Whether to force disable calendar date and site filtering (for Mode A).

    Returns
    -------
    metrics : dict
        Calculated retrieval evaluation metrics.
    """
    if isinstance(query_spec_or_text, dict):
        query_str = query_spec_or_text["query"]
        cond = condition_fn or query_spec_or_text.get("condition_fn")
        regime = target_regime or query_spec_or_text.get("target_regime")
        category = query_spec_or_text.get("category", "General")
        q_mode = mode or query_spec_or_text.get("mode")
    else:
        query_str = str(query_spec_or_text)
        cond = condition_fn
        regime = target_regime
        category = "Custom"
        q_mode = mode

    is_mode_a = disable_spatiotemporal_filters or (q_mode is not None and ("mode a" in str(q_mode).lower() or "phenomenological" in str(q_mode).lower()))

    # Default condition if target_regime provided
    if cond is None and regime:
        cond = lambda row, df: str(row.get("breaker_regime", "")) == regime

    # Total ground-truth support in dataset
    df = rag.df
    total_support = len(df)
    mask_fn = query_spec_or_text.get("mask_fn") if isinstance(query_spec_or_text, dict) else None
    if mask_fn is not None and hasattr(df, "shape"):
        try:
            total_support = int(mask_fn(df).sum())
        except Exception:
            total_support = len(df)
    elif cond is not None and hasattr(df, "shape"):
        try:
            if len(df) <= 500:
                total_support = int(sum(bool(cond(df.iloc[i], df)) for i in range(len(df))))
            else:
                # Sample 500 points to approximate support accurately and instantaneously
                sample_idx = np.linspace(0, len(df) - 1, 500, dtype=int)
                sample_cnt = sum(bool(cond(df.iloc[idx], df)) for idx in sample_idx)
                total_support = int((sample_cnt / 500.0) * len(df))
        except Exception:
            total_support = len(df)

    # Execute RAG query (or slice precomputed result for 4x benchmark speedup)
    if precomputed_rag_res is not None:
        rag_res = precomputed_rag_res
        matches_df = rag_res.matches_df.head(k).copy()
    else:
        target_synth = (query_spec_or_text.get("target_synthetic_values") if isinstance(query_spec_or_text, dict) and allow_synthetic_probes else None)
        spec_override = query_spec_or_text.get("spec") if isinstance(query_spec_or_text, dict) else None
        rag_res = rag.query(
            query_text=query_str,
            top_k=k,
            use_llm_parser=use_llm_parser,
            use_llm_synthesis=False,
            spec=spec_override,
            target_synthetic_values=target_synth,
            mode=q_mode,
            disable_spatiotemporal_filters=is_mode_a,
            **kwargs,
        )
        matches_df = rag_res.matches_df
    n_retrieved = len(matches_df)
    eff_k = max(1, n_retrieved)

    # Calculate binary relevance mask and graded relevance scores for retrieved instances
    graded_fn = query_spec_or_text.get("graded_relevance_fn") if isinstance(query_spec_or_text, dict) else None
    relevance_flags: List[bool] = []
    regime_flags: List[bool] = []
    graded_scores: List[int] = []

    for i in range(n_retrieved):
        row = matches_df.iloc[i]
        is_rel = bool(cond(row, df)) if cond is not None else True
        relevance_flags.append(is_rel)

        if graded_fn is not None:
            try:
                g_score = int(graded_fn(row, df))
            except Exception:
                g_score = 2 if is_rel else 0
        else:
            g_score = 2 if is_rel else 0
        graded_scores.append(g_score)

        if regime and "breaker_regime" in row:
            regime_flags.append(str(row["breaker_regime"]) == regime)
        elif regime:
            regime_flags.append(False)

    n_correct = sum(relevance_flags)
    precision_at_k = float(n_correct / eff_k) if eff_k > 0 else 0.0

    # Regime Purity@k
    if regime:
        regime_purity = float(sum(regime_flags) / eff_k) if eff_k > 0 else 0.0
    else:
        regime_purity = precision_at_k

    # True Recall@k: |Retrieved Ground Truth Events| / |Total Known Ground Truth Events in Window|
    if total_support > 0:
        recall_at_k = float(n_correct / total_support)
        bounded_recall_at_k = float(n_correct / min(eff_k, total_support))
    else:
        recall_at_k = 0.0
        bounded_recall_at_k = 0.0

    # Reciprocal Rank (RR)
    reciprocal_rank = 0.0
    for idx, is_rel in enumerate(relevance_flags):
        if is_rel:
            reciprocal_rank = 1.0 / float(idx + 1)
            break

    # Graded Normalized Discounted Cumulative Gain (Graded NDCG@k)
    k_eff = min(k, n_retrieved)
    if k_eff > 0:
        # Standard graded DCG formulation: (2^rel_i - 1) / log2(i + 2)
        dcg = sum((2.0 ** g - 1.0) / np.log2(idx + 2) for idx, g in enumerate(graded_scores[:k_eff]))

        # Ideal DCG: top-k positions populated with ideal scores (Grade 3 if peak work exists, else Grade 2)
        has_peak_potential = (isinstance(query_spec_or_text, dict) and "peak_work_cond" in query_spec_or_text) or any(g >= 3 for g in graded_scores)
        max_possible_grade = 3 if has_peak_potential else (max(graded_scores) if (len(graded_scores) and max(graded_scores) >= 2) else 2)
        ideal_top_count = min(k_eff, max(1, total_support))
        ideal_grades = [max_possible_grade] * ideal_top_count + [0] * (k_eff - ideal_top_count)
        idcg = sum((2.0 ** ig - 1.0) / np.log2(idx + 2) for idx, ig in enumerate(ideal_grades))
        ndcg_at_k = float(dcg / idcg) if idcg > 0 else 0.0
    else:
        ndcg_at_k = 0.0

    mean_sim = float(np.mean(rag_res.similarities)) if len(rag_res.similarities) > 0 else 0.0

    return {
        "query": query_str,
        "category": category,
        "mode": q_mode or ("Mode A: Pure Phenomenological" if is_mode_a else "Mode B: Full Operational"),
        "target_regime": regime,
        "k": k,
        "precision_at_k": precision_at_k,
        "regime_purity": regime_purity,
        "recall_at_k": recall_at_k,
        "true_recall_at_k": recall_at_k,
        "bounded_recall_at_k": bounded_recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": ndcg_at_k,
        "graded_ndcg_at_k": ndcg_at_k,
        "graded_scores": graded_scores[:k_eff],
        "mean_similarity": mean_sim,
        "n_retrieved": n_retrieved,
        "n_correct": n_correct,
        "total_dataset_support": total_support,
        "retrieved_df": matches_df,
    }


def evaluate_rag_benchmark_suite(
    rag: Optional[PhysicsRAG] = None,
    benchmark_queries: Optional[List[Dict[str, Any]]] = None,
    k_list: Optional[List[int]] = None,
    use_llm_parser: bool = False,
    rag_engine: Optional[PhysicsRAG] = None,
    k_values: Optional[List[int]] = None,
    verbose: bool = False,
    allow_synthetic_probes: bool = False,
    benchmark_mode: Optional[str] = None,
    n_queries: Optional[int] = None,
    max_queries: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute multi-k benchmark evaluation suite across diverse coastal query categories and decoupled modes.

    Parameters
    ----------
    rag : PhysicsRAG, optional
        Instantiated PhysicsRAG engine.
    benchmark_queries : list[dict], optional
        Custom list of query dicts (defaults to DEFAULT_PHYSICS_BENCHMARK_QUERIES).
    k_list : list[int], optional
        List of k rank cutoffs (e.g. [1, 3, 5, 10]).
    use_llm_parser : bool
        Whether to use Gemini structured output parser.
    rag_engine : PhysicsRAG, optional
        Alias for rag.
    k_values : list[int], optional
        Alias for k_list.
    verbose : bool
        Whether to print verbose progress.
    allow_synthetic_probes : bool
        Whether to permit target synthetic probe vectors.
    benchmark_mode : str, optional
        Evaluation mode filter: 'all' (default), 'mode_a' (Pure Phenomenological), or 'mode_b' (Full Operational).

    Returns
    -------
    df_results : pd.DataFrame
        Detailed per-query evaluation table with Mode stratification.
    summary_metrics : dict
        Macro-averaged benchmark summary metrics (Mean P@1, Mean P@5, NDCG, MRR, Regime Purity),
        both overall and partitioned by Mode A vs. Mode B.
    """
    engine = rag_engine if rag_engine is not None else rag
    if engine is None:
        raise ValueError("PhysicsRAG instance must be provided via 'rag' or 'rag_engine'.")
        
    cutoff_ranks = k_values if k_values is not None else (k_list or [1, 3, 5, 10])

    if benchmark_queries is not None:
        raw_queries = benchmark_queries
    elif benchmark_mode is not None and "all" in str(benchmark_mode).lower():
        if "100" in str(benchmark_mode).lower() or (n_queries is not None and n_queries > 60) or (max_queries is not None and max_queries > 60):
            raw_queries = ALL_PHYSICS_BENCHMARK_QUERIES_100
        else:
            raw_queries = ALL_PHYSICS_BENCHMARK_QUERIES_60
    elif benchmark_mode is not None and ("100" in str(benchmark_mode).lower()):
        raw_queries = ALL_PHYSICS_BENCHMARK_QUERIES_100
    elif benchmark_mode is not None and ("ambiguity" in str(benchmark_mode).lower()):
        raw_queries = BENCHMARK_QUERIES_DELIBERATE_AMBIGUITY
    elif benchmark_mode is not None and ("patholog" in str(benchmark_mode).lower()):
        raw_queries = BENCHMARK_QUERIES_PHYSICAL_PATHOLOGIES
    elif benchmark_mode is not None and ("mode_b" in str(benchmark_mode).lower() or "operational" in str(benchmark_mode).lower()):
        raw_queries = ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B
    else:
        # Default: Mode A (Pure Phenomenological Search across all 175k global records)
        raw_queries = DEFAULT_PHYSICS_BENCHMARK_QUERIES

    # Filter queries if benchmark_mode specified (defaulting to Mode A when unspecified and no explicit benchmark_queries)
    if benchmark_queries is not None:
        queries = list(raw_queries)
    elif benchmark_mode is not None:
        bm = str(benchmark_mode).lower()
        if "mode_a" in bm or "mode a" in bm or "phenomenological" in bm:
            queries = [q for q in raw_queries if "mode a" in str(q.get("mode", "")).lower() or "phenomenological" in str(q.get("mode", "")).lower()]
        elif "mode_b" in bm or "mode b" in bm or "operational" in bm:
            queries = [q for q in raw_queries if "mode b" in str(q.get("mode", "")).lower() or "operational" in str(q.get("mode", "")).lower()]
        elif "ambiguity" in bm or "mode_c" in bm or "mode c" in bm:
            queries = [q for q in raw_queries if "mode c" in str(q.get("mode", "")).lower() or "ambiguity" in str(q.get("mode", "")).lower()]
        elif "patholog" in bm or "mode_d" in bm or "mode d" in bm:
            queries = [q for q in raw_queries if "mode d" in str(q.get("mode", "")).lower() or "patholog" in str(q.get("mode", "")).lower()]
        elif "all" in bm or "100" in bm:
            queries = list(raw_queries)
        else:
            queries = [q for q in raw_queries if "mode a" in str(q.get("mode", "")).lower() or "phenomenological" in str(q.get("mode", "")).lower()]
    else:
        queries = list(raw_queries)

    # Configurable sample size limit N (supports 1 <= N <= len(DEFAULT_PHYSICS_BENCHMARK_QUERIES))
    limit_n = max_queries if max_queries is not None else n_queries
    if limit_n is not None and limit_n > 0:
        if benchmark_mode is not None and "all" not in str(benchmark_mode).lower():
            queries = queries[:limit_n]
        else:
            # Balanced sampling across Mode A and Mode B when N <= 60
            q_mode_a = [q for q in queries if "mode a" in str(q.get("mode", "")).lower() or "phenomenological" in str(q.get("mode", "")).lower()]
            q_mode_b = [q for q in queries if "mode b" in str(q.get("mode", "")).lower() or "operational" in str(q.get("mode", "")).lower()]
            if limit_n <= 60 and q_mode_a and q_mode_b:
                n_a = (limit_n + 1) // 2
                n_b = limit_n // 2
                queries = q_mode_a[:n_a] + q_mode_b[:n_b]
            else:
                queries = queries[:limit_n]

    rows: List[Dict[str, Any]] = []

    p_at_k_accum: Dict[int, List[float]] = {k_val: [] for k_val in cutoff_ranks}
    ndcg_accum: Dict[int, List[float]] = {k_val: [] for k_val in cutoff_ranks}
    purity_accum: List[float] = []
    mrr_accum: List[float] = []
    recall_accum: List[float] = []
    bounded_recall_accum: List[float] = []
    sim_accum: List[float] = []

    # Mode-partitioned accumulators
    mode_accum: Dict[str, Dict[str, Any]] = {
        "mode_a": {"p_at_k": {k_val: [] for k_val in cutoff_ranks}, "ndcg": {k_val: [] for k_val in cutoff_ranks}, "purity": [], "mrr": [], "recall": [], "bounded_recall": []},
        "mode_b": {"p_at_k": {k_val: [] for k_val in cutoff_ranks}, "ndcg": {k_val: [] for k_val in cutoff_ranks}, "purity": [], "mrr": [], "recall": [], "bounded_recall": []},
    }

    for q_spec in queries:
        q_mode = q_spec.get("mode", "Mode A: Pure Phenomenological") if isinstance(q_spec, dict) else "Mode A: Pure Phenomenological"
        is_mode_a = "mode a" in str(q_mode).lower() or "phenomenological" in str(q_mode).lower()
        mode_key = "mode_a" if is_mode_a else "mode_b"

        if verbose:
            print(f"Evaluating query: {q_spec['query'] if isinstance(q_spec, dict) else q_spec} [{q_mode}]")
            
        row_entry: Dict[str, Any] = {
            "Query": q_spec["query"] if isinstance(q_spec, dict) else str(q_spec),
            "Category": q_spec.get("category", "General") if isinstance(q_spec, dict) else "General",
            "Mode": q_mode,
            "Target Regime": (q_spec.get("target_regime") if isinstance(q_spec, dict) else None) or "N/A",
        }

        # Query once per question at max_k to eliminate redundant LLM calls across cutoff radii
        query_str = q_spec["query"] if isinstance(q_spec, dict) else str(q_spec)
        target_synth = (q_spec.get("target_synthetic_values") if isinstance(q_spec, dict) and allow_synthetic_probes else None)
        spec_override = q_spec.get("spec") if isinstance(q_spec, dict) else None
        max_k_query = max(cutoff_ranks + [5])
        single_rag_res = engine.query(
            query_text=query_str,
            top_k=max_k_query,
            use_llm_parser=use_llm_parser,
            use_llm_synthesis=False,
            spec=spec_override,
            target_synthetic_values=target_synth,
            mode=q_mode,
            disable_spatiotemporal_filters=is_mode_a,
            **kwargs,
        )

        # Evaluate across multiple k values using the single retrieved result
        for k_val in cutoff_ranks:
            q_res = evaluate_retrieval_query(
                rag=engine,
                query_spec_or_text=q_spec,
                k=k_val,
                use_llm_parser=use_llm_parser,
                allow_synthetic_probes=allow_synthetic_probes,
                precomputed_rag_res=single_rag_res,
                mode=q_mode,
                disable_spatiotemporal_filters=is_mode_a,
            )
            p_val = q_res["precision_at_k"]
            ndcg_val = q_res["ndcg_at_k"]
            row_entry[f"P@{k_val}"] = p_val
            row_entry[f"NDCG@{k_val}"] = ndcg_val
            row_entry[f"Recall@{k_val}"] = q_res["recall_at_k"]
            p_at_k_accum[k_val].append(p_val)
            ndcg_accum[k_val].append(ndcg_val)
            mode_accum[mode_key]["p_at_k"][k_val].append(p_val)
            mode_accum[mode_key]["ndcg"][k_val].append(ndcg_val)

            if k_val == 5:
                row_entry["Regime Purity@5"] = q_res["regime_purity"]
                row_entry["MRR"] = q_res["reciprocal_rank"]
                row_entry["NDCG@5"] = ndcg_val
                row_entry["Recall@5"] = q_res["recall_at_k"]
                row_entry["Bounded Recall@5"] = q_res["bounded_recall_at_k"]
                row_entry["Mean Sim@5"] = q_res["mean_similarity"]
                row_entry["Dataset Support"] = q_res["total_dataset_support"]

                purity_accum.append(q_res["regime_purity"])
                mrr_accum.append(q_res["reciprocal_rank"])
                recall_accum.append(q_res["recall_at_k"])
                bounded_recall_accum.append(q_res["bounded_recall_at_k"])
                sim_accum.append(q_res["mean_similarity"])

                mode_accum[mode_key]["purity"].append(q_res["regime_purity"])
                mode_accum[mode_key]["mrr"].append(q_res["reciprocal_rank"])
                mode_accum[mode_key]["recall"].append(q_res["recall_at_k"])
                mode_accum[mode_key]["bounded_recall"].append(q_res["bounded_recall_at_k"])

        rows.append(row_entry)

    df_results = pd.DataFrame(rows)

    summary_metrics: Dict[str, Any] = {
        f"Mean Precision@{k_val}": float(np.mean(p_at_k_accum[k_val])) if p_at_k_accum.get(k_val) else 0.0
        for k_val in cutoff_ranks
    }
    summary_metrics.update({
        f"Mean NDCG@{k_val}": float(np.mean(ndcg_accum[k_val])) if ndcg_accum.get(k_val) else 0.0
        for k_val in cutoff_ranks
    })

    for k_val in [1, 3, 5, 10]:
        if f"Mean Precision@{k_val}" not in summary_metrics:
            summary_metrics[f"Mean Precision@{k_val}"] = float(np.mean(p_at_k_accum.get(k_val, [0.0])))
        if f"Mean NDCG@{k_val}" not in summary_metrics:
            summary_metrics[f"Mean NDCG@{k_val}"] = float(np.mean(ndcg_accum.get(k_val, [0.0])))

    summary_metrics.update({
        "Mean Regime Purity@5": float(np.mean(purity_accum)) if purity_accum else 0.0,
        "Mean Reciprocal Rank (MRR)": float(np.mean(mrr_accum)) if mrr_accum else 0.0,
        "Mean Recall@5": float(np.mean(recall_accum)) if recall_accum else 0.0,
        "Mean Bounded Recall@5": float(np.mean(bounded_recall_accum)) if bounded_recall_accum else 0.0,
        "Mean Cosine Similarity@5": float(np.mean(sim_accum)) if sim_accum else 0.0,
    })

    # Mode A partitioned summary metrics
    summary_metrics["mode_a_mean_p_at_5"] = float(np.mean(mode_accum["mode_a"]["p_at_k"][5])) if mode_accum["mode_a"]["p_at_k"].get(5) else 0.0
    summary_metrics["mode_a_mean_ndcg_at_5"] = float(np.mean(mode_accum["mode_a"]["ndcg"][5])) if mode_accum["mode_a"]["ndcg"].get(5) else 0.0
    summary_metrics["mode_a_mean_mrr"] = float(np.mean(mode_accum["mode_a"]["mrr"])) if mode_accum["mode_a"]["mrr"] else 0.0
    summary_metrics["mode_a_regime_purity_at_5"] = float(np.mean(mode_accum["mode_a"]["purity"])) if mode_accum["mode_a"]["purity"] else 0.0

    # Mode B partitioned summary metrics
    summary_metrics["mode_b_mean_p_at_5"] = float(np.mean(mode_accum["mode_b"]["p_at_k"][5])) if mode_accum["mode_b"]["p_at_k"].get(5) else 0.0
    summary_metrics["mode_b_mean_ndcg_at_5"] = float(np.mean(mode_accum["mode_b"]["ndcg"][5])) if mode_accum["mode_b"]["ndcg"].get(5) else 0.0
    summary_metrics["mode_b_mean_mrr"] = float(np.mean(mode_accum["mode_b"]["mrr"])) if mode_accum["mode_b"]["mrr"] else 0.0
    summary_metrics["mode_b_regime_purity_at_5"] = float(np.mean(mode_accum["mode_b"]["purity"])) if mode_accum["mode_b"]["purity"] else 0.0

    # Add snake_case aliases for flexible programmatic access
    for k_val in set(cutoff_ranks + [1, 3, 5, 10]):
        if f"Mean Precision@{k_val}" in summary_metrics:
            summary_metrics[f"mean_p_at_{k_val}"] = summary_metrics[f"Mean Precision@{k_val}"]
        if f"Mean NDCG@{k_val}" in summary_metrics:
            summary_metrics[f"mean_ndcg_at_{k_val}"] = summary_metrics[f"Mean NDCG@{k_val}"]
    summary_metrics["mean_mrr"] = summary_metrics["Mean Reciprocal Rank (MRR)"]
    summary_metrics["regime_purity_at_5"] = summary_metrics["Mean Regime Purity@5"]
    summary_metrics["mean_recall_at_5"] = summary_metrics["Mean Recall@5"]
    summary_metrics["mean_bounded_recall_at_5"] = summary_metrics["Mean Bounded Recall@5"]
    return df_results, summary_metrics


def generate_decoupled_mode_summary(
    benchmark_results: Dict[str, Any],
) -> pd.DataFrame:
    """Generate a decoupled Mode A (Pure Phenomenological) vs. Mode B (Full Operational) vs. Overall summary table.

    Parameters
    ----------
    benchmark_results : dict
        Mapping from architecture name (e.g. 'Baseline 1: Raw-Data LLM-RAG', 'Baseline 2: Rule-Based SQL', 'Baseline 3: Physics-RAG')
        to either:
          - (df_results, summary_dict) tuple
          - df_results DataFrame
          - summary_dict dict

    Returns
    -------
    pd.DataFrame
        Publication-ready table comparing architectures across Mode A, Mode B, and Overall.
    """
    summary_rows = []
    for arch_name, res in benchmark_results.items():
        if isinstance(res, tuple):
            df_res = res[0]
        elif isinstance(res, pd.DataFrame):
            df_res = res
        else:
            continue

        if "Mode" not in df_res.columns:
            continue

        modes_to_eval = [
            ("Mode A: Pure Phenomenological", df_res[df_res["Mode"].astype(str).str.contains("Mode A|Phenomenological", case=False, na=False)]),
            ("Mode B: Full Operational", df_res[df_res["Mode"].astype(str).str.contains("Mode B|Operational", case=False, na=False)]),
            ("Overall Benchmark", df_res),
        ]

        for mode_label, sub in modes_to_eval:
            n_q = len(sub)
            if n_q == 0:
                continue
            summary_rows.append({
                "Architecture": arch_name,
                "Evaluation Mode": mode_label,
                "N Queries": n_q,
                "Mean P@1": float(sub["P@1"].mean()) if "P@1" in sub.columns else 0.0,
                "Mean P@5": float(sub["P@5"].mean()) if "P@5" in sub.columns else 0.0,
                "Mean P@10": float(sub["P@10"].mean()) if "P@10" in sub.columns else 0.0,
                "Mean NDCG@5": float(sub["NDCG@5"].mean()) if "NDCG@5" in sub.columns else 0.0,
                "Regime Purity@5": float(sub["Regime Purity@5"].mean()) if "Regime Purity@5" in sub.columns else 0.0,
                "MRR": float(sub["MRR"].mean()) if "MRR" in sub.columns else 0.0,
                "Mean Recall@5": float(sub["Recall@5"].mean()) if "Recall@5" in sub.columns else 0.0,
                "Mean Bounded Recall@5": float(sub["Bounded Recall@5"].mean()) if "Bounded Recall@5" in sub.columns else 0.0,
            })

    return pd.DataFrame(summary_rows)


def evaluate_peak_impact_disequilibrium_benchmark(
    rag_engines: Dict[str, Any],
    df: pd.DataFrame,
    benchmark_queries: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    use_llm_parser: bool = False,
    query_categories: Optional[List[str]] = None,
    parser_mode: str = "pure_gemini",
    aligned_surveys: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Peak Impact & Disequilibrium Ranking across RAG architectures re-anchored to Geomorphic Ground Truth.

    Benchmarks how effectively each retrieval system prioritizes true morphological beach cut,
    cumulative excess wave power flux, and total water level collision regimes rather than
    instantaneous wave height peaks.

    Ground-Truth Geomorphic Criteria:
    1. Total Water Level Collision Regime: TWL = Tide + Surge + R_2% >= z_dune_toe
       (Sallenger 2000 Storm Impact Scale: Swash -> Collision -> Overwash).
    2. Cumulative Excess Wave Power Flux: E_excess = Integral max(0, Upsilon(t) - Upsilon_crit(t)) dt.
    3. Surveyed Cross-Shore Shoreline Displacement: Authentic in-situ RTK-GPS/LIDAR profile retreat (Delta x < 0).

    Parameters
    ----------
    rag_engines : dict
        Mapping of architecture name to initialized RAG engine instance
        (e.g., {"Flat SQL": sql_rag, "Hybrid Cascade": hybrid_rag, ...}).
    df : pd.DataFrame
        Global multi-site dataset containing sensory and feature columns.
    benchmark_queries : list of dict, optional
        Queries to benchmark. Defaults to historical storm event queries from
        DEFAULT_PHYSICS_BENCHMARK_QUERIES (Categories 1 and 7).
    k : int, default 5
        Retrieval cutoff radius.
    use_llm_parser : bool, default False
        Whether to invoke live LLM query parser.
    query_categories : list of str, optional
        Categories to filter from benchmark queries. Defaults to Categories 1 and 7.
    parser_mode : str, default 'pure_gemini'
        Parser mode ('pure_gemini', 'pure_regex', or 'hybrid').
    aligned_surveys : dict of str -> pd.DataFrame, optional
        Dictionary mapping site keys to aligned survey epoch DataFrames. If None,
        automatically discovered from rag_engines or loaded from in-situ survey files.

    Returns
    -------
    df_query_details : pd.DataFrame
        Per-query peak impact metrics across all architectures.
    df_summary : pd.DataFrame
        Macro-averaged architecture comparison table.
    """
    try:
        from src.physics import COASTAL_SITE_PROFILES
    except ImportError:
        from physics import COASTAL_SITE_PROFILES  # type: ignore

    # 0. Automatically discover aligned surveys if not provided
    if aligned_surveys is None:
        for eng in rag_engines.values():
            if getattr(eng, "survey_aligned_epochs", None):
                aligned_surveys = eng.survey_aligned_epochs
                break
        if aligned_surveys is None:
            try:
                from src.data import load_site_field_surveys
                surv_dict = {}
                for s_name in ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]:
                    try:
                        s_df = load_site_field_surveys(s_name)
                        if len(s_df) > 0:
                            surv_dict[s_name] = s_df
                    except Exception:
                        pass
                if surv_dict:
                    aligned_surveys = surv_dict
            except Exception:
                aligned_surveys = None

    if benchmark_queries is None:
        raw_queries = ALL_PHYSICS_BENCHMARK_QUERIES_60
    else:
        raw_queries = benchmark_queries

    target_cats = query_categories or ["1. Historical Storm Events", "7. Extended Historical Storms"]
    storm_queries = [
        q for q in raw_queries
        if any(cat in q.get("category", "") for cat in target_cats)
        or any(w in q.get("query", "").lower() for w in ["storm", "peak", "scarping", "overwash", "cut"])
    ]

    records: List[Dict[str, Any]] = []

    for q in storm_queries:
        mask = q.get("mask_fn")
        sub_gt = df[mask(df)] if mask is not None else pd.DataFrame()
        if len(sub_gt) == 0:
            continue

        # 1. Identify Site Archetype and Geomorphic Parameters
        site_key = "duck_nc"
        site_col = "site_id" if "site_id" in sub_gt.columns else ("site" if "site" in sub_gt.columns else None)
        if site_col and len(sub_gt) > 0:
            raw_s = str(sub_gt[site_col].iloc[0]).lower()
            for k_site in COASTAL_SITE_PROFILES:
                if k_site in raw_s:
                    site_key = k_site
                    break
        else:
            q_txt = q.get("query", "").lower()
            for k_site in COASTAL_SITE_PROFILES:
                if k_site.split("_")[0] in q_txt or COASTAL_SITE_PROFILES[k_site]["name"].split(",")[0].lower() in q_txt:
                    site_key = k_site
                    break

        prof = COASTAL_SITE_PROFILES.get(site_key, {})
        z_crest = float(prof.get("z_crest_m", prof.get("berm_crest_m", 2.5)))
        z_dune_toe = float(prof.get("z_dune_toe_m", z_crest * 0.70))

        # 2. Extract Hydrodynamic & Geomorphic Time Series
        if "total_water_level_m" in sub_gt.columns:
            twl_gt = sub_gt["total_water_level_m"].values.astype(float)
        elif "surge_m" in sub_gt.columns and "runup_r2_m" in sub_gt.columns:
            twl_gt = (sub_gt["surge_m"] + sub_gt["runup_r2_m"]).values.astype(float)
        else:
            twl_gt = sub_gt.get("surge_m", pd.Series(0.0, index=sub_gt.index)).values.astype(float)

        u_gt = sub_gt["upsilon"].values.astype(float) if "upsilon" in sub_gt.columns else np.zeros(len(sub_gt))
        if "upsilon_crit" in sub_gt.columns:
            u_crit_gt = sub_gt["upsilon_crit"].values.astype(float)
        else:
            u_crit_gt = np.full_like(u_gt, 1500.0)

        if "upsilon_excess" in sub_gt.columns:
            u_excess_gt = sub_gt["upsilon_excess"].values.astype(float)
        else:
            u_excess_gt = np.maximum(0.0, u_gt - u_crit_gt)

        # 3. Cumulative Excess Wave Power Flux (Integral of max(0, Upsilon - Upsilon_crit) dt)
        dt_sec = 3600.0
        cum_excess_work_MJ_m = float(np.sum(u_excess_gt) * dt_sec / 1e6)

        # 4. Total Water Level Collision Factor & True Geomorphic Peak Determination
        # Waves attack the subaerial dune/berm face when TWL exceeds dune toe (Sallenger 2000 collision regime)
        collision_ratio = np.maximum(0.0, (twl_gt - z_dune_toe) / z_dune_toe)
        geomorphic_work = u_excess_gt * (1.0 + collision_ratio)
        if np.max(geomorphic_work) == 0:
            geomorphic_work = u_gt * (1.0 + np.maximum(0.0, twl_gt / 2.0))

        peak_geom_idx = int(np.argmax(geomorphic_work))
        peak_row = sub_gt.iloc[peak_geom_idx]
        true_peak_ts = pd.to_datetime(peak_row["timestamp"], utc=True)
        true_peak_hs = float(peak_row.get("hs_m", 0.0))
        true_peak_twl = float(twl_gt[peak_geom_idx])
        true_peak_ups = float(peak_row.get("upsilon", 1.0))
        true_peak_excess_ups = float(u_excess_gt[peak_geom_idx])
        tp_val = float(peak_row.get("tp_s", 10.0))
        true_peak_power = float(0.49 * (true_peak_hs ** 2) * tp_val)

        # 5. Surveyed Cross-Shore Shoreline Displacement (Delta x)
        surveyed_dx = None
        if aligned_surveys and site_key in aligned_surveys:
            s_df = aligned_surveys[site_key]
            if len(s_df) > 0:
                try:
                    s_idx = pd.to_datetime(
                        s_df.index if isinstance(s_df.index, pd.DatetimeIndex) else s_df.get("survey_date", s_df.index),
                        utc=True,
                    )
                    post_mask = s_idx >= (true_peak_ts - pd.Timedelta(days=5))
                    if np.any(post_mask):
                        sub_s = s_df[post_mask]
                        first_post = sub_s.iloc[0]
                        surveyed_dx = float(first_post.get("surveyed_delta_x_m", first_post.get("cross_shore_change_m", 0.0)))
                    else:
                        nearest_idx = int(np.argmin(np.abs((s_idx - true_peak_ts).total_seconds())))
                        surveyed_dx = float(s_df.iloc[nearest_idx].get("surveyed_delta_x_m", s_df.iloc[nearest_idx].get("cross_shore_change_m", 0.0)))
                except Exception:
                    surveyed_dx = None

        cond = q.get("condition_fn")

        row: Dict[str, Any] = {
            "query": q["query"],
            "category": q.get("category", "Historical Storm"),
            "site": site_key,
            "true_peak_timestamp": str(true_peak_ts)[:16],
            "true_peak_hs_m": true_peak_hs,
            "true_peak_twl_m": true_peak_twl,
            "z_dune_toe_m": z_dune_toe,
            "true_peak_upsilon": true_peak_ups,
            "true_peak_excess_upsilon": true_peak_excess_ups,
            "true_peak_power_kW_per_m": true_peak_power,
            "cum_excess_work_MJ_per_m": cum_excess_work_MJ_m,
            "surveyed_delta_x_m": f"{surveyed_dx:+.2f}" if surveyed_dx is not None else "N/A",
        }

        # Parse query intent once per query
        parsed_spec = None
        if use_llm_parser:
            try:
                parsed_spec = GeminiPhysicsQueryParser.parse(query_text=q["query"], parser_mode=parser_mode)
            except Exception:
                parsed_spec = None
        if parsed_spec is None:
            if parser_mode == "pure_gemini":
                parsed_spec = PhysicsQuerySpec(raw_query=q["query"], intent_summary="Pure Gemini parsing unavailable")
            else:
                parsed_spec = GeminiPhysicsQueryParser._fallback_parse(q["query"])

        sub_gt_start = pd.to_datetime(sub_gt["timestamp"].min(), utc=True) - pd.Timedelta(hours=12)
        sub_gt_end = pd.to_datetime(sub_gt["timestamp"].max(), utc=True) + pd.Timedelta(hours=12)

        for name, eng in rag_engines.items():
            res = eng.query(
                q["query"],
                top_k=k,
                use_llm_parser=False,
                use_llm_synthesis=False,
                spec=parsed_spec,
            )
            mdf = res.matches_df
            if len(mdf) > 0:
                top1_ts = pd.to_datetime(mdf["timestamp"].iloc[0], utc=True) if "timestamp" in mdf.columns else None
                dt_hrs = abs((top1_ts - true_peak_ts).total_seconds()) / 3600.0 if top1_ts is not None else 999.0
                top1_hs = float(mdf["hs_m"].iloc[0]) if "hs_m" in mdf.columns else 0.0
                top5_hs = float(mdf["hs_m"].mean()) if "hs_m" in mdf.columns else 0.0
                hs_capture_pct = (top5_hs / true_peak_hs) * 100.0 if true_peak_hs > 0 else 100.0

                # Extract TWL and collision regime in retrieved candidates
                if "total_water_level_m" in mdf.columns:
                    mdf_twl = mdf["total_water_level_m"].values.astype(float)
                elif "surge_m" in mdf.columns and "runup_r2_m" in mdf.columns:
                    mdf_twl = (mdf["surge_m"] + mdf["runup_r2_m"]).values.astype(float)
                else:
                    mdf_twl = mdf.get("surge_m", pd.Series(0.0, index=mdf.index)).values.astype(float)

                top1_twl = float(mdf_twl[0]) if len(mdf_twl) > 0 else 0.0
                mean_top5_twl = float(np.mean(mdf_twl)) if len(mdf_twl) > 0 else 0.0
                collision_capture_pct = float(np.mean(mdf_twl >= z_dune_toe) * 100.0) if len(mdf_twl) > 0 else 0.0

                # Extract Upsilon and excess kinetic flux in retrieved candidates
                mdf_u = mdf["upsilon"].values.astype(float) if "upsilon" in mdf.columns else np.zeros(len(mdf))
                if "upsilon_crit" in mdf.columns:
                    mdf_crit = mdf["upsilon_crit"].values.astype(float)
                else:
                    mdf_crit = np.full_like(mdf_u, 1500.0)
                mdf_excess = np.maximum(0.0, mdf_u - mdf_crit)

                top1_ups = float(mdf_u[0]) if len(mdf_u) > 0 else 0.0
                top5_ups = float(np.mean(mdf_u)) if len(mdf_u) > 0 else 0.0
                top5_excess = float(np.mean(mdf_excess)) if len(mdf_excess) > 0 else 0.0
                excess_work_capture_pct = (top5_excess / max(1.0, true_peak_excess_ups)) * 100.0 if true_peak_excess_ups > 0 else 100.0
                cum_excess_retrieved_MJ = float(np.sum(mdf_excess) * 3600.0 / 1e6)

                top1_power = float(0.49 * (mdf["hs_m"].iloc[0] ** 2) * mdf["tp_s"].iloc[0]) if ("hs_m" in mdf.columns and "tp_s" in mdf.columns) else 0.0
                top5_power = float(np.mean([0.49 * (mdf["hs_m"].iloc[j] ** 2) * mdf["tp_s"].iloc[j] for j in range(len(mdf))])) if ("hs_m" in mdf.columns and "tp_s" in mdf.columns) else 0.0

                # Geomorphic Graded Relevance Scoring (Grades 0 to 3):
                # - Grade 3: Peak Geomorphic Impact (Active TWL Collision Regime TWL >= z_toe AND Excess Upsilon within storm window)
                # - Grade 2: Active Hydrodynamic Storm (Elevated waves Hs >= 3m or Upsilon >= Upsilon_crit within storm window)
                # - Grade 1: Within broad event window but subcritical (no dune collision)
                # - Grade 0: Outside event window or wrong site
                g_scores = []
                for j in range(len(mdf)):
                    r_ts = _get_row_ts(mdf.iloc[j])
                    r_site = _get_row_site(mdf.iloc[j])
                    if r_site and site_key and r_site != site_key:
                        g_scores.append(0)
                        continue
                    if r_ts < sub_gt_start - pd.Timedelta(hours=24) or r_ts > sub_gt_end + pd.Timedelta(hours=24):
                        g_scores.append(0)
                        continue

                    is_in_storm_core = (sub_gt_start <= r_ts <= sub_gt_end)
                    is_collision = (mdf_twl[j] >= z_dune_toe)
                    is_excess_u = (mdf_excess[j] > 0)

                    if is_in_storm_core and is_collision and is_excess_u:
                        g_scores.append(3)
                    elif is_in_storm_core and (is_excess_u or float(mdf["hs_m"].iloc[j]) >= 3.0 or (cond and cond(mdf.iloc[j], df))):
                        g_scores.append(2)
                    elif sub_gt_start - pd.Timedelta(hours=48) <= r_ts <= sub_gt_end + pd.Timedelta(hours=48):
                        g_scores.append(1)
                    else:
                        g_scores.append(0)

                dcg = sum((2.0 ** g - 1.0) / np.log2(j + 2) for j, g in enumerate(g_scores[:k]))
                ideal_top = min(k, max(1, len(sub_gt)))
                ideal_grades = [3] * ideal_top + [0] * max(0, k - ideal_top)
                idcg = sum((2.0 ** ig - 1.0) / np.log2(j + 2) for j, ig in enumerate(ideal_grades))
                ndcg_val = float(dcg / idcg) if idcg > 0 else 0.0

                mrr_val = 0.0
                for j, g in enumerate(g_scores):
                    if g >= 2:
                        mrr_val = 1.0 / (j + 1)
                        break

                p1 = 1.0 if (len(g_scores) and g_scores[0] >= 2) else 0.0
                p5 = float(np.mean([g >= 2 for g in g_scores]))
            else:
                dt_hrs, top1_hs, top5_hs, hs_capture_pct, p5, p1 = 999.0, 0.0, 0.0, 0.0, 0.0, 0.0
                top1_twl, mean_top5_twl, collision_capture_pct = 0.0, 0.0, 0.0
                top1_ups, top5_ups, excess_work_capture_pct, cum_excess_retrieved_MJ = 0.0, 0.0, 0.0, 0.0
                top1_power, top5_power = 0.0, 0.0
                ndcg_val, mrr_val = 0.0, 0.0

            row[f"{name}_P@1"] = p1
            row[f"{name}_P@5"] = p5
            row[f"{name}_NDCG@5"] = ndcg_val
            row[f"{name}_MRR"] = mrr_val
            row[f"{name}_dt_peak_hrs"] = dt_hrs
            row[f"{name}_top1_hs_m"] = top1_hs
            row[f"{name}_mean_hs_m"] = top5_hs
            row[f"{name}_hs_capture_pct"] = hs_capture_pct
            row[f"{name}_top1_twl_m"] = top1_twl
            row[f"{name}_mean_twl_m"] = mean_top5_twl
            row[f"{name}_collision_capture_pct"] = collision_capture_pct
            row[f"{name}_excess_work_capture_pct"] = excess_work_capture_pct
            row[f"{name}_top1_upsilon"] = top1_ups
            row[f"{name}_mean_upsilon"] = top5_ups
            row[f"{name}_cum_excess_MJ_m"] = cum_excess_retrieved_MJ
            row[f"{name}_top1_power_kW_per_m"] = top1_power
            row[f"{name}_mean_power_kW_per_m"] = top5_power

        records.append(row)

    df_query_details = pd.DataFrame(records)

    summary_rows = []
    for name in rag_engines.keys():
        p1_col = f"{name}_P@1"
        p5_col = f"{name}_P@5"
        ndcg_col = f"{name}_NDCG@5"
        mrr_col = f"{name}_MRR"
        dt_col = f"{name}_dt_peak_hrs"
        hs_col = f"{name}_top1_hs_m"
        cap_col = f"{name}_hs_capture_pct"
        twl_col = f"{name}_top1_twl_m"
        coll_col = f"{name}_collision_capture_pct"
        exc_col = f"{name}_excess_work_capture_pct"
        u1_col = f"{name}_top1_upsilon"
        u5_col = f"{name}_mean_upsilon"
        cum_col = f"{name}_cum_excess_MJ_m"
        p1w_col = f"{name}_top1_power_kW_per_m"
        p5w_col = f"{name}_mean_power_kW_per_m"

        summary_rows.append({
            "Architecture": name,
            "Graded NDCG@5": float(df_query_details[ndcg_col].mean()) if ndcg_col in df_query_details else 0.0,
            "Mean Reciprocal Rank (MRR)": float(df_query_details[mrr_col].mean()) if mrr_col in df_query_details else 0.0,
            "Peak Precision@1": float(df_query_details[p1_col].mean()) if p1_col in df_query_details else 0.0,
            "Peak Precision@5": float(df_query_details[p5_col].mean()) if p5_col in df_query_details else 0.0,
            "TWL Collision Capture (%)": float(df_query_details[coll_col].mean()) if coll_col in df_query_details else 0.0,
            "Excess Work Capture (%)": float(df_query_details[exc_col].mean()) if exc_col in df_query_details else 0.0,
            "Median |Δt_peak| Error (hrs)": float(df_query_details[dt_col].median()) if dt_col in df_query_details else 0.0,
            "Mean |Δt_peak| Error (hrs)": float(df_query_details[dt_col].mean()) if dt_col in df_query_details else 0.0,
            "Wave Height Capture (%)": float(df_query_details[cap_col].mean()) if cap_col in df_query_details else 0.0,
            "Mean Top-1 Hs (m)": float(df_query_details[hs_col].mean()) if hs_col in df_query_details else 0.0,
            "Mean Top-1 TWL (m)": float(df_query_details[twl_col].mean()) if twl_col in df_query_details else 0.0,
            "Median Top-1 Upsilon": float(df_query_details[u1_col].median()) if u1_col in df_query_details else 0.0,
            "Mean Top-5 Upsilon": float(df_query_details[u5_col].mean()) if u5_col in df_query_details else 0.0,
            "Median Top-1 Wave Power (kW/m)": float(df_query_details[p1w_col].median()) if p1w_col in df_query_details else 0.0,
            "Mean Top-5 Wave Power (kW/m)": float(df_query_details[p5w_col].mean()) if p5w_col in df_query_details else 0.0,
            "Mean Cumulative Excess Impulse (MJ/m)": float(df_query_details[cum_col].mean()) if cum_col in df_query_details else 0.0,
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_query_details, df_summary



```
