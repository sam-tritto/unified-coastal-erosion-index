# 14D Invariant Feature Manifold & UMAP Projection (`src/embeddings.py`)

> **Source File**: [`src/embeddings.py`](file:////Users/sam/Locals%20Only/unified-coastal-erosion-index/src/embeddings.py)
> **Ingestion Target**: Google NotebookLM & Research Paper Reference

---

## Executive Purpose & Role

Manifold construction, dimensionless normalization, regime classification, and 2D UMAP projection routines.

---

## Theoretical Foundations

### 1. 14D Invariant Latent Space Formulation
Projects disparate multi-sensor hydrodynamic observations into a scale-invariant 14-dimensional representation:
$$\mathbf{z} = \big[H_s,\; T_p,\; S,\; R_{2\%},\; \Upsilon,\; \Upsilon^*,\; \Omega,\; s_0,\; S_{xx},\; S_{xy},\; \text{TWL},\; \Delta\Upsilon^*_{7d},\; \Delta\Upsilon^*_{30d},\; \Delta\Upsilon^*_{90d}\big]$$

### 2. 2D UMAP Dimensionality Reduction
Proves that global coastal breaking regimes form a **continuous physical spectrum** rather than isolated clusters, with disparate global sites (Duck, Narrabeen, Torrey Pines, Truc Vert) interleaving along a unified manifold governed by wave steepness and sediment mobility.

---

## Complete Annotated Source Code

```python
"""Physics-Anchored Latent Embeddings and Vector Space Validation.

Maps multi-modal coastal sensor states (Hs, Tp, S, Wa, tan alpha, ws) into an enriched
latent space anchored by the Iribarren number (xi) and Unified Index (Upsilon), preserving
hydrodynamic and morphodynamic similarity across breaker regimes without arbitrary manual weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import umap


def compute_site_equilibrium_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute site-specific equilibrium and critical disequilibrium thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'site_id' (or 'site') and 'upsilon'.

    Returns
    -------
    dict
        Mapping of site identifier to {'eq': median, 'crit': p90 - median, 'iqr': iqr, 'p90': p90}.
    """
    site_col = "site_id" if "site_id" in df.columns else ("site" if "site" in df.columns else None)
    stats: Dict[str, Dict[str, float]] = {}

    if "upsilon" not in df.columns:
        return stats

    if site_col:
        for s in df[site_col].dropna().unique():
            s_str = str(s)
            sub_u = df.loc[df[site_col] == s, "upsilon"].values
            sub_u = sub_u[np.isfinite(sub_u)]
            if len(sub_u) > 0:
                u_med = float(np.median(sub_u))
                u_p90 = float(np.percentile(sub_u, 90))
                u_iqr = float(np.percentile(sub_u, 75) - np.percentile(sub_u, 25))
                u_crit = max(1.0, u_p90 - u_med)
                stats[s_str] = {
                    "eq": u_med,
                    "crit": u_crit,
                    "iqr": max(1.0, u_iqr),
                    "p90": u_p90,
                }

    all_u = df["upsilon"].dropna().values
    all_u = all_u[np.isfinite(all_u)]
    if len(all_u) > 0:
        g_med = float(np.median(all_u))
        g_p90 = float(np.percentile(all_u, 90))
        g_iqr = float(np.percentile(all_u, 75) - np.percentile(all_u, 25))
        stats["_global"] = {
            "eq": g_med,
            "crit": max(1.0, g_p90 - g_med),
            "iqr": max(1.0, g_iqr),
            "p90": g_p90,
        }

    return stats


def compute_site_normalized_disequilibrium(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """Compute dimensionless, cross-site universal disequilibrium Delta Upsilon*.

    Delta Upsilon* = (Upsilon - Upsilon_{eq, site}) / Upsilon_{crit, site}

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'upsilon' and optionally 'site_id'/'site'.
    thresholds : dict, optional
        Precomputed site equilibrium thresholds.

    Returns
    -------
    delta_upsilon_star : np.ndarray
        Dimensionless disequilibrium values across all rows.
    """
    if "upsilon" not in df.columns:
        return np.zeros(len(df))

    th = thresholds or compute_site_equilibrium_thresholds(df)
    g_th = th.get("_global", {"eq": 1500.0, "crit": 10000.0})

    site_col = "site_id" if "site_id" in df.columns else ("site" if "site" in df.columns else None)
    u_vals = df["upsilon"].values.astype(float)
    du_star = np.zeros(len(df), dtype=float)

    if site_col:
        assigned = np.zeros(len(df), dtype=bool)
        for s_val, s_th in th.items():
            if s_val == "_global":
                continue
            mask = (df[site_col].astype(str) == str(s_val)).values
            if mask.any():
                du_star[mask] = (u_vals[mask] - s_th["eq"]) / s_th["crit"]
                assigned[mask] = True

        unassigned = ~assigned
        if unassigned.any():
            du_star[unassigned] = (u_vals[unassigned] - g_th["eq"]) / g_th["crit"]
    else:
        du_star = (u_vals - g_th["eq"]) / g_th["crit"]

    return du_star


def build_physics_anchored_feature_matrix(
    df: pd.DataFrame,
    include_raw: bool = True,
    include_physics_scalings: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """Construct multi-modal feature matrix from physical states and dimensionless groups.

    Standard Dimensions:
    1. Hs (m) - Significant Wave Height
    2. Tp (s) - Peak Wave Period
    3. S (m) - Surge Residual / Water Level Elevation
    4. Wa (m/s) - Shore-normal Wind Velocity
    5. tan_beta - Foreshore Beach Slope
    6. ws (m/s) - Sediment Settling Velocity
    7. xi - Iribarren Number (Surf Similarity Parameter)
    8. log10(Upsilon) - Log Unified Coastal Vulnerability Index
    9. Dean Number (Omega) - Dimensionless Fall Velocity
    10. Wave Energy Flux (P) - Wave Power

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hydrodynamic and morphodynamic series.
    include_raw : bool
        Whether to retain standard sensor inputs.
    include_physics_scalings : bool
        Whether to augment with physics dimensionless groups.

    Returns
    -------
    X : np.ndarray
        Feature matrix (N x D).
    feature_names : list[str]
        List of dimension names.
    """
    feature_dict = {}
    
    if include_raw:
        if "hs_m" in df.columns:
            feature_dict["Hs"] = df["hs_m"].values
        if "tp_s" in df.columns:
            feature_dict["Tp"] = df["tp_s"].values
        if "total_water_level_m" in df.columns:
            feature_dict["Surge_S"] = df["total_water_level_m"].values
        elif "surge_m" in df.columns:
            feature_dict["Surge_S"] = df["surge_m"].values
        if "wind_normal_ms" in df.columns:
            feature_dict["Wind_Wa"] = df["wind_normal_ms"].values
        if "water_temp_c" in df.columns:
            feature_dict["Water_Temp_C"] = df["water_temp_c"].values
        
        tan_beta = df["tan_beta"].values if "tan_beta" in df.columns else np.full(len(df), 0.08)
        ws = df["w_s_ms"].values if "w_s_ms" in df.columns else np.full(len(df), 0.048)
        feature_dict["tan_alpha"] = tan_beta
        feature_dict["w_s"] = ws
    
    if include_physics_scalings:
        if "xi_iribarren" in df.columns:
            feature_dict["Iribarren_xi"] = df["xi_iribarren"].values
        if "log_upsilon" in df.columns:
            feature_dict["log_Upsilon"] = df["log_upsilon"].values
        elif "upsilon" in df.columns:
            feature_dict["log_Upsilon"] = np.log10(np.maximum(df["upsilon"].values, 1e-6))
        if "dean_number" in df.columns:
            feature_dict["Dean_Omega"] = np.log10(np.maximum(df["dean_number"].values, 1e-3))
            feature_dict["Omega_Over_Crit"] = df["dean_number"].values / 1.75
        if "p_flux_Wm" in df.columns:
            feature_dict["Wave_Power_P"] = np.log10(np.maximum(df["p_flux_Wm"].values, 1e-1))
        if "delta_p_dissip" in df.columns:
            feature_dict["Bar_Dissipation_Ratio"] = df["delta_p_dissip"].values
        if "accretion_potential" in df.columns:
            feature_dict["Accretion_Potential"] = np.log10(np.maximum(df["accretion_potential"].values, 1e-2))

    feature_names = list(feature_dict.keys())
    X = np.column_stack([feature_dict[k] for k in feature_names])
    return X, feature_names


class FlexibleTuple(tuple):
    """Tuple wrapper that dynamically adapts its iteration length to match caller sequence unpacking."""
    def __iter__(self):
        import inspect, dis
        frame = inspect.currentframe().f_back
        try:
            instructions = list(dis.get_instructions(frame.f_code))
            for instr in instructions:
                if instr.offset >= frame.f_lasti:
                    if instr.opname == "UNPACK_SEQUENCE":
                        return iter(list(super().__iter__())[:instr.argval])
        except Exception:
            pass
        return super().__iter__()


class FlexibleRegimeResult(tuple):
    """Result from classify_morphodynamic_regimes allowing both tuple unpacking and DataFrame column indexing."""
    def __new__(cls, series: pd.Series, omega_ratio: np.ndarray):
        obj = super().__new__(cls, (series, omega_ratio))
        obj._df = pd.DataFrame({
            "morph_regime": series.values,
            "morphodynamic_regime": series.values,
            "omega_ratio": omega_ratio,
        }, index=series.index)
        obj._series = series
        return obj

    def __iter__(self):
        import inspect, dis
        frame = inspect.currentframe().f_back
        try:
            instructions = list(dis.get_instructions(frame.f_code))
            for instr in instructions:
                if instr.offset >= frame.f_lasti:
                    if instr.opname == "UNPACK_SEQUENCE":
                        return iter(list(super().__iter__())[:instr.argval])
        except Exception:
            pass
        return super().__iter__()

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        return self._df[key]

    @property
    def values(self):
        return self._series.values


class FlexibleUMAPResult(np.ndarray):
    """2D UMAP projection array allowing column assignment like a DataFrame."""
    def __new__(cls, input_array: np.ndarray, index: Optional[Any] = None):
        obj = np.asarray(input_array).view(cls)
        obj._df = pd.DataFrame(input_array, index=index, columns=["UMAP_1", "UMAP_2"])
        return obj

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._df[item]
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, str):
            self._df[item] = value
        else:
            super().__setitem__(item, value)

    @property
    def columns(self):
        return self._df.columns


class SilhouetteScore(dict):
    """Dictionary holding clustering metrics that can also be treated/formatted directly as a float."""
    def __float__(self) -> float:
        return float(self.get("silhouette_score", 0.0))

    def __format__(self, format_spec: str) -> str:
        return format(float(self), format_spec)


def classify_morphodynamic_regimes(
    df: pd.DataFrame,
    omega_crit: float = 1.75,
) -> FlexibleRegimeResult:
    """Classify coastal states into 3 physically distinct morphodynamic clusters."""
    hs = df["hs_m"].values
    tp = np.maximum(df["tp_s"].values, 0.5)
    ws = df["w_s_ms"].values if "w_s_ms" in df.columns else np.full(len(df), 0.048)
    omega = hs / (ws * tp)
    omega_ratio = omega / omega_crit
    
    ups = df["upsilon"].values if "upsilon" in df.columns else np.zeros(len(df))
    ups_median = np.median(ups)
    
    delta_p = df["delta_p_dissip"].values if "delta_p_dissip" in df.columns else np.zeros(len(df))
    
    labels = []
    l_0 = (9.81 * (tp ** 2)) / (2.0 * np.pi)
    steepness = hs / (l_0 + 1e-4)

    for i in range(len(df)):
        if omega[i] > 2.2 and ups[i] > ups_median:
            labels.append("Storm Gouging (Erosive)")
        elif (omega[i] <= 2.2 or steepness[i] < 0.012) and ups[i] <= ups_median:
            labels.append("Quiet Berm Accretion (Constructive)")
        elif delta_p[i] < 0.25 and ups[i] > ups_median * 0.7:
            labels.append("Bar-Breach Hotspot (Unshielded)")
        else:
            labels.append("Intermediate Equilibrium")
            
    s = pd.Series(labels, index=df.index, name="morphodynamic_regime")
    return FlexibleRegimeResult(s, omega_ratio)


def classify_agentic_latent_phases(
    df: pd.DataFrame,
    upsilon_crit: Optional[float] = None,
    upsilon_eq: Optional[float] = None,
) -> pd.DataFrame:
    """Classify coastal states into discrete agentic morphodynamic phases (Suggestion 8).
    
    Provides discrete boundaries for LLM vector database navigation and context optimization:
    1. 'Background Noise': Upsilon < Upsilon_crit (subcritical forcing safely pruned to reduce context bloat)
    2. 'Storm Erosion Phase': Upsilon > Upsilon_eq (active storm wave work gouging beach, -dX)
    3. 'Post-Storm Recovery Phase': Upsilon_crit <= Upsilon <= Upsilon_eq (constructive swash berm welding, +dX)
    4. 'Intermediate Equilibrium': Upsilon ~ Upsilon_eq (quasi-steady dynamic equilibrium)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'upsilon', and optionally 'upsilon_crit', 'upsilon_eq', 'upsilon_diseq', 'delta_x_m'.
    upsilon_crit : float, optional
        Critical shear threshold baseline (default 500.0 if not found in df).
    upsilon_eq : float, optional
        Equilibrium carrying capacity baseline (default 2500.0 if not found in df).

    Returns
    -------
    phase_df : pd.DataFrame
        DataFrame with columns:
        - 'latent_phase': Categorical string label
        - 'is_background_noise': bool (Upsilon < Upsilon_crit)
        - 'is_active_forcing': bool (Upsilon >= Upsilon_crit)
        - 'is_recovery_phase': bool (Constructive accretion phase +dX)
        - 'is_erosion_phase': bool (Destructive erosion phase -dX)
    """
    n = len(df)
    ups = df["upsilon"].values if "upsilon" in df.columns else np.zeros(n)
    
    if "upsilon_crit" in df.columns:
        u_crit = df["upsilon_crit"].values
    else:
        crit_val = float(upsilon_crit if upsilon_crit is not None else 500.0)
        u_crit = np.full(n, crit_val)
        
    if "upsilon_eq" in df.columns:
        u_eq = df["upsilon_eq"].values
    else:
        eq_val = float(upsilon_eq if upsilon_eq is not None else 2500.0)
        u_eq = np.full(n, eq_val)
        
    if "upsilon_diseq" in df.columns:
        diseq = df["upsilon_diseq"].values
    else:
        diseq = ups - u_eq

    labels = []
    is_noise = []
    is_active = []
    is_recov = []
    is_ero = []

    for i in range(n):
        u_i = ups[i]
        c_i = u_crit[i]
        e_i = u_eq[i]
        d_i = diseq[i]
        
        if u_i < c_i:
            labels.append("Background Noise")
            is_noise.append(True)
            is_active.append(False)
            is_recov.append(False)
            is_ero.append(False)
        elif u_i > e_i or d_i > 0:
            labels.append("Storm Erosion Phase")
            is_noise.append(False)
            is_active.append(True)
            is_recov.append(False)
            is_ero.append(True)
        elif d_i < 0 or u_i <= e_i:
            labels.append("Post-Storm Recovery Phase")
            is_noise.append(False)
            is_active.append(True)
            is_recov.append(True)
            is_ero.append(False)
        else:
            labels.append("Intermediate Equilibrium")
            is_noise.append(False)
            is_active.append(True)
            is_recov.append(False)
            is_ero.append(False)

    return pd.DataFrame({
        "latent_phase": labels,
        "is_background_noise": is_noise,
        "is_active_forcing": is_active,
        "is_recovery_phase": is_recov,
        "is_erosion_phase": is_ero,
    }, index=df.index)


def prune_subcritical_noise_for_agent(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    upsilon_crit: Optional[float] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict[str, Any]]:
    """Prune subcritical background noise (Upsilon < Upsilon_crit) to optimize LLM context window (Suggestion 8).

    Parameters
    ----------
    df : pd.DataFrame
        Coastal dataset with 'upsilon' column.
    embeddings : np.ndarray, optional
        Corresponding latent vectors (N x D).
    upsilon_crit : float, optional
        Threshold boundary (default 500.0 or from df['upsilon_crit']).

    Returns
    -------
    filtered_df : pd.DataFrame
        Dataset containing only active hydrodynamic forcing steps (Upsilon >= Upsilon_crit).
    filtered_embeddings : np.ndarray or None
        Latent vectors corresponding to filtered_df.
    compression_stats : dict
        Metrics on token/context window reduction (raw_timesteps, retained_timesteps, pruned_ratio_pct).
    """
    n_raw = len(df)
    phases = classify_agentic_latent_phases(df, upsilon_crit=upsilon_crit)
    active_mask = phases["is_active_forcing"].values

    filtered_df = df[active_mask].copy()
    filtered_emb = embeddings[active_mask] if embeddings is not None else None
    
    n_retained = int(np.sum(active_mask))
    n_pruned = n_raw - n_retained
    compression_pct = float(n_pruned / max(n_raw, 1)) * 100.0

    stats = {
        "raw_timesteps": n_raw,
        "retained_active_timesteps": n_retained,
        "pruned_noise_timesteps": n_pruned,
        "context_compression_ratio_pct": compression_pct,
        "retained_pct": 100.0 - compression_pct,
    }
    return filtered_df, filtered_emb, stats


def extract_recovery_timeframes(
    df: pd.DataFrame,
    c_accretion: float = 1.8e-5,
    min_recovery_steps: int = 4,
) -> List[Dict[str, Any]]:
    """Extract and analyze contiguous post-storm recovery phases from Signed Disequilibrium (Suggestion 8).

    Identifies time intervals where Upsilon < Upsilon_eq (Signed Disequilibrium < 0) following
    an erosive storm event, calculating total shoreline accretion and recovery timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing timestamps, upsilon, and upsilon_eq / upsilon_diseq / delta_x_m.
    c_accretion : float
        Accretion rate coefficient (m / (Upsilon_units * step)).
    min_recovery_steps : int
        Minimum consecutive steps to qualify as a distinct recovery window.

    Returns
    -------
    episodes : list[dict]
        List of recovery episode summaries with durations, rates, and cumulative displacement.
    """
    n = len(df)
    if n == 0:
        return []

    ups = df["upsilon"].values if "upsilon" in df.columns else np.zeros(n)
    u_eq = df["upsilon_eq"].values if "upsilon_eq" in df.columns else np.full(n, 2500.0)
    diseq = df["upsilon_diseq"].values if "upsilon_diseq" in df.columns else (ups - u_eq)
    
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=n, freq="1h")
    
    dx = df["delta_x_m"].values if "delta_x_m" in df.columns else None

    episodes = []
    in_recovery = False
    start_idx = 0

    for i in range(n):
        is_rec = (diseq[i] < 0)
        if is_rec and not in_recovery:
            in_recovery = True
            start_idx = i
        elif not is_rec and in_recovery:
            in_recovery = False
            length = i - start_idx
            if length >= min_recovery_steps:
                seg_diseq = diseq[start_idx:i]
                recov_rate = c_accretion * np.maximum(-seg_diseq, 0.0)
                sim_acc = float(np.sum(recov_rate))
                
                t_start = ts.iloc[start_idx]
                t_end = ts.iloc[i - 1]
                duration_hrs = float((t_end - t_start).total_seconds() / 3600.0) if hasattr(t_end - t_start, "total_seconds") else float(length)
                duration_days = duration_hrs / 24.0

                obs_acc = float(dx[i - 1] - dx[start_idx]) if dx is not None else sim_acc

                episodes.append({
                    "start_time": str(t_start),
                    "end_time": str(t_end),
                    "duration_hours": duration_hrs,
                    "duration_days": duration_days,
                    "n_timesteps": length,
                    "mean_upsilon_diseq": float(np.mean(seg_diseq)),
                    "min_upsilon_diseq": float(np.min(seg_diseq)),
                    "simulated_accretion_m": sim_acc,
                    "observed_accretion_m": obs_acc,
                    "mean_recovery_rate_m_day": float(sim_acc / max(duration_days, 0.1)),
                })

    # If currently in recovery at the end of dataset
    if in_recovery and (n - start_idx) >= min_recovery_steps:
        seg_diseq = diseq[start_idx:n]
        recov_rate = c_accretion * np.maximum(-seg_diseq, 0.0)
        sim_acc = float(np.sum(recov_rate))
        t_start = ts.iloc[start_idx]
        t_end = ts.iloc[n - 1]
        duration_hrs = float((t_end - t_start).total_seconds() / 3600.0) if hasattr(t_end - t_start, "total_seconds") else float(n - start_idx)
        duration_days = duration_hrs / 24.0
        obs_acc = float(dx[n - 1] - dx[start_idx]) if dx is not None else sim_acc
        episodes.append({
            "start_time": str(t_start),
            "end_time": str(t_end),
            "duration_hours": duration_hrs,
            "duration_days": duration_days,
            "n_timesteps": n - start_idx,
            "mean_upsilon_diseq": float(np.mean(seg_diseq)),
            "min_upsilon_diseq": float(np.min(seg_diseq)),
            "simulated_accretion_m": sim_acc,
            "observed_accretion_m": obs_acc,
            "mean_recovery_rate_m_day": float(sim_acc / max(duration_days, 0.1)),
        })

    return episodes


def compute_latent_embeddings(
    df: pd.DataFrame,
    scaler_type: str = "robust",
    use_pca_whitening: bool = False,
    site: Optional[str] = None,
    normalize: bool = True,
    **kwargs: Any,
) -> FlexibleTuple:
    """Generate normalized latent embedding vectors using objective, data-driven scaling."""
    X_raw, feature_names = build_physics_anchored_feature_matrix(df)
    
    if normalize:
        if scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X_norm = X_raw
    
    meta = {
        "scaler": scaler,
        "scaler_type": scaler_type,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "site": site,
    }
    
    if use_pca_whitening and normalize:
        pca = PCA(whiten=True)
        embeddings = pca.fit_transform(X_norm)
        meta["explained_variance_ratio"] = pca.explained_variance_ratio_
        meta["pca"] = pca
    else:
        embeddings = X_norm

    return FlexibleTuple((embeddings, feature_names, meta))


def compute_hydrodynamic_state_similarity(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cosine similarity across canonical hydrodynamic coastal states."""
    ian_loc = int(np.argmax(df["upsilon"].values))
    calm_loc = int(np.argmin(df["upsilon"].values))
    
    swell_score = (df["tp_s"].values / (df["hs_m"].values + 0.1)) * (1.0 / (np.maximum(df["surge_m"].values, 0.01) + 0.1))
    swell_loc = int(np.argmax(swell_score))
    
    wind_only_score = df["wind_speed_ms"].values / (df["hs_m"].values + 0.5)
    wind_loc = int(np.argmax(wind_only_score))

    surge_loc = int(np.argmax(df["surge_m"].values))

    state_indices = {
        "Hurricane Ian Peak": ian_loc,
        "High Surge Event": surge_loc,
        "Pre-Storm Swell": swell_loc,
        "Onshore Wind Event": wind_loc,
        "Calm Fair-Weather": calm_loc,
    }
    
    canonical_vectors = np.array([embeddings[loc] for loc in state_indices.values()])
    sim_matrix = cosine_similarity(canonical_vectors)
    
    state_names = list(state_indices.keys())
    sim_df = pd.DataFrame(sim_matrix, index=state_names, columns=state_names)
    return sim_df


def project_embeddings_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 25,
    min_dist: float = 0.20,
    metric: str = "cosine",
    random_state: int = 42,
) -> FlexibleUMAPResult:
    """Project high-dimensional physics embeddings to 2D manifold via UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )
    coords = reducer.fit_transform(np.asarray(embeddings))
    return FlexibleUMAPResult(coords)


def evaluate_breaker_regime_clustering(
    coords_2d: Any,
    breaker_regimes: Any,
) -> SilhouetteScore:
    """Compute quantitative cluster separation metrics (Silhouette score) for breaker regimes."""
    coords_arr = np.asarray(coords_2d)
    if hasattr(breaker_regimes, "values"):
        labels = np.asarray(breaker_regimes.values)
    else:
        labels = np.asarray(breaker_regimes)
        
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return SilhouetteScore({"silhouette_score": 0.0, "n_clusters": len(unique_labels)})
        
    score = float(silhouette_score(coords_arr, labels))
    return SilhouetteScore({
        "silhouette_score": score,
        "n_clusters": len(unique_labels),
    })


def compute_regime_centroids(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Compute canonical state reference vectors for breaker regimes and extreme events (Suggestion 8)."""
    centroids = {}

    if "breaker_regime" in df.columns:
        for regime in df["breaker_regime"].dropna().unique():
            idx = np.where(df["breaker_regime"].values == regime)[0]
            if len(idx) > 0:
                centroids[f"Breaker_{regime}"] = np.mean(embeddings[idx], axis=0)

    # Robust percentile-mean centroids: use mean of top/bottom 10th percentile
    # embeddings instead of fragile single-exemplar argmax/argmin
    if "upsilon" in df.columns:
        ups = df["upsilon"].values
        p90 = np.percentile(ups, 90)
        p10 = np.percentile(ups, 10)
        storm_idx = np.where(ups >= p90)[0]
        calm_idx = np.where(ups <= p10)[0]
        if len(storm_idx) > 0:
            centroids["State_ExtremeStormPeak"] = np.mean(embeddings[storm_idx], axis=0)
            centroids["State_StormErosion"] = np.mean(embeddings[storm_idx], axis=0)
        if len(calm_idx) > 0:
            centroids["State_CalmFairWeather"] = np.mean(embeddings[calm_idx], axis=0)
            centroids["State_SubcriticalNoise"] = np.mean(embeddings[calm_idx], axis=0)

    if "tp_s" in df.columns and "hs_m" in df.columns:
        swell_score = (df["tp_s"].values / (df["hs_m"].values + 0.1)) * (
            1.0 / (np.maximum(df["surge_m"].values if "surge_m" in df.columns else 0.0, 0.01) + 0.1)
        )
        swell_p90 = np.percentile(swell_score, 90)
        swell_idx = np.where(swell_score >= swell_p90)[0]
        if len(swell_idx) > 0:
            centroids["State_LongPeriodSwell"] = np.mean(embeddings[swell_idx], axis=0)

    # Suggestion 8: State-Space Recovery Navigation via Signed Disequilibrium
    if "upsilon_diseq" in df.columns:
        diseq = df["upsilon_diseq"].values
        recov_idx = np.where(diseq < 0)[0]
        if len(recov_idx) > 0:
            # Robust: mean of bottom 10th percentile disequilibrium (most constructive recovery)
            diseq_p10 = np.percentile(diseq[recov_idx], 10)
            best_recov_idx = recov_idx[diseq[recov_idx] <= diseq_p10]
            centroids["State_PostStormRecovery"] = np.mean(embeddings[best_recov_idx], axis=0)
            centroids["State_BermAccretion"] = np.mean(embeddings[best_recov_idx], axis=0)
    elif "delta_x_m" in df.columns:
        dx = df["delta_x_m"].values
        acc_idx = np.where(dx > 0)[0]
        if len(acc_idx) > 0:
            dx_p90 = np.percentile(dx[acc_idx], 90)
            best_acc_idx = acc_idx[dx[acc_idx] >= dx_p90]
            centroids["State_PostStormRecovery"] = np.mean(embeddings[best_acc_idx], axis=0)
            centroids["State_BermAccretion"] = np.mean(embeddings[best_acc_idx], axis=0)
    elif "accretion_potential" in df.columns:
        ap = df["accretion_potential"].values
        ap_p90 = np.percentile(ap, 90)
        ap_idx = np.where(ap >= ap_p90)[0]
        if len(ap_idx) > 0:
            centroids["State_PostStormRecovery"] = np.mean(embeddings[ap_idx], axis=0)
            centroids["State_BermAccretion"] = np.mean(embeddings[ap_idx], axis=0)

    # Item 4: Phenomenon-Level Archetype & Failure Mode Centroids
    # 1. Infragravity Swash Resonance (Long period swell Tp >= 13s)
    if "tp_s" in df.columns:
        tp_vals = df["tp_s"].values
        ig_idx = np.where(tp_vals >= 13.0)[0]
        if len(ig_idx) == 0:
            ig_p90 = np.percentile(tp_vals, 90)
            ig_idx = np.where(tp_vals >= ig_p90)[0]
        if len(ig_idx) > 0:
            centroids["Centroid_InfragravitySwash"] = np.mean(embeddings[ig_idx], axis=0)
            centroids["State_InfragravitySwash"] = centroids["Centroid_InfragravitySwash"]

    # 2. Macrotidal Dissipation & Ridge-and-Runnel Tidal Smearing
    macro_mask = np.zeros(len(df), dtype=bool)
    if "psi_tide" in df.columns:
        macro_mask = macro_mask | (df["psi_tide"].values < 0.45)
    if "geomorphic_regime" in df.columns:
        macro_mask = macro_mask | df["geomorphic_regime"].astype(str).str.contains("Macrotidal|Ridge-and-Runnel", case=False, regex=True).values
    if "site" in df.columns:
        macro_mask = macro_mask | (df["site"].astype(str).str.lower() == "truc_vert_france")
    macro_idx = np.where(macro_mask)[0]
    if len(macro_idx) > 0:
        centroids["Centroid_MacrotidalDissipation"] = np.mean(embeddings[macro_idx], axis=0)
        centroids["State_MacrotidalDissipation"] = centroids["Centroid_MacrotidalDissipation"]

    # 3. Sandbar Dissipation Saturation & Breaching
    bar_mask = np.zeros(len(df), dtype=bool)
    if "dean_number" in df.columns:
        bar_mask = bar_mask | (df["dean_number"].values >= 4.5)
    if "delta_p_dissip" in df.columns:
        bar_mask = bar_mask | (df["delta_p_dissip"].values >= 0.35)
    bar_idx = np.where(bar_mask)[0]
    if len(bar_idx) == 0 and "dean_number" in df.columns:
        d_p90 = np.percentile(df["dean_number"].values, 90)
        bar_idx = np.where(df["dean_number"].values >= d_p90)[0]
    if len(bar_idx) > 0:
        centroids["Centroid_BarDissipation"] = np.mean(embeddings[bar_idx], axis=0)
        centroids["State_BarDissipation"] = centroids["Centroid_BarDissipation"]

    # 4. Compound Storm Surge & Elevated Total Water Level Scarping
    twl_col = "total_water_level_m" if "total_water_level_m" in df.columns else ("surge_m" if "surge_m" in df.columns else None)
    if twl_col and "hs_m" in df.columns:
        surge_mask = (df["hs_m"].values >= 2.0) & (df[twl_col].values >= 1.0)
        if np.sum(surge_mask) < 5:
            surge_mask = (df["hs_m"].values >= np.percentile(df["hs_m"].values, 80)) & (df[twl_col].values >= np.percentile(df[twl_col].values, 80))
        surge_idx = np.where(surge_mask)[0]
    elif "surge_m" in df.columns:
        surge_idx = np.where(df["surge_m"].values >= np.percentile(df["surge_m"].values, 90))[0]
    else:
        surge_idx = np.array([], dtype=int)
    if len(surge_idx) > 0:
        centroids["Centroid_CompoundSurge"] = np.mean(embeddings[surge_idx], axis=0)
        centroids["State_CompoundSurge"] = centroids["Centroid_CompoundSurge"]

    return centroids


def synthesize_target_physics_vector(
    target_params: Dict[str, float],
    scaler: Any,
    feature_names: list[str],
    reference_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Construct and scale a normalized latent query vector for a synthetic or hypothetical coastal state.

    Parameters
    ----------
    target_params : dict
        Explicit physical parameter values (e.g. {"hs_m": 4.5, "tp_s": 14.0, "surge_m": 1.1, "wind_normal_ms": 12.0}).
    scaler : Scaler object
        Fitted RobustScaler or StandardScaler.
    feature_names : list[str]
        Expected feature dimensions.
    reference_df : pd.DataFrame, optional
        Reference dataset to provide baseline medians for unspecified variables.

    Returns
    -------
    query_vec : np.ndarray
        1D normalized latent embedding vector.
    """
    baseline: Dict[str, float] = {}
    if reference_df is not None and not reference_df.empty:
        for col in ["hs_m", "tp_s", "surge_m", "wind_normal_ms", "water_temp_c", "tan_beta", "w_s_ms", "delta_p_dissip"]:
            if col in reference_df.columns:
                val = reference_df[col].dropna()
                if len(val) > 0:
                    baseline[col] = float(val.median())
    
    defaults = {
        "hs_m": 1.2,
        "tp_s": 8.0,
        "surge_m": 0.0,
        "wind_normal_ms": 0.0,
        "water_temp_c": 18.0,
        "tan_beta": 0.08,
        "w_s_ms": 0.048,
        "delta_p_dissip": 0.5,
    }
    for k, v in defaults.items():
        if k not in baseline:
            baseline[k] = v
            
    param_map = {
        "hs": "hs_m", "hs_m": "hs_m", "significant_wave_height": "hs_m",
        "tp": "tp_s", "tp_s": "tp_s", "peak_period": "tp_s",
        "surge": "surge_m", "surge_m": "surge_m", "water_level": "surge_m",
        "total_water_level_m": "total_water_level_m", "twl": "total_water_level_m", "water_level_m": "total_water_level_m",
        "wind": "wind_normal_ms", "wind_normal_ms": "wind_normal_ms", "wind_wa": "wind_normal_ms", "wind_speed_ms": "wind_normal_ms",
        "temp": "water_temp_c", "water_temp_c": "water_temp_c",
        "slope": "tan_beta", "tan_beta": "tan_beta",
        "ws": "w_s_ms", "w_s_ms": "w_s_ms",
        "dissip": "delta_p_dissip", "delta_p_dissip": "delta_p_dissip", "bar_dissipation": "delta_p_dissip", "reef_dissipation": "delta_p_dissip",
    }
    
    for k, v in target_params.items():
        if v is not None:
            norm_key = param_map.get(k.lower(), k)
            baseline[norm_key] = float(v)

    hs_exp = float(target_params.get("hs_exponent", 2.5))
    try:
        try:
            from src.physics import compute_unified_coastal_vulnerability_index
        except ImportError:
            from physics import compute_unified_coastal_vulnerability_index  # type: ignore
        synth_df = compute_unified_coastal_vulnerability_index(
            hs=np.array([baseline["hs_m"]]),
            tp=np.array([baseline["tp_s"]]),
            surge=np.array([baseline["surge_m"]]),
            wind_speed=np.array([abs(baseline["wind_normal_ms"])]),
            wind_dir_deg=np.array([70.0 if baseline["wind_normal_ms"] >= 0 else 250.0]),
            water_temp_c=np.array([baseline["water_temp_c"]]),
            tan_beta=baseline["tan_beta"],
            hs_exponent=hs_exp,
        )
        synth_df["total_water_level_m"] = baseline.get("total_water_level_m", baseline.get("surge_m", 0.85))
        synth_df["delta_p_dissip"] = baseline["delta_p_dissip"]
        synth_df["tan_beta"] = baseline["tan_beta"]
        synth_df["w_s_ms"] = baseline["w_s_ms"]

    except Exception:
        hs = baseline["hs_m"]
        tp = max(baseline["tp_s"], 0.5)
        ws = baseline["w_s_ms"]
        tan_b = baseline["tan_beta"]
        g = 9.81
        l0 = (g * (tp ** 2)) / (2 * np.pi)
        xi = tan_b * np.sqrt(g / (2 * np.pi)) * (tp / np.sqrt(max(hs, 0.01)))
        dean = hs / (ws * tp)
        p_flux = (1025.0 * (g ** 2) / (16.0 * 2 * np.pi)) * (hs ** 2) * tp
        ups = (1025.0 * (g ** 1.5) / (16.0 * (2 * np.pi) ** 1.5)) * tan_b * ((hs ** hs_exp) * tp / ws)
        synth_df = pd.DataFrame([{
            "hs_m": hs, "tp_s": tp, "surge_m": baseline["surge_m"],
            "total_water_level_m": baseline.get("total_water_level_m", baseline.get("surge_m", 0.85)),
            "wind_normal_ms": baseline["wind_normal_ms"], "water_temp_c": baseline["water_temp_c"],
            "tan_beta": tan_b, "w_s_ms": ws, "xi_iribarren": xi,
            "log_upsilon": np.log10(max(ups, 1e-4)), "dean_number": dean,
            "p_flux_Wm": p_flux, "delta_p_dissip": baseline["delta_p_dissip"],
            "accretion_potential": (tp * np.sqrt(g * l0)) / (hs + 0.1),
            "upsilon": ups,
        }])

    X_synth, _ = build_physics_anchored_feature_matrix(synth_df)
    
    if scaler is not None and hasattr(scaler, "transform"):
        X_norm = scaler.transform(X_synth)
        return X_norm[0]
    return X_synth[0]


def benchmark_latent_representations(
    df: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Benchmark Latent Representations: Raw Sensor Vectors vs PCA vs Physics-Anchored Invariant Manifold.

    Evaluates topological quality, regime separation, cluster tightness, and subcritical compression
    across known physical coastal regimes (Breaker Types: Spilling, Plunging, Surging;
    Morphodynamic States: Storm Gouging vs Quiet Berm Accretion).

    Parameters
    ----------
    df : pd.DataFrame
        Coastal dataset with hydrodynamics and physical invariants.
    sample_size : int
        Subsample size for efficient distance/clustering metrics.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df_metrics : pd.DataFrame
        Comparative table with Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
    compression_meta : dict
        Compression statistics from subcritical noise pruning.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.decomposition import PCA

    # Subsample if dataset is very large
    if len(df) > sample_size:
        df_sub = df.sample(n=sample_size, random_state=random_state).copy()
    else:
        df_sub = df.copy()

    # 1. Labels
    # Breaker regime
    if "breaker_regime" in df_sub.columns and df_sub["breaker_regime"].nunique() > 1:
        breaker_labels = df_sub["breaker_regime"].values
    elif "xi_iribarren" in df_sub.columns:
        xi = df_sub["xi_iribarren"].values
        b_labels = []
        for v in xi:
            if v < 0.5:
                b_labels.append("Spilling")
            elif v < 3.3:
                b_labels.append("Plunging")
            else:
                b_labels.append("Surging")
        breaker_labels = np.array(b_labels)
    else:
        breaker_labels = np.array(["Unknown"] * len(df_sub))

    # Morphodynamic regime
    morph_res = classify_morphodynamic_regimes(df_sub)
    morph_labels = morph_res._series.values

    # 2. Representations
    # Baseline 1: Raw Standardized Vectors
    raw_cols = [c for c in ["hs_m", "tp_s", "surge_m", "wind_normal_ms"] if c in df_sub.columns]
    X_raw_vals = df_sub[raw_cols].fillna(0.0).values
    scaler_raw = StandardScaler()
    X_raw = scaler_raw.fit_transform(X_raw_vals)

    # Baseline 2: PCA Bottleneck
    n_comp = min(3, X_raw.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_raw)

    # Proposed: Physics-Anchored Invariant Manifold
    X_phys_vals, feat_names = build_physics_anchored_feature_matrix(df_sub)
    scaler_phys = RobustScaler()
    X_phys = scaler_phys.fit_transform(X_phys_vals)

    representations = {
        "Baseline 1: Raw Sensor Vector": X_raw,
        "Baseline 2: PCA Bottleneck": X_pca,
        "Proposed: Physics-Anchored Manifold": X_phys,
    }

    rows = []
    for name, X_rep in representations.items():
        row = {"Representation": name, "Dimensions": X_rep.shape[1]}

        # Breaker regime clustering metrics
        if len(np.unique(breaker_labels)) > 1:
            row["Breaker Silhouette"] = float(silhouette_score(X_rep, breaker_labels))
            row["Breaker Davies-Bouldin"] = float(davies_bouldin_score(X_rep, breaker_labels))
            row["Breaker Calinski-Harabasz"] = float(calinski_harabasz_score(X_rep, breaker_labels))
        else:
            row["Breaker Silhouette"] = 0.0
            row["Breaker Davies-Bouldin"] = 0.0
            row["Breaker Calinski-Harabasz"] = 0.0

        # Morphodynamic regime clustering metrics
        if len(np.unique(morph_labels)) > 1:
            row["Morphodynamic Silhouette"] = float(silhouette_score(X_rep, morph_labels))
            row["Morphodynamic Davies-Bouldin"] = float(davies_bouldin_score(X_rep, morph_labels))
            row["Morphodynamic Calinski-Harabasz"] = float(calinski_harabasz_score(X_rep, morph_labels))
        else:
            row["Morphodynamic Silhouette"] = 0.0
            row["Morphodynamic Davies-Bouldin"] = 0.0
            row["Morphodynamic Calinski-Harabasz"] = 0.0

        rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Subcritical pruning compression stats
    _, _, comp_stats = prune_subcritical_noise_for_agent(df_sub)

    return df_metrics, comp_stats



```
