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
        if "surge_m" in df.columns:
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


def classify_morphodynamic_regimes(
    df: pd.DataFrame,
    omega_crit: float = 1.75,
) -> Tuple[pd.Series, np.ndarray]:
    """Classify coastal states into 3 physically distinct morphodynamic clusters.

    Clusters:
    1. High-Energy Storm Gouging: Omega > 2.0, high Upsilon, high wave energy dissipation.
    2. Quiet Berm Accretion: Omega <= 1.5, low steepness swell, reflective/surging swash.
    3. Bar-Breach Hotspots: Moderate Upsilon, but low Delta P_dissip (unshielded incident energy).

    Returns
    -------
    regime_labels : pd.Series
        Categorical labels for each observation.
    omega_ratio : np.ndarray
        Normalized Dean dimensionless fall velocity (Omega / omega_crit).
    """
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
            
    return pd.Series(labels, index=df.index, name="morphodynamic_regime"), omega_ratio



def compute_latent_embeddings(
    df: pd.DataFrame,
    scaler_type: str = "robust",
    use_pca_whitening: bool = False,
) -> Tuple[np.ndarray, list[str], Dict[str, Any]]:
    """Generate normalized latent embedding vectors using objective, data-driven scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Input coastal forcing dataset.
    scaler_type : str
        'robust' (median/IQR) or 'standard' (z-score) scaling.
    use_pca_whitening : bool
        Whether to apply PCA whitening to decorrelate dimensions.

    Returns
    -------
    embeddings : np.ndarray
        Normalized latent embeddings (N x D).
    feature_names : list[str]
        Feature names.
    meta : dict
        Variance explained and scaler metadata.
    """
    X_raw, feature_names = build_physics_anchored_feature_matrix(df)
    
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X_norm = scaler.fit_transform(X_raw)
    
    meta = {
        "scaler": scaler,
        "scaler_type": scaler_type,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    
    if use_pca_whitening:
        pca = PCA(whiten=True)
        embeddings = pca.fit_transform(X_norm)
        meta["explained_variance_ratio"] = pca.explained_variance_ratio_
        meta["pca"] = pca
    else:
        embeddings = X_norm

    return embeddings, feature_names, meta


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
) -> np.ndarray:
    """Project high-dimensional physics embeddings to 2D manifold via UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )
    return reducer.fit_transform(embeddings)


def evaluate_breaker_regime_clustering(
    coords_2d: np.ndarray,
    breaker_regimes: np.ndarray,
) -> Dict[str, float]:
    """Compute quantitative cluster separation metrics (Silhouette score) for breaker regimes."""
    labels = np.asarray(breaker_regimes)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return {"silhouette_score": 0.0}
        
    score = float(silhouette_score(coords_2d, labels))
    return {
        "silhouette_score": score,
        "n_clusters": len(unique_labels),
    }


def compute_regime_centroids(
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Compute canonical state reference vectors for breaker regimes and extreme events."""
    centroids = {}

    if "breaker_regime" in df.columns:
        for regime in df["breaker_regime"].dropna().unique():
            idx = np.where(df["breaker_regime"].values == regime)[0]
            if len(idx) > 0:
                centroids[f"Breaker_{regime}"] = np.mean(embeddings[idx], axis=0)

    if "upsilon" in df.columns:
        ian_idx = int(np.argmax(df["upsilon"].values))
        centroids["State_ExtremeStormPeak"] = embeddings[ian_idx]
        calm_idx = int(np.argmin(df["upsilon"].values))
        centroids["State_CalmFairWeather"] = embeddings[calm_idx]

    if "tp_s" in df.columns and "hs_m" in df.columns:
        swell_score = (df["tp_s"].values / (df["hs_m"].values + 0.1)) * (
            1.0 / (np.maximum(df["surge_m"].values if "surge_m" in df.columns else 0.0, 0.01) + 0.1)
        )
        swell_idx = int(np.argmax(swell_score))
        centroids["State_LongPeriodSwell"] = embeddings[swell_idx]

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
        "wind": "wind_normal_ms", "wind_normal_ms": "wind_normal_ms", "wind_wa": "wind_normal_ms",
        "temp": "water_temp_c", "water_temp_c": "water_temp_c",
        "slope": "tan_beta", "tan_beta": "tan_beta",
        "ws": "w_s_ms", "w_s_ms": "w_s_ms",
    }
    
    for k, v in target_params.items():
        if v is not None:
            norm_key = param_map.get(k.lower(), k)
            baseline[norm_key] = float(v)

    hs_exp = float(target_params.get("hs_exponent", 2.5))
    try:
        from src.physics import compute_unified_coastal_vulnerability_index
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

