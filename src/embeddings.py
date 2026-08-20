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
        if "p_flux_Wm" in df.columns:
            feature_dict["Wave_Power_P"] = np.log10(np.maximum(df["p_flux_Wm"].values, 1e-1))

    feature_names = list(feature_dict.keys())
    X = np.column_stack([feature_dict[k] for k in feature_names])
    return X, feature_names


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
        "scaler_type": scaler_type,
        "n_features": len(feature_names),
    }
    
    if use_pca_whitening:
        pca = PCA(whiten=True)
        embeddings = pca.fit_transform(X_norm)
        meta["explained_variance_ratio"] = pca.explained_variance_ratio_
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
