"""Physics-Inspired Retrieval-Augmented Generation (Physics-RAG) Engine.

Enables natural language querying of multi-modal coastal datasets and latent embeddings.
Combines semantic query parsing, physics-anchored vector similarity search, physical regime
filtering, and automated physics context generation.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from src.embeddings import (
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
        compute_regime_centroids,
    )
except ImportError:
    from src.embeddings import (
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
    )

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
            surge_col = df["surge_m"].values if "surge_m" in df.columns else np.zeros(len(df))
            swell_score = (df["tp_s"].values / (df["hs_m"].values + 0.1)) * (
                1.0 / (np.maximum(surge_col, 0.01) + 0.1)
            )
            swell_idx = int(np.argmax(swell_score))
            centroids["State_LongPeriodSwell"] = embeddings[swell_idx]

        return centroids


class PhysicsQueryParser:
    """Parses natural language requests into physical criteria and target embedding profiles."""

    REGIME_KEYWORDS = {
        "spilling": {"breaker_regime": "Spilling", "xi_max": 0.5},
        "plunging": {"breaker_regime": "Plunging", "xi_min": 0.5, "xi_max": 3.3},
        "surging": {"breaker_regime": "Surging", "xi_min": 3.3},
        "collapsing": {"breaker_regime": "Surging", "xi_min": 3.3},
        "dissipative": {"breaker_regime": "Spilling", "xi_max": 0.5},
        "reflective": {"breaker_regime": "Surging", "xi_min": 3.3},
    }

    MORPHO_KEYWORDS = {
        "erosion": {"morphology": "Erosive", "delta_x_max": 0.0},
        "erosive": {"morphology": "Erosive", "delta_x_max": 0.0},
        "beach cut": {"morphology": "Erosive", "delta_x_max": -0.1},
        "accretion": {"morphology": "Accretive", "delta_x_min": 0.0},
        "accretive": {"morphology": "Accretive", "delta_x_min": 0.0},
        "recovery": {"morphology": "Accretive", "delta_x_min": 0.0},
        "berm building": {"morphology": "Accretive", "delta_x_min": 0.0},
    }

    EVENT_KEYWORDS = {
        "hurricane ian": {"event": "hurricane_ian", "high_vuln": True},
        "ian peak": {"event": "hurricane_ian_peak", "high_vuln": True},
        "storm peak": {"high_vuln": True},
        "extreme": {"high_vuln": True},
        "surge": {"focus": "surge"},
        "swell": {"focus": "swell"},
        "onshore wind": {"focus": "onshore_wind", "wind_normal_min": 0.0},
        "gale": {"focus": "wind_gale"},
        "calm": {"focus": "calm", "low_vuln": True},
        "fair weather": {"focus": "calm", "low_vuln": True},
    }

    @classmethod
    def parse(cls, query_text: str) -> Dict[str, Any]:
        """Parse natural language query into physical search parameters.

        Parameters
        ----------
        query_text : str
            User query (e.g. 'Show me all the Spilling breakers', 'Find peak storm surge erosion').

        Returns
        -------
        parsed : dict
            Parsed physical constraints, target regimes, and intent.
        """
        text = query_text.lower().strip()
        parsed: Dict[str, Any] = {
            "raw_query": query_text,
            "breaker_regime": None,
            "morphology": None,
            "high_vuln": False,
            "low_vuln": False,
            "focus": None,
            "filters": {},
            "target_state": None,
        }

        # Check breaker regime keywords
        for kw, specs in cls.REGIME_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", text):
                parsed["breaker_regime"] = specs["breaker_regime"]
                if "xi_max" in specs:
                    parsed["filters"]["xi_max"] = specs["xi_max"]
                if "xi_min" in specs:
                    parsed["filters"]["xi_min"] = specs["xi_min"]
                break

        # Check morphodynamic keywords
        for kw, specs in cls.MORPHO_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", text):
                parsed["morphology"] = specs["morphology"]
                if "delta_x_max" in specs:
                    parsed["filters"]["delta_x_max"] = specs["delta_x_max"]
                if "delta_x_min" in specs:
                    parsed["filters"]["delta_x_min"] = specs["delta_x_min"]
                break

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

        # Infer target reference state
        if parsed["high_vuln"] or "peak" in text or "storm" in text or "ian" in text:
            parsed["target_state"] = "State_ExtremeStormPeak"
        elif parsed["low_vuln"] or "calm" in text or "fair" in text:
            parsed["target_state"] = "State_CalmFairWeather"
        elif parsed["focus"] == "swell" or "swell" in text:
            parsed["target_state"] = "State_LongPeriodSwell"
        elif parsed["breaker_regime"]:
            parsed["target_state"] = f"Breaker_{parsed['breaker_regime']}"

        return parsed


class PhysicsRAGResult:
    """Structured container for Physics-RAG search results with explainability."""

    def __init__(
        self,
        query: str,
        matches_df: pd.DataFrame,
        similarities: np.ndarray,
        explanation: str,
        parsed_criteria: Dict[str, Any],
    ):
        self.query = query
        self.matches_df = matches_df
        self.similarities = similarities
        self.explanation = explanation
        self.parsed_criteria = parsed_criteria

    def __repr__(self) -> str:
        return f"<PhysicsRAGResult for query='{self.query}', n_matches={len(self.matches_df)}>"

    def summary(self) -> str:
        """Return formatted text summary of the retrieved physics states."""
        lines = [
            f"=== Physics-Inspired RAG Results ===",
            f"Query: \"{self.query}\"",
            f"Retrieved Instances: {len(self.matches_df)}",
            "",
            f"Physical Explanation:",
            f"{self.explanation}",
            "",
            "Top Matching Observations:",
        ]
        
        display_cols = [
            c for c in [
                "timestamp", "hs_m", "tp_s", "surge_m", "wind_normal_ms",
                "xi_iribarren", "breaker_regime", "upsilon", "delta_x_m", "similarity"
            ] if c in self.matches_df.columns
        ]
        
        table_str = self.matches_df[display_cols].to_string(index=False)
        lines.append(table_str)
        return "\n".join(lines)


class PhysicsRAG:
    """Physics-Inspired Retrieval-Augmented Generation Engine for Coastal Systems."""

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        feature_names: List[str],
        scaler_meta: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the PhysicsRAG engine.

        Parameters
        ----------
        df : pd.DataFrame
            Underlying coastal hydrodynamic & morphodynamic dataset.
        embeddings : np.ndarray
            Physics-anchored latent vectors (N x D).
        feature_names : list[str]
            Feature dimension names.
        scaler_meta : dict, optional
            Metadata containing fitted scalers and transforms.
        """
        self.df = df.copy()
        self.embeddings = embeddings
        self.feature_names = feature_names
        self.scaler_meta = scaler_meta or {}
        self.centroids = compute_regime_centroids(self.embeddings, self.df)

    @classmethod
    def from_dataset(
        cls,
        df: pd.DataFrame,
        scaler_type: str = "robust",
    ) -> "PhysicsRAG":
        """Factory method to construct PhysicsRAG directly from a DataFrame."""
        embeddings, feature_names, meta = compute_latent_embeddings(df, scaler_type=scaler_type)
        return cls(df=df, embeddings=embeddings, feature_names=feature_names, scaler_meta=meta)

    def _build_query_vector(self, parsed: Dict[str, Any]) -> np.ndarray:
        """Construct reference latent query vector from parsed intent."""
        target_state = parsed.get("target_state")
        
        # 1. Use precomputed regime centroid if available
        if target_state and target_state in self.centroids:
            return self.centroids[target_state]

        # 2. Check if a breaker regime centroid is requested
        if parsed.get("breaker_regime"):
            key = f"Breaker_{parsed['breaker_regime']}"
            if key in self.centroids:
                return self.centroids[key]

        # 3. Default to mean dataset embedding
        return np.mean(self.embeddings, axis=0)

    def _apply_hard_and_soft_filters(
        self,
        parsed: Dict[str, Any],
    ) -> np.ndarray:
        """Generate boolean mask for candidates satisfying physical constraints."""
        mask = np.ones(len(self.df), dtype=bool)
        filters = parsed.get("filters", {})

        # Breaker regime exact match
        if parsed.get("breaker_regime") and "breaker_regime" in self.df.columns:
            mask = mask & (self.df["breaker_regime"].values == parsed["breaker_regime"])

        # Iribarren limits
        if "xi_min" in filters and "xi_iribarren" in self.df.columns:
            mask = mask & (self.df["xi_iribarren"].values >= filters["xi_min"])
        if "xi_max" in filters and "xi_iribarren" in self.df.columns:
            mask = mask & (self.df["xi_iribarren"].values <= filters["xi_max"])

        # Morphology limits
        if "delta_x_min" in filters and "delta_x_m" in self.df.columns:
            mask = mask & (self.df["delta_x_m"].values >= filters["delta_x_min"])
        if "delta_x_max" in filters and "delta_x_m" in self.df.columns:
            mask = mask & (self.df["delta_x_m"].values <= filters["delta_x_max"])

        # Shore-normal wind limits
        if "wind_normal_min" in filters and "wind_normal_ms" in self.df.columns:
            mask = mask & (self.df["wind_normal_ms"].values >= filters["wind_normal_min"])

        # High / Low vulnerability percentiles
        if parsed.get("high_vuln") and "upsilon" in self.df.columns:
            p75 = np.percentile(self.df["upsilon"].values, 75)
            mask = mask & (self.df["upsilon"].values >= p75)
        elif parsed.get("low_vuln") and "upsilon" in self.df.columns:
            p25 = np.percentile(self.df["upsilon"].values, 25)
            mask = mask & (self.df["upsilon"].values <= p25)

        # Fallback if filters are overly restrictive (return all if mask is empty)
        if not np.any(mask):
            mask = np.ones(len(self.df), dtype=bool)

        return mask

    def _generate_explanation(
        self,
        parsed: Dict[str, Any],
        top_df: pd.DataFrame,
    ) -> str:
        """Synthesize physical explainability narrative for the query results."""
        if top_df.empty:
            return "No matching coastal state instances found for the specified physical parameters."

        regime = parsed.get("breaker_regime")
        avg_hs = top_df["hs_m"].mean() if "hs_m" in top_df.columns else 0.0
        avg_tp = top_df["tp_s"].mean() if "tp_s" in top_df.columns else 0.0
        avg_xi = top_df["xi_iribarren"].mean() if "xi_iribarren" in top_df.columns else 0.0
        avg_upsilon = top_df["upsilon"].mean() if "upsilon" in top_df.columns else 0.0
        avg_surge = top_df["surge_m"].mean() if "surge_m" in top_df.columns else 0.0
        avg_dx = top_df["delta_x_m"].mean() if "delta_x_m" in top_df.columns else 0.0

        parts = []

        if regime == "Spilling":
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
    ) -> PhysicsRAGResult:
        """Execute a natural language query against the coastal dataset & latent embeddings.

        Parameters
        ----------
        query_text : str
            Natural language query (e.g. 'Show me all the Spilling breakers',
            'Find peak storm surge erosion during Hurricane Ian',
            'Show calm reflective swell conditions').
        top_k : int
            Maximum number of instances to return.
        min_similarity : float
            Cosine similarity cutoff.

        Returns
        -------
        PhysicsRAGResult
            Object containing matching DataFrame, similarities, explanation, and criteria.
        """
        # 1. Parse natural language into physical intent
        parsed = PhysicsQueryParser.parse(query_text)

        # 2. Build target query embedding vector
        query_vec = self._build_query_vector(parsed)

        # 3. Compute vector similarities across latent embedding space
        sims = cosine_similarity(query_vec.reshape(1, -1), self.embeddings)[0]

        # 4. Filter candidates
        filter_mask = self._apply_hard_and_soft_filters(parsed)
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

        # 7. Generate physical explanation
        explanation = self._generate_explanation(parsed, top_df)

        return PhysicsRAGResult(
            query=query_text,
            matches_df=top_df,
            similarities=selected_sims,
            explanation=explanation,
            parsed_criteria=parsed,
        )
