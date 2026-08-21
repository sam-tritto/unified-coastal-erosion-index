"""Physics-Inspired Retrieval-Augmented Generation (Physics-RAG) Engine with Gemini Flash.

Enables natural language querying of multi-modal coastal datasets and latent embeddings.
Combines Gemini-powered structured query parsing, physics-anchored vector similarity search,
dynamic synthetic vector projections, physical regime filtering, and grounded oceanographic narrative synthesis.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from src.embeddings import (
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
        compute_regime_centroids,
        synthesize_target_physics_vector,
    )
except ImportError:
    from embeddings import (  # type: ignore
        build_physics_anchored_feature_matrix,
        compute_latent_embeddings,
        compute_regime_centroids,
        synthesize_target_physics_vector,
    )


class PhysicsQuerySpec(BaseModel):
    """Structured physical constraints and target embedding specifications."""

    raw_query: str = Field(description="The original natural language query string.")
    intent_summary: str = Field(
        default="",
        description="Summary of physical intent and oceanographic phenomena requested."
    )
    breaker_regime: Optional[Literal["Spilling", "Plunging", "Surging"]] = Field(
        default=None,
        description="Target breaking wave regime (Spilling for ξ < 0.5, Plunging for 0.5 <= ξ < 3.3, Surging for ξ >= 3.3)."
    )
    morphology: Optional[Literal["Erosive", "Accretive", "Equilibrium"]] = Field(
        default=None,
        description="Morphodynamic response (Erosive: shoreline recession, Accretive: berm buildup)."
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
    target_synthetic_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target numerical values to synthesize an explicit query vector (e.g. {'hs_m': 4.5, 'tp_s': 14.0, 'surge_m': 1.0})."
    )
    target_state: Optional[str] = Field(
        default=None,
        description="Canonical reference state name (State_ExtremeStormPeak, State_CalmFairWeather, State_LongPeriodSwell, Breaker_Spilling, etc.)"
    )
    focus: Optional[str] = Field(default=None, description="Focal phenomenon (e.g. 'surge', 'swell', 'wind', 'calm', 'storm').")


class PhysicsQueryParser:
    """Deterministic rule-based parser for physical criteria and target embedding profiles."""

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
        "scarping": {"morphology": "Erosive", "delta_x_max": -0.1},
        "accretion": {"morphology": "Accretive", "delta_x_min": 0.0},
        "accretive": {"morphology": "Accretive", "delta_x_min": 0.0},
        "recovery": {"morphology": "Accretive", "delta_x_min": 0.0},
        "berm building": {"morphology": "Accretive", "delta_x_min": 0.0},
        "berm": {"morphology": "Accretive", "delta_x_min": 0.0},
    }

    EVENT_KEYWORDS = {
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
    }

    @classmethod
    def parse(cls, query_text: str) -> Dict[str, Any]:
        """Parse natural language query into physical search parameters dict."""
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
            "target_synthetic_values": None,
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

        if synth_vals:
            parsed["target_synthetic_values"] = synth_vals

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


class GeminiPhysicsQueryParser:
    """Uses Gemini Flash (or configured model) with structured output to parse natural language queries into PhysicsQuerySpec."""

    SYSTEM_PROMPT = """You are an expert coastal oceanographer and hydrodynamic search specialist.
Your task is to parse a user's natural language query about coastal conditions (waves, surge, morphology, breaker regimes, storms like Hurricane Ian, berm accretion, wind forcing) into a strictly typed JSON schema according to PhysicsQuerySpec.

Guidelines:
- If the query mentions 'spilling' or 'dissipative', set breaker_regime="Spilling" and xi_max=0.5.
- If the query mentions 'plunging' or 'curling', set breaker_regime="Plunging", xi_min=0.5, xi_max=3.3.
- If the query mentions 'surging', 'collapsing', or 'reflective', set breaker_regime="Surging" and xi_min=3.3.
- If the query mentions 'erosion', 'beach cut', 'scarping', set morphology="Erosive", delta_x_max=0.0.
- If the query mentions 'accretion', 'recovery', 'berm building', set morphology="Accretive", delta_x_min=0.0.
- If the query mentions 'Hurricane Ian', set target_state="State_ExtremeStormPeak", min_upsilon_percentile=75.0, date_start="2022-09-27", date_end="2022-10-02".
- If the query mentions 'storm peak', 'extreme storm', 'major storm', set min_upsilon_percentile=75.0, target_state="State_ExtremeStormPeak".
- If the query mentions 'calm', 'fair weather', 'mild', set max_upsilon_percentile=25.0, target_state="State_CalmFairWeather".
- If the query specifies numerical values (e.g. 'waves over 4m', '14s period', 'surge > 0.8m'), populate hs_min, tp_min, surge_min, and populate target_synthetic_values with these numbers.
- Provide a clear intent_summary describing the physical oceanographic phenomenon.
"""

    @classmethod
    def parse(
        cls,
        query_text: str,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> PhysicsQuerySpec:
        """Parse natural language query using Gemini Flash structured output with automatic fallback."""
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

            if client is not None:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"Parse the following coastal oceanography query into the PhysicsQuerySpec schema:\nQuery: {query_text}",
                    config={
                        "system_instruction": cls.SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": PhysicsQuerySpec,
                        "temperature": 0.0,
                    },
                )
                if response and response.text:
                    import json
                    parsed_json = json.loads(response.text)
                    parsed_json["raw_query"] = query_text
                    return PhysicsQuerySpec(**parsed_json)
        except Exception:
            pass

        return cls._fallback_parse(query_text)

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
            target_state=parsed_dict.get("target_state"),
            focus=parsed_dict.get("focus"),
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
                "xi_iribarren", "breaker_regime", "upsilon", "delta_x_m", "similarity"
            ] if c in top_df.columns
        ]
        table_str = top_df[display_cols].to_string(index=False)

        prompt = f"""You are an authoritative senior coastal oceanographer reviewing hydrodynamic search results from the USACE FRF Duck dataset.
Synthesize a concise, 2-to-3 paragraph physical diagnostic narrative for the query: "{query_text}" based strictly on the retrieved observations below.

Query Intent: {spec.intent_summary or spec.raw_query}
Target Regime: {spec.breaker_regime or 'General Hydrodynamics'}

Retrieved Observation Data:
{table_str}

Summary Statistics:
- Mean Significant Wave Height (Hs): {top_df['hs_m'].mean() if 'hs_m' in top_df.columns else 0:.2f} m
- Mean Peak Period (Tp): {top_df['tp_s'].mean() if 'tp_s' in top_df.columns else 0:.1f} s
- Mean Surge Residual (S): {top_df['surge_m'].mean() if 'surge_m' in top_df.columns else 0:.2f} m
- Mean Iribarren Number (ξ): {top_df['xi_iribarren'].mean() if 'xi_iribarren' in top_df.columns else 0:.2f}
- Mean Unified Index (Υ): {top_df['upsilon'].mean() if 'upsilon' in top_df.columns else 0:.1e}
- Mean Shoreline Displacement (ΔX): {top_df['delta_x_m'].mean() if 'delta_x_m' in top_df.columns else 0:.2f} m

Instructions:
1. Explain the wave dissipation kinetics and surf similarity / breaker regime (Iribarren number ξ).
2. Connect wave power flux and total water level (surge + runup) to morphodynamic response (beach cut vs. accretion).
3. Cite the quantitative numbers directly from the data. Do NOT speculate or hallucinate outside the retrieved records. Keep tone rigorous, objective, and professional.
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
    except Exception:
        pass
    return None


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
    ):
        self.query = query
        self.matches_df = matches_df
        self.similarities = similarities
        self.explanation = explanation
        self.parsed_criteria = parsed_criteria
        self.llm_explanation = llm_explanation
        self.query_vector = query_vector
        self.metrics = metrics or {}

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
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
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
        model_name : str
            Gemini model identifier (default: 'gemini-2.5-flash').
        api_key : str, optional
            Gemini API key (defaults to GEMINI_API_KEY / GOOGLE_API_KEY in environment).
        """
        self.df = df.copy()
        self.embeddings = embeddings
        self.feature_names = feature_names
        self.scaler_meta = scaler_meta or {}
        self.model_name = model_name
        self.api_key = api_key
        self.scaler = self.scaler_meta.get("scaler")
        if self.scaler is None and hasattr(self.df, "columns"):
            try:
                X_raw, _ = build_physics_anchored_feature_matrix(self.df)
                scaler_type = self.scaler_meta.get("scaler_type", "robust")
                scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
                scaler.fit(X_raw)
                self.scaler = scaler
                self.scaler_meta["scaler"] = scaler
            except Exception:
                pass
        self.centroids = compute_regime_centroids(self.embeddings, self.df)

    @classmethod
    def from_dataset(
        cls,
        df: pd.DataFrame,
        scaler_type: str = "robust",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
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
        )

    def _build_query_vector(self, spec: PhysicsQuerySpec) -> np.ndarray:
        """Construct reference latent query vector from structured intent."""
        # 1. Synthesize explicit vector from target values if user specified explicit numbers
        if spec.target_synthetic_values:
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

        # 3. Check if breaker regime centroid is requested
        if spec.breaker_regime:
            key = f"Breaker_{spec.breaker_regime}"
            if key in self.centroids:
                return self.centroids[key]

        # 4. Default to mean dataset embedding
        return np.mean(self.embeddings, axis=0)

    def _apply_hard_and_soft_filters(self, spec: PhysicsQuerySpec) -> np.ndarray:
        """Generate boolean mask for candidates satisfying physical constraints."""
        mask = np.ones(len(self.df), dtype=bool)

        # Breaker regime exact match
        if spec.breaker_regime and "breaker_regime" in self.df.columns:
            mask = mask & (self.df["breaker_regime"].values == spec.breaker_regime)

        # Wave height limits
        if spec.hs_min is not None and "hs_m" in self.df.columns:
            mask = mask & (self.df["hs_m"].values >= spec.hs_min)
        if spec.hs_max is not None and "hs_m" in self.df.columns:
            mask = mask & (self.df["hs_m"].values <= spec.hs_max)

        # Wave period limits
        if spec.tp_min is not None and "tp_s" in self.df.columns:
            mask = mask & (self.df["tp_s"].values >= spec.tp_min)
        if spec.tp_max is not None and "tp_s" in self.df.columns:
            mask = mask & (self.df["tp_s"].values <= spec.tp_max)

        # Surge limits
        if spec.surge_min is not None and "surge_m" in self.df.columns:
            mask = mask & (self.df["surge_m"].values >= spec.surge_min)
        if spec.surge_max is not None and "surge_m" in self.df.columns:
            mask = mask & (self.df["surge_m"].values <= spec.surge_max)

        # Wind limits
        if spec.wind_normal_min is not None and "wind_normal_ms" in self.df.columns:
            mask = mask & (self.df["wind_normal_ms"].values >= spec.wind_normal_min)
        if spec.wind_normal_max is not None and "wind_normal_ms" in self.df.columns:
            mask = mask & (self.df["wind_normal_ms"].values <= spec.wind_normal_max)

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

        # Date window filter
        if (spec.date_start or spec.date_end) and "timestamp" in self.df.columns:
            try:
                ts = pd.to_datetime(self.df["timestamp"])
                if spec.date_start:
                    mask = mask & (ts >= pd.to_datetime(spec.date_start))
                if spec.date_end:
                    mask = mask & (ts <= pd.to_datetime(spec.date_end))
            except Exception:
                pass

        # Fallback if filters are overly restrictive (return all candidates)
        if not np.any(mask):
            mask = np.ones(len(self.df), dtype=bool)

        return mask

    def _generate_explanation(
        self,
        spec: PhysicsQuerySpec,
        top_df: pd.DataFrame,
    ) -> str:
        """Synthesize deterministic physical explainability narrative for query results."""
        if top_df.empty:
            return "No matching coastal state instances found for the specified physical parameters."

        regime = spec.breaker_regime
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
        use_llm_parser: bool = True,
        use_llm_synthesis: bool = True,
    ) -> PhysicsRAGResult:
        """Execute a natural language query against coastal dataset & latent embeddings.

        Parameters
        ----------
        query_text : str
            Natural language query (e.g. 'Show spilling breakers during storm',
            'Find waves over 4m with long period swell and surge > 0.8m',
            'Show calm fair-weather conditions with beach accretion').
        top_k : int
            Maximum number of instances to return.
        min_similarity : float
            Cosine similarity cutoff.
        use_llm_parser : bool
            Whether to attempt Gemini Flash structured query parsing.
        use_llm_synthesis : bool
            Whether to attempt Gemini Flash narrative synthesis.

        Returns
        -------
        PhysicsRAGResult
            Object containing matching DataFrame, similarities, explanation, and criteria.
        """
        # 1. Parse natural language into structured physical intent
        if use_llm_parser:
            spec = GeminiPhysicsQueryParser.parse(
                query_text=query_text,
                model_name=self.model_name,
                api_key=self.api_key,
            )
        else:
            spec = GeminiPhysicsQueryParser._fallback_parse(query_text)

        # 2. Build target query embedding vector
        query_vec = self._build_query_vector(spec)

        # 3. Compute vector similarities across latent embedding space
        sims = cosine_similarity(query_vec.reshape(1, -1), self.embeddings)[0]

        # 4. Filter candidates
        filter_mask = self._apply_hard_and_soft_filters(spec)
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
        )


# ==============================================================================
# Standard Physics-RAG Evaluation Suite & Metrics Engine
# ==============================================================================

DEFAULT_PHYSICS_BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "Show me all the Spilling breakers",
        "category": "Breaker Regime",
        "target_regime": "Spilling",
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Spilling" or float(row.get("xi_iribarren", 1.0)) < 0.5,
        "description": "Dissipative wave breaking regime with surf similarity ξ < 0.5",
    },
    {
        "query": "Find plunging curling waves",
        "category": "Breaker Regime",
        "target_regime": "Plunging",
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Plunging" or (0.5 <= float(row.get("xi_iribarren", 1.0)) < 3.3),
        "description": "Intermediate wave breaking regime with 0.5 <= ξ < 3.3",
    },
    {
        "query": "Show reflective surging shorebreak",
        "category": "Breaker Regime",
        "target_regime": "Surging",
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 3.3,
        "description": "Reflective wave surging regime with ξ >= 3.3",
    },
    {
        "query": "Find high energy storm waves over 3m",
        "category": "Wave Energetics",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 3.0,
        "description": "Severe sea states with significant wave height Hs >= 3.0m",
    },
    {
        "query": "Show long period swell with period over 12s",
        "category": "Wave Energetics",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 12.0,
        "description": "Long-period swell wave fields with peak period Tp >= 12.0s",
    },
    {
        "query": "Show severe beach erosion and shoreline cut",
        "category": "Morphodynamics",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("delta_x_m", row.get("shoreline_pos_obs_m", 0.0))) < float(df["delta_x_m"].median() if "delta_x_m" in df.columns else 0.0),
        "description": "Erosive morphodynamic response with negative shoreline displacement ΔX < 0",
    },
    {
        "query": "Find berm accretion and fair weather recovery",
        "category": "Morphodynamics",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("delta_x_m", row.get("shoreline_pos_obs_m", 0.0))) > float(df["delta_x_m"].median() if "delta_x_m" in df.columns else 0.0),
        "description": "Accretive morphodynamic state with positive shoreline recovery ΔX > 0",
    },
    {
        "query": "Find positive storm surge elevation above 0.5m",
        "category": "Total Water Level",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("surge_m", 0.0)) >= 0.5,
        "description": "Elevated storm surge elevation with Surge >= 0.5m",
    },
    {
        "query": "Show strong onshore wind conditions",
        "category": "Atmospheric Forcing",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("wind_normal_ms", 0.0)) >= 4.0,
        "description": "Onshore wind forcing with shore-normal wind speed >= 4.0 m/s",
    },
    {
        "query": "Find extreme vulnerability storm peak states",
        "category": "Vulnerability Index",
        "target_regime": None,
        "condition_fn": lambda row, df: float(row.get("upsilon", 0.0)) >= float(np.percentile(df["upsilon"].values, 70.0)),
        "description": "Top tier Unified Coastal Vulnerability Index Υ (>= 70th percentile)",
    },
]


def evaluate_retrieval_query(
    rag: PhysicsRAG,
    query_spec_or_text: Union[str, Dict[str, Any]],
    condition_fn: Optional[Any] = None,
    target_regime: Optional[str] = None,
    k: int = 5,
    use_llm_parser: bool = False,
) -> Dict[str, Any]:
    """Calculate Precision@k, Regime Purity@k, Recall@k, and Reciprocal Rank for a query.

    Parameters
    ----------
    rag : PhysicsRAG
        Instantiated PhysicsRAG engine.
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
    else:
        query_str = str(query_spec_or_text)
        cond = condition_fn
        regime = target_regime
        category = "Custom"

    # Default condition if target_regime provided
    if cond is None and regime:
        cond = lambda row, df: str(row.get("breaker_regime", "")) == regime

    # Total ground-truth support in dataset
    df = rag.df
    total_support = 0
    if cond is not None:
        try:
            total_support = int(sum(bool(cond(df.iloc[i], df)) for i in range(len(df))))
        except Exception:
            total_support = len(df)

    # Execute RAG query
    rag_res = rag.query(
        query_text=query_str,
        top_k=k,
        use_llm_parser=use_llm_parser,
        use_llm_synthesis=False,
    )

    matches_df = rag_res.matches_df
    n_retrieved = len(matches_df)
    eff_k = max(1, n_retrieved)

    # Calculate binary relevance mask for retrieved instances
    relevance_flags: List[bool] = []
    regime_flags: List[bool] = []

    for i in range(n_retrieved):
        row = matches_df.iloc[i]
        is_rel = bool(cond(row, df)) if cond is not None else True
        relevance_flags.append(is_rel)

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

    # Recall@k (relative to min(k, total_support))
    if total_support > 0:
        recall_at_k = float(n_correct / min(eff_k, total_support))
    else:
        recall_at_k = 1.0 if n_correct > 0 else 0.0

    # Reciprocal Rank (RR)
    reciprocal_rank = 0.0
    for idx, is_rel in enumerate(relevance_flags):
        if is_rel:
            reciprocal_rank = 1.0 / float(idx + 1)
            break

    mean_sim = float(np.mean(rag_res.similarities)) if len(rag_res.similarities) > 0 else 0.0

    return {
        "query": query_str,
        "category": category,
        "target_regime": regime,
        "k": k,
        "precision_at_k": precision_at_k,
        "regime_purity": regime_purity,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "mean_similarity": mean_sim,
        "n_retrieved": n_retrieved,
        "n_correct": n_correct,
        "total_dataset_support": total_support,
        "retrieved_df": matches_df,
    }


def evaluate_rag_benchmark_suite(
    rag: PhysicsRAG,
    benchmark_queries: Optional[List[Dict[str, Any]]] = None,
    k_list: List[int] = [1, 3, 5, 10],
    use_llm_parser: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Execute multi-k benchmark evaluation suite across diverse coastal query categories.

    Parameters
    ----------
    rag : PhysicsRAG
        Instantiated PhysicsRAG engine.
    benchmark_queries : list[dict], optional
        Custom list of query dicts (defaults to DEFAULT_PHYSICS_BENCHMARK_QUERIES).
    k_list : list[int]
        List of k rank cutoffs (e.g. [1, 3, 5, 10]).
    use_llm_parser : bool
        Whether to use Gemini structured output parser.

    Returns
    -------
    df_results : pd.DataFrame
        Detailed per-query evaluation table.
    summary_metrics : dict
        Macro-averaged benchmark summary metrics (Mean P@1, Mean P@5, Mean Regime Purity, Mean MRR).
    """
    queries = benchmark_queries or DEFAULT_PHYSICS_BENCHMARK_QUERIES
    rows: List[Dict[str, Any]] = []

    p_at_k_accum: Dict[int, List[float]] = {k_val: [] for k_val in k_list}
    purity_accum: List[float] = []
    mrr_accum: List[float] = []
    recall_accum: List[float] = []
    sim_accum: List[float] = []

    for q_spec in queries:
        row_entry: Dict[str, Any] = {
            "Query": q_spec["query"],
            "Category": q_spec.get("category", "General"),
            "Target Regime": q_spec.get("target_regime") or "N/A",
        }

        # Evaluate across multiple k values
        for k_val in k_list:
            q_res = evaluate_retrieval_query(
                rag=rag,
                query_spec_or_text=q_spec,
                k=k_val,
                use_llm_parser=use_llm_parser,
            )
            p_val = q_res["precision_at_k"]
            row_entry[f"P@{k_val}"] = p_val
            p_at_k_accum[k_val].append(p_val)

            if k_val == 5:
                row_entry["Regime Purity@5"] = q_res["regime_purity"]
                row_entry["MRR"] = q_res["reciprocal_rank"]
                row_entry["Recall@5"] = q_res["recall_at_k"]
                row_entry["Mean Sim@5"] = q_res["mean_similarity"]
                row_entry["Dataset Support"] = q_res["total_dataset_support"]

                purity_accum.append(q_res["regime_purity"])
                mrr_accum.append(q_res["reciprocal_rank"])
                recall_accum.append(q_res["recall_at_k"])
                sim_accum.append(q_res["mean_similarity"])

        rows.append(row_entry)

    df_results = pd.DataFrame(rows)

    summary_metrics = {
        "Mean Precision@1": float(np.mean(p_at_k_accum.get(1, [0.0]))),
        "Mean Precision@3": float(np.mean(p_at_k_accum.get(3, [0.0]))),
        "Mean Precision@5": float(np.mean(p_at_k_accum.get(5, [0.0]))),
        "Mean Precision@10": float(np.mean(p_at_k_accum.get(10, [0.0]))),
        "Mean Regime Purity@5": float(np.mean(purity_accum)) if purity_accum else 0.0,
        "Mean Reciprocal Rank (MRR)": float(np.mean(mrr_accum)) if mrr_accum else 0.0,
        "Mean Recall@5": float(np.mean(recall_accum)) if recall_accum else 0.0,
        "Mean Cosine Similarity@5": float(np.mean(sim_accum)) if sim_accum else 0.0,
    }

    return df_results, summary_metrics

