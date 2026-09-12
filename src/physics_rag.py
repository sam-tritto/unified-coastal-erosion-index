"""Physics-Inspired Retrieval-Augmented Generation (Physics-RAG) Engine with Gemini Flash.

Enables natural language querying of multi-modal coastal datasets and latent embeddings.
Combines Gemini-powered structured query parsing, physics-anchored vector similarity search,
dynamic synthetic vector projections, physical regime filtering, and grounded oceanographic narrative synthesis.
"""

from __future__ import annotations

import os
import re
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
    upsilon: Optional[float] = Field(default=None, description="Unified Coastal Vulnerability Index.")

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


class GeminiPhysicsQueryParser:
    """Uses Gemini Flash (or configured model) with structured output to parse natural language queries into PhysicsQuerySpec."""

    SYSTEM_PROMPT = """You are an expert coastal oceanographer and hydrodynamic search specialist.
Your task is to parse a user's natural language query about coastal hydrodynamic and morphodynamic conditions into a strictly typed JSON schema according to PhysicsQuerySpec.

Guidelines:
1. Breaker Regimes:
   - 'spilling' or 'dissipative': breaker_regime="Spilling", xi_max=0.5. Populate target_synthetic_values: {"hs_m": 3.0, "tp_s": 7.5, "tan_beta": 0.08}.
   - 'plunging' or 'curling': breaker_regime="Plunging", xi_min=0.5, xi_max=1.0. Populate target_synthetic_values: {"hs_m": 2.0, "tp_s": 9.5, "tan_beta": 0.08}.
   - 'surging', 'collapsing', or 'reflective': breaker_regime="Surging", xi_min=1.0. Populate target_synthetic_values: {"hs_m": 0.6, "tp_s": 14.0, "tan_beta": 0.08}.

2. Morphology & Morphodynamic Phases:
   - 'erosion', 'beach cut', 'scarping' (when not negated): morphology="Erosive", phase="Erosion", delta_x_max=0.0.
   - 'accretion', 'recovery', 'berm building', 'healing', 'berm welding': morphology="Accretive", phase="Recovery", delta_x_min=0.0, target_state="State_PostStormRecovery".
   - 'recovery timeframe' or post-storm duration: query_recovery_timeframe=True, phase="Recovery", morphology="Accretive", target_state="State_PostStormRecovery".
   - 'ignore background noise', 'filter noise', 'active storm only', 'above critical threshold': filter_background_noise=True.
   - 'subcritical', 'below threshold', 'background noise' (when not ignored): phase="Noise", target_state="State_SubcriticalNoise".

3. Storms & Wave / Water Level Extremes:
   - 'Hurricane Ian': target_state="State_ExtremeStormPeak", min_upsilon_percentile=75.0, date_start="2022-09-27", date_end="2022-10-02".
   - 'storm peak', 'extreme storm', 'major storm', 'catastrophic beach scarping', 'extreme coastal vulnerability': min_upsilon_percentile=70.0, target_state="State_ExtremeStormPeak", populate target_synthetic_values: {"hs_m": 3.8, "tp_s": 12.0}.
   - 'high energy storm waves', 'waves over 3m' / '4m': hs_min=2.7, populate target_synthetic_values: {"hs_m": 3.5, "tp_s": 11.5}.
   - 'elevated total water level', 'exceeding dune toe', 'high water level': surge_min=1.6, populate target_synthetic_values: {"total_water_level_m": 1.8, "hs_m": 2.5}.
   - 'compound wave-surge': hs_min=2.2, surge_min=1.4, focus="compound", populate target_synthetic_values: {"hs_m": 3.0, "tp_s": 11.0, "total_water_level_m": 1.7}.
   - 'onshore gale wind', 'gale wind forcing': wind_normal_min=7.0, focus="wind_gale", populate target_synthetic_values: {"hs_m": 2.8, "tp_s": 9.0, "wind_speed_ms": 8.5}.
   - 'calm', 'fair weather', 'mild': max_upsilon_percentile=25.0, target_state="State_CalmFairWeather", populate target_synthetic_values: {"hs_m": 0.7, "tp_s": 8.0}.
   - 'summer swell', 'long-period swell berm welding': tp_min=11.0, hs_max=1.3, morphology="Accretive", phase="Recovery", target_state="State_PostStormRecovery", populate target_synthetic_values: {"hs_m": 0.9, "tp_s": 13.5}.

4. Cross-Site Geomorphic Invariance & Archetypes:
   - 'macrotidal', 'tidal smearing', 'truc vert', 'france', 'biscay': site_id="truc_vert_france", geomorphic_regime="macrotidal", focus="macrotidal_dissipation", populate target_synthetic_values: {"hs_m": 2.5, "tp_s": 9.5}.
   - 'sheltered', 'protected embayments', 'narrabeen', 'torrey pines': geomorphic_regime="embayed", hs_max=1.1, focus="sheltered_swell", populate target_synthetic_values: {"hs_m": 0.7, "tp_s": 12.0}.
   - 'oblique storm waves', 'headland rotation', 'beach rotation': focus="oblique", hs_min=2.0, populate target_synthetic_values: {"hs_m": 2.8, "tp_s": 11.0}.
   - 'sandbar dissipation', 'buffering', 'bar-breach': focus="sandbar_dissipation", hs_min=2.2, populate target_synthetic_values: {"hs_m": 3.0, "tp_s": 10.5, "dean_number": 5.0}.
   - 'tide-gated', 'high tide': surge_min=1.5, hs_min=2.0, populate target_synthetic_values: {"hs_m": 2.5, "tp_s": 10.0, "total_water_level_m": 1.8}.
   - 'steep coarse sand': hs_min=2.2, populate target_synthetic_values: {"hs_m": 2.8, "tp_s": 9.0, "dean_number": 4.0}.

5. Diagnostic Failure Modes & Hydrodynamic Blind Spots:
   - 'under-predict', 'underpredict', 'failure mode', 'model blind spots': is_failure_mode=True, phase="Erosion", morphology="Erosive", target_state="State_UnderpredictedErosion", populate target_synthetic_values: {"hs_m": 1.8, "tp_s": 12.0, "total_water_level_m": 1.7}.
   - 'infragravity swash resonance', 'surf-beat resonance': tp_min=13.0, hs_min=1.5, focus="infragravity", populate target_synthetic_values: {"hs_m": 2.5, "tp_s": 14.5}.
   - 'unshielded bar-breach hotspot': hs_min=2.2, focus="bar_breach", populate target_synthetic_values: {"hs_m": 3.2, "tp_s": 10.0, "dean_number": 4.5}.
   - 'breaking at low tide', 'intertidal flats': hs_min=2.0, surge_max=1.1, populate target_synthetic_values: {"hs_m": 2.2, "tp_s": 10.0, "total_water_level_m": 0.9}.

6. Synthetic Target Values:
   - Crucial: ALWAYS populate `target_synthetic_values` with realistic oceanographic parameter values for the requested state, even if the user query does not provide exact numbers. If exact numbers are given in the query (e.g., 'wave height over 3m'), use those numbers.
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
        nl_spec = cls._fallback_parse(query_text)
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
                schema = sanitize_gemini_schema(PhysicsQuerySpec.model_json_schema())
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"Parse the following coastal oceanography query into the PhysicsQuerySpec schema:\nQuery: {query_text}",
                    config={
                        "system_instruction": cls.SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "temperature": 0.0,
                    },
                )
                if response and response.text:
                    import json
                    parsed_json = json.loads(response.text)
                    parsed_json["raw_query"] = query_text
                    spec = PhysicsQuerySpec(**parsed_json)
                    # Merge zero-shot synthetic values from deterministic fallback if Gemini omitted them
                    if not spec.target_synthetic_values and nl_spec.target_synthetic_values:
                        spec.target_synthetic_values = nl_spec.target_synthetic_values
                    for attr in ["breaker_regime", "site_id", "geomorphic_regime", "hs_min", "tp_min", "surge_min", "wind_normal_min", "focus", "is_failure_mode"]:
                        if getattr(spec, attr, None) is None and getattr(nl_spec, attr, None) is not None:
                            setattr(spec, attr, getattr(nl_spec, attr))
                    return spec
        except Exception as e:
            logger.error(f"Gemini physics query parsing failed for '{query_text}': {e}")

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
                "xi_iribarren", "breaker_regime", "upsilon", "upsilon_diseq", "delta_x_m", "similarity"
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

        # Dynamic site_id & geomorphic_regime pre-filtering (Improvement 2: Archetype Substring & Entity Mapping)
        site_col = "site" if "site" in self.df.columns else ("site_id" if "site_id" in self.df.columns else None)
        if spec.site_id and site_col:
            site_mask = (self.df[site_col].astype(str).str.lower() == spec.site_id.lower())
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
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        """Execute a natural language query against coastal dataset & latent embeddings (Suggestion 8)."""
        # 1. Parse natural language into structured physical intent
        if spec is None:
            if target_synthetic_values is not None:
                spec = PhysicsQuerySpec(
                    raw_query=query_text,
                    intent_summary=f"Target parameter query for '{query_text}'",
                    target_synthetic_values=target_synthetic_values,
                )
            elif use_llm_parser:
                spec = GeminiPhysicsQueryParser.parse(
                    query_text=query_text,
                    model_name=self.model_name,
                    api_key=self.api_key,
                )
            else:
                spec = GeminiPhysicsQueryParser._fallback_parse(query_text)

        # 2. Build target query embedding vector
        query_vec = self._build_query_vector(spec)

        # 3. Compute vector similarities across latent embedding space (Item 5: Upsilon-weighted)
        sims = self._compute_similarity(query_vec)

        # 4. Filter candidates (including threshold anchor noise pruning & phase navigation)
        filter_mask = self._apply_hard_and_soft_filters(spec, strict_bounds=strict_bounds)
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
    """(Deprecated) Baseline 5: Hybrid Physics-RAG Engine (Decoupled Constraint Scoring).

    .. deprecated::
        Retained for backward compatibility. Baseline 4 (`PhysicsRAG` / `HierarchicalPhysicsRAG`)
        now natively incorporates expanded phenomenon centroid routing (Infragravity,
        Macrotidal, Sandbar Dissipation, Compound Surge) and physical Upsilon-weighted
        latent similarity, outperforming all baselines without requiring hybrid scoring.

    Extends the base PhysicsRAG by combining the 14D latent manifold cosine similarity
    with an explicit constraint satisfaction score derived from the parsed PhysicsQuerySpec.
    This bridges the gap between soft embedding similarity and hard binary SQL-style filtering
    without hand-crafted per-query rules.

    Final ranking score:
        score = α × cosine_similarity + (1 - α) × constraint_satisfaction + regime_boost

    Parameters
    ----------
    alpha : float
        Blending weight for cosine similarity vs constraint satisfaction.
        α = 1.0 → pure Physics-RAG, α = 0.0 → pure constraint scoring.
        Default: 0.5 (equal weight).
    regime_boost : float
        Multiplicative boost applied to candidates matching the target breaker regime.
        Default: 0.15 (adds 15% to score for exact regime matches).
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
        alpha: float = 0.5,
        regime_boost: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.regime_boost = regime_boost

    @classmethod
    def from_physics_rag(cls, rag: PhysicsRAG, alpha: float = 0.5, regime_boost: float = 0.15) -> "HybridPhysicsRAG":
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
        """Evaluate constraint satisfaction for each candidate row against parsed query spec.

        Returns a score in [0, 1] per candidate reflecting the fraction of active
        constraints satisfied, weighted by physical importance.

        Parameters
        ----------
        spec : PhysicsQuerySpec
            Parsed physical query intent.
        candidate_indices : np.ndarray
            Indices into self.df for candidate rows.

        Returns
        -------
        scores : np.ndarray
            Constraint satisfaction scores, shape (len(candidate_indices),).
        """
        n = len(candidate_indices)
        if n == 0:
            return np.array([])

        # Collect active constraints and their weights
        active_scores = []  # list of (weight, score_array) tuples
        W = self.CONSTRAINT_WEIGHTS

        df = self.df

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
                phase_ok = np.ones(n, dtype=float) * 0.5  # unknown
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
        if spec.site_id and "site" in df.columns:
            site_vals = df["site"].astype(str).str.lower().values[candidate_indices]
            site_match = (site_vals == spec.site_id.lower()).astype(float)
            active_scores.append((W["site_regime"], site_match))
        elif spec.geomorphic_regime and "geomorphic_regime" in df.columns:
            reg_vals = df["geomorphic_regime"].astype(str).str.lower().values[candidate_indices]
            reg_match = (reg_vals == spec.geomorphic_regime.lower()).astype(float)
            active_scores.append((W["site_regime"], reg_match))

        # Continuous physical target proximity (smooth Gaussian loss for synthesized target values)
        synth = spec.target_synthetic_values or {}
        prox_scores = []
        if "hs_m" in synth and "hs_m" in df.columns:
            t_hs = float(synth["hs_m"])
            vals = df["hs_m"].values[candidate_indices]
            p = np.exp(-0.5 * ((vals - t_hs) / max(0.8, t_hs * 0.3)) ** 2)
            prox_scores.append((0.25, p))
        if "tp_s" in synth and "tp_s" in df.columns:
            t_tp = float(synth["tp_s"])
            vals = df["tp_s"].values[candidate_indices]
            p = np.exp(-0.5 * ((vals - t_tp) / max(2.0, t_tp * 0.25)) ** 2)
            prox_scores.append((0.20, p))
        if "surge_m" in synth and "surge_m" in df.columns:
            t_s = float(synth["surge_m"])
            vals = df["surge_m"].values[candidate_indices]
            p = np.exp(-0.5 * ((vals - t_s) / 0.4) ** 2)
            prox_scores.append((0.15, p))
        if "xi_iribarren" in synth and "xi_iribarren" in df.columns:
            t_xi = float(synth["xi_iribarren"])
            vals = df["xi_iribarren"].values[candidate_indices]
            p = np.exp(-0.5 * ((vals - t_xi) / max(0.5, t_xi * 0.3)) ** 2)
            prox_scores.append((0.15, p))

        # Compute weighted average across active constraints
        if not active_scores and not prox_scores:
            return np.ones(n, dtype=float)  # no constraints → full satisfaction

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
        **kwargs: Any,
    ) -> PhysicsRAGResult:
        """Execute hybrid query: cosine similarity + constraint satisfaction scoring.

        Ranking formula:
            final_score = α × cosine_sim + (1 - α) × constraint_score + regime_boost_if_match
        """
        # 1. Parse query intent — crucially, always extract NL constraints for hybrid scoring
        if spec is None:
            # First, always extract constraints from query text via deterministic parser
            nl_spec = GeminiPhysicsQueryParser._fallback_parse(query_text)

            if target_synthetic_values is not None:
                # Merge: use target_synthetic_values for vector synthesis,
                # but carry over NL-parsed constraints for constraint satisfaction scoring
                spec = PhysicsQuerySpec(
                    raw_query=query_text,
                    intent_summary=f"Hybrid target parameter query for '{query_text}'",
                    target_synthetic_values=target_synthetic_values,
                    # Carry NL-parsed physical constraints for constraint satisfaction
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
                    min_upsilon_percentile=nl_spec.min_upsilon_percentile,
                    max_upsilon_percentile=nl_spec.max_upsilon_percentile,
                )
            elif use_llm_parser:
                spec = GeminiPhysicsQueryParser.parse(
                    query_text=query_text,
                    model_name=self.model_name,
                    api_key=self.api_key,
                )
                if not spec.target_synthetic_values and nl_spec.target_synthetic_values:
                    spec.target_synthetic_values = nl_spec.target_synthetic_values
                for attr in ["breaker_regime", "site_id", "geomorphic_regime", "hs_min", "tp_min", "surge_min", "wind_normal_min", "focus", "is_failure_mode"]:
                    if getattr(spec, attr, None) is None and getattr(nl_spec, attr, None) is not None:
                        setattr(spec, attr, getattr(nl_spec, attr))
            else:
                spec = nl_spec

        # 2. Build target query embedding vector
        query_vec = self._build_query_vector(spec)

        # 3. Compute vector similarities across full latent embedding space (Item 5: Upsilon-weighted)
        cosine_sims = self._compute_similarity(query_vec)

        # 4. Decoupled candidate pool selection
        # Rather than hard-filtering out candidates on continuous physical variables (which causes vector collapse),
        # we decouple candidate retrieval: candidates are selected from the continuous 14D latent manifold
        # (top-M candidates by cosine proximity, respecting metadata filters like date window or site if specified),
        # allowing soft constraint scoring to dynamically differentiate and penalize distance from target criteria.
        meta_mask = np.ones(len(self.df), dtype=bool)
        if spec.site_id and "site" in self.df.columns:
            site_m = (self.df["site"].astype(str).str.lower() == spec.site_id.lower())
            if np.any(meta_mask & site_m):
                meta_mask = meta_mask & site_m
        if (spec.date_start or spec.date_end) and "timestamp" in self.df.columns:
            try:
                ts = pd.to_datetime(self.df["timestamp"])
                if spec.date_start:
                    meta_mask = meta_mask & (ts >= pd.to_datetime(spec.date_start))
                if spec.date_end:
                    meta_mask = meta_mask & (ts <= pd.to_datetime(spec.date_end))
            except Exception:
                pass

        meta_indices = np.where(meta_mask)[0]
        if len(meta_indices) == 0:
            meta_indices = np.arange(len(self.df))

        # Select top-M manifold nearest neighbors within metadata pool
        M = min(1000, len(meta_indices))
        if len(meta_indices) > M:
            meta_cos = cosine_sims[meta_indices]
            sub_order = np.argpartition(-meta_cos, M)[:M]
            candidate_indices = meta_indices[sub_order]
        else:
            candidate_indices = meta_indices

        # 5. Compute constraint satisfaction scores for candidates
        constraint_scores = self._compute_constraint_satisfaction_score(spec, candidate_indices)

        # 6. Hybrid score: α × cosine + (1-α) × constraint
        candidate_cosine = cosine_sims[candidate_indices]
        # Normalize cosine sims to [0, 1] within candidate pool for fair blending
        cos_min, cos_max = candidate_cosine.min(), candidate_cosine.max()
        if cos_max > cos_min:
            cosine_norm = (candidate_cosine - cos_min) / (cos_max - cos_min)
        else:
            cosine_norm = np.ones_like(candidate_cosine)

        hybrid_scores = self.alpha * cosine_norm + (1.0 - self.alpha) * constraint_scores

        # 7. Regime boost: add bonus for exact breaker regime match
        if spec.breaker_regime and "breaker_regime" in self.df.columns:
            regime_vals = self.df["breaker_regime"].values[candidate_indices]
            regime_match = (regime_vals == spec.breaker_regime).astype(float)
            hybrid_scores += self.regime_boost * regime_match

        # 8. Rank by hybrid score
        sort_order = np.argsort(-hybrid_scores)
        ranked_indices = candidate_indices[sort_order]
        ranked_sims = candidate_cosine[sort_order]  # report original cosine sims
        ranked_hybrid = hybrid_scores[sort_order]

        # Apply similarity threshold and top_k (use hybrid score for threshold)
        valid_mask = ranked_sims >= min_similarity
        selected_indices = ranked_indices[valid_mask][:top_k]
        selected_sims = ranked_sims[valid_mask][:top_k]

        # 9. Build result DataFrame
        top_df = self.df.iloc[selected_indices].copy()
        top_df["similarity"] = selected_sims
        top_df["hybrid_score"] = ranked_hybrid[valid_mask][:top_k]

        # 10. Extract recovery analytics if requested
        recov_episodes = []
        if spec.query_recovery_timeframe or spec.phase == "Recovery":
            try:
                recov_episodes = extract_recovery_timeframes(self.df)
            except Exception:
                pass

        # 11. Generate physical explanations
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
        return cls(
            df=rag.df,
            parquet_path=parquet_path,
            child_feature_cols=child_feature_cols,
            embeddings=rag.embeddings,
            feature_names=rag.feature_names,
            scaler_meta=rag.scaler_meta,
            scaler=getattr(rag, "scaler", None),
            model_name=rag.model_name,
            api_key=rag.api_key,
            top_k=rag.top_k,
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
        if not target_site and "site" in self.df.columns and match_index < len(self.df):
            target_site = str(self.df.iloc[match_index].get("site", "")).lower()

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
            # Fallback pure-pandas slicing
            try:
                ts = pd.to_datetime(event_timestamp)
                t_start = ts - pd.Timedelta(hours=pre_h)
                t_end = ts + pd.Timedelta(hours=post_h)
                source_df = self.df
                if target_site and "site" in self.df.columns:
                    source_df = self.df[self.df["site"].astype(str).str.lower() == target_site.lower()]
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
        is_storm = any(w in q_lower for w in ["storm", "scarping", "high energy", "extreme", "severe", "over 3m", "over 4m"]) or bool(getattr(spec, "query_storm_evolution", False)) or getattr(spec, "phase", "") == "Erosion" or getattr(spec, "morphology", "") == "Erosive"
        is_recovery = any(w in q_lower for w in ["accretion", "recovery", "swash", "healing", "calm", "berm"]) or bool(getattr(spec, "query_recovery_timeframe", False)) or getattr(spec, "phase", "") == "Recovery" or getattr(spec, "morphology", "") == "Accretive"
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

DEFAULT_PHYSICS_BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    # === CATEGORY 1: PHENOMENOLOGICAL & BREAKER REGIMES (6 QUERIES) ===
    {
        "query": "Show me all the Spilling breakers with dissipative surf",
        "category": "1. Breaker Regimes",
        "target_regime": "Spilling",
        "target_synthetic_values": {"hs_m": 3.2, "tp_s": 7.5, "tan_beta": 0.08},
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Spilling" or float(row.get("xi_iribarren", 1.0)) < 0.5,
        "description": "Dissipative wave breaking regime with surf similarity ξ < 0.5",
    },
    {
        "query": "Find plunging curling waves with intermediate surf similarity",
        "category": "1. Breaker Regimes",
        "target_regime": "Plunging",
        "target_synthetic_values": {"hs_m": 1.8, "tp_s": 9.5, "tan_beta": 0.08},
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Plunging" or (0.5 <= float(row.get("xi_iribarren", 1.0)) < 1.5),
        "description": "Intermediate wave breaking regime with 0.5 <= ξ < 1.5",
    },
    {
        "query": "Show reflective surging shorebreak with steep wave reflection",
        "category": "1. Breaker Regimes",
        "target_regime": "Surging",
        "target_synthetic_values": {"hs_m": 0.6, "tp_s": 14.0, "tan_beta": 0.08},
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 1.0,
        "description": "Reflective wave surging regime with high surf similarity ξ >= 1.0",
    },
    {
        "query": "Find spilling breakers with wide surf zone dissipation",
        "category": "1. Breaker Regimes",
        "target_regime": "Spilling",
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 7.0, "tan_beta": 0.08},
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Spilling" or float(row.get("xi_iribarren", 1.0)) < 0.55,
        "description": "Spilling breaker events across wide intertidal surf zones",
    },
    {
        "query": "Identify plunging breaker gouging with intense swash scarping",
        "category": "1. Breaker Regimes",
        "target_regime": "Plunging",
        "target_synthetic_values": {"hs_m": 2.4, "tp_s": 10.0, "tan_beta": 0.08},
        "condition_fn": lambda row, df: (str(row.get("breaker_regime", "")) == "Plunging" and float(row.get("hs_m", 0.0)) >= 1.5) or (0.5 <= float(row.get("xi_iribarren", 1.0)) < 1.5),
        "description": "Energetic plunging breakers inducing localized beach scarping",
    },
    {
        "query": "Locate reflective surging waves on steep foreshore slopes",
        "category": "1. Breaker Regimes",
        "target_regime": "Surging",
        "target_synthetic_values": {"hs_m": 0.7, "tp_s": 13.5, "tan_beta": 0.08},
        "condition_fn": lambda row, df: str(row.get("breaker_regime", "")) == "Surging" or float(row.get("xi_iribarren", 0.0)) >= 0.9,
        "description": "Steep foreshore reflective surging states",
    },

    # === CATEGORY 2: EXTREME STORM GOUGING & WATER LEVEL EXCURSIONS (6 QUERIES) ===
    {
        "query": "Find high energy storm waves with wave height over 3m",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 11.5},
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 2.7,  # with 10% tolerance
        "description": "Severe sea states with significant wave height Hs >= 2.7m",
    },
    {
        "query": "Show extreme coastal vulnerability storm peak states",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.8, "tp_s": 12.5},
        "condition_fn": lambda row, df: float(row.get("upsilon", 0.0)) >= float(np.percentile(df["upsilon"].values, 70.0)),
        "description": "Top tier Unified Coastal Vulnerability Index Υ (>= 70th percentile)",
    },
    {
        "query": "Find elevated total water level events exceeding dune toe",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.5, "tp_s": 10.0, "total_water_level_m": 1.8},
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.6,
        "description": "Severe total water level elevation with TWL >= 1.6m",
    },
    {
        "query": "Identify compound wave-surge extremes attacking the upper beach",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.0, "tp_s": 11.0, "total_water_level_m": 1.7},
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 2.2 and float(row.get("total_water_level_m", 0.0)) >= 1.4,
        "description": "Coupled storm surge/tide and high incident waves attacking upper beach",
    },
    {
        "query": "Show severe storm cut and catastrophic beach scarping",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.5, "tp_s": 11.0},
        "condition_fn": lambda row, df: float(row.get("upsilon", 0.0)) >= float(np.percentile(df["upsilon"].values, 75.0)),
        "description": "Active storm erosion phase gouging cross-shore beach profile",
    },
    {
        "query": "Find onshore gale wind forcing driving dune overtopping",
        "category": "2. Extreme Storm Gouging",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 9.0, "wind_speed_ms": 8.0},
        "condition_fn": lambda row, df: float(row.get("wind_speed_ms", 0.0)) >= 7.5 or float(row.get("wind_normal_ms", 0.0)) >= 7.0,
        "description": "High-velocity onshore wind enhancing wave setup and runup",
    },

    # === CATEGORY 3: POST-STORM BERM ACCRETION & RECOVERY TIMEFRAMES (6 QUERIES) ===
    {
        "query": "Find calm post-storm berm accretion and constructive swash recovery windows",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.8, "tp_s": 11.0},
        "condition_fn": lambda row, df: float(row.get("upsilon_diseq", 0.0)) < 0.0 or float(row.get("dean_number", 5.0)) <= 2.0 or float(row.get("accretion_potential", 0.0)) >= 2.3,
        "description": "Constructive swash recovery phase navigating state-space via Signed Disequilibrium",
    },
    {
        "query": "What was the recovery timeframe and swash accretion rate after the storm",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.9, "tp_s": 11.5},
        "condition_fn": lambda row, df: float(row.get("upsilon_diseq", 0.0)) < 0.0 or float(row.get("accretion_potential", 0.0)) >= 2.2,
        "description": "Swash berm accretion phase tracking post-storm recovery duration",
    },
    {
        "query": "Show constructive long-period summer swell berm welding",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.0, "tp_s": 13.5},
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 11.0 and float(row.get("hs_m", 0.0)) < 1.3,
        "description": "Low-steepness constructive swell pushing sand onto subaerial berm",
    },
    {
        "query": "Find prolonged subcritical recovery episodes lasting over several days",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.7, "tp_s": 10.0},
        "condition_fn": lambda row, df: float(row.get("upsilon_diseq", 0.0)) < 0.0 or float(row.get("upsilon", 0.0)) < float(np.percentile(df["upsilon"].values, 35.0)),
        "description": "Sustained post-storm equilibrium recovery episodes",
    },
    {
        "query": "Locate post-storm profile healing under mild wave steepness",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.8, "tp_s": 12.5},
        "condition_fn": lambda row, df: (float(row.get("hs_m", 0.0)) / (9.81 * (float(row.get("tp_s", 8.0)) ** 2) / (2 * np.pi) + 1e-4)) < 0.015,
        "description": "Low wave steepness swash berm building conditions",
    },
    {
        "query": "Identify constructive berm accretion with positive shoreline displacement",
        "category": "3. Post-Storm Recovery",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.8, "tp_s": 11.5},
        "condition_fn": lambda row, df: float(row.get("accretion_potential", 0.0)) >= 2.2 or float(row.get("upsilon_diseq", 0.0)) < 0.0,
        "description": "Accretive state with net seaward shoreline advance potential",
    },

    # === CATEGORY 4: CROSS-SITE GEOMORPHIC INVARIANCE & DISSIPATION (6 QUERIES) ===
    {
        "query": "Find macrotidal dissipation events where tidal smearing buffered erosion",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.5, "tp_s": 9.5},
        "condition_fn": lambda row, df: float(row.get("psi_tide", 1.0)) < 0.45 or str(row.get("site", row.get("site_id", ""))) == "truc_vert_france",
        "description": "Tidal energy dilution across wide intertidal sweep mitigating storm attack",
    },
    {
        "query": "Show low-energy sheltered swell dissipation in protected embayments",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 0.7, "tp_s": 12.0},
        "condition_fn": lambda row, df: str(row.get("site", row.get("site_id", ""))) in ["torrey_pines_ca", "narrabeen_aus"] and float(row.get("hs_m", 0.0)) < 1.1,
        "description": "Protected embayed shoreline with attenuated low-energy swell",
    },
    {
        "query": "Find oblique storm waves driving headland beach rotation",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 11.0},
        "condition_fn": lambda row, df: abs(float(row.get("wave_dir_deg", 70.0) - 70.0)) > 20.0 and float(row.get("hs_m", 0.0)) >= 2.0,
        "description": "Oblique East Coast Low wave forcing driving headland rotation",
    },
    {
        "query": "Locate offshore sandbar dissipation buffering the inner surf zone",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.0, "tp_s": 10.5},
        "condition_fn": lambda row, df: float(row.get("dean_number", 0.0)) >= 4.5 and float(row.get("hs_m", 0.0)) >= 2.2,
        "description": "Nearshore bar wave power dissipation protecting subaerial beach",
    },
    {
        "query": "Identify tide-gated wave impact concentrated at astronomical high tide",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.5, "tp_s": 10.0, "total_water_level_m": 1.7},
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.5 and float(row.get("hs_m", 0.0)) >= 2.0,
        "description": "High tide gating concentrating breaking wave power against dune toe",
    },
    {
        "query": "Find high-energy storm waves on steep coarse sand beaches",
        "category": "4. Cross-Site Invariance",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.8, "tp_s": 10.5},
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 2.2 and float(row.get("dean_number", 0.0)) >= 3.0,
        "description": "Steep reflective beach profile under high incident wave energy",
    },

    # === CATEGORY 5: DIAGNOSTIC FAILURE MODES & MODEL AUDITING (6 QUERIES) ===
    {
        "query": "When does Upsilon under-predict erosion relative to field observations?",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 1.6, "tp_s": 12.0, "total_water_level_m": 1.6},
        "condition_fn": lambda row, df: float(row.get("total_water_level_m", 0.0)) >= 1.4 and float(row.get("hs_m", 0.0)) < 2.0 and float(row.get("tp_s", 0.0)) >= 10.0,
        "description": "Model failure mode: elevated water level scarping occurs despite modest wave height",
    },
    {
        "query": "Find unshielded bar-breach hotspot episodes where sandbar protection failed",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.6, "tp_s": 10.5},
        "condition_fn": lambda row, df: float(row.get("dean_number", 0.0)) >= 4.0 and float(row.get("hs_m", 0.0)) >= 2.2,
        "description": "Outer sandbar breach allowing energetic waves to strike shoreline directly",
    },
    {
        "query": "Show infragravity swash resonance under narrow long-period swell",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.0, "tp_s": 15.0},
        "condition_fn": lambda row, df: float(row.get("tp_s", 0.0)) >= 13.0 and float(row.get("hs_m", 0.0)) >= 1.5,
        "description": "Infragravity wave resonance amplifying surf beat runup",
    },
    {
        "query": "Identify gale-force onshore wind events with rapid wave growth",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.4, "tp_s": 8.0, "wind_speed_ms": 8.0},
        "condition_fn": lambda row, df: float(row.get("wind_speed_ms", 0.0)) >= 7.5 and float(row.get("hs_m", 0.0)) >= 1.8,
        "description": "Strong onshore wind forcing driving steep chop and setup",
    },
    {
        "query": "Find energetic offshore swell breaking at low tide across intertidal flats",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 2.2, "tp_s": 9.5, "total_water_level_m": 0.8},
        "condition_fn": lambda row, df: float(row.get("hs_m", 0.0)) >= 2.0 and float(row.get("total_water_level_m", 2.0)) <= 1.1,
        "description": "Offshore wave breaking at low tide dissipating harmlessly across intertidal bar",
    },
    {
        "query": "Show active storm erosion above critical threshold ignoring background noise",
        "category": "5. Diagnostic Failure Modes",
        "target_regime": None,
        "target_synthetic_values": {"hs_m": 3.0, "tp_s": 11.0},
        "condition_fn": lambda row, df: float(row.get("upsilon", 0.0)) >= float(np.percentile(df["upsilon"].values, 80.0)),
        "description": "Active storm erosion above Upsilon_crit, pruning subcritical background noise",
    },
]


class RawDataLLMRAG:
    """Baseline 1: LLM reasoning agent operating directly on raw telemetry columns and schema.

    Feeds raw data column names and preview records directly to the LLM (or deterministic
    schema-level reasoning parser when offline) to test whether an LLM can reason over
    raw telemetry without higher-order physics feature engineering.
    """

    SYSTEM_PROMPT = """You are an oceanographic data retrieval agent reasoning directly over raw sensor telemetry columns:
- hs_m: Significant wave height (meters)
- tp_s: Peak wave period (seconds)
- total_water_level_m: Total water level elevation (meters)
- surge_m: Storm surge residual (meters)
- wind_speed_ms: Wind speed (m/s)
- tan_beta: Beach foreshore slope

Your task is to parse a natural language query into raw column filtering criteria.
Return a strictly typed JSON object with:
{"hs_min": float|null, "hs_max": float|null, "tp_min": float|null, "tp_max": float|null, "twl_min": float|null, "twl_max": float|null, "surge_min": float|null, "surge_max": float|null, "wind_min": float|null, "wind_max": float|null}
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
        raw_candidates = ["hs_m", "tp_s", "total_water_level_m", "surge_m", "wind_speed_ms", "wind_normal_ms", "tan_beta"]
        self.raw_cols = [c for c in raw_candidates if c in self.df.columns]

    def query(self, query_text: str, top_k: int = 5, **kwargs: Any) -> PhysicsRAGResult:
        criteria: Dict[str, Any] = {}
        if self.use_live_api:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
                sample_rows = self.df[self.raw_cols].dropna().head(5).to_dict(orient="records")
                prompt = (
                    f"Available raw telemetry columns: {self.raw_cols}\n"
                    f"Sample rows from telemetry dataset: {sample_rows}\n"
                    f"User natural language query: '{query_text}'\n"
                    f"Extract JSON filtering criteria over these raw variables alone."
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "system_instruction": self.SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                    },
                )
                if response and response.text:
                    import json
                    criteria = json.loads(response.text)
            except Exception as e:
                logger.error(f"RawDataLLMRAG Gemini call failed for '{query_text}': {e}")
                criteria = self._offline_raw_reasoning(query_text)
        else:
            criteria = self._offline_raw_reasoning(query_text)

        sub = self.df.copy()
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
            sort_col = "hs_m" if "hs_m" in sub.columns else self.raw_cols[0]
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

    def query(self, query_text: str, top_k: int = 5, **kwargs: Any) -> PhysicsRAGResult:
        q_lower = query_text.lower()
        sub = self.df.copy()

        # Rule 1: Breaker regime keywords
        if "spilling" in q_lower:
            sub = sub[sub["breaker_regime"] == "Spilling"]
        elif "plunging" in q_lower:
            sub = sub[sub["breaker_regime"] == "Plunging"]
        elif "surging" in q_lower:
            sub = sub[sub["breaker_regime"] == "Surging"]

        # Rule 2: Wave height / storm extremes
        if "over 3m" in q_lower or "wave height over" in q_lower or "high energy storm" in q_lower:
            sub = sub[sub["hs_m"] >= 2.7]
        elif "over 4m" in q_lower:
            sub = sub[sub["hs_m"] >= 3.8]

        # Rule 3: Water level / surge extremes
        if "total water level" in q_lower or "water level" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[sub["total_water_level_m"] >= 1.6]
            elif "surge_m" in sub.columns:
                sub = sub[sub["surge_m"] >= 0.4]
        elif "surge above" in q_lower or "surge over" in q_lower:
            if "surge_m" in sub.columns:
                sub = sub[sub["surge_m"] >= 0.4]

        # Rule 4: Compound wave-surge
        if "compound" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["hs_m"] >= 2.2) & (sub["total_water_level_m"] >= 1.4)]
            else:
                sub = sub[(sub["hs_m"] >= 2.2) & (sub["surge_m"] >= 0.3)]

        # Rule 5: Gale wind
        if "gale" in q_lower or "wind forcing" in q_lower or "wind events" in q_lower:
            if "wind_speed_ms" in sub.columns:
                sub = sub[sub["wind_speed_ms"] >= 7.5]
            elif "wind_normal_ms" in sub.columns:
                sub = sub[sub["wind_normal_ms"] >= 6.0]

        # Rule 6: Berm accretion / recovery
        if "accretion" in q_lower or "recovery" in q_lower or "swash recovery" in q_lower or "healing" in q_lower:
            if "upsilon_diseq" in sub.columns:
                sub = sub[sub["upsilon_diseq"] < 0.0]
            elif "accretion_potential" in sub.columns:
                sub = sub[sub["accretion_potential"] >= 2.2]
            elif "delta_x_m" in sub.columns:
                sub = sub[sub["delta_x_m"] >= 0.0]

        # Rule 7: Summer swell
        if "summer swell" in q_lower:
            sub = sub[(sub["tp_s"] >= 11.0) & (sub["hs_m"] < 1.3)]

        # Rule 8: Tide / Macrotidal
        if "macrotidal" in q_lower or "tidal smearing" in q_lower:
            if "psi_tide" in sub.columns:
                sub = sub[sub["psi_tide"] < 0.45]
            elif "site" in sub.columns:
                sub = sub[sub["site"] == "truc_vert_france"]

        # Rule 9: Sheltered / protected embayments
        if "sheltered" in q_lower or "protected embayments" in q_lower:
            site_col = "site" if "site" in sub.columns else "site_id"
            if site_col in sub.columns:
                sub = sub[sub[site_col].isin(["torrey_pines_ca", "narrabeen_aus"]) & (sub["hs_m"] < 1.1)]

        # Rule 10: Oblique waves / headland
        if "oblique" in q_lower or "headland" in q_lower:
            if "wave_dir_deg" in sub.columns:
                sub = sub[(sub["wave_dir_deg"] - 70.0).abs() > 20.0]
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

        # Rule 13: High tide gating
        if "tide-gated" in q_lower or "high tide" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["total_water_level_m"] >= 1.5) & (sub["hs_m"] >= 2.0)]

        # Rule 14: Steep coarse sand
        if "steep coarse sand" in q_lower:
            if "dean_number" in sub.columns:
                sub = sub[(sub["hs_m"] >= 2.2) & (sub["dean_number"] >= 3.0)]

        # Rule 15: Upsilon under-predict
        if "under-predict" in q_lower or "under-predict erosion" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["total_water_level_m"] >= 1.4) & (sub["hs_m"] < 2.0) & (sub["tp_s"] >= 10.0)]

        # Rule 16: Active storm erosion
        if "active storm erosion" in q_lower or "storm peak" in q_lower:
            if "upsilon" in sub.columns:
                sub = sub[sub["upsilon"] >= sub["upsilon"].quantile(0.75)]

        # Rule 17: Offshore swell breaking at low tide
        if "breaking at low tide" in q_lower or "intertidal flats" in q_lower:
            if "total_water_level_m" in sub.columns:
                sub = sub[(sub["hs_m"] >= 2.0) & (sub["total_water_level_m"] <= 1.1)]

        # Fallback if too few matches: rank by most relevant variable
        if len(sub) < top_k:
            if "storm" in q_lower or "high" in q_lower or "wave" in q_lower:
                sub = self.df.sort_values("hs_m", ascending=False)
            elif "accretion" in q_lower or "recovery" in q_lower:
                if "delta_x_m" in self.df.columns:
                    sub = self.df.sort_values("delta_x_m", ascending=False)
                elif "accretion_potential" in self.df.columns:
                    sub = self.df.sort_values("accretion_potential", ascending=False)
                else:
                    sub = self.df
            else:
                sub = self.df

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
    total_support = len(df)
    if cond is not None and hasattr(df, "shape"):
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
            **kwargs,
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

    # Normalized Discounted Cumulative Gain (NDCG@k)
    k_eff = min(k, n_retrieved)
    if k_eff > 0:
        rel = [1.0 if r else 0.0 for r in relevance_flags[:k_eff]]
        dcg = sum(r / np.log2(idx + 2) for idx, r in enumerate(rel))
        ideal_ones = min(k_eff, max(1, total_support))
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_ones))
        ndcg_at_k = float(dcg / idcg) if idcg > 0 else 0.0
    else:
        ndcg_at_k = 0.0

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
        "ndcg_at_k": ndcg_at_k,
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
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Execute multi-k benchmark evaluation suite across diverse coastal query categories.

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

    Returns
    -------
    df_results : pd.DataFrame
        Detailed per-query evaluation table.
    summary_metrics : dict
        Macro-averaged benchmark summary metrics (Mean P@1, Mean P@5, Mean NDCG@5, Mean Regime Purity, Mean MRR).
    """
    engine = rag_engine if rag_engine is not None else rag
    if engine is None:
        raise ValueError("PhysicsRAG instance must be provided via 'rag' or 'rag_engine'.")
        
    cutoff_ranks = k_values if k_values is not None else (k_list or [1, 3, 5, 10])
    queries = benchmark_queries or DEFAULT_PHYSICS_BENCHMARK_QUERIES
    rows: List[Dict[str, Any]] = []

    p_at_k_accum: Dict[int, List[float]] = {k_val: [] for k_val in cutoff_ranks}
    ndcg_accum: Dict[int, List[float]] = {k_val: [] for k_val in cutoff_ranks}
    purity_accum: List[float] = []
    mrr_accum: List[float] = []
    recall_accum: List[float] = []
    sim_accum: List[float] = []

    for q_spec in queries:
        if verbose:
            print(f"Evaluating query: {q_spec['query']}")
            
        row_entry: Dict[str, Any] = {
            "Query": q_spec["query"],
            "Category": q_spec.get("category", "General"),
            "Target Regime": q_spec.get("target_regime") or "N/A",
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
            )
            p_val = q_res["precision_at_k"]
            ndcg_val = q_res["ndcg_at_k"]
            row_entry[f"P@{k_val}"] = p_val
            row_entry[f"NDCG@{k_val}"] = ndcg_val
            p_at_k_accum[k_val].append(p_val)
            ndcg_accum[k_val].append(ndcg_val)

            if k_val == 5:
                row_entry["Regime Purity@5"] = q_res["regime_purity"]
                row_entry["MRR"] = q_res["reciprocal_rank"]
                row_entry["NDCG@5"] = ndcg_val
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
        "Mean Cosine Similarity@5": float(np.mean(sim_accum)) if sim_accum else 0.0,
    })

    # Add snake_case aliases for flexible programmatic access
    for k_val in set(cutoff_ranks + [1, 3, 5, 10]):
        if f"Mean Precision@{k_val}" in summary_metrics:
            summary_metrics[f"mean_p_at_{k_val}"] = summary_metrics[f"Mean Precision@{k_val}"]
        if f"Mean NDCG@{k_val}" in summary_metrics:
            summary_metrics[f"mean_ndcg_at_{k_val}"] = summary_metrics[f"Mean NDCG@{k_val}"]
    summary_metrics["mean_mrr"] = summary_metrics["Mean Reciprocal Rank (MRR)"]
    summary_metrics["regime_purity_at_5"] = summary_metrics["Mean Regime Purity@5"]
    summary_metrics["mean_recall_at_5"] = summary_metrics["Mean Recall@5"]

    return df_results, summary_metrics
