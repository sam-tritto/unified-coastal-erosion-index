"""Unit and integration tests for Physics-RAG with Gemini Flash and Embeddings Engine."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.embeddings import (
    build_physics_anchored_feature_matrix,
    compute_latent_embeddings,
    compute_regime_centroids,
    synthesize_target_physics_vector,
)
from src.physics_rag import (
    PhysicsQuerySpec,
    PhysicsQueryParser,
    GeminiPhysicsQueryParser,
    PhysicsRAG,
    PhysicsRAGResult,
    explain_with_gemini,
)


@pytest.fixture
def sample_coastal_df() -> pd.DataFrame:
    """Construct synthetic multi-modal coastal dataset for testing."""
    n = 100
    np.random.seed(42)
    dates = pd.date_range("2022-09-20", periods=n, freq="6h")
    
    hs = np.random.uniform(0.5, 5.0, n)
    tp = np.random.uniform(4.0, 16.0, n)
    surge = np.random.uniform(-0.3, 1.5, n)
    wind_normal = np.random.uniform(-10.0, 15.0, n)
    water_temp = np.random.uniform(15.0, 24.0, n)
    tan_beta = np.full(n, 0.08)
    w_s = np.full(n, 0.048)
    
    g = 9.81
    xi = tan_beta * np.sqrt(g / (2 * np.pi)) * (tp / np.sqrt(hs))
    regimes = np.where(xi < 0.5, "Spilling", np.where(xi < 3.3, "Plunging", "Surging"))
    
    dean = hs / (w_s * tp)
    p_flux = (1025.0 * (g ** 2) / (16.0 * 2 * np.pi)) * (hs ** 2) * tp
    ups = (1025.0 * (g ** 1.5) / (16.0 * (2 * np.pi) ** 1.5)) * tan_beta * ((hs ** 2.5) * tp / w_s) * (1.0 + np.maximum(surge / hs, 0.0))
    dx = np.where(ups > np.median(ups), -np.random.uniform(0.1, 1.2, n), np.random.uniform(0.05, 0.5, n))
    
    return pd.DataFrame({
        "timestamp": dates,
        "hs_m": hs,
        "tp_s": tp,
        "surge_m": surge,
        "wind_normal_ms": wind_normal,
        "water_temp_c": water_temp,
        "tan_beta": tan_beta,
        "w_s_ms": w_s,
        "xi_iribarren": xi,
        "breaker_regime": regimes,
        "dean_number": dean,
        "p_flux_Wm": p_flux,
        "upsilon": ups,
        "log_upsilon": np.log10(np.maximum(ups, 1e-4)),
        "delta_p_dissip": np.random.uniform(0.2, 0.8, n),
        "accretion_potential": np.random.uniform(10.0, 100.0, n),
        "delta_x_m": dx,
    })


def test_physics_query_spec_schema():
    """Verify PhysicsQuerySpec model validation and defaults."""
    spec = PhysicsQuerySpec(
        raw_query="Find spilling breakers with waves over 3m",
        breaker_regime="Spilling",
        hs_min=3.0,
        xi_max=0.5,
    )
    assert spec.raw_query == "Find spilling breakers with waves over 3m"
    assert spec.breaker_regime == "Spilling"
    assert spec.hs_min == 3.0
    assert spec.xi_max == 0.5
    assert spec.morphology is None


def test_physics_query_parser_regimes():
    """Test deterministic rule-based query parser across breaker regimes."""
    res_spill = PhysicsQueryParser.parse("Show me dissipative spilling breakers")
    assert res_spill["breaker_regime"] == "Spilling"
    assert res_spill["filters"].get("xi_max") == 0.5

    res_plunge = PhysicsQueryParser.parse("Find plunging curling waves")
    assert res_plunge["breaker_regime"] == "Plunging"
    assert res_plunge["filters"].get("xi_min") == 0.5
    assert res_plunge["filters"].get("xi_max") == 3.3

    res_surge = PhysicsQueryParser.parse("Show reflective surging shorebreak")
    assert res_surge["breaker_regime"] == "Surging"
    assert res_surge["filters"].get("xi_min") == 3.3


def test_physics_query_parser_numerical_extraction():
    """Test extracting numerical targets from natural language queries."""
    res = PhysicsQueryParser.parse("Find cases with 4.5m wave height and 14s period and 0.9m surge")
    assert "target_synthetic_values" in res
    assert res["target_synthetic_values"]["hs_m"] == 4.5
    assert res["target_synthetic_values"]["tp_s"] == 14.0
    assert res["target_synthetic_values"]["surge_m"] == 0.9


def test_gemini_query_parser_fallback():
    """Test GeminiPhysicsQueryParser fallback when offline or no API key is provided."""
    spec = GeminiPhysicsQueryParser.parse("Find storm surge during Hurricane Ian with beach erosion", api_key=None)
    assert isinstance(spec, PhysicsQuerySpec)
    assert spec.morphology == "Erosive"
    assert spec.target_state == "State_ExtremeStormPeak"
    assert spec.min_upsilon_percentile == 75.0


def test_gemini_query_parser_mock_client():
    """Test GeminiPhysicsQueryParser when Gemini client returns structured JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"raw_query": "high energy storm", "intent_summary": "Major storm gouging", "breaker_regime": "Plunging", "hs_min": 3.5, "min_upsilon_percentile": 80.0}'
    mock_client.models.generate_content.return_value = mock_response

    spec = GeminiPhysicsQueryParser.parse(
        query_text="high energy storm",
        client=mock_client,
    )
    assert spec.breaker_regime == "Plunging"
    assert spec.hs_min == 3.5
    assert spec.min_upsilon_percentile == 80.0
    assert "Major storm" in spec.intent_summary


def test_synthesize_target_physics_vector(sample_coastal_df):
    """Test dynamic query vector synthesis from arbitrary sensor values."""
    embeddings, feature_names, meta = compute_latent_embeddings(sample_coastal_df, scaler_type="robust")
    scaler = meta["scaler"]
    
    target_params = {"hs_m": 4.2, "tp_s": 14.5, "surge_m": 1.1, "wind_normal_ms": 8.5}
    vec = synthesize_target_physics_vector(
        target_params=target_params,
        scaler=scaler,
        feature_names=feature_names,
        reference_df=sample_coastal_df,
    )
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == embeddings.shape[1]
    assert np.all(np.isfinite(vec))


def test_physics_rag_initialization_and_search(sample_coastal_df):
    """Test PhysicsRAG initialization, filtering, and retrieval results."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    
    # 1. Spilling breaker query
    res = rag.query("Show me all the Spilling breakers", top_k=5, use_llm_parser=False, use_llm_synthesis=False)
    assert isinstance(res, PhysicsRAGResult)
    assert len(res.matches_df) <= 5
    assert (res.matches_df["breaker_regime"] == "Spilling").all()
    assert res.similarities[0] <= 1.0001
    
    # 2. Check summary text formatting
    summary_text = res.summary()
    assert "=== Physics-Inspired RAG Results ===" in summary_text
    assert "Spilling" in summary_text


def test_physics_rag_synthetic_vector_query(sample_coastal_df):
    """Test PhysicsRAG query with explicit numerical values synthesizing target vector."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    
    res = rag.query("Find states similar to 4.5m wave height and 14s period", top_k=5)
    assert len(res.matches_df) > 0
    assert res.query_vector is not None
    assert isinstance(res.query_vector, np.ndarray)


def test_physics_rag_date_filtering(sample_coastal_df):
    """Test date window filtering in PhysicsRAG."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    
    spec = PhysicsQuerySpec(
        raw_query="Late September storm",
        date_start="2022-09-25",
        date_end="2022-09-28",
    )
    mask = rag._apply_hard_and_soft_filters(spec)
    filtered_df = sample_coastal_df[mask]
    
    assert len(filtered_df) > 0
    ts = pd.to_datetime(filtered_df["timestamp"])
    assert (ts >= pd.to_datetime("2022-09-25")).all()
    assert (ts <= pd.to_datetime("2022-09-28")).all()


def test_explain_with_gemini_mock(sample_coastal_df):
    """Test explain_with_gemini with mock LLM client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Retrieved states represent extreme dissipative storm surge conditions during Hurricane Ian with high wave power."
    mock_client.models.generate_content.return_value = mock_response

    spec = PhysicsQuerySpec(raw_query="Hurricane Ian peak", breaker_regime="Plunging")
    top_df = sample_coastal_df.head(5)

    narrative = explain_with_gemini(
        query_text="Hurricane Ian peak",
        spec=spec,
        top_df=top_df,
        client=mock_client,
    )
    assert narrative is not None
    assert "extreme dissipative storm surge" in narrative
