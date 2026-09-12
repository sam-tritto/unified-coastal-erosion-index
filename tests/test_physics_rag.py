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
    classify_agentic_latent_phases,
    prune_subcritical_noise_for_agent,
    extract_recovery_timeframes,
)
from src.physics_rag import (
    PhysicsQuerySpec,
    PhysicsQueryParser,
    GeminiPhysicsQueryParser,
    PhysicsRAG,
    PhysicsRAGResult,
    explain_with_gemini,
    explain_hierarchical_with_gemini,
    ParentWindowContext,
    HierarchicalRAGResult,
    HierarchicalPhysicsRAG,
    DEFAULT_PHYSICS_BENCHMARK_QUERIES,
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
    u_crit = np.full(n, 500.0)
    u_excess = np.maximum(0.0, ups - u_crit)
    u_eq = np.full(n, 2500.0)
    u_diseq = ups - u_eq
    signed_rate = -1.2e-5 * np.maximum(ups - u_eq, 0.0) + 1.8e-5 * np.maximum(u_eq - ups, 0.0)
    dx = np.where(ups > u_eq, -np.random.uniform(0.1, 1.2, n), np.random.uniform(0.05, 0.5, n))
    
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
        "upsilon_crit": u_crit,
        "upsilon_excess": u_excess,
        "upsilon_eq": u_eq,
        "upsilon_diseq": u_diseq,
        "signed_equilibrium_rate": signed_rate,
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


def test_evaluate_retrieval_query_precision_and_purity(sample_coastal_df):
    """Test evaluating Precision@k, Regime Purity@k, Recall@k, and Reciprocal Rank for single queries."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    # 1. Regime query (Spilling)
    metrics_spill = rag.evaluate_query(
        query="Show me all the Spilling breakers",
        target_regime="Spilling",
        k=5,
        use_llm_parser=False,
    )
    assert metrics_spill["precision_at_k"] == 1.0
    assert metrics_spill["regime_purity"] == 1.0
    assert metrics_spill["recall_at_k"] == 1.0
    assert metrics_spill["reciprocal_rank"] == 1.0
    assert metrics_spill["n_retrieved"] == 5
    assert metrics_spill["total_dataset_support"] > 0

    # 2. Surge query (> 0.5m)
    metrics_surge = rag.evaluate_query(
        query="Find positive storm surge elevation above 0.5m",
        condition_fn=lambda row, df: float(row.get("surge_m", 0.0)) >= 0.5,
        k=5,
        use_llm_parser=False,
    )
    assert metrics_surge["precision_at_k"] == 1.0
    assert metrics_surge["reciprocal_rank"] == 1.0
    assert metrics_surge["n_correct"] == 5


def test_suggestion_8_threshold_anchor_noise_pruning(sample_coastal_df):
    """Verify Suggestion 8: Threshold Anchor (Upsilon_crit) prunes subcritical noise to reduce context-window bloat."""
    phases = classify_agentic_latent_phases(sample_coastal_df, upsilon_crit=500.0)
    assert "latent_phase" in phases.columns
    assert "is_background_noise" in phases.columns
    assert "is_active_forcing" in phases.columns
    
    # Prune subcritical noise
    filtered_df, filtered_emb, stats = prune_subcritical_noise_for_agent(sample_coastal_df, upsilon_crit=500.0)
    assert len(filtered_df) == stats["retained_active_timesteps"]
    assert stats["raw_timesteps"] == len(sample_coastal_df)
    assert stats["pruned_noise_timesteps"] >= 0
    assert stats["context_compression_ratio_pct"] >= 0.0
    assert (filtered_df["upsilon"] >= 500.0).all()


def test_suggestion_8_state_space_recovery_navigation(sample_coastal_df):
    """Verify Suggestion 8: State-Space Navigation retrieves recovery phases (+dX) via Signed Disequilibrium."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    
    # Check centroid for recovery
    assert "State_PostStormRecovery" in rag.centroids
    
    # Query for post-storm beach recovery phases
    res = rag.query(
        "Find post-storm beach recovery phases and recovery timeframes",
        top_k=5,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )
    assert isinstance(res, PhysicsRAGResult)
    assert len(res.matches_df) > 0
    # Verified: all retrieved states have negative disequilibrium (constructive recovery) or positive dx
    assert (res.matches_df["upsilon_diseq"] < 0).all() or (res.matches_df["delta_x_m"] >= 0).all()
    assert "recovery" in res.explanation.lower() or "berm" in res.explanation.lower()


def test_suggestion_8_extract_recovery_timeframes(sample_coastal_df):
    """Verify Suggestion 8: extract_recovery_timeframes calculates duration, rates, and cumulative accretion."""
    episodes = extract_recovery_timeframes(sample_coastal_df, min_recovery_steps=2)
    assert isinstance(episodes, list)
    if len(episodes) > 0:
        ep = episodes[0]
        assert "duration_hours" in ep
        assert "duration_days" in ep
        assert "simulated_accretion_m" in ep
        assert "mean_recovery_rate_m_day" in ep
        assert ep["mean_upsilon_diseq"] < 0


def test_suggestion_8_rag_query_with_noise_filtering(sample_coastal_df):
    """Verify Suggestion 8: Natural language query with background noise pruning."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    
    res = rag.query(
        "Show active storm erosion above critical threshold ignoring background noise",
        top_k=5,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )
    assert len(res.matches_df) > 0
    assert (res.matches_df["upsilon"] >= 500.0).all()
    assert "subcritical" in res.explanation.lower() or "pruned" in res.explanation.lower()


def test_evaluate_rag_benchmark_suite(sample_coastal_df):
    """Test full multi-k RAG benchmark evaluation suite."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    df_results, summary = rag.evaluate_benchmark(
        k_list=[1, 3, 5],
        use_llm_parser=False,
        allow_synthetic_probes=True,
    )

    assert not df_results.empty
    assert len(df_results) == len(DEFAULT_PHYSICS_BENCHMARK_QUERIES)  # 30 grand benchmark queries
    assert "Query" in df_results.columns
    assert "Category" in df_results.columns
    assert "P@1" in df_results.columns
    assert "P@3" in df_results.columns
    assert "P@5" in df_results.columns
    assert "Regime Purity@5" in df_results.columns
    assert "MRR" in df_results.columns
    assert "Recall@5" in df_results.columns
    assert "Mean Sim@5" in df_results.columns

    # Check macro-averaged summary metrics
    assert "Mean Precision@1" in summary
    assert "Mean Precision@5" in summary
    assert "Mean Regime Purity@5" in summary
    assert "Mean Reciprocal Rank (MRR)" in summary
    assert summary["Mean Precision@5"] > 0.50
    assert summary["Mean Regime Purity@5"] > 0.40
    assert summary["Mean Reciprocal Rank (MRR)"] > 0.60


def test_parent_window_context_model(sample_coastal_df):
    """Test ParentWindowContext data model and serialization."""
    pw = ParentWindowContext(
        match_index=0,
        event_timestamp="2022-09-21 12:00:00",
        similarity=0.95,
        window_start="2022-09-18 12:00:00",
        window_end="2022-09-22 12:00:00",
        pre_window_hours=72.0,
        post_window_hours=24.0,
        parent_df=sample_coastal_df.iloc[:10],
        summary_stats={"peak_hs_m": 4.2, "net_delta_x_m": -0.8},
    )
    assert pw.similarity == 0.95
    assert pw.summary_stats["peak_hs_m"] == 4.2
    assert len(pw.parent_df) == 10


def test_hierarchical_physics_rag_initialization_and_window_hydration(sample_coastal_df, tmp_path):
    """Test HierarchicalPhysicsRAG DuckDB registration, window slicing, and Parquet caching."""
    parquet_file = str(tmp_path / "test_sensors.parquet")
    
    hrag = HierarchicalPhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    hrag.export_parent_parquet(parquet_file)
    assert (tmp_path / "test_sensors.parquet").exists()

    # Hydrate window around middle timestamp
    mid_ts = sample_coastal_df.iloc[30]["timestamp"]
    pw = hrag.hydrate_parent_window(
        event_timestamp=mid_ts,
        pre_window_hours=48.0,
        post_window_hours=24.0,
    )

    assert isinstance(pw, ParentWindowContext)
    assert not pw.parent_df.empty
    assert "peak_hs_m" in pw.summary_stats
    assert "hs_rise_rate_m_per_hr" in pw.summary_stats
    assert "net_delta_x_m" in pw.summary_stats
    assert "mean_upsilon" in pw.summary_stats


def test_hierarchical_physics_rag_query(sample_coastal_df):
    """Test full hierarchical parent-child query pipeline with DuckDB hydration."""
    hrag = HierarchicalPhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    res = hrag.query(
        "Find high energy storm waves over 3m with beach erosion",
        top_k=3,
        pre_window_hours=72.0,
        post_window_hours=24.0,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )

    assert isinstance(res, HierarchicalRAGResult)
    assert len(res.child_matches_df) == 3
    assert len(res.parent_windows) == 3
    assert all(isinstance(pw, ParentWindowContext) for pw in res.parent_windows)
    
    summary_text = res.summary()
    assert "Hierarchical Parent-Child Physics-RAG" in summary_text
    assert "Hydrated Parent Context Windows" in summary_text
    assert "Peak Hs" in summary_text


def test_physics_rag_query_hierarchical_compatibility(sample_coastal_df):
    """Verify PhysicsRAG backwards compatibility with query_hierarchical()."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    res = rag.query_hierarchical(
        "Show me all the Spilling breakers",
        top_k=2,
        pre_window_hours=48.0,
        post_window_hours=12.0,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )

    assert isinstance(res, HierarchicalRAGResult)
    assert len(res.child_matches_df) == 2
    assert len(res.parent_windows) == 2


def test_explain_hierarchical_with_gemini_mock(sample_coastal_df):
    """Test explain_hierarchical_with_gemini using a mock client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The antecedent 72h wave buildup shows sustained wave power flux driving significant foreshore scarping."
    mock_client.models.generate_content.return_value = mock_response

    hrag = HierarchicalPhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    res = hrag.query(
        "Find high energy storm waves over 3m",
        top_k=2,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )

    spec = PhysicsQuerySpec(raw_query="Find high energy storm waves over 3m", hs_min=3.0)
    narrative = explain_hierarchical_with_gemini(
        query_text="Find high energy storm waves over 3m",
        spec=spec,
        child_df=res.child_matches_df,
        parent_windows=res.parent_windows,
        client=mock_client,
    )

    assert narrative is not None
    assert "antecedent" in narrative.lower()


def test_query_failure_modes(sample_coastal_df):
    """Verify automated query for hydrodynamic failure modes and model under-prediction across archetypes."""
    df_test = sample_coastal_df.copy()
    df_test["site"] = "duck_nc"
    # Inject a strong under-prediction episode (severe negative delta_x with zero/low Upsilon)
    df_test.loc[10, "delta_x_m"] = -5.0
    df_test.loc[10, "upsilon"] = 0.0
    df_test.loc[10, "hs_m"] = 4.0
    df_test.loc[10, "surge_m"] = 1.2

    hrag = HierarchicalPhysicsRAG.from_dataset(df_test, scaler_type="robust")
    res = hrag.query_failure_modes(
        query_text="When does Upsilon under-predict erosion?",
        top_k_per_site=2,
        pre_window_hours=24.0,
        post_window_hours=12.0,
    )

    assert isinstance(res, dict)
    assert "failure_episodes_df" in res
    assert "parent_windows" in res
    assert "diagnostic_summary" in res
    df_fail = res["failure_episodes_df"]
    assert isinstance(df_fail, pd.DataFrame)
    assert not df_fail.empty
    assert "Physical Root Cause" in df_fail.columns
    assert "Discrepancy Score" in df_fail.columns


def test_expanded_phenomenon_centroids(sample_coastal_df):
    """Verify precomputation of Item 4 phenomenon-level centroids."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")
    centroids = rag.centroids

    # Standard centroids
    assert "Breaker_Plunging" in centroids or "Breaker_Spilling" in centroids
    assert "State_ExtremeStormPeak" in centroids or "State_StormErosion" in centroids

    # Expanded Item 4 phenomenon centroids
    assert "Centroid_InfragravitySwash" in centroids or "State_InfragravitySwash" in centroids


def test_phenomenon_centroid_routing(sample_coastal_df):
    """Verify _build_query_vector routes to phenomenon centroids when target values are absent."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    # Infragravity focus query without explicit target values
    spec_ig = PhysicsQuerySpec(raw_query="Show infragravity swash resonance", focus="Infragravity")
    vec_ig = rag._build_query_vector(spec_ig)
    assert vec_ig is not None
    assert len(vec_ig) == rag.embeddings.shape[1]
    if "Centroid_InfragravitySwash" in rag.centroids:
        assert np.allclose(vec_ig, rag.centroids["Centroid_InfragravitySwash"])


def test_upsilon_weighted_similarity(sample_coastal_df):
    """Verify Item 5 Upsilon-weighted cosine similarity computation."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust", use_weighted_similarity=True)
    assert rag.use_weighted_similarity is True
    assert hasattr(rag, "feature_weights")
    assert rag.feature_weights["log_Upsilon"] == 1.2
    assert rag.feature_weights["Water_Temp_C"] == 0.2

    query_vec = rag.embeddings[0].copy()
    sims_weighted = rag._compute_similarity(query_vec)
    assert len(sims_weighted) == len(sample_coastal_df)
    assert np.isclose(sims_weighted[0], 1.0, atol=1e-5)



