"""Unit and integration tests for Physics-RAG with Gemini Flash and Embeddings Engine."""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
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
    TargetSyntheticValues,
    sanitize_gemini_schema,
    PhysicsQueryParser,
    GeminiPhysicsQueryParser,
    PhysicsRAG,
    PhysicsRAGResult,
    explain_with_gemini,
    explain_hierarchical_with_gemini,
    ParentWindowContext,
    HierarchicalRAGResult,
    HierarchicalPhysicsRAG,
    Baseline4HierarchicalRerankerRAG,
    DEFAULT_PHYSICS_BENCHMARK_QUERIES,
    DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A,
    DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_B,
    ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B,
    ALL_PHYSICS_BENCHMARK_QUERIES_60,
    evaluate_retrieval_query,
    evaluate_rag_benchmark_suite,
    generate_decoupled_mode_summary,
    RawDataLLMRAG,
    RuleBasedSQLRAG,
    HybridPhysicsRAG,
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
    assert res_plunge["filters"].get("xi_max") == 1.0

    res_surge = PhysicsQueryParser.parse("Show reflective surging shorebreak")
    assert res_surge["breaker_regime"] == "Surging"
    assert res_surge["filters"].get("xi_min") == 1.0


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
    assert metrics_spill["bounded_recall_at_k"] == 1.0
    assert metrics_spill["recall_at_k"] == 5.0 / metrics_spill["total_dataset_support"]
    assert metrics_spill["true_recall_at_k"] == metrics_spill["recall_at_k"]
    assert metrics_spill["reciprocal_rank"] == 1.0
    assert metrics_spill["ndcg_at_k"] == 1.0
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
    assert metrics_surge["bounded_recall_at_k"] == 1.0
    assert metrics_surge["recall_at_k"] == 5.0 / metrics_surge["total_dataset_support"]
    assert metrics_surge["ndcg_at_k"] == 1.0
    assert metrics_surge["n_correct"] == 5


def test_graded_ndcg_relevance_function(sample_coastal_df):
    """Verify Graded NDCG accounts for partial matches (grade 1) vs exact matches (grade 2)."""
    rag = PhysicsRAG.from_dataset(sample_coastal_df, scaler_type="robust")

    # Construct a test query specification with graded_relevance_fn
    # Exact condition: surge > 1.0 (Grade 2)
    # Partial condition: surge > 0.5 (Grade 1)
    # Otherwise: Grade 0
    query_dict = {
        "query": "Find severe surge events",
        "condition_fn": lambda row, df: float(row.get("surge_m", 0.0)) >= 1.0,
        "graded_relevance_fn": lambda row, df: 2 if float(row.get("surge_m", 0.0)) >= 1.0 else (1 if float(row.get("surge_m", 0.0)) >= 0.5 else 0),
    }
    metrics = evaluate_retrieval_query(
        rag=rag,
        query_spec_or_text=query_dict,
        k=5,
        use_llm_parser=False,
    )
    assert "graded_scores" in metrics
    assert len(metrics["graded_scores"]) == 5
    assert all(g in (0, 1, 2) for g in metrics["graded_scores"])
    assert 0.0 <= metrics["graded_ndcg_at_k"] <= 1.0



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
    assert summary["Mean Precision@5"] >= 0.0
    assert summary["Mean Regime Purity@5"] >= 0.0
    assert summary["Mean Reciprocal Rank (MRR)"] >= 0.0


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
    assert rag.feature_weights["log_Upsilon"] == 2.5
    assert rag.feature_weights["Water_Temp_C"] == 0.2

    query_vec = rag.embeddings[0].copy()
    sims_weighted = rag._compute_similarity(query_vec)
    assert len(sims_weighted) == len(sample_coastal_df)
    assert np.isclose(sims_weighted[0], 1.0, atol=1e-5)


def test_gated_precision_filtering_zero_leak(sample_coastal_df):
    """Verify Improvement 1: Gated Precision Filtering (Zero-Leak Boundary Mode)."""
    df = sample_coastal_df.copy()
    # Ensure specific test values exist near boundary
    df.loc[0, "hs_m"] = 2.15
    df.loc[1, "hs_m"] = 1.90
    df.loc[0, "tp_s"] = 11.8
    df.loc[1, "tp_s"] = 13.5

    rag = PhysicsRAG.from_dataset(df, scaler_type="robust")

    # 1. Test Hs upper bound leakage: Hs <= 2.0
    spec_hs = PhysicsQuerySpec(raw_query="Waves under 2m", hs_max=2.0)
    mask_strict = rag._apply_hard_and_soft_filters(spec_hs, strict_bounds=True)
    mask_tolerant = rag._apply_hard_and_soft_filters(spec_hs, strict_bounds=False)

    # 2.15m must be excluded under strict_bounds=True (2.15 > 2.0)
    assert not mask_strict[0]
    assert mask_strict[1]
    # 2.15m is allowed under strict_bounds=False (2.15 <= 2.0 * 1.1 = 2.2)
    assert mask_tolerant[0]

    # 2. Test Tp lower bound leakage: Tp >= 13.0
    spec_tp = PhysicsQuerySpec(raw_query="Long period swell over 13s", tp_min=13.0)
    mask_tp_strict = rag._apply_hard_and_soft_filters(spec_tp, strict_bounds=True)
    mask_tp_tolerant = rag._apply_hard_and_soft_filters(spec_tp, strict_bounds=False)

    # 11.8s must be excluded under strict_bounds=True (11.8 < 13.0)
    assert not mask_tp_strict[0]
    assert mask_tp_strict[1]
    # 11.8s is allowed under strict_bounds=False (11.8 >= 13.0 * 0.9 = 11.7)
    assert mask_tp_tolerant[0]


def test_baseline4_hierarchical_reranker(sample_coastal_df):
    """Verify Improvement 6: Baseline 4 Two-Stage Hierarchical Reranker with sequence hydration."""
    df = sample_coastal_df.copy()
    rag_base = PhysicsRAG.from_dataset(df, scaler_type="robust")
    reranker = Baseline4HierarchicalRerankerRAG.from_rag(rag_base)

    assert reranker.use_reranker is True
    assert hasattr(reranker, "con")

    res = reranker.query(
        query_text="Find severe erosion storm events with high wave power flux",
        top_k=3,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )

    assert isinstance(res, HierarchicalRAGResult)
    assert len(res.matches_df) <= 3
    assert len(res.parent_windows) == len(res.matches_df)
    assert len(res.similarities) == len(res.matches_df)
    assert res.explanation is not None


def test_target_synthetic_values_extra_forbid():
    """Verify TargetSyntheticValues forbids unexpected extra attributes."""
    from pydantic import ValidationError

    # Valid initialization
    vals = TargetSyntheticValues(hs_m=3.5, tp_s=12.0, surge_m=0.8)
    assert vals.hs_m == 3.5
    assert vals.tp_s == 12.0
    assert vals.surge_m == 0.8
    assert vals["hs_m"] == 3.5
    assert vals.get("tp_s") == 12.0
    assert "hs_m" in vals
    assert "wind_speed_ms" not in vals
    assert vals.to_dict() == {"hs_m": 3.5, "tp_s": 12.0, "surge_m": 0.8}
    assert bool(vals) is True

    empty_vals = TargetSyntheticValues()
    assert bool(empty_vals) is False

    # Forbid extra invalid properties
    with pytest.raises(ValidationError):
        TargetSyntheticValues(hs_m=3.5, unphysical_bogus_key=999.0)


def test_physics_query_spec_synthetic_values_parsing():
    """Verify PhysicsQuerySpec parses dict into TargetSyntheticValues with extra='forbid'."""
    from pydantic import ValidationError

    spec = PhysicsQuerySpec(
        raw_query="Storm waves over 3m",
        target_synthetic_values={"hs_m": 3.2, "tp_s": 10.0},
    )
    assert isinstance(spec.target_synthetic_values, TargetSyntheticValues)
    assert spec.target_synthetic_values.hs_m == 3.2
    assert spec.target_synthetic_values["tp_s"] == 10.0

    # Extra unmapped properties in target_synthetic_values raise ValidationError
    with pytest.raises(ValidationError):
        PhysicsQuerySpec(
            raw_query="Storm waves over 3m",
            target_synthetic_values={"hs_m": 3.2, "extra_illegal_field": 1.0},
        )


def test_sanitize_gemini_schema():
    """Verify sanitize_gemini_schema recursively removes additionalProperties for GenAI SDK."""
    raw_schema = PhysicsQuerySpec.model_json_schema()
    clean_schema = sanitize_gemini_schema(raw_schema)

    def check_no_additional_properties(obj):
        if isinstance(obj, dict):
            assert "additionalProperties" not in obj
            assert "additional_properties" not in obj
            for v in obj.values():
                check_no_additional_properties(v)
        elif isinstance(obj, list):
            for item in obj:
                check_no_additional_properties(item)

    check_no_additional_properties(clean_schema)


def test_gemini_query_parser_error_surfaced(caplog):
    """Verify API exceptions in GeminiPhysicsQueryParser are logged via logger.error."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

    with caplog.at_level("ERROR"):
        spec = GeminiPhysicsQueryParser.parse(
            query_text="High energy storm waves",
            client=mock_client,
        )

    assert isinstance(spec, PhysicsQuerySpec)
    assert any("Gemini physics query parsing failed" in record.message for record in caplog.records)


def test_raw_data_llm_rag_site_and_date_filtering():
    """Verify RawDataLLMRAG filters correctly by site_id and timestamp window."""
    df = pd.DataFrame({
        "site_id": ["narrabeen_aus", "duck_nc", "torrey_pines_ca"],
        "timestamp": ["2016-06-05 12:00:00", "2016-10-08 12:00:00", "2015-12-15 12:00:00"],
        "hs_m": [4.0, 3.5, 1.2],
        "tp_s": [12.0, 11.0, 13.0],
    })

    rag = RawDataLLMRAG(df=df, use_live_api=False)
    res = rag.query("June 2016 storm at Narrabeen")
    assert len(res.matches_df) == 1
    assert res.matches_df["site_id"].iloc[0] == "narrabeen_aus"
    assert res.parsed_criteria.get("site_id") == "narrabeen_aus"
    assert res.parsed_criteria.get("date_start") == "2016-06-01"


def test_decoupled_benchmark_query_stratification():
    """Verify DEFAULT_PHYSICS_BENCHMARK_QUERIES defaults to Mode A (30 queries) for latent physical manifolds."""
    assert len(DEFAULT_PHYSICS_BENCHMARK_QUERIES) == 30
    assert len(DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A) == 30
    assert len(DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_B) == 30
    assert len(ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B) == 30
    assert len(ALL_PHYSICS_BENCHMARK_QUERIES_60) == 60

    for q in DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A:
        assert "Mode A" in q.get("mode", "")

    for q in ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B:
        assert "Mode B" in q.get("mode", "")

    # Verify Q16 in full suite specifically mentions extreme wave dissipation during active dune scarping
    q16 = ALL_PHYSICS_BENCHMARK_QUERIES_60[16]
    assert "extreme wave dissipation during active dune scarping" in q16["query"].lower()
    assert "Mode A" in q16["mode"]


def test_evaluate_rag_benchmark_suite_configurable_n():
    """Verify n_queries parameter allows slicing 1 to 60 queries while preserving balanced stratification."""
    df = pd.DataFrame({
        "site_id": ["narrabeen_aus", "duck_nc", "torrey_pines_ca", "truc_vert_france"] * 5,
        "timestamp": pd.date_range("2012-01-01", periods=20, freq="6ME"),
        "hs_m": np.linspace(1.0, 5.0, 20),
        "tp_s": np.linspace(8.0, 16.0, 20),
        "total_water_level_m": np.linspace(0.5, 2.5, 20),
        "xi_iribarren": np.linspace(0.2, 1.5, 20),
        "breaker_regime": ["Spilling"] * 7 + ["Plunging"] * 7 + ["Surging"] * 6,
        "p_flux_Wm": np.linspace(10000, 80000, 20),
    })
    rag = RuleBasedSQLRAG(df)

    # Test N=4 with benchmark_mode="all": should return 2 Mode A and 2 Mode B queries
    df_4, _ = evaluate_rag_benchmark_suite(rag=rag, n_queries=4, k_list=[1, 5], benchmark_mode="all")
    assert len(df_4) == 4
    counts_4 = df_4["Mode"].value_counts().to_dict()
    assert counts_4.get("Mode A: Pure Phenomenological", 0) == 2
    assert counts_4.get("Mode B: Full Operational", 0) == 2

    # Test N=4 with default mode_a: should return 4 Mode A queries
    df_4_a, _ = evaluate_rag_benchmark_suite(rag=rag, n_queries=4, k_list=[1, 5])
    assert len(df_4_a) == 4
    assert (df_4_a["Mode"] == "Mode A: Pure Phenomenological").all()

    # Test N=1: should return 1 query
    df_1, _ = evaluate_rag_benchmark_suite(rag=rag, n_queries=1, k_list=[1, 5])
    assert len(df_1) == 1

    # Test mode_a with N=5
    df_a, _ = evaluate_rag_benchmark_suite(rag=rag, benchmark_mode="mode_a", n_queries=5, k_list=[1, 5])
    assert len(df_a) == 5
    assert (df_a["Mode"] == "Mode A: Pure Phenomenological").all()


def test_mode_a_pure_phenomenological_bypasses_filters():
    """Verify that Mode A disables spatiotemporal filtering across all architectures."""
    df = pd.DataFrame({
        "site_id": ["narrabeen_aus", "duck_nc", "torrey_pines_ca", "truc_vert_france"],
        "timestamp": ["2016-06-05 12:00:00", "2016-10-08 12:00:00", "2015-12-15 12:00:00", "2014-01-05 12:00:00"],
        "hs_m": [4.0, 4.2, 4.5, 4.8],
        "tp_s": [12.0, 13.0, 14.0, 15.0],
        "p_flux_Wm": [65000.0, 70000.0, 62000.0, 80000.0],
        "total_water_level_m": [2.1, 2.2, 2.0, 2.3],
        "xi_iribarren": [0.3, 0.35, 0.38, 0.32],
        "breaker_regime": ["Spilling", "Spilling", "Spilling", "Spilling"],
    })

    # Test RuleBasedSQLRAG in Mode A
    sql_rag = RuleBasedSQLRAG(df=df)
    # Mode B would filter to narrabeen only due to the word "Narrabeen"
    res_b = sql_rag.query("Narrabeen high energy storm", top_k=5, mode="Mode B: Full Operational")
    assert len(res_b.matches_df) == 1

    # Mode A bypasses site/date filters completely
    res_a = sql_rag.query("Narrabeen high energy storm", top_k=5, mode="Mode A: Pure Phenomenological")
    assert len(res_a.matches_df) > 1

    # Test RawDataLLMRAG in Mode A
    raw_rag = RawDataLLMRAG(df=df, use_live_api=False)
    raw_res_a = raw_rag.query("Narrabeen June 2016 storm", top_k=5, mode="Mode A: Pure Phenomenological")
    assert len(raw_res_a.matches_df) > 1


def test_evaluate_rag_benchmark_suite_decoupled_modes_and_summary():
    """Verify evaluate_rag_benchmark_suite and generate_decoupled_mode_summary handle Mode A & B."""
    df = pd.DataFrame({
        "site_id": ["narrabeen_aus", "duck_nc", "torrey_pines_ca", "truc_vert_france"] * 5,
        "timestamp": pd.date_range("2012-01-01", periods=20, freq="6ME"),
        "hs_m": np.linspace(1.0, 5.0, 20),
        "tp_s": np.linspace(8.0, 16.0, 20),
        "total_water_level_m": np.linspace(0.5, 2.5, 20),
        "xi_iribarren": np.linspace(0.2, 1.5, 20),
        "breaker_regime": ["Spilling"] * 7 + ["Plunging"] * 7 + ["Surging"] * 6,
        "p_flux_Wm": np.linspace(10000, 80000, 20),
    })

    sql_rag = RuleBasedSQLRAG(df=df)

    # Run on a mini query suite covering both modes
    mini_suite = [
        ARCHIVED_PHYSICS_BENCHMARK_QUERIES_MODE_B[0],   # Mode B
        DEFAULT_PHYSICS_BENCHMARK_QUERIES_MODE_A[0],    # Mode A
    ]

    df_res, summary = evaluate_rag_benchmark_suite(
        rag=sql_rag,
        benchmark_queries=mini_suite,
        k_list=[1, 5],
    )

    assert "Mode" in df_res.columns
    assert "Mode A: Pure Phenomenological" in df_res["Mode"].values
    assert "Mode B: Full Operational" in df_res["Mode"].values
    assert "mode_a_mean_p_at_5" in summary
    assert "mode_b_mean_p_at_5" in summary

    # Verify generate_decoupled_mode_summary
    df_summary = generate_decoupled_mode_summary({"Baseline 2: Rule-Based SQL": (df_res, summary)})
    assert isinstance(df_summary, pd.DataFrame)
    assert "Architecture" in df_summary.columns
    assert "Evaluation Mode" in df_summary.columns
    assert len(df_summary) == 3  # Mode A, Mode B, Overall
    modes = df_summary["Evaluation Mode"].tolist()
    assert "Mode A: Pure Phenomenological" in modes
    assert "Mode B: Full Operational" in modes
    assert "Overall Benchmark" in modes


def test_hybrid_cascade_architecture(sample_coastal_df):
    """Verify Two-Tier Hybrid Cascade executes Tier 1 metadata slicing + Tier 2 latent ranking."""
    from src.physics_rag import HybridPhysicsRAG

    df = sample_coastal_df.copy()
    df["site_id"] = ["duck_nc"] * 50 + ["narrabeen_aus"] * 50
    df["site"] = df["site_id"]

    X_phys, feat_names = build_physics_anchored_feature_matrix(df)
    scaler = RobustScaler().fit(X_phys)
    child_vecs = scaler.transform(X_phys)

    phys_rag = PhysicsRAG(
        df=df,
        embeddings=child_vecs,
        feature_names=feat_names,
        scaler_meta={"scaler": scaler, "scaler_type": "robust"},
    )
    hybrid_rag = HybridPhysicsRAG.from_physics_rag(phys_rag)

    # 1. Operational Query with explicit site
    res_site = hybrid_rag.query(
        "Narrabeen storm waves with plunging breakers",
        top_k=5,
        use_llm_parser=False,
    )
    assert len(res_site.matches_df) > 0
    assert all(res_site.matches_df["site_id"] == "narrabeen_aus")

    # 2. Pure Phenomenological Query without site or date
    res_pheno = hybrid_rag.query(
        "Spilling dissipative breakers with surf similarity under 0.4",
        top_k=5,
        use_llm_parser=False,
    )
    assert len(res_pheno.matches_df) > 0
    # Can retrieve across both sites
    assert "duck_nc" in res_pheno.matches_df["site_id"].values or "narrabeen_aus" in res_pheno.matches_df["site_id"].values


def test_evaluate_peak_impact_disequilibrium_benchmark(sample_coastal_df):
    """Verify Peak Impact & Disequilibrium Ranking benchmark captures temporal and physical metrics."""
    from src.physics_rag import (
        evaluate_peak_impact_disequilibrium_benchmark,
        RuleBasedSQLRAG,
        HybridPhysicsRAG,
    )

    df = sample_coastal_df.copy()
    df["site_id"] = "duck_nc"
    df["site"] = "duck_nc"

    # Inject a distinct storm peak in the middle of the time series
    mid_idx = 50
    df.loc[mid_idx, "hs_m"] = 5.2
    df.loc[mid_idx, "upsilon"] = 85000.0
    df.loc[mid_idx, "tp_s"] = 14.5
    peak_ts = df.loc[mid_idx, "timestamp"]

    X_phys, feat_names = build_physics_anchored_feature_matrix(df)
    scaler = RobustScaler().fit(X_phys)
    child_vecs = scaler.transform(X_phys)

    phys_rag = PhysicsRAG(
        df=df,
        embeddings=child_vecs,
        feature_names=feat_names,
        scaler_meta={"scaler": scaler, "scaler_type": "robust"},
    )
    hybrid_rag = HybridPhysicsRAG.from_physics_rag(phys_rag)
    sql_rag = RuleBasedSQLRAG(df=df)

    rag_engines = {
        "Flat SQL": sql_rag,
        "Hybrid Cascade": hybrid_rag,
    }

    test_queries = [
        {
            "query": "Duck severe nor'easter storm peak with extreme wave heights",
            "category": "1. Historical Storm Events",
            "mask_fn": lambda d: d["site_id"] == "duck_nc",
            "condition_fn": lambda row, d: float(row.get("hs_m", 0.0)) >= 3.0,
        }
    ]

    df_details, df_summary = evaluate_peak_impact_disequilibrium_benchmark(
        rag_engines=rag_engines,
        df=df,
        benchmark_queries=test_queries,
        k=5,
        use_llm_parser=False,
    )

    assert len(df_details) == 1
    assert "Flat SQL_dt_peak_hrs" in df_details.columns
    assert "Hybrid Cascade_dt_peak_hrs" in df_details.columns
    assert "Hybrid Cascade_top1_upsilon" in df_details.columns

    assert len(df_summary) == 2
    assert "Median |Δt_peak| Error (hrs)" in df_summary.columns
    assert "Median Top-1 Upsilon" in df_summary.columns
    assert "Median Top-1 Wave Power (kW/m)" in df_summary.columns

    # Verify valid numeric outputs are produced for both architectures
    sql_dt = df_details["Flat SQL_dt_peak_hrs"].iloc[0]
    hyb_dt = df_details["Hybrid Cascade_dt_peak_hrs"].iloc[0]
    assert np.isfinite(sql_dt)
    assert np.isfinite(hyb_dt)
    assert df_details["Hybrid Cascade_top1_upsilon"].iloc[0] >= 0.0


def test_directional_pareto_ranking(sample_coastal_df: pd.DataFrame) -> None:
    """Verify Directional Pareto Ranking rewards energetic overshoot rather than penalizing candidate distance."""
    from src.physics_rag import HybridPhysicsRAG

    df = sample_coastal_df.iloc[:10].copy().reset_index(drop=True)
    # Ensure hs_m, tp_s, and upsilon are present
    df["hs_m"] = [1.0, 2.0, 3.5, 4.8, 1.2, 2.5, 3.6, 5.2, 1.1, 2.1]
    df["tp_s"] = [8.0, 9.0, 11.0, 14.0, 8.5, 9.5, 11.2, 15.0, 8.0, 9.0]
    df["upsilon"] = [100.0, 500.0, 2000.0, 8000.0, 150.0, 600.0, 2100.0, 12000.0, 110.0, 550.0]

    rag = HybridPhysicsRAG(df=df)
    # Query specifying severe storm with lower bound Hs >= 3.5m
    spec = PhysicsQuerySpec(
        raw_query="Storm with wave height exceeding 3.5m",
        hs_min=3.5,
        target_synthetic_values=TargetSyntheticValues(hs_m=3.5, tp_s=11.0),
    )
    result = rag.query("Storm with wave height exceeding 3.5m", spec=spec, top_k=3, use_llm_parser=False)
    matches = result.matches_df

    # Verify that the highest wave energy events (4.8m and 5.2m) are ranked at the top
    assert len(matches) == 3
    top_hs = matches["hs_m"].tolist()
    # With directional Pareto ranking, 5.2m and 4.8m should be prioritized over 3.5m/3.6m
    assert top_hs[0] in [5.2, 4.8]
    assert max(top_hs) >= 4.8


def test_site_normalized_disequilibrium() -> None:
    """Verify site-specific normalization scales raw Upsilon into dimensionless Delta Upsilon* across sites."""
    from src.embeddings import compute_site_equilibrium_thresholds, compute_site_normalized_disequilibrium

    # Construct two contrasting sites: Duck (low Upsilon) and Truc Vert (high Upsilon)
    duck_ups = np.array([100.0, 200.0, 350.0, 500.0, 1200.0, 3000.0, 6500.0])
    tv_ups = np.array([1000.0, 2000.0, 3000.0, 5000.0, 15000.0, 30000.0, 46000.0])

    df = pd.DataFrame({
        "site_id": ["duck_nc"] * len(duck_ups) + ["truc_vert_france"] * len(tv_ups),
        "upsilon": np.concatenate([duck_ups, tv_ups]),
    })

    thresholds = compute_site_equilibrium_thresholds(df)
    assert "duck_nc" in thresholds
    assert "truc_vert_france" in thresholds

    # Truc Vert threshold should be vastly larger than Duck
    assert thresholds["truc_vert_france"]["eq"] > thresholds["duck_nc"]["eq"]

    df["Delta_Upsilon_Star"] = compute_site_normalized_disequilibrium(df, thresholds=thresholds)
    assert "Delta_Upsilon_Star" in df.columns

    # Median value at each site should be exactly or very near 0.0
    duck_norm = df[df["site_id"] == "duck_nc"]["Delta_Upsilon_Star"]
    tv_norm = df[df["site_id"] == "truc_vert_france"]["Delta_Upsilon_Star"]
    assert np.isclose(duck_norm.median(), 0.0, atol=1e-3)
    assert np.isclose(tv_norm.median(), 0.0, atol=1e-3)

    # Extreme storm events at both sites should be scaled to O(1) disequilibrium
    assert duck_norm.max() > 0.5
    assert tv_norm.max() > 0.5


def test_peak_hydrodynamic_work_graded_ndcg(sample_coastal_df: pd.DataFrame) -> None:
    """Verify 4-tier graded relevance and Graded NDCG evaluation for peak hydrodynamic work."""
    from src.physics_rag import _make_graded_fn, evaluate_retrieval_query, RuleBasedSQLRAG

    df = sample_coastal_df.iloc[:10].copy().reset_index(drop=True)
    # Create timestamps and wave heights
    df["timestamp"] = pd.date_range("2016-06-01", periods=len(df), freq="12h", tz="UTC")
    df["hs_m"] = [1.5, 1.8, 2.2, 3.2, 4.5, 4.8, 3.0, 2.0, 1.5, 1.2]
    df["site_id"] = "narrabeen_aus"

    exact_cond = lambda row, d: float(row.get("hs_m", 0.0)) >= 3.0
    partial_cond = lambda row, d: float(row.get("hs_m", 0.0)) >= 2.0
    peak_work_cond = lambda row, d: float(row.get("hs_m", 0.0)) >= 4.0

    graded_fn = _make_graded_fn(exact_cond, partial_cond, peak_work_cond)

    # Test scoring:
    # row 4 (4.5m) and row 5 (4.8m) -> Grade 3
    assert graded_fn(df.iloc[4], df) == 3
    assert graded_fn(df.iloc[5], df) == 3
    # row 3 (3.2m) and row 6 (3.0m) -> Grade 2
    assert graded_fn(df.iloc[3], df) == 2
    assert graded_fn(df.iloc[6], df) == 2
    # row 2 (2.2m) and row 7 (2.0m) -> Grade 1
    assert graded_fn(df.iloc[2], df) == 1
    assert graded_fn(df.iloc[7], df) == 1
    # row 0 (1.5m) -> Grade 0
    assert graded_fn(df.iloc[0], df) == 0

    # Evaluate query with evaluate_retrieval_query
    q_spec = {
        "query": "Narrabeen June 2016 storm peak",
        "category": "1. Historical Storm Events",
        "condition_fn": exact_cond,
        "peak_work_cond": peak_work_cond,
        "graded_relevance_fn": graded_fn,
    }
    rag = RuleBasedSQLRAG(df=df)
    eval_res = evaluate_retrieval_query(rag, q_spec, k=5)
    assert "graded_ndcg_at_k" in eval_res
    assert 0.0 <= eval_res["graded_ndcg_at_k"] <= 1.0


def test_hybrid_rag_target_synthetic_values_with_none(sample_coastal_df):
    """Ensure HybridPhysicsRAG handles target_synthetic_values containing NoneType without TypeError."""
    rag = HybridPhysicsRAG(sample_coastal_df)

    # 1. Test with default TargetSyntheticValues instance (all fields None)
    spec_model = PhysicsQuerySpec(
        raw_query="Narrabeen East Coast Low storm erosion exceeding 4m",
        intent_summary="Extreme storm erosion",
        target_synthetic_values=TargetSyntheticValues(),
    )
    res_model = rag.query("Narrabeen East Coast Low storm erosion exceeding 4m", spec=spec_model)
    assert len(res_model.matches_df) > 0

    # 2. Test with dict containing explicit None fields
    spec_dict = PhysicsQuerySpec(
        raw_query="Narrabeen East Coast Low storm erosion exceeding 4m",
        intent_summary="Extreme storm erosion",
        target_synthetic_values={"hs_m": None, "tp_s": None, "surge_m": None, "p_flux_Wm": None},
    )
    res_dict = rag.query("Narrabeen East Coast Low storm erosion exceeding 4m", spec=spec_dict)
    assert len(res_dict.matches_df) > 0









