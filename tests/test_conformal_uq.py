"""Unit and integration tests for Conformalized Quantile Regression (CQR) in Physics-RAG."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.conformal_uq import (
    PinballLoss,
    compute_pinball_loss_np,
    ConformalizedQuantileRegressor,
    format_conformal_prompt_section,
    HAS_TORCH,
)
from src.physics_rag import (
    PhysicsRAG,
    HierarchicalPhysicsRAG,
    PhysicsRAGResult,
    HierarchicalRAGResult,
    PhysicsQuerySpec,
    ParentWindowContext,
    explain_with_gemini,
    explain_hierarchical_with_gemini,
)


@pytest.fixture
def synthetic_coastal_cqr_data() -> pd.DataFrame:
    """Generate synthetic coastal hydrodynamic observations with heteroscedastic volatility."""
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2023-01-01", periods=n, freq="6h")

    hs = np.random.uniform(0.5, 5.0, n)
    tp = np.random.uniform(4.0, 16.0, n)
    surge = np.random.uniform(-0.3, 1.5, n)
    wind_norm = np.random.uniform(-10.0, 15.0, n)
    tan_beta = 0.08
    g = 9.81
    xi = tan_beta * np.sqrt(g / (2 * np.pi)) * (tp / np.sqrt(hs))
    dean = hs / (0.048 * tp)
    p_flux = (1025.0 * (g ** 2) / (16.0 * 2 * np.pi)) * (hs ** 2) * tp
    ups = (1025.0 * (g ** 1.5) / (16.0 * (2 * np.pi) ** 1.5)) * tan_beta * ((hs ** 2.5) * tp / 0.048)
    u_eq = 2500.0
    u_diseq = ups - u_eq

    # Heteroscedastic noise: extreme volatility during storm surge and large wave heights
    volatility = 0.15 + 0.6 * (hs / 5.0) ** 2 + 0.4 * np.maximum(surge, 0.0)
    # Erosive beach cut during storms, subtle accretion during calm periods
    true_delta_x = -0.4 * hs - 0.7 * np.maximum(surge, 0.0) + 0.1 * (hs < 1.2) + np.random.normal(0, volatility)

    return pd.DataFrame({
        "timestamp": dates,
        "hs_m": hs,
        "tp_s": tp,
        "surge_m": surge,
        "wind_normal_ms": wind_norm,
        "xi_iribarren": xi,
        "dean_number": dean,
        "p_flux_Wm": p_flux,
        "upsilon": ups,
        "upsilon_diseq": u_diseq,
        "log_upsilon": np.log10(np.maximum(ups, 1.0)),
        "delta_x_m": true_delta_x,
    })


def test_pinball_loss_formula():
    """Verify pinball loss matches mathematical check function: max(tau * diff, (tau - 1) * diff)."""
    y_true = np.array([2.0, -1.0, 0.5])
    y_pred = np.array([1.5, -0.5, 1.0])

    for tau in [0.05, 0.50, 0.95]:
        loss_np = compute_pinball_loss_np(y_true, y_pred, tau)
        # Check manually
        diff = y_true - y_pred
        expected = np.mean(np.where(diff >= 0, tau * diff, (tau - 1.0) * diff))
        assert np.isclose(loss_np, expected), f"Mismatch for tau={tau}"

    if HAS_TORCH:
        import torch
        loss_fn = PinballLoss(quantiles=(0.05, 0.50, 0.95))
        preds_t = torch.tensor(np.column_stack([y_pred, y_pred, y_pred]), dtype=torch.float32)
        targets_t = torch.tensor(y_true, dtype=torch.float32)
        loss_torch = loss_fn(preds_t, targets_t).item()

        expected_mean = np.mean([
            compute_pinball_loss_np(y_true, y_pred, 0.05),
            compute_pinball_loss_np(y_true, y_pred, 0.50),
            compute_pinball_loss_np(y_true, y_pred, 0.95),
        ])
        assert np.isclose(loss_torch, expected_mean, atol=1e-5)


def test_cqr_fit_and_calibration(synthetic_coastal_cqr_data):
    """Test fitting, calibration, and finite-sample coverage guarantee of CQR."""
    df = synthetic_coastal_cqr_data
    train_df = df.iloc[:350]
    cal_df = df.iloc[350:475]
    test_df = df.iloc[475:]

    cqr = ConformalizedQuantileRegressor(
        alpha=0.10,
        target_col="delta_x_m",
        model_backend="gradient_boosting",
        random_state=42,
    )
    cqr.fit(train_df)
    assert not cqr.is_calibrated

    # Calibrate non-conformity scores
    cqr.calibrate(cal_df)
    assert cqr.is_calibrated
    assert cqr.calibration_scores is not None
    assert len(cqr.calibration_scores) == len(cal_df)

    # Evaluate out-of-sample coverage
    metrics = cqr.evaluate_coverage(test_df)
    assert metrics["nominal_coverage"] == pytest.approx(0.90, rel=1e-2)
    # Coverage should be at or close to 90% (e.g. >= 85% on finite test sample)
    assert metrics["empirical_coverage"] >= 0.85
    assert metrics["mean_interval_width"] > 0.0
    assert metrics["winkler_score"] > 0.0


def test_cqr_dynamic_interval_stretching(synthetic_coastal_cqr_data):
    """Verify that prediction intervals dynamically stretch during high-volatility storm surges."""
    df = synthetic_coastal_cqr_data
    train_df = df.iloc[:350]
    cal_df = df.iloc[350:475]

    cqr = ConformalizedQuantileRegressor(alpha=0.10, random_state=42)
    cqr.fit(train_df).calibrate(cal_df)

    # Construct two contrasting hydrodynamic states:
    # 1. Calm swash equilibrium: Hs=0.6m, Surge=-0.1m, small Upsilon
    calm_state = pd.DataFrame([{
        "hs_m": 0.6,
        "tp_s": 8.0,
        "surge_m": -0.1,
        "wind_normal_ms": 2.0,
        "xi_iribarren": 1.2,
        "dean_number": 1.5,
        "p_flux_Wm": 2500.0,
        "upsilon": 400.0,
        "upsilon_diseq": -2100.0,
        "log_upsilon": np.log10(400.0),
    }])

    # 2. Extreme hurricane storm surge shock: Hs=4.8m, Surge=1.4m, massive Upsilon
    storm_state = pd.DataFrame([{
        "hs_m": 4.8,
        "tp_s": 14.0,
        "surge_m": 1.4,
        "wind_normal_ms": 14.0,
        "xi_iribarren": 0.4,
        "dean_number": 7.0,
        "p_flux_Wm": 85000.0,
        "upsilon": 18000.0,
        "upsilon_diseq": 15500.0,
        "log_upsilon": np.log10(18000.0),
    }])

    pred_calm = cqr.predict_conformal_interval(calm_state)
    pred_storm = cqr.predict_conformal_interval(storm_state)

    width_calm = pred_calm["interval_width"].iloc[0]
    width_storm = pred_storm["interval_width"].iloc[0]

    # The storm interval must be significantly wider due to heteroscedastic pinball bounds
    assert width_storm > width_calm, f"Storm width ({width_storm:.2f}) should exceed calm width ({width_calm:.2f})"
    assert pred_calm["conformal_lower"].iloc[0] < pred_calm["conformal_upper"].iloc[0]
    assert pred_storm["conformal_lower"].iloc[0] < pred_storm["conformal_upper"].iloc[0]


def test_format_conformal_prompt_section():
    """Verify that prompt formatting outputs structured, statistically grounded sections."""
    payload = {
        "target_variable": "delta_x_m",
        "nominal_coverage": 0.90,
        "point_estimate": -2.45,
        "conformal_lower": -3.80,
        "conformal_upper": -1.10,
        "interval_width": 2.70,
        "raw_q_lo": -3.60,
        "raw_q_hi": -1.30,
        "q_conformal": 0.20,
        "volatility_regime": "Extreme Storm Surge Volatility (Substantial Dynamic Interval Expansion)",
    }
    prompt_sec = format_conformal_prompt_section(payload)
    assert "Decision-Grade Uncertainty Quantification" in prompt_sec
    assert "90.0%" in prompt_sec
    assert "[ -3.80 m, -1.10 m ]" in prompt_sec
    assert "Extreme Storm Surge Volatility" in prompt_sec


def test_rag_result_summary_conformal_uq():
    """Verify PhysicsRAGResult and HierarchicalRAGResult summaries contain conformal UQ."""
    df_match = pd.DataFrame([{"hs_m": 2.5, "delta_x_m": -1.2, "timestamp": "2023-05-01"}])
    uq_payload = {
        "target_variable": "delta_x_m",
        "nominal_coverage": 0.90,
        "point_estimate": -1.20,
        "conformal_lower": -2.50,
        "conformal_upper": +0.10,
        "interval_width": 2.60,
        "q_conformal": 0.15,
        "volatility_regime": "Moderate Baseline Hydrodynamics",
    }

    # Test PhysicsRAGResult
    res = PhysicsRAGResult(
        query="storm scour",
        matches_df=df_match,
        similarities=np.array([0.92]),
        explanation="Storm scour event",
        parsed_criteria=PhysicsQuerySpec(raw_query="storm scour"),
        conformal_uq=uq_payload,
    )
    summary = res.summary()
    assert "Decision-Grade Uncertainty Quantification" in summary
    assert "[ -2.50 m, +0.10 m ]" in summary

    # Test HierarchicalRAGResult
    hres = HierarchicalRAGResult(
        query="storm scour",
        child_matches_df=df_match,
        parent_windows=[],
        similarities=np.array([0.92]),
        explanation="Hierarchical explanation",
        parsed_criteria=PhysicsQuerySpec(raw_query="storm scour"),
        conformal_uq=uq_payload,
    )
    h_summary = hres.summary()
    assert "Decision-Grade Uncertainty Quantification" in h_summary
    assert "[ -2.50 m, +0.10 m ]" in h_summary


def test_explain_with_gemini_injects_conformal_bounds():
    """Verify explain_with_gemini passes conformal bounds into the Gemini prompt."""
    spec = PhysicsQuerySpec(raw_query="Erosive nor'easter storm cut")
    df = pd.DataFrame([{
        "timestamp": "2023-03-01",
        "hs_m": 3.5,
        "tp_s": 12.0,
        "surge_m": 0.8,
        "delta_x_m": -2.1,
    }])
    uq_payload = {
        "target_variable": "delta_x_m",
        "nominal_coverage": 0.90,
        "point_estimate": -2.10,
        "conformal_lower": -3.40,
        "conformal_upper": -0.80,
        "interval_width": 2.60,
        "q_conformal": 0.25,
        "volatility_regime": "Elevated Wave Energy / Disequilibrium Shock",
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Grounded physical analysis with guaranteed conformal interval."
    mock_client.models.generate_content.return_value = mock_resp

    narrative = explain_with_gemini(
        query_text="Erosive nor'easter storm cut",
        spec=spec,
        top_df=df,
        client=mock_client,
        conformal_uq=uq_payload,
    )
    assert narrative == "Grounded physical analysis with guaranteed conformal interval."

    # Inspect prompt passed to client
    called_prompt = mock_client.models.generate_content.call_args[1]["contents"]
    assert "Decision-Grade Uncertainty Quantification" in called_prompt
    assert "[ -3.40 m, -0.80 m ]" in called_prompt
    assert "Elevated Wave Energy / Disequilibrium Shock" in called_prompt


def test_hierarchical_rag_query_with_cqr(synthetic_coastal_cqr_data):
    """Integration test: HierarchicalPhysicsRAG querying with conformal_regressor."""
    df = synthetic_coastal_cqr_data
    train_df = df.iloc[:350]
    cal_df = df.iloc[350:475]

    cqr = ConformalizedQuantileRegressor(alpha=0.10, random_state=42)
    cqr.fit(train_df).calibrate(cal_df)

    rag = HierarchicalPhysicsRAG(
        df=df,
        conformal_regressor=cqr,
        use_reranker=False,
    )

    result = rag.query(
        query_text="Extreme storm surge erosion",
        top_k=3,
        use_llm_parser=False,
        use_llm_synthesis=False,
    )

    assert isinstance(result, HierarchicalRAGResult)
    assert result.conformal_uq is not None
    assert "conformal_lower" in result.conformal_uq
    assert "conformal_upper" in result.conformal_uq
    assert result.conformal_uq["conformal_lower"] < result.conformal_uq["conformal_upper"]

    # Verify summary includes UQ
    sum_text = result.summary()
    assert "Decision-Grade Uncertainty Quantification" in sum_text
