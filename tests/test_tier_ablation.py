"""Unit tests for 4-tier constructive physical ablation in src/models.py."""

import pytest
import numpy as np
import pandas as pd

from src.data import generate_frf_duck_hurricane_ian_dataset
from src.models import run_constructive_tier_ablation


@pytest.fixture
def synthetic_tier_dataset() -> pd.DataFrame:
    """Generate synthetic coastal dataset with explicit physical tiers."""
    df = generate_frf_duck_hurricane_ian_dataset(random_seed=42)
    # Ensure tier columns exist
    df["upsilon_tier1"] = df["upsilon"] if "upsilon" in df.columns else df["hs_m"] ** 2.5
    df["upsilon_tier2"] = df["upsilon_tier1"] * 1.05 + 0.1
    df["upsilon_tier3"] = df["upsilon_tier2"] * 1.02 - 0.05
    df["upsilon_tier4"] = df["upsilon_tier3"] * 0.99 + 0.02
    df["dean_number"] = df["tp_s"] / (df["hs_m"] + 0.5)
    return df


def test_run_constructive_tier_ablation_mlp(synthetic_tier_dataset: pd.DataFrame):
    """Verify constructive tier ablation using PyTorch MLP with out-of-sample bootstrap CIs."""
    summary_df, details = run_constructive_tier_ablation(
        df=synthetic_tier_dataset,
        target_col="shoreline_pos_obs_m",
        train_split=0.75,
        model_type="mlp",
        hidden_dim=32,
        epochs=3,
        batch_size=128,
        n_bootstraps=200,
        ci=0.95,
        random_seed=42,
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 4
    assert "Tier" in summary_df.columns
    assert "Test RMSE (m)" in summary_df.columns
    assert "Test RMSE 95% CI" in summary_df.columns
    assert "ΔRMSE (m)" in summary_df.columns
    assert "ΔRMSE 95% CI" in summary_df.columns
    assert "p-value (DM)" in summary_df.columns
    assert "Significance" in summary_df.columns
    assert "Test BSS" in summary_df.columns

    # Verify that point RMSE is contained within its bootstrap 95% CI
    for _, row in summary_df.iterrows():
        rmse = row["Test RMSE (m)"]
        ci_low = row["RMSE CI Lower"]
        ci_high = row["RMSE CI Upper"]
        assert ci_low <= rmse <= ci_high, f"RMSE {rmse} not in [{ci_low}, {ci_high}]"

    # Verify baseline tier classification
    assert summary_df.iloc[0]["Significance"] == "Baseline (Tier 1)"

    # Check details dictionary
    assert "test_predictions" in details
    assert "bootstrap_rmses" in details
    assert "paired_deltas" in details
    assert len(details["test_predictions"]) == 4


def test_run_constructive_tier_ablation_ridge_and_rolling_cv(synthetic_tier_dataset: pd.DataFrame):
    """Verify constructive tier ablation using Ridge regression and rolling-origin CV folds."""
    summary_df, details = run_constructive_tier_ablation(
        df=synthetic_tier_dataset,
        target_col="shoreline_pos_obs_m",
        train_split=0.70,
        model_type="ridge",
        n_bootstraps=150,
        ci=0.95,
        random_seed=42,
        rolling_origin_folds=3,
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 4

    # Verify rolling-origin CV results
    assert "rolling_origin_results" in details
    ro_res = details["rolling_origin_results"]
    assert "tier_fold_rmses" in ro_res
    assert "tier_mean_cv_rmse" in ro_res
    assert len(ro_res["tier_mean_cv_rmse"]) == 4


def test_custom_tier_feature_sets_and_significance():
    """Verify that identical features or pure noise produce 'Noise Margin / Not Significant'."""
    n = 200
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, n)
    # Target is smooth wave
    y = np.sin(t) + rng.normal(0, 0.1, n)
    df = pd.DataFrame({
        "shoreline_pos_obs_m": y,
        "feat_base": np.sin(t) + rng.normal(0, 0.05, n),
        "feat_noise": rng.normal(0, 1.0, n),
    })

    custom_tiers = {
        "Tier 1 Base": ["feat_base"],
        "Tier 2 + Noise": ["feat_base", "feat_noise"],
    }

    summary_df, details = run_constructive_tier_ablation(
        df=df,
        target_col="shoreline_pos_obs_m",
        tier_feature_sets=custom_tiers,
        train_split=0.75,
        model_type="ridge",
        n_bootstraps=200,
        random_seed=42,
    )

    assert len(summary_df) == 2
    # Tier 2 adds pure noise, so it should be within noise margin or not significant
    tier2_sig = summary_df.iloc[1]["Significance"]
    assert "Noise Margin" in tier2_sig or "Degradation" in tier2_sig or "p=" in tier2_sig
