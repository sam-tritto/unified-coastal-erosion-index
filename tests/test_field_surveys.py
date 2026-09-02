"""Unit and integration tests for in-situ field survey data ingestion and validation."""

import pytest
import numpy as np
import pandas as pd
from src.data import (
    load_frf_duck_field_surveys,
    align_hydrodynamics_with_field_surveys,
    load_field_survey_validation_dataset,
)
from src.models import (
    evaluate_field_survey_correlation,
    williams_test,
    diebold_mariano_test,
    bootstrap_metric_ci,
)


def test_williams_test():
    """Verify that the Hotelling-Williams test correctly computes t-statistic and p-value for dependent correlations."""
    # When r1y == r2y, t_stat should be 0.0 and p_value should be 1.0
    t_stat, p_val = williams_test(r1y=0.5, r2y=0.5, r12=0.6, n=50)
    assert np.isclose(t_stat, 0.0)
    assert np.isclose(p_val, 1.0)

    # When r1y significantly exceeds r2y with moderate inter-correlation
    t_stat, p_val = williams_test(r1y=0.65, r2y=0.20, r12=0.30, n=80)
    assert t_stat > 3.0
    assert p_val < 0.01

    # Edge cases: n <= 3
    t_edge, p_edge = williams_test(r1y=0.5, r2y=0.2, r12=0.1, n=3)
    assert t_edge == 0.0
    assert p_edge == 1.0


def test_diebold_mariano_test():
    """Verify that Diebold-Mariano test calculates loss differential statistics with HLN correction."""
    rng = np.random.default_rng(42)
    n = 60
    y_true = rng.normal(0, 1, n)
    # Model 1 has smaller errors than Model 2
    y_pred1 = y_true + rng.normal(0, 0.3, n)
    y_pred2 = y_true + rng.normal(0, 0.8, n)

    dm_stat, p_val = diebold_mariano_test(y_true, y_pred1, y_pred2, loss="squared", h=1, hln_adjust=True)
    assert dm_stat < 0.0  # Model 1 has lower MSE
    assert p_val < 0.05  # Significant difference

    # When both models are identical
    dm_same, p_same = diebold_mariano_test(y_true, y_pred1, y_pred1, loss="squared")
    assert np.isclose(dm_same, 0.0)
    assert np.isclose(p_same, 1.0)


def test_bootstrap_metric_ci():
    """Verify that bootstrap confidence intervals compute valid bounds covering point estimates."""
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 10, n)
    y = 2.0 * x + rng.normal(0, 1.0, n)

    point_r, r_low, r_high = bootstrap_metric_ci(x, y, metric_type="pearson_r", n_bootstraps=500, ci=0.95, random_seed=42)
    assert r_low <= point_r <= r_high
    assert r_low > 0.80  # Strong positive correlation

    point_r2, r2_low, r2_high = bootstrap_metric_ci(x, y, metric_type="r2", n_bootstraps=500, ci=0.95, random_seed=42)
    assert r2_low <= point_r2 <= r2_high
    assert r2_low > 0.70


def test_field_survey_correlation_evaluation():
    """Verify that empirical correlation evaluation calculates skill metrics, bootstrap CIs, Williams test, and variance lift."""
    df_aligned, meta = load_field_survey_validation_dataset(year=2022, transect_id="FRF_0058")
    assert not df_aligned.empty
    assert len(df_aligned) >= 10

    df_skill = evaluate_field_survey_correlation(df_aligned, target_col="surveyed_delta_x_m", n_bootstraps=200)
    assert not df_skill.empty
    assert "Predictor" in df_skill.columns
    assert "Pearson r" in df_skill.columns
    assert "r_95ci_lower" in df_skill.columns
    assert "r_95ci_upper" in df_skill.columns
    assert "Spearman rho" in df_skill.columns
    assert "R2 Score" in df_skill.columns
    assert "r2_95ci_lower" in df_skill.columns
    assert "r2_95ci_upper" in df_skill.columns
    assert "RMSE (m)" in df_skill.columns
    assert "Williams t-stat" in df_skill.columns
    assert "Williams p-value" in df_skill.columns
    assert "Diebold-Mariano stat" in df_skill.columns
    assert "Diebold-Mariano p-value" in df_skill.columns
    assert "Delta R2 vs Top" in df_skill.columns
    assert "Pct Variance Lift (%)" in df_skill.columns

    # Verify that Signed Equilibrium Upsilon, Peak, and Cumulative Upsilon are present in the skill ranking
    predictors = df_skill["Predictor"].tolist()
    assert any("Signed Equilibrium" in p for p in predictors)
    assert any("Peak Storm Upsilon" in p for p in predictors)
    assert any("Cumulative Storm Work" in p for p in predictors)
    assert any("Upsilon" in p for p in predictors)

    # Verify primary physical indices (Signed Equilibrium, Upsilon, Dean) rank at or near the top
    top_predictors = df_skill.head(3)["Predictor"].tolist()
    assert any("Signed Equilibrium" in p or "Upsilon" in p or "Dean" in p for p in top_predictors)


def test_load_global_sites_field_surveys():
    """Verify that all global sites (Duck, Narrabeen, Torrey Pines, Truc Vert, Oahu North Shore) load real in-situ surveys."""
    from src.data import load_site_field_surveys

    sites = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france", "oahu_north_shore_hi"]
    for site in sites:
        df = load_site_field_surveys(site=site)
        assert not df.empty, f"Site {site} surveys should not be empty"
        assert len(df) >= 20, f"Site {site} should have at least 20 survey records"
        assert "shoreline_pos_mhw_m" in df.columns
        assert "cross_shore_change_m" in df.columns
        assert "site" in df.columns
        assert (df["site"] == site).all()


def test_bayesian_benchmark_on_field_surveys():
    """Verify that Bayesian multi-model comparison runs on real field survey dataset with Yates and XBeach competitors."""
    from src.models import run_benchmark_model_comparison

    df_aligned, meta = load_field_survey_validation_dataset(start_year=2022, end_year=2022, site="duck_nc", transect_id="FRF_0058")
    assert not df_aligned.empty
    assert "yates_simulated_delta_x_m" in df_aligned.columns
    assert "xbeach_simulated_delta_x_m" in df_aligned.columns

    df_comp, traces = run_benchmark_model_comparison(
        df=df_aligned,
        train_split=0.75,
        target_col="surveyed_delta_x_m",
        draws=100,
        tune=100,
        chains=2,
        cores=1,
        include_nonlinear=False,
        include_multivariate=False,
        include_hybrid=False,
        compute_loo=False,
    )
    assert not df_comp.empty
    models_tested = df_comp["Model"].tolist()
    assert any("Upsilon" in m for m in models_tested)
    assert any("Yates" in m for m in models_tested)
    assert any("XBeach" in m for m in models_tested)


def test_narrabeen_three_transect_field_surveys_rotation():
    """Verify that Narrabeen surveys contain authentic 3-transect records (PF1, PF2, PF4) displaying rotation."""
    from src.data import load_site_field_surveys

    df_narr = load_site_field_surveys(site="narrabeen_aus")
    assert not df_narr.empty
    transects = df_narr["transect_id"].unique().tolist()
    assert "PF1" in transects
    assert "PF2" in transects
    assert "PF4" in transects
    assert "alongshore_y_m" in df_narr.columns

    # Check Feb 2020 ECL survey epoch
    feb_surveys = df_narr[df_narr.index == "2020-02-11 11:15:00"]
    assert len(feb_surveys) == 3

    pf1_change = feb_surveys[feb_surveys["transect_id"] == "PF1"]["cross_shore_change_m"].iloc[0]
    pf2_change = feb_surveys[feb_surveys["transect_id"] == "PF2"]["cross_shore_change_m"].iloc[0]
    pf4_change = feb_surveys[feb_surveys["transect_id"] == "PF4"]["cross_shore_change_m"].iloc[0]

    # PF1 (North) accretes, PF2 (Center) erodes, PF4 (South) erodes heavily
    assert pf1_change > 0.0, f"PF1 should accrete during ECL: got {pf1_change}"
    assert pf2_change < 0.0, f"PF2 should erode during ECL: got {pf2_change}"
    assert pf4_change < 0.0, f"PF4 should erode during ECL: got {pf4_change}"
    assert pf1_change > pf2_change > pf4_change



