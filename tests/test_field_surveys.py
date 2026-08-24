"""Unit and integration tests for in-situ field survey data ingestion and validation."""

import pytest
import numpy as np
import pandas as pd
from src.data import (
    load_frf_duck_field_surveys,
    align_hydrodynamics_with_field_surveys,
    load_field_survey_validation_dataset,
)
from src.models import evaluate_field_survey_correlation


def test_load_frf_duck_field_surveys_schema():
    """Verify that USACE FRF Duck field survey dataset loads with valid schema and transects."""
    df = load_frf_duck_field_surveys(transect_id=None)
    assert not df.empty
    assert len(df) >= 20
    assert "transect_id" in df.columns
    assert "shoreline_pos_mhw_m" in df.columns
    assert "cross_shore_change_m" in df.columns
    assert "uncertainty_m" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)

    # Verify multi-transect filtering
    df_58 = load_frf_duck_field_surveys(transect_id="FRF_0058")
    assert not df_58.empty
    assert (df_58["transect_id"] == "FRF_0058").all()
    assert df_58["shoreline_pos_mhw_m"].min() > 80.0  # Plausible subaerial MHW position

    df_109 = load_frf_duck_field_surveys(transect_id="FRF_0109")
    assert not df_109.empty
    assert (df_109["transect_id"] == "FRF_0109").all()


def test_align_hydrodynamics_with_field_surveys():
    """Verify synchronization between continuous hydrodynamics and discrete survey epochs."""
    # Create synthetic continuous hydrodynamics spanning 2022
    time_idx = pd.date_range("2022-01-01", "2022-12-31", freq="1h")
    n = len(time_idx)
    rng = np.random.default_rng(42)

    df_hydro = pd.DataFrame(
        {
            "hs_m": 1.2 + 0.8 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.abs(rng.normal(0, 0.3, n)),
            "tp_s": np.full(n, 9.0),
            "surge_m": 0.2 + 0.3 * np.cos(np.linspace(0, 6 * np.pi, n)),
            "upsilon": 5000.0 + 3000.0 * np.sin(np.linspace(0, 4 * np.pi, n)),
            "log_upsilon": np.log10(5000.0 + 3000.0 * np.sin(np.linspace(0, 4 * np.pi, n))),
            "dean_number": np.full(n, 2.5),
            "p_flux_Wm": np.full(n, 12000.0),
        },
        index=time_idx,
    )

    df_surveys = load_frf_duck_field_surveys(transect_id="FRF_0058")
    df_aligned = align_hydrodynamics_with_field_surveys(df_hydro, df_surveys, memory_hours=120)

    assert len(df_aligned) == len(df_surveys)
    assert "antecedent_mean_upsilon" in df_aligned.columns
    assert "antecedent_upsilon_memory" in df_aligned.columns
    assert "antecedent_max_hs_m" in df_aligned.columns
    assert "surveyed_pos_mhw_m" in df_aligned.columns
    assert "surveyed_delta_x_m" in df_aligned.columns
    assert not df_aligned["antecedent_mean_upsilon"].isnull().any()
    assert df_aligned.attrs["data_provenance"]["is_synthetic_morphodynamics"] is False


def test_field_survey_correlation_evaluation():
    """Verify that empirical correlation evaluation calculates skill metrics against true surveys."""
    df_aligned, meta = load_field_survey_validation_dataset(year=2022, transect_id="FRF_0058")
    assert not df_aligned.empty
    assert len(df_aligned) >= 10

    df_skill = evaluate_field_survey_correlation(df_aligned, target_col="surveyed_delta_x_m")
    assert not df_skill.empty
    assert "Predictor" in df_skill.columns
    assert "Pearson r" in df_skill.columns
    assert "Spearman rho" in df_skill.columns
    assert "R2 Score" in df_skill.columns
    assert "RMSE (m)" in df_skill.columns

    # Verify that Upsilon is present in the skill ranking
    predictors = df_skill["Predictor"].tolist()
    assert any("Upsilon" in p for p in predictors)


def test_load_global_sites_field_surveys():
    """Verify that all global sites (Duck, Narrabeen, Torrey Pines, Truc Vert) load real in-situ surveys."""
    from src.data import load_site_field_surveys

    sites = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]
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


