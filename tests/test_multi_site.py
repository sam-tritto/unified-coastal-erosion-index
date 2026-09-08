"""Unit and integration tests for multi-site and multi-year data ingestion."""

import pytest
import numpy as np
import pandas as pd
from src.data import (
    load_annual_dataset,
    load_multi_year_dataset,
    load_dataset,
)
from src.physics import COASTAL_SITE_PROFILES


def test_load_all_coastal_sites():
    """Verify that all global coastal sites load with correct geomorphology and physical parameters."""
    sites = ["duck_nc", "torrey_pines_ca", "narrabeen_aus", "truc_vert_france", "oahu_north_shore_hi"]
    for s in sites:
        df, label = load_annual_dataset(year=2022, site=s, morph_model="xbeach", try_live_api=False)
        assert not df.empty
        assert len(df) >= 8000
        assert "upsilon" in df.columns
        assert "log_upsilon" in df.columns
        assert "shoreline_pos_obs_m" in df.columns
        assert "psi_reef" in df.columns
        assert df["site"].iloc[0] == s

        # Verify site-specific slope and reef dissipation flag
        expected_slope = COASTAL_SITE_PROFILES[s]["tan_beta"]
        assert np.isclose(df["tan_beta"].iloc[0], expected_slope, atol=0.01)
        if COASTAL_SITE_PROFILES[s].get("is_reef", False):
            assert np.all(df["psi_reef"] <= 1.0)
            assert np.any(df["psi_reef"] < 1.0)
        else:
            assert np.all(df["psi_reef"] == 1.0)


def test_load_multi_year_sequence():
    """Verify loading multi-year continuous sequences (e.g. 2022-2023)."""
    df, label = load_multi_year_dataset(site="duck_nc", start_year=2022, end_year=2023, morph_model="xbeach", try_live_api=False)
    assert not df.empty
    assert len(df) >= 17000  # 2 full years of hourly records
    assert df.index.min().year == 2022
    assert df.index.max().year == 2023
    assert "upsilon" in df.columns
    assert "shoreline_pos_obs_m" in df.columns


def test_load_dataset_parameterized_dispatcher():
    """Verify high-level load_dataset dispatcher with different site and year options."""
    # 1. Single year Torrey Pines
    df_tp, _ = load_dataset(site="torrey_pines_ca", year=2023, morph_model="xbeach", try_live_api=False)
    assert not df_tp.empty
    assert df_tp["site"].iloc[0] == "torrey_pines_ca"

    # 2. Multi-year Narrabeen
    df_nb, _ = load_dataset(site="narrabeen_aus", start_year=2022, end_year=2023, try_live_api=False)
    assert not df_nb.empty
    assert len(df_nb) >= 17000

    # 3. Oahu's North Shore single-year and multi-year
    df_oahu, _ = load_dataset(site="oahu_north_shore_hi", year=2023, morph_model="xbeach", try_live_api=False)
    assert not df_oahu.empty
    assert df_oahu["site"].iloc[0] == "oahu_north_shore_hi"

    # 4. Sliced window
    df_slice, _ = load_dataset(site="duck_nc", start_date="2022-06-01", end_date="2022-08-31", try_live_api=False)
    assert not df_slice.empty
    assert df_slice.index.min() >= pd.Timestamp("2022-06-01")
    assert df_slice.index.max() <= pd.Timestamp("2022-08-31 23:59:59")


def test_export_unified_feature_engineered_datasets(tmp_path):
    """Verify exporting clean parquet datasets across all 5 sites with 4 physical tiers."""
    from src.data import export_unified_feature_engineered_datasets
    
    out_dir = str(tmp_path / "processed")
    sites = ["duck_nc", "narrabeen_aus", "oahu_north_shore_hi", "torrey_pines_ca", "truc_vert_france"]
    res = export_unified_feature_engineered_datasets(sites=sites, output_dirs=(out_dir,), year=2022)
    
    assert len(res) == 5
    for s in sites:
        assert s in res
        df = res[s]
        assert not df.empty
        assert "upsilon_tier1" in df.columns
        assert "upsilon_tier2" in df.columns
        assert "upsilon_tier3" in df.columns
        assert "upsilon_tier4" in df.columns
        assert "dean_number" in df.columns
        assert "directional_spread_deg" in df.columns
        assert "p_l_alongshore_Wm" in df.columns
        assert "runup_r_ig_m" in df.columns
        assert df["site"].iloc[0] == s


def test_option_b_authentic_multisite_datasets():
    """Verify that Option B authentic multi-site datasets (2012-2016, 5 years) exist, are non-synthetic, and contain all physics terms."""
    import os
    sites = ["duck_nc", "narrabeen_aus", "torrey_pines_ca", "truc_vert_france"]

    # 1. Verify 5-year sensor parquets (43,848 hours each: 366*2 + 365*3 = 1827 days = 43,848 hours)
    expected_hours = 43848
    for s in sites:
        parquet_path = f"data/processed/{s}_2012_2016_sensors.parquet"
        assert os.path.exists(parquet_path), f"Missing parquet for {s} at {parquet_path}"
        df_parquet = pd.read_parquet(parquet_path)
        assert len(df_parquet) == expected_hours, f"Expected {expected_hours} hours for {s}, got {len(df_parquet)}"

        # Verify new physics terms
        for col in [
            "wave_steepness_s0", "psi_steep",
            "s_xx_Wm", "s_xy_Wm", "psi_sxx",
            "psi_impact", "impact_regime",
            "upsilon_full",
            "upsilon_decay_7d", "upsilon_diseq_7d",
            "upsilon_decay_30d", "upsilon_diseq_30d",
            "upsilon_decay_90d", "upsilon_diseq_90d",
            "upsilon_decay_365d", "upsilon_diseq_365d",
        ]:
            assert col in df_parquet.columns, f"Missing physics column {col} in {s}"

    # 2. Verify authentic unified survey catalog (N=1228)
    survey_path = "data/site_field_surveys_2012_2017.csv"
    assert os.path.exists(survey_path)
    df_surveys = pd.read_csv(survey_path)
    assert len(df_surveys) >= 1200
    assert set(df_surveys["site"].unique()) == set(sites)
    assert np.all(df_surveys["is_in_situ_observed"] == True)
    assert not df_surveys["shoreline_pos_mhw_m"].isna().any()


