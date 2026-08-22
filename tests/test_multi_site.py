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
    sites = ["duck_nc", "torrey_pines_ca", "narrabeen_aus", "truc_vert_france"]
    for s in sites:
        df, label = load_annual_dataset(year=2022, site=s, morph_model="xbeach")
        assert not df.empty
        assert len(df) >= 8000
        assert "upsilon" in df.columns
        assert "log_upsilon" in df.columns
        assert "shoreline_pos_obs_m" in df.columns
        assert df["site"].iloc[0] == s

        # Verify site-specific slope is respected
        expected_slope = COASTAL_SITE_PROFILES[s]["tan_beta"]
        assert np.isclose(df["tan_beta"].iloc[0], expected_slope, atol=0.01)


def test_load_multi_year_sequence():
    """Verify loading multi-year continuous sequences (e.g. 2022-2023)."""
    df, label = load_multi_year_dataset(site="duck_nc", start_year=2022, end_year=2023, morph_model="xbeach")
    assert not df.empty
    assert len(df) >= 17000  # 2 full years of hourly records
    assert df.index.min().year == 2022
    assert df.index.max().year == 2023
    assert "upsilon" in df.columns
    assert "shoreline_pos_obs_m" in df.columns


def test_load_dataset_parameterized_dispatcher():
    """Verify high-level load_dataset dispatcher with different site and year options."""
    # 1. Single year Torrey Pines
    df_tp, _ = load_dataset(site="torrey_pines_ca", year=2023, morph_model="xbeach")
    assert not df_tp.empty
    assert df_tp["site"].iloc[0] == "torrey_pines_ca"

    # 2. Multi-year Narrabeen
    df_nb, _ = load_dataset(site="narrabeen_aus", start_year=2022, end_year=2023)
    assert not df_nb.empty
    assert len(df_nb) >= 17000

    # 3. Sliced window
    df_slice, _ = load_dataset(site="duck_nc", start_date="2022-06-01", end_date="2022-08-31")
    assert not df_slice.empty
    assert df_slice.index.min() >= pd.Timestamp("2022-06-01")
    assert df_slice.index.max() <= pd.Timestamp("2022-08-31 23:59:59")
