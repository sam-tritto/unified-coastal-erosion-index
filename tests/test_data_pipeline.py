"""Unit tests for sensor data gap capping, JONSWAP hindcast, and multi-source buoy fallback pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.physics import compute_jonswap_wind_wave_hindcast
from src.data import (
    fetch_ndbc_wave_data,
    fetch_multi_source_wave_data,
    FALLBACK_WAVE_BUOY_IDS,
)


def test_jonswap_wind_wave_hindcast_physics():
    """Verify JONSWAP wind-wave hindcast yields physically realistic wave height and period."""
    # Strong onshore wind: 20 m/s at 70 deg True (shore-normal)
    hs_storm, tp_storm = compute_jonswap_wind_wave_hindcast(
        wind_speed_ms=20.0,
        wind_dir_deg=70.0,
        fetch_km=120.0,
        shore_normal_deg=70.0,
    )
    assert 2.5 < hs_storm < 5.5, f"Storm Hs out of expected physical bounds: {hs_storm}"
    assert 7.0 < tp_storm < 13.0, f"Storm Tp out of expected physical bounds: {tp_storm}"

    # Light calm wind: 2 m/s
    hs_calm, tp_calm = compute_jonswap_wind_wave_hindcast(
        wind_speed_ms=2.0,
        wind_dir_deg=70.0,
        fetch_km=120.0,
        shore_normal_deg=70.0,
    )
    assert 0.40 <= hs_calm < 1.0, f"Calm Hs should default near background swell: {hs_calm}"
    assert tp_calm >= 7.0

    # Vectorized array input
    speeds = np.array([5.0, 15.0, 25.0])
    dirs = np.array([70.0, 70.0, 70.0])
    hs_arr, tp_arr = compute_jonswap_wind_wave_hindcast(speeds, dirs)
    assert len(hs_arr) == 3
    assert hs_arr[0] < hs_arr[1] < hs_arr[2], "Wave height should increase monotonically with wind speed"


def test_interpolation_capping_preserves_gaps():
    """Verify that capping interpolation prevents artificial linear ramps over multi-hour dropouts."""
    # Synthetic series with a 10-hour gap
    idx = pd.date_range("2022-01-01 00:00:00", periods=20, freq="1h")
    df = pd.DataFrame({"hs_m": [1.0] * 5 + [np.nan] * 10 + [3.0] * 5}, index=idx)

    # Capped interpolation with limit=2
    df_capped = df.interpolate(method="time", limit=2)
    # The first 2 NaNs after index 4 should be filled, but index 7-14 should remain NaN
    assert not np.isnan(df_capped["hs_m"].iloc[5])
    assert not np.isnan(df_capped["hs_m"].iloc[6])
    assert np.isnan(df_capped["hs_m"].iloc[7])
    assert np.isnan(df_capped["hs_m"].iloc[14])


def test_multi_source_wave_fallback_simulation():
    """Verify multi-source fallback combines primary and fallback observations cleanly."""
    idx = pd.date_range("2022-01-01 00:00:00", periods=10, freq="1h")
    
    # Primary buoy missing middle hours
    df_primary = pd.DataFrame({"hs_m": [1.0, 1.1, np.nan, np.nan, np.nan, np.nan, 1.2, 1.3, 1.2, 1.1]}, index=idx)
    # Fallback buoy has data for the gap
    df_fallback = pd.DataFrame({"hs_m": [np.nan, np.nan, 2.0, 2.1, 2.2, 2.0, np.nan, np.nan, np.nan, np.nan]}, index=idx)
    
    combined = df_primary.combine_first(df_fallback)
    assert combined["hs_m"].isna().sum() == 0
    assert combined["hs_m"].iloc[2] == 2.0
    assert combined["hs_m"].iloc[0] == 1.0
