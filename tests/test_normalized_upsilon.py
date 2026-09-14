"""Unit tests for Dimensionless Normalized Upsilon (ΔΥ*) Differential Formulation."""

import numpy as np
import pandas as pd
import pytest

from src.physics import (
    simulate_normalized_equilibrium_upsilon_evolution,
    simulate_signed_equilibrium_upsilon_evolution,
)
from src.models import calibrate_normalized_equilibrium_parameters


def test_simulate_normalized_equilibrium_zero_drift():
    """Verify that when forcing equals equilibrium (ΔΥ* == 0), displacement is zero."""
    n = 100
    ups = np.full(n, 2000.0)
    x_pos, x_obs = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=ups,
        upsilon_eq=2000.0,
        upsilon_crit=5000.0,
        c_erosion_star=0.05,
        c_accretion_star=0.02,
        initial_x=0.0,
    )
    assert len(x_pos) == n
    assert np.allclose(x_pos, 0.0)
    assert np.allclose(x_obs, 0.0)


def test_simulate_normalized_equilibrium_monotonicity():
    """Verify that higher storm forcing (higher ΔΥ*) produces strictly greater erosion."""
    n = 50
    # Moderate storm: ΔΥ* = 1.0
    u_mod = np.full(n, 3000.0)
    # Severe storm: ΔΥ* = 3.0
    u_sev = np.full(n, 5000.0)

    x_mod, _ = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=u_mod,
        upsilon_eq=2000.0,
        upsilon_crit=1000.0,
        c_erosion_star=0.05,
        c_accretion_star=0.02,
    )
    x_sev, _ = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=u_sev,
        upsilon_eq=2000.0,
        upsilon_crit=1000.0,
        c_erosion_star=0.05,
        c_accretion_star=0.02,
    )

    # Both should erode (negative displacement)
    assert x_mod[-1] < 0.0
    assert x_sev[-1] < 0.0
    # Severe storm should erode more than moderate storm
    assert x_sev[-1] < x_mod[-1]


def test_simulate_normalized_scale_invariance():
    """Verify that scaling Upsilon, Upsilon_eq, and Upsilon_crit by a constant factor 10x yields identical trajectories."""
    n = 100
    rng = np.random.default_rng(42)
    ups_duck = rng.uniform(100.0, 6000.0, n)
    ups_truc = ups_duck * 10.0  # 10x higher absolute energy flux

    x_duck, _ = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=ups_duck,
        upsilon_eq=500.0,
        upsilon_crit=2000.0,
        c_erosion_star=0.03,
        c_accretion_star=0.01,
        random_seed=42,
    )
    x_truc, _ = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=ups_truc,
        upsilon_eq=5000.0,
        upsilon_crit=20000.0,
        c_erosion_star=0.03,
        c_accretion_star=0.01,
        random_seed=42,
    )

    # Normalization eliminates the 10x scale disparity, yielding identical physical displacements
    np.testing.assert_allclose(x_duck, x_truc, atol=1e-10)


def test_simulate_normalized_direct_delta_upsilon_star():
    """Verify simulation when providing pre-computed delta_upsilon_star directly."""
    du_star = np.array([0.0, 1.5, 2.0, -0.5, -0.8, 0.0])
    x_pos, _ = simulate_normalized_equilibrium_upsilon_evolution(
        delta_upsilon_star=du_star,
        c_erosion_star=0.10,
        c_accretion_star=0.05,
    )
    assert len(x_pos) == len(du_star)
    # Step 1: du=1.5 > 0 -> rate = -0.10 * 1.5 = -0.15
    assert np.isclose(x_pos[1], -0.15)
    # Step 2: du=2.0 > 0 -> rate = -0.10 * 2.0 = -0.20 -> cumulative = -0.35
    assert np.isclose(x_pos[2], -0.35)
    # Step 3: du=-0.5 < 0 -> rate = +0.05 * 0.5 = +0.025 -> cumulative = -0.325
    assert np.isclose(x_pos[3], -0.325)


def test_simulate_normalized_clipping_bounds():
    """Verify that displacement is safely clamped within physical boundaries."""
    n = 200
    ups_extreme = np.full(n, 50000.0)
    x_pos, _ = simulate_normalized_equilibrium_upsilon_evolution(
        upsilon=ups_extreme,
        upsilon_eq=1000.0,
        upsilon_crit=1000.0,
        c_erosion_star=0.50,
        x_bounds=(-15.0, 10.0),
    )
    assert np.all(x_pos >= -15.0)
    assert np.all(x_pos <= 10.0)
    assert np.isclose(x_pos[-1], -15.0)


def test_calibrate_normalized_equilibrium_parameters():
    """Verify multi-site rate coefficient calibration for normalized disequilibrium."""
    df1 = pd.DataFrame({
        "surveyed_delta_x_m": [-1.5, -2.0, 0.5, 0.8],
        "upsilon_norm_diseq_simulated_delta_x_m": [-1.0, -1.8, 0.4, 0.7],
    })
    df2 = pd.DataFrame({
        "surveyed_delta_x_m": [-0.8, -1.2, 0.3, 0.6],
        "upsilon_norm_diseq_simulated_delta_x_m": [-0.6, -1.0, 0.2, 0.5],
    })
    cal = calibrate_normalized_equilibrium_parameters([df1, df2])
    assert "c_erosion_star" in cal
    assert "c_accretion_star" in cal
    assert "slope" in cal
    assert cal["slope"] > 0.5  # Positive correlation
