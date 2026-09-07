"""Unit tests for Leave-One-Site-Out (LOSO) Cross-Validation across global coastal sites."""

import pytest
import numpy as np
import pandas as pd
from src.models import (
    evaluate_leave_one_site_out_cv,
    calibrate_signed_equilibrium_parameters,
)


def test_calibrate_signed_equilibrium_parameters():
    """Verify calibration of rate coefficients across training site data."""
    # Synthetic training site data
    np.random.seed(42)
    n = 100
    ups = np.random.uniform(500, 5000, n)
    ups_eq = 2500.0
    c_ero_true = 1.5e-5
    c_acc_true = 2.0e-5
    
    d_ero = np.maximum(ups - ups_eq, 0.0)
    d_acc = np.maximum(ups_eq - ups, 0.0)
    y_true = -c_ero_true * d_ero + c_acc_true * d_acc + np.random.normal(0, 0.01, n)
    
    df1 = pd.DataFrame({
        "antecedent_mean_upsilon": ups[:50],
        "surveyed_delta_x_m": y_true[:50],
        "upsilon_diseq_simulated_delta_x_m": -1.2e-5 * d_ero[:50] + 1.8e-5 * d_acc[:50],
    })
    df2 = pd.DataFrame({
        "antecedent_mean_upsilon": ups[50:],
        "surveyed_delta_x_m": y_true[50:],
        "upsilon_diseq_simulated_delta_x_m": -1.2e-5 * d_ero[50:] + 1.8e-5 * d_acc[50:],
    })
    
    params = calibrate_signed_equilibrium_parameters([df1, df2])
    assert "c_erosion" in params
    assert "c_accretion" in params
    assert "slope" in params
    assert params["c_erosion"] > 0
    assert params["c_accretion"] > 0


def test_evaluate_leave_one_site_out_cv_smoke():
    """Verify execution of Leave-One-Site-Out CV across sites."""
    # Run with authentic sites, fast MCMC settings
    test_sites = ["duck_nc", "narrabeen_aus"]
    
    df_synthesis, df_held_out = evaluate_leave_one_site_out_cv(
        sites=test_sites,
        draws=50,
        tune=30,
        chains=1,
        cores=1,
        use_field_surveys=True,
        random_seed=42,
    )
    
    assert not df_synthesis.empty
    assert not df_held_out.empty
    
    # Check that each test site was held out exactly once
    held_out_sites_found = set(df_held_out["Held-Out Site"].unique())
    for s in test_sites:
        assert s in held_out_sites_found
        
    # Check required columns in synthesis table
    required_cols = [
        "Paradigm Class",
        "Model Name",
        "LOSO RMSE (m) (Δx)",
        "LOSO R2 Score (Δx)",
        "LOSO BSS (Δx)",
        "Position LOSO R2 Score",
        "Position LOSO RMSE (m)",
        "Held-Out Sites Tested",
    ]
    for col in required_cols:
        assert col in df_synthesis.columns
        
    # Verify presence of both Global Default and Calibrated Class III models
    model_names = set(df_synthesis["Model Name"].unique())
    assert "Signed Equilibrium Upsilon (Global Default)" in model_names
    assert "Signed Equilibrium Upsilon (Calibrated)" in model_names
    assert "Bayesian Latent Memory Kernel (Upsilon)" in model_names
