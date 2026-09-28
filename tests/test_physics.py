"""Unit tests for physics formulations, non-circular data pipelines, latent models, and embeddings."""

import numpy as np
import pytest
from src.physics import (
    compute_settling_velocity_soulsby,
    compute_kinematic_viscosity_seawater,
    compute_cross_array_dissipation_ratio,
    compute_constructive_accretion_potential,
    compute_deepwater_wavelength,
    compute_stockdon_wave_runup,
    compute_iribarren_number,
    classify_breaker_regime,
    compute_wave_energy_flux,
    compute_dean_number,
    compute_shore_normal_wind,
    compute_asymmetric_wind_amplification,
    compute_tidal_smearing_factor,
    compute_instantaneous_tide_phase_factor,
    compute_tide_phase_gating,
    compute_directional_wave_factor,
    compute_longshore_transport_divergence,
    compute_reef_dissipation_modulator,
    compute_unified_coastal_vulnerability_index,
    compute_signed_coastal_vulnerability_index,
    compute_expanding_window_equilibrium,
    compute_upsilon_disequilibrium,
    compute_signed_equilibrium_rate,
    simulate_signed_equilibrium_upsilon_evolution,
    compute_critical_shear_threshold,
    compute_shields_critical_shear_stress_soulsby,
    sweep_critical_threshold_sensitivity,
    compute_excess_upsilon,
    compute_critical_threshold_shear_activation,
    compute_excess_vulnerability_index,
    compute_coupled_2d_embayment_displacement,
    compute_alongshore_transport_gradient_rotation,
    compute_embayed_beach_rotation,
    simulate_coupled_2d_embayment_evolution,
    compute_dynamic_foreshore_slope,
    compute_dynamic_grain_size,
    compute_site_reference_flux,
    compute_dimensionless_upsilon,
    compute_wave_steepness,
    compute_radiation_stress_tensor,
    compute_sallenger_impact_ratio,
    compute_multiscale_memory_decay,
    COASTAL_SITE_PROFILES,
)
from src.data import (
    load_dataset,
    load_multi_storm_dataset,
    generate_frf_duck_hurricane_ian_dataset,
    simulate_equilibrium_shoreline_evolution,
    simulate_xbeach_inspired_morphodynamics,
)
from src.embeddings import (
    compute_latent_embeddings,
    compute_hydrodynamic_state_similarity,
    project_embeddings_umap,
    evaluate_breaker_regime_clustering,
    classify_morphodynamic_regimes,
)
from src.models import (
    build_latent_memory_bayesian_model,
    build_multivariate_linear_model,
    build_nonlinear_response_model,
    build_physics_anchored_hybrid_model,
    build_asymmetric_dual_regime_model,
    build_learnable_critical_threshold_bayesian_model,
    build_learnable_exponent_bayesian_model,
    run_hs_exponent_sensitivity_analysis,
    compute_antecedent_memory_forcing,
    evaluate_model_performance,
    predict_bayesian_model,
    run_arviz_loo_comparison,
    run_benchmark_model_comparison,
    fit_bayesian_model,
    compute_mcmc_summary_table,
    run_bayesian_memory_window_ablation,
    run_bayesian_memory_prior_sensitivity_analysis,
    run_heteroscedastic_noise_ablation,
)


def test_soulsby_settling_velocity():
    # For D50 = 0.35 mm sand in seawater, ws should be ~0.048 - 0.052 m/s
    ws = compute_settling_velocity_soulsby(0.00035)
    assert 0.045 < ws < 0.055, f"Soulsby ws out of expected physical bounds: {ws}"

    ws_fine = compute_settling_velocity_soulsby(0.00010)
    ws_coarse = compute_settling_velocity_soulsby(0.00100)
    assert ws_fine < ws < ws_coarse


def test_deepwater_wavelength_and_stockdon_runup():
    L0 = compute_deepwater_wavelength(10.0)
    assert np.isclose(L0, 156.13, atol=0.5)

    # Wave runup for moderate swell
    r2 = compute_stockdon_wave_runup(hs=2.0, tp=10.0, tan_beta=0.08)
    assert 0.8 < r2 < 2.5


def test_iribarren_and_regimes():
    xi_spill = compute_iribarren_number(hs=4.0, tp=6.0, tan_beta=0.02)
    assert xi_spill < 0.5
    assert classify_breaker_regime(xi_spill) == "Spilling"

    xi_plunge = compute_iribarren_number(hs=1.5, tp=10.0, tan_beta=0.08)
    assert 0.5 <= xi_plunge < 3.3
    assert classify_breaker_regime(xi_plunge) == "Plunging"

    xi_surge = compute_iribarren_number(hs=0.4, tp=16.0, tan_beta=0.15)
    assert xi_surge >= 3.3
    assert classify_breaker_regime(xi_surge) == "Surging"


def test_shore_normal_wind():
    wa_direct = compute_shore_normal_wind(10.0, 70.0, shore_normal_deg=70.0)
    assert np.isclose(wa_direct, 10.0)

    wa_along = compute_shore_normal_wind(10.0, 160.0, shore_normal_deg=70.0)
    assert np.isclose(wa_along, 0.0, atol=1e-5)


def test_unified_index_non_collapsing():
    # Test boundary condition: zero surge, zero onshore wind, large wave swell
    df_idx = compute_unified_coastal_vulnerability_index(
        hs=np.array([3.0]),
        tp=np.array([14.0]),
        surge=np.array([0.0]),
        wind_speed=np.array([0.0]),
        wind_dir_deg=np.array([0.0]),
    )
    # The robust formulation must NOT collapse to zero
    assert df_idx["upsilon"].iloc[0] > 1000.0


def test_non_circular_morphodynamics_and_dataset():
    df, source = load_dataset(try_live_api=False)
    assert len(df) > 500
    assert "upsilon" in df.columns
    assert "shoreline_pos_obs_m" in df.columns

    # Verify morphodynamics was generated by equilibrium model (not Upsilon)
    x_true, x_obs = simulate_equilibrium_shoreline_evolution(df["hs_m"].values, df["tp_s"].values)
    assert len(x_true) == len(df)


def test_xbeach_inspired_morphodynamics():
    df = generate_frf_duck_hurricane_ian_dataset()
    x_true, x_obs = simulate_xbeach_inspired_morphodynamics(
        hs=df["hs_m"].values,
        tp=df["tp_s"].values,
        surge=df["surge_m"].values,
        wind_speed=df["wind_speed_ms"].values,
        wind_dir_deg=df["wind_dir_deg"].values,
    )
    assert len(x_true) == len(df)
    assert len(x_obs) == len(df)
    # Storm should cause negative shoreline retreat
    assert np.min(x_true) < -2.0


def test_multi_storm_dataset_loader():
    df_default, label_default = load_multi_storm_dataset(try_live_api=False)
    assert len(df_default) > 100
    assert "YATES" in label_default.upper()
    assert "storm_event" in df_default.columns
    assert "dominant_forcing" in df_default.columns

    df_xbeach, label_xbeach = load_multi_storm_dataset(try_live_api=False, morph_model="xbeach")
    assert len(df_xbeach) > 100
    assert "XBEACH" in label_xbeach.upper()


def test_latent_bayesian_model_structure():
    df = generate_frf_duck_hurricane_ian_dataset()
    model, meta = build_latent_memory_bayesian_model(
        forcing=df["upsilon"].values[:100],
        delta_x_obs=df["shoreline_pos_obs_m"].values[:100],
        max_lag=24,
    )
    assert "gamma" in model.named_vars
    assert "effective_forcing" in model.named_vars
    assert "beta_forcing" in model.named_vars


def test_multivariate_and_nonlinear_bayesian_models():
    df = generate_frf_duck_hurricane_ian_dataset()
    sub_df = df.iloc[:80]
    
    # Test multivariate model
    mv_model, mv_meta = build_multivariate_linear_model(
        forcings={
            "hs": sub_df["hs_m"].values,
            "surge": sub_df["surge_m"].values,
            "wind": sub_df["wind_normal_ms"].values,
            "dean": sub_df["dean_number"].values,
        },
        delta_x_obs=sub_df["shoreline_pos_obs_m"].values,
    )
    assert "beta_hs" in mv_model.named_vars
    assert "beta_surge" in mv_model.named_vars
    assert "beta_wind" in mv_model.named_vars
    assert "beta_dean" in mv_model.named_vars

    # Test nonlinear polynomial model
    nl_model, nl_meta = build_nonlinear_response_model(
        forcing=sub_df["upsilon"].values,
        delta_x_obs=sub_df["shoreline_pos_obs_m"].values,
        degree=3,
    )
    assert "beta_poly" in nl_model.named_vars
    assert "tau_global" in nl_model.named_vars


def test_physics_anchored_hybrid_model():
    df = generate_frf_duck_hurricane_ian_dataset()
    sub_df = df.iloc[:80]
    hybrid_model, meta = build_physics_anchored_hybrid_model(
        upsilon=sub_df["upsilon"].values,
        forcings={
            "hs": sub_df["hs_m"].values,
            "surge": sub_df["surge_m"].values,
            "wind": sub_df["wind_normal_ms"].values,
        },
        delta_x_obs=sub_df["shoreline_pos_obs_m"].values,
        max_lag=12,
    )
    assert "beta_upsilon" in hybrid_model.named_vars
    assert "delta_hs" in hybrid_model.named_vars
    assert "delta_surge" in hybrid_model.named_vars
    assert "delta_wind" in hybrid_model.named_vars
    assert "gamma" in hybrid_model.named_vars


def test_data_driven_embeddings_and_silhouette():
    df = generate_frf_duck_hurricane_ian_dataset()
    emb, names, meta = compute_latent_embeddings(df)
    assert emb.shape[0] == len(df)
    assert emb.shape[1] == len(names)

    # Cosine similarity matrix
    sim_df = compute_hydrodynamic_state_similarity(emb, df)
    assert sim_df.shape == (5, 5)
    assert np.allclose(np.diag(sim_df.values), 1.0)

    # UMAP projection
    coords_2d = project_embeddings_umap(emb, n_neighbors=15)
    assert coords_2d.shape == (len(df), 2)

    # Silhouette evaluation
    sil = evaluate_breaker_regime_clustering(coords_2d, df["breaker_regime"].values)
    assert sil["silhouette_score"] > -0.5


def test_dynamic_viscosity_and_temperature_settling():
    # Kinematic viscosity in winter (5C) vs summer (25C)
    nu_cold = compute_kinematic_viscosity_seawater(5.0)
    nu_warm = compute_kinematic_viscosity_seawater(25.0)
    assert nu_cold > nu_warm
    assert np.isclose(nu_cold, 1.52e-6, atol=0.1e-6)
    assert np.isclose(nu_warm, 0.95e-6, atol=0.1e-6)

    # Dynamic Soulsby settling velocity: cold winter water yields slower settling
    ws_winter = compute_settling_velocity_soulsby(0.00035, water_temp_c=5.0)
    ws_summer = compute_settling_velocity_soulsby(0.00035, water_temp_c=25.0)
    assert ws_winter < ws_summer
    assert 0.035 < ws_winter < 0.045
    assert 0.048 < ws_summer < 0.058


def test_cross_array_dissipation_and_accretion_potential():
    p_outer = 45000.0  # 45 kW/m in 26m depth
    p_inner = 15000.0  # 15 kW/m in 8m depth (dissipated over sandbar)
    delta_p, shielding = compute_cross_array_dissipation_ratio(p_outer, p_inner)
    assert np.isclose(delta_p, (45000 - 15000) / 45000, atol=1e-3)
    assert np.isclose(shielding, 1.0 - delta_p, atol=1e-3)

    # Constructive accretion potential: high for long period low steepness swell
    acc_swell = compute_constructive_accretion_potential(hs=0.8, tp=15.0)
    acc_storm = compute_constructive_accretion_potential(hs=4.5, tp=8.0)
    assert acc_swell > acc_storm


def test_asymmetric_dual_regime_model():
    df = generate_frf_duck_hurricane_ian_dataset()
    sub_df = df.iloc[:80]
    model, meta = build_asymmetric_dual_regime_model(
        hs=sub_df["hs_m"].values,
        tp=sub_df["tp_s"].values,
        ws=sub_df["w_s_ms"].values,
        upsilon=sub_df["upsilon"].values,
        delta_x_obs=sub_df["shoreline_pos_obs_m"].values,
        shielding_factor=sub_df["shielding_factor"].values,
        accretion_potential=sub_df["accretion_potential"].values,
        max_lag=12,
    )
    assert "omega_crit" in model.named_vars
    assert "k_eros" in model.named_vars
    assert "k_accrete" in model.named_vars
    assert "gamma" in model.named_vars
    assert "eff_eros" in model.named_vars
    assert "eff_acc" in model.named_vars


def test_morphodynamic_regime_classification():
    df = generate_frf_duck_hurricane_ian_dataset()
    regimes, omega_ratio = classify_morphodynamic_regimes(df, omega_crit=1.75)
    assert len(regimes) == len(df)
    assert len(omega_ratio) == len(df)
    unique_regimes = set(regimes.unique())
    assert "Storm Gouging (Erosive)" in unique_regimes
    assert "Quiet Berm Accretion (Constructive)" in unique_regimes


def test_bayesian_temporal_holdout_and_prediction():
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:80]
    split_idx = 60
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    model, meta = build_latent_memory_bayesian_model(
        forcing=df_train["upsilon"].values,
        delta_x_obs=df_train["shoreline_change_rate_mh"].values,
        max_lag=12,
    )
    trace = fit_bayesian_model(model, draws=50, tune=50, chains=2, random_seed=42)

    # Full continuous prediction across train and out-of-sample test splits
    mu_mean, mu_samples, y_pred = predict_bayesian_model(
        trace=trace,
        model_type="single",
        meta=meta,
        df=df,
        return_samples=True,
        random_seed=42,
    )
    assert len(mu_mean) == len(df)
    assert mu_samples.shape == (100, len(df))
    assert y_pred.shape == (100, len(df))

    # Evaluate out-of-sample holdout on shoreline_change_rate_mh
    y_test_obs = df_test["shoreline_change_rate_mh"].values
    perf_test = evaluate_model_performance(y_true=y_test_obs, y_pred=mu_mean[split_idx:])
    assert perf_test["RMSE (m)"] < 10.0
    assert "R2 Score" in perf_test
    assert "Brier Skill Score (BSS)" in perf_test


def test_bayesian_benchmark_with_holdout_and_loo():
    """Verify Bayesian model comparison across physical formulations using PSIS-LOO-CV.

    Note on ArviZ Pareto k warnings (k > 0.5):
    Under acute storm sequences (e.g. Hurricane Ian wave power / surge peaks), heavy-tailed
    morphodynamic shocks induce high Pareto shape parameters (k > 0.5), indicating high
    importance-weight variance in PSIS-LOO-CV. As documented in Section IV-E of the research
    paper, this posterior sensitivity to episodic outliers highlights the structural limitation
    of parametric Bayesian inference and scientifically motivates transitioning to distribution-free
    Conformalized Quantile Regression (CQR).
    """
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:80]
    df_comp, traces = run_benchmark_model_comparison(
        df=df,
        target_col="shoreline_change_rate_mh",
        train_split=0.75,
        draws=50,
        tune=50,
        chains=2,
        cores=1,
        random_seed=42,
        include_nonlinear=False,
        include_multivariate=True,
        include_hybrid=True,
        compute_loo=True,
    )
    assert not df_comp.empty
    assert "Test RMSE (m)" in df_comp.columns
    assert "Test R2 Score" in df_comp.columns
    assert "Train RMSE (m)" in df_comp.columns
    assert "ELPD LOO" in df_comp.columns
    assert "LOO Rank" in df_comp.columns
    assert df_comp.attrs["train_split"] == 0.75
    assert df_comp.attrs["split_idx"] == 60
    assert len(traces) >= 3


def test_parameterized_hs_exponent_scaling():
    """Verify that hs_exponent n in [1.0, 3.0] scales Upsilon monotonically for storm waves."""
    hs_storm = np.array([4.0])
    tp = np.array([12.0])
    surge = np.array([0.8])
    wind = np.array([15.0])
    wind_dir = np.array([70.0])

    ups_values = []
    for n in [1.0, 1.5, 2.0, 2.5, 3.0]:
        df = compute_unified_coastal_vulnerability_index(
            hs=hs_storm,
            tp=tp,
            surge=surge,
            wind_speed=wind,
            wind_dir_deg=wind_dir,
            hs_exponent=n,
        )
        assert "hs_exponent" in df.columns
        assert df["hs_exponent"].iloc[0] == n
        ups_values.append(df["upsilon"].iloc[0])

    # Higher exponent n on Hs > 1.0m must produce strictly higher Upsilon
    for i in range(len(ups_values) - 1):
        assert ups_values[i + 1] > ups_values[i]


def test_learnable_exponent_bayesian_model():
    """Test construction and MCMC sampling of the learnable Hs exponent Bayesian model."""
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:60]
    model, meta = build_learnable_exponent_bayesian_model(
        hs=df["hs_m"].values,
        tp=df["tp_s"].values,
        ws=df["w_s_ms"].values,
        surge=df["surge_m"].values,
        runup=df["runup_r2_m"].values,
        wind_normal=df["wind_normal_ms"].values,
        delta_x_obs=df["shoreline_pos_obs_m"].values,
        max_lag=12,
        exponent_prior="uniform",
        lower_bound=1.0,
        upper_bound=3.0,
    )
    assert meta["model_name"] == "learnable_hs_exponent"
    assert "hs_exponent" in model.named_vars

    trace = fit_bayesian_model(model, draws=50, tune=50, chains=2, cores=1, random_seed=42)
    assert "hs_exponent" in trace.posterior
    post_exp = trace.posterior["hs_exponent"].values.flatten()
    assert np.all(post_exp >= 1.0)
    assert np.all(post_exp <= 3.0)
    assert len(post_exp) == 100


def test_hs_exponent_sensitivity_analysis():
    """Verify systematic sensitivity analysis over fixed and learnable exponents."""
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:40]
    df_comp, traces = run_hs_exponent_sensitivity_analysis(
        df=df,
        exponents=(1.5, 2.5),
        max_lag=6,
        draws=30,
        tune=30,
        chains=2,
        include_learnable=True,
        random_seed=42,
    )
    assert "fixed_n_1.5" in traces
    assert "fixed_n_2.5" in traces
    assert "learnable_n" in traces
    assert len(traces) == 3
    assert not df_comp.empty
    assert "elpd_loo" in df_comp.columns or "rank" in df_comp.columns


def test_twl_surge_and_runup_amplification():
    """Verify that decoupled TWL amplification:
    (a) strictly increases Υ with surge and runup,
    (b) produces naturally smaller relative surge amplification under large waves (due to runup baseline)
        than small waves, without distorting the Hs^n power-law scaling.
    """
    tp = np.array([12.0])
    wind = np.array([10.0])
    wind_dir = np.array([70.0])

    # --- (a) Surge must always increase Υ ----------------------------------------
    for hs_val, surge_val in [(1.0, 0.5), (2.0, 0.5), (5.0, 1.0)]:
        df_no = compute_unified_coastal_vulnerability_index(
            hs=np.array([hs_val]), tp=tp, surge=np.array([0.0]),
            wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
        )
        df_with = compute_unified_coastal_vulnerability_index(
            hs=np.array([hs_val]), tp=tp, surge=np.array([surge_val]),
            wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
        )
        ratio = df_with["upsilon"].iloc[0] / df_no["upsilon"].iloc[0]
        assert ratio > 1.0, f"Hs={hs_val}, surge={surge_val}: expected ratio > 1.0, got {ratio:.3f}"

    # --- (b) Under large waves (Hs=5m) 1m surge produces expected non-dimensional amplification -------
    df_no_surge = compute_unified_coastal_vulnerability_index(
        hs=np.array([5.0]), tp=tp, surge=np.array([0.0]),
        wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
    )
    df_with_surge = compute_unified_coastal_vulnerability_index(
        hs=np.array([5.0]), tp=tp, surge=np.array([1.0]),
        wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
    )
    ratio_large = df_with_surge["upsilon"].iloc[0] / df_no_surge["upsilon"].iloc[0]
    assert 1.0 < ratio_large < 1.50, (
        f"Under large waves (Hs=5m) surge amplification ratio should be in (1.0, 1.50), "
        f"got {ratio_large:.2f}"
    )

    # --- (c) Small-wave surge impact > large-wave surge impact -------------------
    df_small_no = compute_unified_coastal_vulnerability_index(
        hs=np.array([0.5]), tp=tp, surge=np.array([0.0]),
        wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
    )
    df_small_with = compute_unified_coastal_vulnerability_index(
        hs=np.array([0.5]), tp=tp, surge=np.array([1.0]),
        wind_speed=wind, wind_dir_deg=wind_dir, s_ref=1.0, r_ref=2.0,
    )
    ratio_small = df_small_with["upsilon"].iloc[0] / df_small_no["upsilon"].iloc[0]
    assert ratio_small > ratio_large, (
        f"Surge amplification should be larger under small waves: "
        f"ratio_small={ratio_small:.2f}, ratio_large={ratio_large:.2f}"
    )


def test_forward_physics_bayesian_kinematic_consistency():
    """Verify that compute_unified_coastal_vulnerability_index matches the PyTensor
    kinematic_base * (Hs^n) algebraic formulation in build_learnable_exponent_bayesian_model.
    """
    hs = np.array([1.2, 2.5, 4.0])
    tp = np.array([8.0, 11.0, 14.0])
    surge = np.array([0.2, 0.6, 1.1])
    wind = np.array([5.0, 12.0, 18.0])
    wind_dir = np.array([60.0, 70.0, 80.0])
    tan_beta = 0.08

    for n_exp in [1.5, 2.0, 2.5, 3.0]:
        df = compute_unified_coastal_vulnerability_index(
            hs=hs,
            tp=tp,
            surge=surge,
            wind_speed=wind,
            wind_dir_deg=wind_dir,
            tan_beta=tan_beta,
            hs_exponent=n_exp,
        )
        upsilon_physics = df["upsilon"].values

        # Compute via Bayesian kinematic base formula with asymmetric wind
        prefactor = (1025.0 * (9.81 ** 1.5)) / (16.0 * ((2.0 * np.pi) ** 1.5))
        twl_arr = 1.0 + np.maximum(surge / 1.0, 0.0) + np.maximum(df["runup_r2_m"].values / 2.0, 0.0)
        wind_arr_mod = compute_asymmetric_wind_amplification(df["wind_normal_ms"].values)
        ws_arr = df["w_s_ms"].values

        kinematic_base = prefactor * tan_beta * (tp / ws_arr) * twl_arr * wind_arr_mod
        upsilon_reconstructed = kinematic_base * (hs ** n_exp)

        np.testing.assert_allclose(upsilon_physics, upsilon_reconstructed, rtol=1e-5)


def test_asymmetric_wind_scaling():
    """Verify that wind amplification follows physically asymmetric onshore/offshore scaling."""
    hs = np.array([2.0])
    tp = np.array([10.0])
    surge = np.array([0.3])
    shore_normal = 70.0  # Duck, NC

    # 1. Calm wind baseline (W = 0)
    df_calm = compute_unified_coastal_vulnerability_index(
        hs=hs, tp=tp, surge=surge, wind_speed=np.array([0.0]), wind_dir_deg=np.array([70.0]),
    )
    ups_calm = df_calm["upsilon"].iloc[0]

    # 2. Moderate onshore wind (+10 m/s at 70 deg)
    df_onshore = compute_unified_coastal_vulnerability_index(
        hs=hs, tp=tp, surge=surge, wind_speed=np.array([10.0]), wind_dir_deg=np.array([70.0]),
    )
    ups_onshore = df_onshore["upsilon"].iloc[0]

    # 3. Moderate offshore wind (-10 m/s at 250 deg)
    df_offshore = compute_unified_coastal_vulnerability_index(
        hs=hs, tp=tp, surge=surge, wind_speed=np.array([10.0]), wind_dir_deg=np.array([250.0]),
    )
    ups_offshore = df_offshore["upsilon"].iloc[0]

    # Asymmetric quadratic modulation around calm baseline:
    # Onshore gain = c_on * (10/15)^2 = 1.0 * 0.4444 = 0.4444
    # Offshore reduction = c_off * (10/15)^2 = 0.5 * 0.4444 = 0.2222
    onshore_gain = (ups_onshore - ups_calm) / ups_calm
    offshore_reduction = (ups_calm - ups_offshore) / ups_calm
    np.testing.assert_allclose(onshore_gain, 1.0 * (10.0 / 15.0) ** 2, rtol=1e-3)
    np.testing.assert_allclose(offshore_reduction, 0.5 * (10.0 / 15.0) ** 2, rtol=1e-3)
    # Physical asymmetry: onshore response is twice as large as offshore steepness flattening
    assert onshore_gain > offshore_reduction
    np.testing.assert_allclose(onshore_gain / offshore_reduction, 2.0, rtol=1e-3)

    # 4. Extreme gale capping (+30 m/s vs -30 m/s) with default bounds (0.6 onshore, 0.3 offshore)
    df_extreme_on = compute_unified_coastal_vulnerability_index(
        hs=hs, tp=tp, surge=surge, wind_speed=np.array([30.0]), wind_dir_deg=np.array([70.0]),
    )
    df_extreme_off = compute_unified_coastal_vulnerability_index(
        hs=hs, tp=tp, surge=surge, wind_speed=np.array([30.0]), wind_dir_deg=np.array([250.0]),
    )
    np.testing.assert_allclose(df_extreme_on["upsilon"].iloc[0] / ups_calm, 1.0 + 0.6, rtol=1e-3)
    np.testing.assert_allclose(df_extreme_off["upsilon"].iloc[0] / ups_calm, 1.0 - 0.3, rtol=1e-3)

    # 5. Direct test of compute_asymmetric_wind_amplification
    assert compute_asymmetric_wind_amplification(0.0) == 1.0
    assert abs(compute_asymmetric_wind_amplification(15.0) - 1.6) < 1e-6
    assert abs(compute_asymmetric_wind_amplification(-15.0) - 0.7) < 1e-6
    # Vectorized test
    w_vec = np.array([-30.0, -15.0, 0.0, 10.0, 30.0])
    res_vec = compute_asymmetric_wind_amplification(w_vec)
    np.testing.assert_allclose(res_vec, [0.7, 0.7, 1.0, 1.0 + (10/15)**2, 1.6], rtol=1e-5)


def test_coastal_site_profiles_database():
    """Verify global coastal site geomorphology database lookups."""
    assert "duck_nc" in COASTAL_SITE_PROFILES
    assert "narrabeen_aus" in COASTAL_SITE_PROFILES
    assert "torrey_pines_ca" in COASTAL_SITE_PROFILES
    assert "oahu_north_shore_hi" in COASTAL_SITE_PROFILES

    # Run Upsilon parameterized by named site
    df_narr = compute_unified_coastal_vulnerability_index(
        hs=np.array([2.5]),
        tp=np.array([11.0]),
        surge=np.array([0.4]),
        wind_speed=np.array([12.0]),
        wind_dir_deg=np.array([85.0]),
        site="narrabeen_aus",
    )
    assert df_narr["tan_beta"].iloc[0] == COASTAL_SITE_PROFILES["narrabeen_aus"]["tan_beta"]
    assert df_narr["d50_m"].iloc[0] == COASTAL_SITE_PROFILES["narrabeen_aus"]["d50_m"]
    assert df_narr["upsilon"].iloc[0] > 0.0

    # Test Oahu North Shore profile lookup
    df_oahu = compute_unified_coastal_vulnerability_index(
        hs=np.array([4.0]),
        tp=np.array([16.0]),
        surge=np.array([0.2]),
        wind_speed=np.array([10.0]),
        wind_dir_deg=np.array([325.0]),
        site="oahu_north_shore_hi",
    )
    assert df_oahu["tan_beta"].iloc[0] == COASTAL_SITE_PROFILES["oahu_north_shore_hi"]["tan_beta"]
    assert df_oahu["d50_m"].iloc[0] == COASTAL_SITE_PROFILES["oahu_north_shore_hi"]["d50_m"]
    assert df_oahu["upsilon"].iloc[0] > 0.0


def test_dynamic_foreshore_slope_and_grain_size():
    """Verify dynamic storm-flattening slope and storm-coarsening grain size response."""
    hs_storm_series = np.array([1.0, 1.0, 4.5, 4.5, 4.5, 1.0, 1.0])
    
    # 1. Dynamic slope: storm waves must reduce slope toward dissipative profile
    slopes = compute_dynamic_foreshore_slope(hs_storm_series, base_slope=0.08, min_slope=0.035, max_slope=0.12)
    assert slopes[3] < slopes[0]  # Storm peak is flatter than initial calm
    assert np.all(slopes >= 0.035)
    assert np.all(slopes <= 0.12)

    # 2. Dynamic grain size: storm winnowing coarsens active lag deposit
    d50s = compute_dynamic_grain_size(hs_storm_series, base_d50_m=0.00035, coarsening_factor=1.30)
    assert d50s[3] > d50s[0]  # Storm peak is coarser than calm
    assert np.all(d50s >= 0.00020)
    assert np.all(d50s <= 0.00060)

    # 3. Full Upsilon pipeline with dynamic_slope=True and dynamic_grain_size=True
    df_dyn = compute_unified_coastal_vulnerability_index(
        hs=hs_storm_series,
        tp=np.full_like(hs_storm_series, 10.0),
        surge=np.full_like(hs_storm_series, 0.5),
        wind_speed=np.full_like(hs_storm_series, 10.0),
        wind_dir_deg=np.full_like(hs_storm_series, 70.0),
        dynamic_slope=True,
        dynamic_grain_size=True,
    )
    assert len(df_dyn["tan_beta"].unique()) > 1
    assert len(df_dyn["d50_m"].unique()) > 1
    assert "upsilon" in df_dyn.columns


def test_signed_accretion_channel():
    """Verify signed morphodynamic channel (+ = Accretion, - = Erosion) under calm vs storm conditions."""
    df = generate_frf_duck_hurricane_ian_dataset()
    assert "signed_upsilon" in df.columns
    assert "accretion_potential" in df.columns

    # Test direct function evaluation
    ups_signed = compute_signed_coastal_vulnerability_index(
        upsilon=df["upsilon"].values,
        accretion_potential=df["accretion_potential"].values,
        dean_number=df["dean_number"].values,
        omega_crit=2.5,
    )
    assert len(ups_signed) == len(df)

    # 1. Storm conditions (Hs > 3m) must produce strictly negative forcing (erosion)
    storm_mask = df["hs_m"].values > 3.0
    assert np.all(ups_signed[storm_mask] < 0.0)

    # 2. Calm swell conditions (Hs < 1.2m, Dean < 2.0) must produce positive forcing (accretion)
    calm_mask = (df["hs_m"].values < 1.2) & (df["dean_number"].values < 2.0)
    assert np.all(ups_signed[calm_mask] > 0.0)

    # 3. Scalar inputs
    scalar_signed = compute_signed_coastal_vulnerability_index(
        upsilon=1000.0,
        accretion_potential=150.0,
        dean_number=1.2,
        omega_crit=2.5,
    )
    assert isinstance(scalar_signed, float)
    assert scalar_signed > 0.0


def test_tidal_smearing_factor():
    """Verify tidal energy dissipation & dilution factor Psi_tide = 1 / (1 + TR / (2 * Hs))."""
    # 1. Microtidal (Duck, NC: TR = 1.0m, Hs = 4.0m) -> Psi_tide = 1 / (1 + 1/8) = 0.8889
    psi_duck = compute_tidal_smearing_factor(tide_range_m=1.0, hs=4.0)
    np.testing.assert_allclose(psi_duck, 1.0 / (1.0 + 1.0 / 8.0), rtol=1e-4)
    assert psi_duck > 0.85  # Concentrated berm attack

    # 2. Macrotidal (Truc Vert: TR = 4.5m, Hs = 3.0m) -> Psi_tide = 1 / (1 + 4.5/6) = 0.5714
    psi_truc = compute_tidal_smearing_factor(tide_range_m=4.5, hs=3.0)
    np.testing.assert_allclose(psi_truc, 1.0 / (1.0 + 4.5 / 6.0), rtol=1e-4)
    assert psi_truc < 0.60  # Energy diluted across intertidal sweep

    # 3. Microtidal limit (TR -> 0)
    psi_micro_limit = compute_tidal_smearing_factor(tide_range_m=0.0, hs=3.0)
    np.testing.assert_allclose(psi_micro_limit, 1.0, rtol=1e-5)

    # 4. Array inputs
    tr_arr = np.array([0.0, 1.0, 4.5])
    hs_arr = np.array([2.0, 4.0, 3.0])
    psi_arr = compute_tidal_smearing_factor(tide_range_m=tr_arr, hs=hs_arr)
    assert len(psi_arr) == 3
    assert psi_arr[0] > psi_arr[1] > psi_arr[2]


def test_instantaneous_tide_phase_factor():
    """Verify Instantaneous High-Tide Wave Energy Gating factor:
    Psi_phase(t) = [ (eta_tide(t) - MSL) / (TR / 2) + 1 ]^p.
    """
    tr = 4.0  # 4m tidal range -> semi-amplitude = 2.0m

    # 1. Low tide (eta = -2.0m) -> Psi_phase = [-2.0/2.0 + 1]^1 = [ -1 + 1 ] = 0.0
    psi_low = compute_instantaneous_tide_phase_factor(tide=-2.0, tide_range_m=tr, p=1.0)
    np.testing.assert_allclose(psi_low, 0.0, atol=1e-5)

    # 2. Mid tide (eta = 0.0m = MSL) -> Psi_phase = [0/2.0 + 1]^1 = 1.0
    psi_mid = compute_instantaneous_tide_phase_factor(tide=0.0, tide_range_m=tr, p=1.0)
    np.testing.assert_allclose(psi_mid, 1.0, atol=1e-5)

    # 3. High tide (eta = +2.0m) -> Psi_phase = [+2.0/2.0 + 1]^1 = 2.0
    psi_high = compute_instantaneous_tide_phase_factor(tide=2.0, tide_range_m=tr, p=1.0)
    np.testing.assert_allclose(psi_high, 2.0, atol=1e-5)

    # 4. Exponent p = 2.0 -> high tide = 2^2 = 4.0, low tide = 0^2 = 0.0
    psi_high_quad = compute_instantaneous_tide_phase_factor(tide=2.0, tide_range_m=tr, p=2.0)
    np.testing.assert_allclose(psi_high_quad, 4.0, atol=1e-5)

    # 5. Extreme spring low tide / drawdown (eta = -2.5m < -2.0m) -> clipped at 0.0
    psi_spring_low = compute_instantaneous_tide_phase_factor(tide=-2.5, tide_range_m=tr, p=1.0)
    np.testing.assert_allclose(psi_spring_low, 0.0, atol=1e-5)

    # 6. Microtidal limit (TR -> 0) -> neutral baseline 1.0
    psi_micro = compute_instantaneous_tide_phase_factor(tide=0.5, tide_range_m=0.0, p=1.0)
    np.testing.assert_allclose(psi_micro, 1.0, atol=1e-5)

    # 7. Alias function equivalence
    np.testing.assert_allclose(
        compute_tide_phase_gating(tide=1.0, tide_range_m=2.0),
        compute_instantaneous_tide_phase_factor(tide=1.0, tide_range_m=2.0),
    )

    # 8. Array evaluation
    tide_series = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    psi_series = compute_instantaneous_tide_phase_factor(tide=tide_series, tide_range_m=4.0, p=1.0)
    expected_series = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(psi_series, expected_series, atol=1e-5)


def test_tidal_inundation_factor_clipping():
    """Verify exponential tidal inundation modulator Psi_tide(eta, Hs) strictly saturates
    at numerical floor (0.1) and ceiling (10.0) under extreme surges and calm conditions."""
    from src.physics import compute_tidal_inundation_factor, compute_tidal_inundation_modulator

    # 1. Normal condition: eta = 0.5m, MHW = 0.0m, Hs = 1.0m -> exp(0.5/1.0) = exp(0.5) ~ 1.6487
    val_norm = compute_tidal_inundation_factor(tide=0.5, mhw=0.0, hs=1.0)
    np.testing.assert_allclose(val_norm, np.exp(0.5), rtol=1e-4)

    # 2. Extreme hurricane storm surge: eta - MHW = +2.0m, Hs = 0.6m -> exp(2.0/0.6) = exp(3.33) ~ 28.0
    # Must be safely capped at a_max = 10.0
    val_surge = compute_tidal_inundation_factor(tide=2.0, mhw=0.0, hs=0.6, a_max=10.0)
    assert val_surge == 10.0

    # 3. Minimum wave floor: Hs = 0.2m (below 0.5m minimum floor) -> denominator is 0.5m
    # eta - MHW = 3.0m -> raw exp(3.0/0.5) = exp(6.0) ~ 403.4 -> capped at 10.0
    val_calm_surge = compute_tidal_inundation_factor(tide=3.0, mhw=0.0, hs=0.2, a_max=10.0)
    assert val_calm_surge == 10.0

    # 4. Extreme negative tide / drawdown: eta - MHW = -5.0m, Hs = 1.0m -> raw exp(-5.0) ~ 0.0067 -> floored at a_min = 0.1
    val_drawdown = compute_tidal_inundation_factor(tide=-5.0, mhw=0.0, hs=1.0, a_min=0.1)
    assert val_drawdown == 0.1

    # 5. Alias equivalence
    val_alias = compute_tidal_inundation_modulator(tide=2.0, mhw=0.0, hs=0.6, a_max=10.0)
    assert val_alias == 10.0



def test_tide_phase_gating_macrotidal_truc_vert():
    """Verify that Psi_phase eliminates false-positive erosion signals when huge waves occur
    at low tide on macrotidal beaches like Truc Vert (TR = 4.5m).
    """
    hs = np.array([5.0])  # 5.0m major storm swell
    tp = np.array([14.0])
    surge = np.array([0.2])
    wind = np.array([12.0])
    wind_dir = np.array([275.0])

    # Truc Vert: TR = 4.5m -> semi-amplitude = 2.25m
    # Scenario A: Storm peak at Low Tide (eta_tide = -2.25m)
    df_low = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind,
        wind_dir_deg=wind_dir,
        tide=np.array([-2.25]),
        site="truc_vert_france",
    )
    assert df_low["psi_phase"].iloc[0] == pytest.approx(0.0, abs=1e-4)
    assert df_low["upsilon"].iloc[0] == pytest.approx(0.0, abs=1e-4)  # No false-positive erosion at low tide!

    # Scenario B: Storm peak at Mid Tide (eta_tide = 0.0m)
    df_mid = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind,
        wind_dir_deg=wind_dir,
        tide=np.array([0.0]),
        site="truc_vert_france",
    )
    assert df_mid["psi_phase"].iloc[0] == pytest.approx(1.0, abs=1e-4)
    assert df_mid["upsilon"].iloc[0] > 0.0

    # Scenario C: Storm peak at High Tide (eta_tide = +2.25m)
    df_high = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind,
        wind_dir_deg=wind_dir,
        tide=np.array([2.25]),
        site="truc_vert_france",
    )
    assert df_high["psi_phase"].iloc[0] == pytest.approx(2.0, abs=1e-4)
    assert df_high["upsilon"].iloc[0] == pytest.approx(2.0 * df_mid["upsilon"].iloc[0], rel=1e-3)
    assert df_high["upsilon"].iloc[0] > df_mid["upsilon"].iloc[0] > df_low["upsilon"].iloc[0]


def test_directional_wave_factor_and_rectification():
    """Verify rectified shore-normal directional wave factor Psi_dir = max(0, cos(theta_rel))^2."""
    shore_normal = 70.0  # Duck, NC

    # 1. Direct shore-normal incidence (theta_wave = 70 deg) -> Psi_dir = 1.0
    psi_normal = compute_directional_wave_factor(wave_dir_deg=70.0, shore_normal_deg=shore_normal, m=2.0)
    np.testing.assert_allclose(psi_normal, 1.0, rtol=1e-5)

    # 2. Oblique wave incidence (45 deg angle) -> cos(45)^2 = 0.50 (matching radiation stress S_xx)
    psi_oblique = compute_directional_wave_factor(wave_dir_deg=115.0, shore_normal_deg=shore_normal, m=2.0)
    np.testing.assert_allclose(psi_oblique, 0.50, rtol=1e-4)

    # 3. Shadowed / offshore wave incidence (120 deg angle) -> rectified to 0.0 (no negative erosion/accretion)
    psi_shadowed = compute_directional_wave_factor(wave_dir_deg=190.0, shore_normal_deg=shore_normal, m=2.0)
    np.testing.assert_allclose(psi_shadowed, 0.0, atol=1e-5)

    # 4. Array broadcast
    angles = np.array([70.0, 115.0, 190.0, 25.0])
    psi_vec = compute_directional_wave_factor(wave_dir_deg=angles, shore_normal_deg=shore_normal, m=2.0)
    assert len(psi_vec) == 4
    np.testing.assert_allclose(psi_vec[0], 1.0, rtol=1e-4)
    np.testing.assert_allclose(psi_vec[1], 0.5, rtol=1e-4)
    np.testing.assert_allclose(psi_vec[2], 0.0, atol=1e-5)
    np.testing.assert_allclose(psi_vec[3], 0.5, rtol=1e-4)


def test_longshore_transport_divergence():
    """Verify alongshore transport / rotational potential Phi_longshore = sin(2 * theta_rel)."""
    shore_normal = 85.0  # Narrabeen, Australia

    # 1. Shore-normal incidence -> Phi_longshore = 0
    phi_normal = compute_longshore_transport_divergence(85.0, shore_normal_deg=shore_normal)
    np.testing.assert_allclose(phi_normal, 0.0, atol=1e-5)

    # 2. +45 deg oblique ECL storm -> Phi_longshore = sin(90) = +1.0
    phi_pos = compute_longshore_transport_divergence(130.0, shore_normal_deg=shore_normal)
    np.testing.assert_allclose(phi_pos, 1.0, rtol=1e-4)

    # 3. -45 deg oblique storm -> Phi_longshore = sin(-90) = -1.0
    phi_neg = compute_longshore_transport_divergence(40.0, shore_normal_deg=shore_normal)
    np.testing.assert_allclose(phi_neg, -1.0, rtol=1e-4)


def test_unified_index_global_continuous_scaling():
    """Verify that Upsilon self-scales across global archetypes (Duck, Torrey Pines, Truc Vert, Narrabeen)
    as a continuous dimensionless manifold.
    """
    # 1. Duck: Microtidal storm barrier (Surge dominant, concentrated tidal impact)
    df_duck = compute_unified_coastal_vulnerability_index(
        hs=np.array([4.0]),
        tp=np.array([10.0]),
        surge=np.array([1.5]),
        wind_speed=np.array([15.0]),
        wind_dir_deg=np.array([70.0]),
        wave_dir_deg=np.array([70.0]),
        site="duck_nc",
    )
    assert df_duck["psi_tide"].iloc[0] > 0.85
    assert df_duck["psi_dir"].iloc[0] == 1.0
    assert df_duck["upsilon"].iloc[0] > 0.0

    # 2. Truc Vert: Macrotidal high-energy (Tidal smearing dilutes energy)
    df_truc = compute_unified_coastal_vulnerability_index(
        hs=np.array([3.0]),
        tp=np.array([12.0]),
        surge=np.array([0.3]),
        wind_speed=np.array([10.0]),
        wind_dir_deg=np.array([275.0]),
        wave_dir_deg=np.array([275.0]),
        site="truc_vert_france",
    )
    assert df_truc["psi_tide"].iloc[0] < 0.60
    assert df_truc["upsilon"].iloc[0] > 0.0

    # 3. Torrey Pines: Pacific long-period swell (Runup dominant, surge ~ 0)
    df_tp = compute_unified_coastal_vulnerability_index(
        hs=np.array([2.5]),
        tp=np.array([16.0]),
        surge=np.array([0.05]),
        wind_speed=np.array([3.0]),
        wind_dir_deg=np.array([265.0]),
        wave_dir_deg=np.array([265.0]),
        site="torrey_pines_ca",
    )
    assert df_tp["runup_r2_m"].iloc[0] > 1.5
    assert df_tp["total_water_level_m"].iloc[0] > 1.5

    # 4. Narrabeen: Embayed rotational coast (Oblique waves split into cross-shore vs alongshore)
    df_narr = compute_unified_coastal_vulnerability_index(
        hs=np.array([3.0]),
        tp=np.array([11.0]),
        surge=np.array([0.4]),
        wind_speed=np.array([12.0]),
        wind_dir_deg=np.array([130.0]),
        wave_dir_deg=np.array([130.0]),  # +45 deg oblique East Coast Low
        site="narrabeen_aus",
    )
    np.testing.assert_allclose(df_narr["psi_dir"].iloc[0], 0.50, rtol=1e-3)
    np.testing.assert_allclose(df_narr["phi_longshore"].iloc[0], 1.0, rtol=1e-3)

    # 5. Oahu North Shore: Reef-Fronted Coast (Depth-limited breaking & flat friction)
    df_oahu = compute_unified_coastal_vulnerability_index(
        hs=np.array([5.0]),
        tp=np.array([16.0]),
        surge=np.array([0.1]),
        tide=np.array([0.4]),
        wind_speed=np.array([8.0]),
        wind_dir_deg=np.array([325.0]),
        wave_dir_deg=np.array([325.0]),
        site="oahu_north_shore_hi",
    )
    assert df_oahu["psi_reef"].iloc[0] < 1.0
    assert df_oahu["hs_shore_m"].iloc[0] < 5.0
    assert df_oahu["h_reef_m"].iloc[0] == 1.5  # 0.4 + 0.1 - (-1.0) = 1.5m
    assert df_oahu["k_c_crest"].iloc[0] == pytest.approx(0.78 * 1.5 / 5.0, rel=1e-3)


def test_reef_dissipation_modulator_physics():
    """Verify exact mathematical formulations of depth-limited breaking Kc and flat friction Kf."""
    # Scenario: 4.0m deepwater swell hitting reef with 1.0m water depth (Zreef = -1.0m, tide=0, surge=0)
    # Wr = 150m, cf = 0.08, gamma_b = 0.78, l_d_ref_m = 25.0
    psi_reef, h_r, k_c, k_f, hs_shore = compute_reef_dissipation_modulator(
        hs=4.0,
        tide=0.0,
        surge=0.0,
        z_reef_m=-1.0,
        reef_flat_width_m=150.0,
        c_f=0.08,
        gamma_b=0.78,
        l_d_ref_m=25.0,
        is_reef=True,
    )
    # 1. Reef depth hr = max(0, 0 + 0 - (-1.0)) = 1.0m
    assert h_r == pytest.approx(1.0, rel=1e-4)

    # 2. Crest clipping: Kc = min(1.0, 0.78 * 1.0 / 4.0) = 0.195
    assert k_c == pytest.approx(0.195, rel=1e-4)

    # 3. Flat friction: Kf = exp(-0.08 * 150.0 / (25.0 * 1.0)) = exp(-0.48) = 0.61878
    expected_kf = np.exp(-12.0 / 25.0)
    assert k_f == pytest.approx(expected_kf, rel=1e-4)

    # 4. Total modulator Psi_reef = Kc * Kf
    expected_psi = 0.195 * expected_kf
    assert psi_reef == pytest.approx(expected_psi, rel=1e-4)
    assert hs_shore == pytest.approx(4.0 * expected_psi, rel=1e-4)


def test_reef_dissipation_tide_sensitivity():
    """Verify that higher water levels reduce friction and increase wave transmission across the reef."""
    hs_in = 3.0
    # Low tide: tide = -0.5m -> hr = 0.5m
    psi_low, hr_low, kc_low, kf_low, hs_low = compute_reef_dissipation_modulator(
        hs=hs_in, tide=-0.5, surge=0.0, z_reef_m=-1.0, reef_flat_width_m=100.0, c_f=0.05,
    )
    # High tide: tide = +1.0m -> hr = 2.0m
    psi_high, hr_high, kc_high, kf_high, hs_high = compute_reef_dissipation_modulator(
        hs=hs_in, tide=1.0, surge=0.0, z_reef_m=-1.0, reef_flat_width_m=100.0, c_f=0.05,
    )

    assert hr_high > hr_low
    assert kc_high > kc_low
    assert kf_high > kf_low
    assert psi_high > psi_low
    assert hs_high > hs_low


def test_reef_dry_exposure_and_non_reef_bypass():
    """Verify that dry reef blocks 100% of wave energy and non-reef coasts pass 100% through."""
    # 1. Dry reef: water level below reef crest (tide = -1.5m, Zreef = -1.0m -> hr = 0)
    psi_dry, hr_dry, kc_dry, kf_dry, hs_dry = compute_reef_dissipation_modulator(
        hs=4.0, tide=-1.5, surge=0.0, z_reef_m=-1.0, reef_flat_width_m=150.0, is_reef=True,
    )
    assert hr_dry == 0.0
    assert kc_dry == 0.0
    assert kf_dry == 0.0
    assert psi_dry == 0.0
    assert hs_dry == 0.0

    # 2. Non-reef bypass: is_reef = False
    psi_nr, hr_nr, kc_nr, kf_nr, hs_nr = compute_reef_dissipation_modulator(
        hs=4.0, tide=0.5, surge=0.2, z_reef_m=-1.0, is_reef=False,
    )
    assert psi_nr == 1.0
    assert kc_nr == 1.0
    assert kf_nr == 1.0
    assert hs_nr == 4.0


def test_upsilon_disequilibrium_and_signed_rate():
    """Verify Signed Equilibrium Formulation: Upsilon_diseq = Upsilon - Upsilon_eq and dX/dt response."""
    ups_eq = 2500.0

    # 1. Storm state (Upsilon = 8000 > 2500)
    storm_ups = 8000.0
    storm_diseq = compute_upsilon_disequilibrium(storm_ups, upsilon_eq=ups_eq)
    storm_rate = compute_signed_equilibrium_rate(storm_ups, upsilon_eq=ups_eq, c_erosion=1.2e-5, c_accretion=1.8e-5)
    assert storm_diseq == 5500.0
    assert storm_rate < 0.0  # Erosion
    assert np.isclose(storm_rate, -1.2e-5 * 5500.0)

    # 2. Fair-weather recovery state (Upsilon = 500 < 2500)
    calm_ups = 500.0
    calm_diseq = compute_upsilon_disequilibrium(calm_ups, upsilon_eq=ups_eq)
    calm_rate = compute_signed_equilibrium_rate(calm_ups, upsilon_eq=ups_eq, c_erosion=1.2e-5, c_accretion=1.8e-5)
    assert calm_diseq == -2000.0
    assert calm_rate > 0.0  # Accretion / berm recovery
    assert np.isclose(calm_rate, 1.8e-5 * 2000.0)

    # 3. Exact equilibrium state (Upsilon = 2500)
    eq_rate = compute_signed_equilibrium_rate(ups_eq, upsilon_eq=ups_eq)
    assert eq_rate == 0.0


def test_simulate_signed_equilibrium_upsilon_evolution():
    """Verify continuous shoreline morphodynamics driven by Signed Equilibrium Upsilon."""
    # Sequence of calm fair-weather (Upsilon=400), storm (Upsilon=10000), calm recovery (Upsilon=400)
    ups_seq = np.array([400.0] * 50 + [10000.0] * 20 + [400.0] * 50)
    x_pos, x_obs = simulate_signed_equilibrium_upsilon_evolution(
        upsilon=ups_seq,
        c_erosion=1.2e-5,
        c_accretion=1.8e-5,
        upsilon_eq=2500.0,
    )

    assert len(x_pos) == len(ups_seq)
    # Initial phase: positive accretion
    assert x_pos[49] > x_pos[0]
    # Storm phase: erosion drop
    assert x_pos[69] < x_pos[49]
    # Recovery phase: accretion rebuild
    assert x_pos[-1] > x_pos[69]


def test_compute_expanding_window_equilibrium_causality():
    """Verify Suggestion 7: Causal expanding and trailing window equilibrium has zero future data leakage."""
    # Create arbitrary forcing series
    rng = np.random.default_rng(123)
    n = 100
    ups_base = rng.uniform(500.0, 5000.0, size=n)

    # 1. Expanding window without warm-up blending (min_periods=1)
    eq_expanding = compute_expanding_window_equilibrium(ups_base, window_hours=None, min_periods=1)
    assert len(eq_expanding) == n
    # Verify exact cumulative mean formula at step t
    for t in range(n):
        expected_mean = float(np.mean(ups_base[: t + 1]))
        np.testing.assert_allclose(eq_expanding[t], expected_mean, rtol=1e-5)

    # 2. Strict Proof of Zero Look-Ahead Leakage:
    # Modifying future values (t >= 50) MUST NOT alter past equilibrium (t < 50)
    ups_altered = ups_base.copy()
    ups_altered[50:] = 50000.0  # Massive future super-storm
    eq_altered = compute_expanding_window_equilibrium(ups_altered, window_hours=None, min_periods=1)
    np.testing.assert_allclose(eq_altered[:50], eq_expanding[:50], rtol=1e-10)

    # 3. Trailing window moving average (e.g. 24 hours)
    w = 24
    eq_trailing = compute_expanding_window_equilibrium(ups_base, window_hours=w, min_periods=1)
    for t in range(n):
        expected_trail = float(np.mean(ups_base[max(0, t - w + 1) : t + 1]))
        np.testing.assert_allclose(eq_trailing[t], expected_trail, rtol=1e-5)

    # 4. Trailing window zero leakage test
    eq_trailing_alt = compute_expanding_window_equilibrium(ups_altered, window_hours=w, min_periods=1)
    np.testing.assert_allclose(eq_trailing_alt[:50], eq_trailing[:50], rtol=1e-10)

    # 5. Causal EMA mode
    eq_ema = compute_expanding_window_equilibrium(ups_base, alpha_ema=0.1, initial_baseline=2500.0)
    assert len(eq_ema) == n
    assert eq_ema[0] == 2500.0
    assert np.isclose(eq_ema[1], 0.1 * ups_base[1] + 0.9 * 2500.0)


def test_simulate_signed_equilibrium_with_expanding_eq():
    """Verify simulation with expanding and trailing equilibrium modes (Suggestion 7)."""
    ups_seq = np.array([300.0] * 40 + [8000.0] * 20 + [300.0] * 40)

    # Expanding window simulation
    x_pos_exp, x_obs_exp = simulate_signed_equilibrium_upsilon_evolution(
        upsilon=ups_seq,
        upsilon_eq_mode="expanding",
        c_erosion=1.2e-5,
        c_accretion=1.8e-5,
    )
    assert len(x_pos_exp) == len(ups_seq)
    assert x_pos_exp[39] > x_pos_exp[0]  # Accretion
    assert x_pos_exp[59] < x_pos_exp[39]  # Storm erosion
    assert x_pos_exp[-1] > x_pos_exp[59]  # Recovery accretion

    # Trailing window simulation
    x_pos_trail, _ = simulate_signed_equilibrium_upsilon_evolution(
        upsilon=ups_seq,
        upsilon_eq_mode="trailing",
        window_hours=24,
    )
    assert len(x_pos_trail) == len(ups_seq)


def test_build_latent_equilibrium_bayesian_model():
    """Verify PyMC Latent Equilibrium Model graph construction and parameter estimation without leakage."""
    from src.models import build_latent_equilibrium_bayesian_model

    rng = np.random.default_rng(42)
    n = 60
    forcing = rng.uniform(500.0, 6000.0, size=n)
    delta_x = rng.normal(0.0, 0.5, size=n)

    model, meta = build_latent_equilibrium_bayesian_model(
        forcing=forcing,
        delta_x_obs=delta_x,
        model_name="test_latent_eq",
        max_lag=12,
        scale_forcing=False,
        upsilon_eq_prior_mu=2500.0,
        upsilon_eq_prior_sigma=500.0,
    )

    assert meta["model_name"] == "test_latent_eq"
    assert meta["n_obs"] == n
    assert "gamma" in model.named_vars
    assert "upsilon_eq" in model.named_vars
    assert "upsilon_diseq" in model.named_vars
    assert "delta_erosion" in model.named_vars
    assert "delta_accretion" in model.named_vars
    assert "mu" in model.named_vars
    assert "y_obs" in model.named_vars


def test_critical_shear_threshold_scaling():
    """Verify physical scaling of Upsilon_crit(tan(beta), D_50)."""
    # 1. Baseline reference conditions (Duck, NC: tan_beta = 0.08, D50 = 0.35 mm) -> Upsilon_crit = 500.0
    u_crit_base = compute_critical_shear_threshold(tan_beta=0.08, d50_m=0.00035, upsilon_crit_0=500.0)
    np.testing.assert_allclose(u_crit_base, 500.0, rtol=1e-5)

    # 2. Coarser sand (D50 = 0.70 mm = 2x) -> 2x higher threshold to mobilize
    u_crit_coarse = compute_critical_shear_threshold(tan_beta=0.08, d50_m=0.00070, upsilon_crit_0=500.0)
    np.testing.assert_allclose(u_crit_coarse, 1000.0, rtol=1e-5)

    # 3. Steeper foreshore/dune slope (tan_beta = 0.16 = 2x) -> 2x higher resistance threshold
    u_crit_steep = compute_critical_shear_threshold(tan_beta=0.16, d50_m=0.00035, upsilon_crit_0=500.0)
    np.testing.assert_allclose(u_crit_steep, 1000.0, rtol=1e-5)

    # 4. Fine sand + gentle dissipative slope (Torrey Pines: tan_beta = 0.04, D50 = 0.20 mm)
    u_crit_tp = compute_critical_shear_threshold(tan_beta=0.04, d50_m=0.00020, upsilon_crit_0=500.0)
    expected_tp = 500.0 * (0.00020 / 0.00035) * (0.04 / 0.08)
    np.testing.assert_allclose(u_crit_tp, expected_tp, rtol=1e-5)

    # 5. Coarse sand + steep reflective slope (Oahu North Shore: tan_beta = 0.09, D50 = 0.45 mm)
    u_crit_oahu = compute_critical_shear_threshold(tan_beta=0.09, d50_m=0.00045, upsilon_crit_0=500.0)
    expected_oahu = 500.0 * (0.00045 / 0.00035) * (0.09 / 0.08)
    np.testing.assert_allclose(u_crit_oahu, expected_oahu, rtol=1e-5)

    # 6. Exponent sensitivity: a, b in [0.5, 1.5]
    u_crit_sublinear = compute_critical_shear_threshold(tan_beta=0.08, d50_m=0.00070, upsilon_crit_0=500.0, a=0.5, b=0.5)
    np.testing.assert_allclose(u_crit_sublinear, 500.0 * np.sqrt(2.0), rtol=1e-5)

    u_crit_superlinear = compute_critical_shear_threshold(tan_beta=0.16, d50_m=0.00035, upsilon_crit_0=500.0, a=1.0, b=1.5)
    np.testing.assert_allclose(u_crit_superlinear, 500.0 * (2.0 ** 1.5), rtol=1e-5)

    # 7. Exact Soulsby Shields nonlinear mode
    u_crit_soulsby = compute_critical_shear_threshold(tan_beta=0.08, d50_m=0.00070, upsilon_crit_0=500.0, mode="soulsby_shields")
    assert u_crit_soulsby > 500.0

    # 8. Array broadcast
    tb_vec = np.array([0.04, 0.08, 0.12])
    d50_vec = np.array([0.00020, 0.00035, 0.00050])
    u_crit_vec = compute_critical_shear_threshold(tan_beta=tb_vec, d50_m=d50_vec, upsilon_crit_0=500.0)
    assert len(u_crit_vec) == 3
    assert u_crit_vec[0] < u_crit_vec[1] < u_crit_vec[2]


def test_soulsby_shields_critical_shear_stress():
    """Verify Soulsby (1997) nonlinear Shields parameter and critical shear stress computation."""
    theta_cr, tau_cr = compute_shields_critical_shear_stress_soulsby(d50_m=0.00035)
    assert 0.03 <= theta_cr <= 0.10
    assert tau_cr > 0.0

    # Coarser sand has higher dimensional critical shear stress
    _, tau_cr_fine = compute_shields_critical_shear_stress_soulsby(d50_m=0.00015)
    _, tau_cr_coarse = compute_shields_critical_shear_stress_soulsby(d50_m=0.00100)
    assert tau_cr_fine < tau_cr < tau_cr_coarse

    # Vectorized computation
    d50_arr = np.array([0.00015, 0.00035, 0.00080])
    th_vec, tau_vec = compute_shields_critical_shear_stress_soulsby(d50_m=d50_arr)
    assert len(th_vec) == 3
    assert len(tau_vec) == 3
    assert np.all(np.diff(tau_vec) > 0)


def test_critical_threshold_sensitivity_sweep():
    """Verify 2D sensitivity grid sweep over exponents a, b in [0.5, 1.5]."""
    ups_series = np.linspace(100.0, 3000.0, 50)
    df_sweep = sweep_critical_threshold_sensitivity(
        upsilon=ups_series,
        tan_beta=0.09,
        d50_m=0.00045,
        a_values=(0.5, 1.0, 1.5),
        b_values=(0.5, 1.0, 1.5),
        upsilon_crit_0=500.0,
    )
    assert len(df_sweep) == 9
    assert "mean_upsilon_crit" in df_sweep.columns
    assert "activation_fraction" in df_sweep.columns
    assert "mean_excess_upsilon" in df_sweep.columns
    assert np.all(df_sweep["activation_fraction"] > 0.0)
    assert np.all(df_sweep["activation_fraction"] <= 1.0)
    assert np.all(df_sweep["mean_excess_upsilon"] >= 0.0)


def test_learnable_critical_threshold_bayesian_model_structure():
    """Verify construction of PyMC learnable critical threshold model."""
    upsilon_raw = np.linspace(200.0, 2500.0, 40)
    delta_x_obs = np.random.RandomState(42).randn(40) * 0.5

    model, meta = build_learnable_critical_threshold_bayesian_model(
        upsilon_raw=upsilon_raw,
        delta_x_obs=delta_x_obs,
        tan_beta=0.08,
        d50_m=0.00035,
        max_lag=6,
    )
    assert "a" in model.named_vars
    assert "b" in model.named_vars
    assert "upsilon_crit_0" in model.named_vars
    assert "upsilon_crit" in model.named_vars
    assert "upsilon_excess" in model.named_vars
    assert "y_obs" in model.named_vars
    assert meta["max_lag"] == 6


def test_excess_upsilon_activation_and_noise_cutoff():
    """Verify Upsilon_excess = max(0, Upsilon - Upsilon_crit) cuts off calm background noise
    and preserves active storm forcing.
    """
    u_crit = 500.0

    # 1. Calm fair-weather condition (Upsilon = 250 < 500) -> Upsilon_excess = 0.0 (Noise eliminated)
    calm_excess = compute_excess_upsilon(upsilon=250.0, upsilon_crit=u_crit)
    assert calm_excess == 0.0

    # 2. Exact critical threshold (Upsilon = 500) -> Upsilon_excess = 0.0
    thresh_excess = compute_excess_upsilon(upsilon=500.0, upsilon_crit=u_crit)
    assert thresh_excess == 0.0

    # 3. Active storm condition (Upsilon = 3500 > 500) -> Upsilon_excess = 3000.0
    storm_excess = compute_excess_upsilon(upsilon=3500.0, upsilon_crit=u_crit)
    assert storm_excess == 3000.0

    # 4. Extreme storm (Upsilon = 20000 > 500) -> Upsilon_excess = 19500.0
    extreme_excess = compute_excess_upsilon(upsilon=20000.0, upsilon_crit=u_crit)
    assert extreme_excess == 19500.0

    # 5. Direct computation with tan_beta and d50_m
    direct_excess = compute_excess_upsilon(upsilon=1200.0, tan_beta=0.08, d50_m=0.00035, upsilon_crit_0=500.0)
    assert direct_excess == 700.0

    # 6. Physical aliases equivalence
    np.testing.assert_allclose(
        compute_critical_threshold_shear_activation(1200.0, upsilon_crit=500.0),
        compute_excess_vulnerability_index(1200.0, upsilon_crit=500.0),
    )

    # 7. Array evaluation
    ups_series = np.array([100.0, 300.0, 500.0, 1000.0, 5000.0])
    excess_series = compute_excess_upsilon(ups_series, upsilon_crit=500.0)
    expected_series = np.array([0.0, 0.0, 0.0, 500.0, 4500.0])
    np.testing.assert_allclose(excess_series, expected_series, atol=1e-5)


def test_unified_index_critical_threshold_integration():
    """Verify integration of Upsilon_crit and Upsilon_excess in compute_unified_coastal_vulnerability_index."""
    hs_series = np.array([0.4, 0.5, 4.0, 5.0, 0.3])  # Calm, calm, storm, storm, calm
    tp_series = np.array([5.0, 6.0, 14.0, 15.0, 5.0])
    surge_series = np.array([0.0, 0.0, 1.2, 1.5, 0.0])
    wind_series = np.array([2.0, 3.0, 18.0, 22.0, 2.0])

    df = compute_unified_coastal_vulnerability_index(
        hs=hs_series,
        tp=tp_series,
        surge=surge_series,
        wind_speed=wind_series,
        wind_dir_deg=np.full_like(hs_series, 70.0),
        site="duck_nc",
        upsilon_crit_0=500.0,
    )

    assert "upsilon_crit" in df.columns
    assert "upsilon_excess" in df.columns
    assert "log_upsilon_excess" in df.columns

    # Duck reference critical threshold
    np.testing.assert_allclose(df["upsilon_crit"].iloc[0], 500.0, rtol=1e-4)

    # Calm days (index 0, 4) should have Upsilon < Upsilon_crit -> Upsilon_excess == 0.0
    assert df["upsilon_excess"].iloc[0] == 0.0
    assert df["upsilon_excess"].iloc[4] == 0.0

    # Storm days (index 2, 3) should have Upsilon > Upsilon_crit -> Upsilon_excess > 0.0
    assert df["upsilon_excess"].iloc[2] > 0.0
    assert df["upsilon_excess"].iloc[3] > 0.0
    np.testing.assert_allclose(
        df["upsilon_excess"].iloc[2],
        df["upsilon"].iloc[2] - df["upsilon_crit"].iloc[2],
        rtol=1e-4,
    )


def test_coupled_2d_embayment_displacement_formulation():
    """Verify the Coupled 2D Embayment Displacement Proxy Delta_X_net = Delta_X_cross + alpha_rot * Phi_ls * ((y - y_c) / L)."""
    # 1. Shore-normal wave approach (Phi_longshore = 0.0) -> Pure cross-shore response
    dx_cross = -5.0
    dx_net_norm = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=dx_cross,
        phi_longshore=0.0,
        y_alongshore_m=3400.0,  # North Narrabeen
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    assert dx_net_norm == dx_cross

    # 2. Oblique East Coast Low (ECL) storm from SE (Phi_longshore = +1.0)
    # North transect (PF1, y=3400m): (3400 - 1800) / 3600 = 1600 / 3600 = +4/9
    # Rotational term = 12.0 * 1.0 * (4/9) = +5.3333 m
    # Net displacement = -3.0 + 5.3333 = +2.3333 m (accretion at north downdrift headland!)
    dx_north = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=-3.0,
        phi_longshore=1.0,
        y_alongshore_m=3400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    np.testing.assert_allclose(dx_north, -3.0 + 12.0 * (1600.0 / 3600.0), rtol=1e-4)
    assert dx_north > 0.0  # Accretion at northern end

    # Central fulcrum (PF2, y=1800m): (1800 - 1800) / 3600 = 0.0 -> Net = -3.0 m
    dx_center = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=-3.0,
        phi_longshore=1.0,
        y_alongshore_m=1800.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    assert dx_center == -3.0

    # South transect (PF4, y=400m): (400 - 1800) / 3600 = -1400 / 3600 = -7/18
    # Rotational term = 12.0 * 1.0 * (-7/18) = -4.6667 m
    # Net displacement = -3.0 - 4.6667 = -7.6667 m (severe scour at south updrift end!)
    dx_south = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=-3.0,
        phi_longshore=1.0,
        y_alongshore_m=400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    np.testing.assert_allclose(dx_south, -3.0 - 12.0 * (1400.0 / 3600.0), rtol=1e-4)
    assert dx_south < -3.0

    # 3. Reversed NE wave approach (Phi_longshore = -1.0) flips rotation dipole
    dx_north_ne = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=-3.0,
        phi_longshore=-1.0,
        y_alongshore_m=3400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    dx_south_ne = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=-3.0,
        phi_longshore=-1.0,
        y_alongshore_m=400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    assert dx_north_ne < -3.0  # North erodes
    assert dx_south_ne > -3.0  # South receives sand

    # 4. Aliases equivalence
    np.testing.assert_allclose(
        compute_alongshore_transport_gradient_rotation(-3.0, 1.0, 3400.0),
        compute_embayed_beach_rotation(-3.0, 1.0, 3400.0),
    )

    # 5. Vectorized array evaluation
    y_coords = np.array([400.0, 1800.0, 3400.0])
    dx_arr = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=np.full(3, -2.0),
        phi_longshore=np.full(3, 1.0),
        y_alongshore_m=y_coords,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
    )
    assert dx_arr[0] < -2.0  # South eroded
    assert dx_arr[1] == -2.0  # Center unchanged
    assert dx_arr[2] > -2.0  # North accreted / widened


def test_simulate_coupled_2d_embayment_evolution_narrabeen():
    """Verify multi-transect time-series simulation of Narrabeen headland beach rotation."""
    # Synthetic 200-hour East Coast Low storm time-series
    hours = 200
    ups_series = np.full(hours, 6500.0)  # High storm Upsilon forcing (>> 2500 equilibrium)
    phi_series = np.full(hours, 1.0)     # Oblique SE storm waves (Phi_longshore = +1.0)

    # Simulate PF1 (North Narrabeen / Alley, y=3400m)
    x_north, _ = simulate_coupled_2d_embayment_evolution(
        upsilon=ups_series,
        phi_longshore=phi_series,
        y_alongshore_m=3400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
        initial_x=0.0,
    )

    # Simulate PF2 (Central Fulcrum, y=1800m)
    x_center, _ = simulate_coupled_2d_embayment_evolution(
        upsilon=ups_series,
        phi_longshore=phi_series,
        y_alongshore_m=1800.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
        initial_x=0.0,
    )

    # Simulate PF4 (Collaroy / South, y=400m)
    x_south, _ = simulate_coupled_2d_embayment_evolution(
        upsilon=ups_series,
        phi_longshore=phi_series,
        y_alongshore_m=400.0,
        y_center_m=1800.0,
        l_beach_m=3600.0,
        alpha_rot=12.0,
        initial_x=0.0,
    )

    # At final timestep:
    # 1. Central transect experiences pure cross-shore disequilibrium erosion (x_center < 0)
    assert x_center[-1] < 0.0
    # 2. North transect is displaced positive (accretion / widening) relative to center
    assert x_north[-1] > x_center[-1]
    # 3. South transect is displaced negative (severe narrowing / scour) relative to center
    assert x_south[-1] < x_center[-1]
    # 4. Net rotation gradient dipole: x_north > x_center > x_south
    assert x_north[-1] > x_south[-1]


def test_coastal_site_profiles_embayment_parameters():
    """Verify that all global site profiles have complete alongshore and embayment specifications."""
    required_keys = ["is_embayment", "l_beach_m", "y_center_m", "alpha_rot", "default_transects", "transect_coords_m"]
    for site_key, profile in COASTAL_SITE_PROFILES.items():
        for k in required_keys:
            assert k in profile, f"Site '{site_key}' missing required alongshore/embayment key '{k}'"
        assert profile["l_beach_m"] > 0.0
        assert profile["y_center_m"] >= 0.0
        assert profile["alpha_rot"] >= 0.0
        assert len(profile["default_transects"]) >= 2
        assert len(profile["transect_coords_m"]) >= 2


def test_unified_index_alongshore_embayment_integration():
    """Verify compute_unified_coastal_vulnerability_index calculation of alongshore rotation gradient."""
    hs = np.array([3.5, 4.0, 4.5])
    tp = np.array([12.0, 13.0, 14.0])
    surge = np.array([0.5, 0.6, 0.7])
    wind = np.array([14.0, 16.0, 18.0])
    wind_dir = np.full(3, 130.0)
    wave_dir = np.full(3, 130.0)  # +45 deg oblique to Narrabeen 85 deg shore-normal -> Phi_ls = 1.0

    # North Transect (y = 3400m)
    df_north = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind,
        wind_dir_deg=wind_dir,
        wave_dir_deg=wave_dir,
        site="narrabeen_aus",
        y_alongshore_m=3400.0,
    )

    # South Transect (y = 400m)
    df_south = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind,
        wind_dir_deg=wind_dir,
        wave_dir_deg=wave_dir,
        site="narrabeen_aus",
        y_alongshore_m=400.0,
    )

    assert "phi_rot_gradient" in df_north.columns
    assert "delta_x_rot_proxy" in df_north.columns
    assert "y_alongshore_m" in df_north.columns

    # North has positive gradient and positive rotation proxy (sand trapped at north headland)
    assert df_north["phi_rot_gradient"].iloc[0] > 0.0
    assert df_north["delta_x_rot_proxy"].iloc[0] > 0.0

    # South has negative gradient and negative rotation proxy (sand exported northward)
    assert df_south["phi_rot_gradient"].iloc[0] < 0.0
    assert df_south["delta_x_rot_proxy"].iloc[0] < 0.0


def test_simulate_shorefor_evolution():
    from src.data import simulate_shorefor_evolution
    hs = np.array([1.0, 2.5, 4.0, 1.5, 1.0])
    tp = np.array([8.0, 10.0, 14.0, 9.0, 8.0])
    x_pos, x_obs = simulate_shorefor_evolution(hs, tp, w_s_ms=0.045, random_seed=42)
    assert len(x_pos) == len(hs)
    assert len(x_obs) == len(hs)
    assert np.all(np.isfinite(x_pos))
    assert np.all(np.isfinite(x_obs))


def test_evaluate_cross_paradigm_walk_forward_cv():
    from src.models import evaluate_cross_paradigm_walk_forward_cv
    df_synthesis, df_folds = evaluate_cross_paradigm_walk_forward_cv(
        sites=["duck_nc"],
        n_splits=2,
        draws=20,
        tune=20,
        chains=2,
        random_seed=42,
    )
    assert not df_synthesis.empty
    assert not df_folds.empty
    assert "CV R2 Score" in df_synthesis.columns
    assert "Fold R2 (std)" in df_synthesis.columns
    assert "CV BSS" in df_synthesis.columns
    assert "Full RMSE (m)" in df_synthesis.columns
    assert "Full R2 Score" in df_synthesis.columns
    models = df_synthesis["Model Name"].tolist()
    assert "Bayesian Latent Memory Kernel (Upsilon)" in models
    assert "Signed Equilibrium Upsilon" in models
    assert "Yates (2009) Equilibrium" in models
    assert "ShoreFor (2013) Memory" in models
    assert "XBeach 1D Process Model" in models


def test_heteroscedastic_uncertainty_approach_a():
    """Verify Approach A: Bayesian noise parameter sigma scales dynamically with storm intensity (rectified & exponential)."""
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:60]
    
    # 1. Test model graph construction with heteroscedastic=True / 'rectified'
    model_hetero, meta_hetero = build_latent_memory_bayesian_model(
        forcing=df["upsilon"].values,
        delta_x_obs=df["shoreline_change_rate_mh"].values,
        max_lag=12,
        heteroscedastic="rectified",
    )
    assert meta_hetero["heteroscedastic_mode"] == "rectified"
    assert "sigma_base" in model_hetero.named_vars
    assert "sigma_storm" in model_hetero.named_vars
    assert "sigma" in model_hetero.named_vars

    # 2. Test model graph construction with heteroscedastic='exponential' (log-link)
    model_exp, meta_exp = build_latent_memory_bayesian_model(
        forcing=df["upsilon"].values,
        delta_x_obs=df["shoreline_change_rate_mh"].values,
        max_lag=12,
        heteroscedastic="exponential",
    )
    assert meta_exp["heteroscedastic_mode"] == "exponential"
    assert "sigma_base" in model_exp.named_vars
    assert "beta_sigma" in model_exp.named_vars
    assert "sigma" in model_exp.named_vars

    # 3. Test model graph construction with heteroscedastic=False (homoscedastic baseline)
    model_homo, meta_homo = build_latent_memory_bayesian_model(
        forcing=df["upsilon"].values,
        delta_x_obs=df["shoreline_change_rate_mh"].values,
        max_lag=12,
        heteroscedastic=False,
    )
    assert meta_homo["heteroscedastic_mode"] == "homoscedastic"
    assert "sigma_base" not in model_homo.named_vars
    assert "sigma" in model_homo.named_vars

    # 4. Fit exponential model and verify sampling & posterior predictive
    trace_exp = fit_bayesian_model(model_exp, draws=50, tune=50, chains=2, random_seed=42)
    assert "sigma_base" in trace_exp.posterior
    assert "beta_sigma" in trace_exp.posterior

    mu_mean, mu_samples, y_pred = predict_bayesian_model(
        trace=trace_exp,
        model_type="single",
        meta=meta_exp,
        df=df,
        return_samples=True,
        random_seed=42,
    )
    assert len(mu_mean) == len(df)
    assert mu_samples.shape == (100, len(df))
    assert y_pred.shape == (100, len(df))

    # Summary table must include beta_sigma
    summary_exp = compute_mcmc_summary_table(trace_exp)
    assert "sigma_base" in summary_exp.index
    assert "beta_sigma" in summary_exp.index


def test_run_heteroscedastic_noise_ablation():
    """Verify systematic likelihood noise model ablation (homoscedastic vs. rectified vs. exponential) via PSIS-LOO-CV."""
    df = generate_frf_duck_hurricane_ian_dataset().iloc[:60]
    df_comp, traces = run_heteroscedastic_noise_ablation(
        df=df,
        target_col="shoreline_change_rate_mh",
        forcing_col="upsilon",
        max_lag=12,
        draws=50,
        tune=50,
        chains=2,
        random_seed=42,
    )
    assert len(traces) == 3
    assert "Homoscedastic" in traces
    assert "Rectified Linear (ReLU)" in traces
    assert "Log-Link Exponential (EGARCH)" in traces
    assert not df_comp.empty
    assert "elpd_loo" in df_comp.columns
    assert "rank" in df_comp.columns


def test_run_bayesian_memory_window_ablation():
    """Verify systematic Bayesian memory window (max_lag) ablation across candidate horizons."""
    df, _ = load_dataset(try_live_api=False)
    df_sub = df.iloc[:100].copy()
    lags = (12, 24, 48)
    df_comp, traces = run_bayesian_memory_window_ablation(
        df=df_sub,
        lags=lags,
        target_col="shoreline_pos_obs_m",
        draws=50,
        tune=50,
        chains=2,
        random_seed=42,
    )
    assert len(traces) == len(lags)
    for lag in lags:
        assert f"lag_{lag}h" in traces
        assert "gamma" in traces[f"lag_{lag}h"].posterior
    assert len(df_comp) == len(lags)
    assert "elpd_loo" in df_comp.columns or "elpd_waic" in df_comp.columns or "rank" in df_comp.columns


def test_latent_bayesian_model_custom_gamma_prior():
    """Verify custom Beta prior parameters on latent memory gamma."""
    forcing = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_obs = np.array([-0.1, -0.5, -0.8, -1.2, -1.5])
    model, meta = build_latent_memory_bayesian_model(
        forcing=forcing,
        delta_x_obs=y_obs,
        gamma_prior=(2.0, 5.0),
    )
    assert meta["gamma_prior"] == (2.0, 5.0)
    assert "gamma" in [v.name for v in model.free_RVs]


def test_run_bayesian_memory_prior_sensitivity_analysis():
    """Verify systematic prior-sensitivity check over Beta(2, 5), Beta(2, 2), and Beta(5, 20)."""
    df, _ = load_dataset(try_live_api=False)
    df_sub = df.iloc[:100].copy()
    candidate_priors = [
        ("Beta(2, 5)", (2.0, 5.0)),
        ("Beta(2, 2)", (2.0, 2.0)),
        ("Beta(5, 20)", (5.0, 20.0)),
    ]
    df_comp, traces, df_gamma_summary = run_bayesian_memory_prior_sensitivity_analysis(
        df=df_sub,
        priors=candidate_priors,
        target_col="shoreline_pos_obs_m",
        max_lag=24,
        draws=50,
        tune=50,
        chains=2,
        random_seed=42,
    )
    assert len(traces) == len(candidate_priors)
    for prior_name, _ in candidate_priors:
        clean_suffix = prior_name.replace(" ", "").replace("(", "_").replace(")", "").replace(",", "_")
        m_name = f"prior_{clean_suffix}"
        assert m_name in traces
        assert "gamma" in traces[m_name].posterior

    assert len(df_gamma_summary) == len(candidate_priors)
    assert "post_mean" in df_gamma_summary.columns
    assert "post_median" in df_gamma_summary.columns
    assert "hdi_3%" in df_gamma_summary.columns
    assert "hdi_97%" in df_gamma_summary.columns
    assert "half_life_hours" in df_gamma_summary.columns

    for _, row in df_gamma_summary.iterrows():
        assert 0.0 < row["post_mean"] < 1.0
        assert 0.0 < row["hdi_3%"] <= row["hdi_97%"] < 1.0

    assert len(df_comp) == len(candidate_priors)
    assert "elpd_loo" in df_comp.columns or "elpd_waic" in df_comp.columns or "rank" in df_comp.columns


def test_evaluate_model_performance_baselines():
    """Verify evaluate_model_performance supports zero-change, persistence, mean, and custom baselines."""
    # Synthetic shoreline position data with non-zero mean
    y_true = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])
    y_pred = np.array([10.5, 11.8, 11.2, 14.7, 14.2, 15.8])

    # 1. Default (zero baseline)
    perf_default = evaluate_model_performance(y_true, y_pred)
    assert "RMSE (m)" in perf_default
    assert "MAE (m)" in perf_default
    assert "R2 Score" in perf_default
    assert "Brier Skill Score (BSS)" in perf_default
    assert "BSS (Zero-Change)" in perf_default
    assert "BSS (Persistence)" in perf_default

    # BSS default is against zero baseline
    assert np.isclose(perf_default["Brier Skill Score (BSS)"], perf_default["BSS (Zero-Change)"])

    # Verify persistence baseline calculation
    y_b_persist = np.array([10.0, 10.0, 12.0, 11.0, 15.0, 14.0])
    mse_persist_expected = float(np.mean((y_true - y_b_persist) ** 2))
    mse_model = float(np.mean((y_true - y_pred) ** 2))
    bss_persist_expected = float(1.0 - (mse_model / mse_persist_expected))
    assert np.isclose(perf_default["MSE Persistence"], mse_persist_expected)
    assert np.isclose(perf_default["BSS (Persistence)"], bss_persist_expected)

    # 2. Explicit baseline_mode="persistence" or y_baseline="persistence"
    perf_persist_mode = evaluate_model_performance(y_true, y_pred, baseline_mode="persistence")
    assert np.isclose(perf_persist_mode["Brier Skill Score (BSS)"], bss_persist_expected)
    assert np.isclose(perf_persist_mode["MSE Baseline"], mse_persist_expected)

    perf_persist_str = evaluate_model_performance(y_true, y_pred, y_baseline="persistence")
    assert np.isclose(perf_persist_str["Brier Skill Score (BSS)"], bss_persist_expected)

    # 3. Explicit custom baseline array
    y_custom_base = y_true - 1.0
    mse_custom_expected = 1.0
    bss_custom_expected = float(1.0 - (mse_model / mse_custom_expected))
    perf_custom = evaluate_model_performance(y_true, y_pred, y_baseline=y_custom_base)
    assert np.isclose(perf_custom["Brier Skill Score (BSS)"], bss_custom_expected)
    assert np.isclose(perf_custom["MSE Baseline"], mse_custom_expected)
    assert "BSS (Zero-Change)" in perf_custom
    assert "BSS (Persistence)" in perf_custom


def test_dimensional_analysis_and_units():
    """Verify dimensional prefactor C_0, wave-sediment core, and Upsilon kinetic units [kg/s]."""
    rho = 1025.0       # kg/m^3
    g = 9.81           # m/s^2
    tau = 2.0 * np.pi

    # Prefactor C_0 = rho * g^1.5 / (16 * (2*pi)^1.5) [kg / (m^1.5 * s^3)]
    c0 = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    assert 120.0 < c0 < 140.0  # ~130.5 kg / (m^1.5 * s^3)

    # Core at n=2.5: Hs^2.5 * Tp / ws [m^1.5 * s^2]
    hs = 2.0   # m
    tp = 10.0  # s
    ws = 0.05  # m/s
    core = (hs ** 2.5) * tp / ws
    assert core > 0.0

    # Total product Upsilon = C_0 * tan_beta * core [kg/s]
    tan_beta = 0.08
    ups_manual = c0 * tan_beta * core
    assert ups_manual > 0.0

    # Reference flux calculation must match manual formula
    u_ref = compute_site_reference_flux(
        tan_beta=tan_beta,
        hs_mean=hs,
        tp_mean=tp,
        ws=ws,
        hs_exponent=2.5,
        rho=rho,
        g=g,
        tau=tau,
    )
    assert np.isclose(u_ref, ups_manual)


def test_site_reference_flux_and_dimensionless_upsilon():
    """Verify compute_site_reference_flux and compute_dimensionless_upsilon with scalars and arrays."""
    # 1. Scalar calculation
    u_ref = compute_site_reference_flux(tan_beta=0.08, hs_mean=1.5, tp_mean=8.0, ws=0.04)
    assert isinstance(u_ref, (float, np.floating))
    assert u_ref > 0.0

    u_raw = 2.0 * u_ref
    u_star = compute_dimensionless_upsilon(u_raw, u_ref)
    assert np.isclose(u_star, 2.0)

    # 2. Array calculation
    u_raw_arr = np.array([0.5 * u_ref, 1.0 * u_ref, 3.0 * u_ref])
    u_star_arr = compute_dimensionless_upsilon(u_raw_arr, u_ref)
    np.testing.assert_allclose(u_star_arr, [0.5, 1.0, 3.0])


def test_unified_vulnerability_index_dimensionless_outputs():
    """Verify compute_unified_coastal_vulnerability_index outputs upsilon_ref and upsilon_star."""
    hs = np.array([1.0, 2.0, 3.5])
    tp = np.array([8.0, 10.0, 12.0])
    surge = np.array([0.1, 0.3, 0.8])
    wind_spd = np.array([5.0, 12.0, 20.0])
    wind_dir = np.array([70.0, 70.0, 70.0])

    df = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind_spd,
        wind_dir_deg=wind_dir,
        site="duck_nc",
    )

    assert "upsilon_ref" in df.columns
    assert "upsilon_star" in df.columns
    assert "upsilon_star_excess" in df.columns
    assert "upsilon_star_crit" in df.columns

    # Reference flux must be positive
    assert np.all(df["upsilon_ref"] > 0.0)

    # Dimensionless Upsilon* must equal upsilon / upsilon_ref
    expected_u_star = df["upsilon"].values / df["upsilon_ref"].values
    np.testing.assert_allclose(df["upsilon_star"].values, expected_u_star, rtol=1e-5)

    # Storm condition (Hs=3.5m) should yield higher Upsilon* than calm condition (Hs=1.0m)
    assert df["upsilon_star"].iloc[2] > df["upsilon_star"].iloc[0]


def test_cross_archetype_dimensionless_calibration_invariance():
    """Verify that normalizing by site reference flux yields dimensionless baseline Upsilon* ~ O(1) across all 5 archetypes."""
    sites = ["duck_nc", "narrabeen_aus", "oahu_north_shore_hi", "torrey_pines_ca", "truc_vert_france"]

    for site_key in sites:
        prof = COASTAL_SITE_PROFILES[site_key]
        # Simulate typical site conditions
        hs = np.array([1.5, 2.0, 2.5])
        tp = np.array([9.0, 10.0, 11.0])
        surge = np.array([0.2, 0.3, 0.4])
        wind_spd = np.array([8.0, 10.0, 12.0])
        wind_dir = np.full_like(hs, prof["shore_normal_deg"])

        df = compute_unified_coastal_vulnerability_index(
            hs=hs,
            tp=tp,
            surge=surge,
            wind_speed=wind_spd,
            wind_dir_deg=wind_dir,
            site=site_key,
        )

        mean_u_star = float(df["upsilon_star"].mean())
        # Normalized mean relative forcing across typical conditions must be O(1), e.g. within [0.2, 5.0]
        assert 0.2 < mean_u_star < 5.0, f"Site {site_key} Upsilon* out of bounds: {mean_u_star}"
        assert np.all(df["upsilon_star_excess"] >= 0.0)


def test_wave_steepness_formulation():
    """Verify wave steepness calculation s0 = Hs / L0 and bounded continuous modulator Psi_steep."""
    # Calm swell: Hs = 1.0m, Tp = 14.0s -> very low steepness (< 0.01)
    s0_swell, psi_swell = compute_wave_steepness(1.0, 14.0)
    assert s0_swell < 0.01
    assert 0.2 <= psi_swell < 1.0

    # Steep storm seas: Hs = 4.5m, Tp = 8.0s -> high steepness (> 0.04)
    s0_storm, psi_storm = compute_wave_steepness(4.5, 8.0)
    assert s0_storm > 0.04
    assert psi_storm > 1.0

    # Vectorized check
    hs = np.array([1.0, 2.0, 4.5])
    tp = np.array([14.0, 10.0, 8.0])
    s0_arr, psi_arr = compute_wave_steepness(hs, tp)
    assert len(s0_arr) == 3
    assert np.all(np.diff(s0_arr) > 0)  # Monotonically increasing steepness


def test_radiation_stress_tensor_formulation():
    """Verify Longuet-Higgins & Stewart (1964) deepwater radiation stress S_xx, S_xy, and Psi_Sxx."""
    # Shore-normal wave approach (theta_wave == shore_normal) -> theta_rel = 0
    # S_xx = 0.5 * E, S_xy = 0, Psi_Sxx = 0.5
    s_xx, s_xy, psi_sxx, e_energy = compute_radiation_stress_tensor(
        hs=2.0, tp=10.0, wave_dir_deg=70.0, shore_normal_deg=70.0
    )
    assert np.isclose(psi_sxx, 0.5)
    assert np.isclose(s_xy, 0.0)
    assert np.isclose(s_xx, 0.5 * e_energy)
    assert e_energy > 0.0

    # Highly oblique approach (theta_wave - shore_normal = 45 deg)
    # cos^2(45) = 0.5 -> S_xx = 0.25 * E, sin(90) = 1.0 -> S_xy = 0.25 * E
    s_xx_ob, s_xy_ob, psi_sxx_ob, e_ob = compute_radiation_stress_tensor(
        hs=2.0, tp=10.0, wave_dir_deg=115.0, shore_normal_deg=70.0
    )
    assert np.isclose(psi_sxx_ob, 0.25)
    assert np.isclose(s_xy_ob, 0.25 * e_ob)
    assert s_xx_ob < s_xx  # Oblique waves transmit less shore-normal momentum flux


def test_sallenger_impact_ratio_formulation():
    """Verify Sallenger (2000) storm impact scale classification and continuous ratio Psi_impact."""
    # Swash regime: TWL = 1.8m < berm crest 2.5m
    psi_swash, regime_swash = compute_sallenger_impact_ratio(twl=1.8, berm_crest_m=2.5, hs=2.0)
    assert np.isclose(psi_swash, 1.0)
    assert regime_swash == "swash"

    # Collision regime: TWL = 2.9m > berm crest 2.5m
    psi_coll, regime_coll = compute_sallenger_impact_ratio(twl=2.9, berm_crest_m=2.5, hs=2.0)
    assert psi_coll > 1.0
    assert regime_coll == "collision"

    # Overwash regime: TWL = 4.0m >> berm crest 2.5m
    psi_over, regime_over = compute_sallenger_impact_ratio(twl=4.0, berm_crest_m=2.5, hs=2.0)
    assert psi_over > psi_coll
    assert regime_over == "overwash"


def test_multiscale_memory_decay_formulation():
    """Verify multi-scale antecedent exponential memory decay convolutions for tau in {7, 30, 90, 365} days."""
    # Constant series: decay should equal the constant value, disequilibrium should be 0.0
    n = 200
    const_ups = np.full(n, 2500.0)
    out_const = compute_multiscale_memory_decay(const_ups)
    for tau in [7, 30, 90, 365]:
        np.testing.assert_allclose(out_const[f"upsilon_decay_{tau}d"], 2500.0, rtol=1e-3)
        np.testing.assert_allclose(out_const[f"upsilon_diseq_{tau}d"], 0.0, atol=1e-3)

    # Step function / storm pulse: baseline 1000, sudden jump to 5000 at index 100
    step_ups = np.full(n, 1000.0)
    step_ups[100:] = 5000.0
    out_step = compute_multiscale_memory_decay(step_ups)
    # 7-day memory should adapt faster to the new high level than 365-day memory
    assert out_step["upsilon_decay_7d"][-1] > out_step["upsilon_decay_365d"][-1]
    # Immediately after the step (index 100), disequilibrium is positive
    assert out_step["upsilon_diseq_7d"][100] > 0.0


def test_compute_unified_coastal_vulnerability_index_new_physics_columns():
    """Verify that compute_unified_coastal_vulnerability_index includes all 4 new physical terms."""
    hs = np.array([1.5, 2.5, 3.5])
    tp = np.array([8.0, 10.0, 12.0])
    surge = np.array([0.1, 0.4, 0.9])
    wind_spd = np.array([6.0, 12.0, 18.0])
    wind_dir = np.array([70.0, 70.0, 70.0])

    df = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind_spd,
        wind_dir_deg=wind_dir,
        site="duck_nc",
    )

    expected_cols = [
        "wave_steepness_s0", "psi_steep",
        "s_xx_Wm", "s_xy_Wm", "psi_sxx", "e_energy_Jm2",
        "psi_impact", "impact_regime",
        "upsilon_full",
        "upsilon_decay_7d", "upsilon_decay_30d", "upsilon_decay_90d", "upsilon_decay_365d",
        "upsilon_diseq_7d", "upsilon_diseq_30d", "upsilon_diseq_90d", "upsilon_diseq_365d",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected physics column: {col}"





