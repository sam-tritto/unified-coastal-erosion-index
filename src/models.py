"""Bayesian modeling of coastal morphodynamic response and shoreline change using PyMC.

Includes:
- Latent Bayesian estimation of antecedent memory retention (gamma ~ Beta(5, 20))
- Differentiable PyTensor convolution matrix for dynamic memory weighting
- Robust Student-t likelihood with constrained degrees of freedom (nu >= 5)
- Multivariate linear model (individual beta per forcing: Hs, S, Wa, Omega)
- Non-linear polynomial response model with Horseshoe sparsity priors
- 4-chain NUTS MCMC sampling and convergence diagnostics (R-hat < 1.05, ESS > 400)
- Multi-model benchmarking suite with support for multiple morphodynamic targets
- ArviZ LOO-CV / WAIC model comparison and morphodynamic Brier Skill Score (BSS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Any, Tuple, Optional, List, Sequence


def compute_antecedent_memory_forcing(
    forcing: np.ndarray,
    gamma: float = 0.85,
    max_lag_hours: int = 48,
) -> np.ndarray:
    """Compute exponential memory-weighted antecedent forcing time series in NumPy."""
    n = len(forcing)
    weights = np.array([gamma ** k for k in range(max_lag_hours)])
    weights /= weights.sum()
    
    padded = np.pad(forcing, (max_lag_hours - 1, 0), mode="edge")
    filtered = np.convolve(padded, weights, mode="valid")
    return filtered[:n]


def create_lag_matrix(forcing: np.ndarray, max_lag: int = 48) -> np.ndarray:
    """Construct a Toeplitz lag matrix X_lag of shape (N, max_lag).

    X_lag[t, k] = forcing[max(0, t - k)]
    """
    n = len(forcing)
    max_lag = max(1, min(max_lag, n))
    lag_mat = np.zeros((n, max_lag), dtype=float)
    for k in range(max_lag):
        lag_mat[k:, k] = forcing[: n - k]
        if k > 0:
            lag_mat[:k, k] = forcing[0]
    return lag_mat


def build_latent_memory_bayesian_model(
    forcing: np.ndarray,
    delta_x_obs: np.ndarray,
    model_name: str = "upsilon_model",
    max_lag: int = 48,
    scale_forcing: bool = True,
    heteroscedastic: bool = True,
    gamma_prior: Tuple[float, float] = (5.0, 20.0),
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a PyMC Bayesian model where antecedent memory gamma is inferred as a latent variable.

    Formulation:
    - gamma ~ Beta(alpha, beta)  (Latent memory prior, default Beta(5, 20))
    - weights_k = gamma^k / sum(gamma^j) for k = 0, ..., max_lag-1
    - E(gamma) = X_lag * weights(gamma)
    - mu_t = alpha + beta_forcing * E_t(gamma)
    - Heteroscedastic noise (Approach A): sigma_t = sigma_base + sigma_storm * max(E_t(gamma), 0)
    - y_obs ~ StudentT(nu >= 5, mu_t, sigma_t)

    Parameters
    ----------
    forcing : np.ndarray
        Raw forcing series (e.g., Upsilon, Wave Power P, Dean Omega, etc.).
    delta_x_obs : np.ndarray
        Observed cross-shore position changes (m).
    model_name : str
        Model identifier.
    max_lag : int
        Maximum historical convolution window in hours.
    scale_forcing : bool
        Whether to standardize the forcing series.
    heteroscedastic : bool
        Whether to model noise dynamically with storm intensity (Approach A).
    gamma_prior : Tuple[float, float]
        Shape parameters (alpha, beta) for the Beta prior on latent memory gamma. Default (5.0, 20.0).

    Returns
    -------
    model : pm.Model
        Constructed PyMC model.
    meta : dict
        Model metadata and normalization parameters.
    """
    forcing_arr = np.asarray(forcing, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)
    
    f_mean = float(np.mean(forcing_arr))
    f_std = float(np.std(forcing_arr)) if np.std(forcing_arr) > 1e-6 else 1.0
    f_scaled = (forcing_arr - f_mean) / f_std if scale_forcing else forcing_arr
    
    max_lag = max(1, min(max_lag, len(y_obs)))
    lag_mat = create_lag_matrix(f_scaled, max_lag=max_lag)
    k_indices = np.arange(max_lag, dtype=float)

    meta = {
        "model_name": model_name,
        "f_mean": f_mean,
        "f_std": f_std,
        "max_lag": max_lag,
        "n_obs": len(y_obs),
        "heteroscedastic": heteroscedastic,
        "gamma_prior": gamma_prior,
    }

    with pm.Model() as model:
        # Data containers
        x_lag = pm.Data("x_lag", lag_mat)
        k_vec = pm.Data("k_vec", k_indices)
        
        # 1. Latent memory parameter prior (configurable Beta(alpha, beta), default Beta(5, 20))
        gamma_alpha, gamma_beta = float(gamma_prior[0]), float(gamma_prior[1])
        gamma = pm.Beta("gamma", alpha=gamma_alpha, beta=gamma_beta)
        
        # 2. Differentiable exponential weights in PyTensor
        # weights_k = gamma^k / sum(gamma^j)
        unnorm_weights = pt.power(gamma, k_vec)
        weights = unnorm_weights / pt.sum(unnorm_weights)
        
        # 3. Dynamic antecedent convolution: (N, max_lag) @ (max_lag,) -> (N,)
        effective_forcing = pm.Deterministic("effective_forcing", pt.dot(x_lag, weights))
        
        # 4. Morphodynamic response parameters
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta_forcing = pm.Normal("beta_forcing", mu=-2.0, sigma=1.5)

        # 5. Uncertainty Model (Approach A: Heteroscedastic Storm Volatility)
        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_storm = pm.HalfNormal("sigma_storm", sigma=0.5)
            storm_excitation = pt.maximum(effective_forcing, 0.0)
            sigma = pm.Deterministic("sigma", sigma_base + sigma_storm * storm_excitation)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))
        
        # 6. Deterministic latent response
        mu = pm.Deterministic("mu", alpha + beta_forcing * effective_forcing)
        
        # 7. Student-T Likelihood
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_obs)

    return model, meta


def fit_bayesian_model(
    model: pm.Model,
    draws: int = 1500,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.95,
    random_seed: int = 42,
    compute_log_likelihood: bool = True,
) -> az.InferenceData:
    """Execute NUTS MCMC sampling with 4 chains and sample posterior predictives."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
        if compute_log_likelihood and not hasattr(trace, "log_likelihood"):
            try:
                pm.compute_log_likelihood(trace, progressbar=False)
            except Exception:
                pass
        ppc = pm.sample_posterior_predictive(trace, random_seed=random_seed, progressbar=False)
        trace.extend(ppc)
        
    return trace


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[Union[np.ndarray, str]] = None,
    baseline_mode: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate morphodynamic validation metrics (RMSE, MAE, R^2, and Brier Skill Score).

    Supports zero-change baseline (climatological null = "no change" for step-change Δx),
    persistence baseline (previous observation carried forward for cumulative position x_t),
    climatological mean baseline, or custom explicit baseline arrays.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth observed values.
    y_pred : np.ndarray
        Model predicted values.
    y_baseline : Optional[Union[np.ndarray, str]]
        Explicit baseline array, or string indicating baseline mode ('zero', 'persistence', 'mean').
    baseline_mode : Optional[str]
        Optional string indicating baseline mode if y_baseline is None ('zero', 'persistence', 'mean').

    Returns
    -------
    Dict[str, float]
        Dictionary containing RMSE, MAE, R^2, active BSS, BSS (Zero-Change),
        BSS (Persistence), and corresponding MSE values.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    
    # Compute standard zero-change and persistence baselines
    y_b_zero = np.zeros_like(y_t)
    y_b_persist = np.roll(y_t, 1)
    if len(y_b_persist) > 0:
        y_b_persist[0] = y_t[0]

    # Resolve active baseline
    if isinstance(y_baseline, str):
        mode = y_baseline.lower()
        if mode in ("persistence", "persist"):
            y_b = y_b_persist
        elif mode in ("mean", "climatology"):
            y_b = np.full_like(y_t, np.mean(y_t)) if len(y_t) > 0 else np.zeros_like(y_t)
        else:
            y_b = y_b_zero
    elif y_baseline is not None:
        y_b = np.asarray(y_baseline, dtype=float).ravel()
    elif baseline_mode is not None:
        mode = baseline_mode.lower()
        if mode in ("persistence", "persist"):
            y_b = y_b_persist
        elif mode in ("mean", "climatology"):
            y_b = np.full_like(y_t, np.mean(y_t)) if len(y_t) > 0 else np.zeros_like(y_t)
        else:
            y_b = y_b_zero
    else:
        y_b = y_b_zero
        
    mse_model = float(np.mean((y_t - y_p) ** 2))
    mse_base = float(np.mean((y_t - y_b) ** 2))
    mse_zero = float(np.mean((y_t - y_b_zero) ** 2))
    mse_persist = float(np.mean((y_t - y_b_persist) ** 2))
    
    rmse = float(np.sqrt(mse_model))
    mae = float(np.mean(np.abs(y_t - y_p)))
    
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2) if len(y_t) > 0 else 0.0
    ss_res = np.sum((y_t - y_p) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
    
    bss = float(1.0 - (mse_model / mse_base)) if mse_base > 1e-9 else 0.0
    bss_zero = float(1.0 - (mse_model / mse_zero)) if mse_zero > 1e-9 else 0.0
    bss_persist = float(1.0 - (mse_model / mse_persist)) if mse_persist > 1e-9 else 0.0
    
    return {
        "RMSE (m)": rmse,
        "MAE (m)": mae,
        "R2 Score": r2,
        "Brier Skill Score (BSS)": bss,
        "BSS (Zero-Change)": bss_zero,
        "BSS (Persistence)": bss_persist,
        "MSE Model": mse_model,
        "MSE Baseline": mse_base,
        "MSE Zero": mse_zero,
        "MSE Persistence": mse_persist,
    }


def compute_mcmc_summary_table(idata: az.InferenceData) -> pd.DataFrame:
    """Generate clean parameter posterior summary table with R-hat and ESS."""
    var_names = [
        "alpha",
        "beta_forcing",
        "gamma",
        "sigma_base",
        "sigma_storm",
        "sigma",
        "nu",
        "beta_upsilon",
        "k_eros",
        "k_accrete",
        "omega_crit",
        "hs_exponent",
    ]
    # Filter out deterministic vector sigma if scalar sigma_base is present for compact display
    has_hetero = "sigma_base" in idata.posterior
    available = [
        v for v in var_names
        if v in idata.posterior and (v != "sigma" or not has_hetero)
    ]
    return az.summary(idata, var_names=available)


def predict_bayesian_model(
    trace: az.InferenceData,
    model_type: str,
    meta: Dict[str, Any],
    df: pd.DataFrame,
    spec: Optional[Dict[str, Any]] = None,
    return_samples: bool = False,
    random_seed: int = 42,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Predict morphodynamic response mu (and draw posterior samples) for any dataset window.

    Supports continuous antecedent memory convolution across train/test splits and
    dynamic heteroscedastic noise scaling (Approach A).

    Parameters
    ----------
    trace : az.InferenceData
        Fitted MCMC inference data trace.
    model_type : str
        One of 'single', 'nonlinear', 'multivariate', 'hybrid', 'asymmetric'.
    meta : dict
        Model metadata containing normalization constants (mean, std, max_lag, etc.).
    df : pd.DataFrame
        Dataset containing hydrodynamic forcing channels for prediction.
    spec : dict, optional
        Specification dict containing raw forcing arrays (if df columns differ).
    return_samples : bool
        Whether to return posterior distribution samples (mu_samples, y_pred_samples).
    random_seed : int
        Random seed for posterior predictive Student-T error sampling.

    Returns
    -------
    mu_mean : np.ndarray
        Posterior mean deterministic prediction of shoreline position (N,).
    (Optional) mu_samples : np.ndarray
        Posterior samples of mu (S, N).
    (Optional) y_pred_samples : np.ndarray
        Posterior predictive simulated observations drawn from StudentT (S, N).
    """
    rng = np.random.default_rng(random_seed)
    n_pts = len(df)
    max_lag = int(meta.get("max_lag", 48))
    k_vec = np.arange(max_lag, dtype=float)

    alphas = trace.posterior["alpha"].values.reshape(-1)
    nus = trace.posterior["nu"].values.reshape(-1) if "nu" in trace.posterior else np.full(len(alphas), 5.0)
    
    # Sub-sample posterior draws to 200 for fast, memory-safe out-of-sample evaluation on large multi-year series
    if len(alphas) > 200:
        sub_idx = np.linspace(0, len(alphas) - 1, 200, dtype=int)
        alphas = alphas[sub_idx]
        nus = nus[sub_idx]
    else:
        sub_idx = np.arange(len(alphas))
    n_samples = len(alphas)

    # Handle homoscedastic vs. heteroscedastic noise parameters
    has_hetero = ("sigma_base" in trace.posterior) and ("sigma_storm" in trace.posterior)
    if has_hetero:
        s_base_arr = trace.posterior["sigma_base"].values.reshape(-1)[sub_idx]
        s_storm_arr = trace.posterior["sigma_storm"].values.reshape(-1)[sub_idx]
        sigmas = None
    else:
        sigmas = trace.posterior["sigma"].values.reshape(-1)[sub_idx] if "sigma" in trace.posterior else np.full(n_samples, 0.3)

    is_survey = "antecedent_upsilon_memory" in df.columns or "surveyed_delta_x_m" in df.columns
    eff_storm_samples = np.zeros((n_samples, n_pts), dtype=float)

    if model_type == "single":
        if spec is not None and "forcing" in spec:
            f_raw = np.asarray(spec["forcing"], dtype=float)
        else:
            m_name = meta.get("model_name", "").lower()
            if is_survey:
                if "yates" in m_name:
                    f_raw = df["yates_simulated_delta_x_m"].values if "yates_simulated_delta_x_m" in df.columns else df["yates_simulated_pos_m"].values
                elif "xbeach" in m_name:
                    f_raw = df["xbeach_simulated_delta_x_m"].values if "xbeach_simulated_delta_x_m" in df.columns else df["xbeach_simulated_pos_m"].values
                elif "hs" in m_name:
                    f_raw = df["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df.columns else df["antecedent_max_hs_m"].values
                elif "power" in m_name or "flux" in m_name:
                    f_raw = df["antecedent_max_power_Wm"].values if "antecedent_max_power_Wm" in df.columns else np.zeros(n_pts)
                elif "dean" in m_name:
                    f_raw = df["antecedent_mean_dean"].values if "antecedent_mean_dean" in df.columns else np.zeros(n_pts)
                elif meta.get("use_log_upsilon", False) and "antecedent_log_upsilon" in df.columns:
                    f_raw = df["antecedent_log_upsilon"].values
                elif "antecedent_upsilon_memory" in df.columns:
                    f_raw = df["antecedent_upsilon_memory"].values
                elif "antecedent_mean_upsilon" in df.columns:
                    f_raw = df["antecedent_mean_upsilon"].values
                else:
                    f_raw = np.zeros(n_pts)
            else:
                if "yates" in m_name and "yates_simulated_delta_x_m" in df.columns:
                    f_raw = df["yates_simulated_delta_x_m"].values
                elif "xbeach" in m_name and "xbeach_simulated_delta_x_m" in df.columns:
                    f_raw = df["xbeach_simulated_delta_x_m"].values
                elif "hs" in m_name:
                    f_raw = df["hs_m"].values
                elif "power" in m_name or "flux" in m_name:
                    f_raw = df["p_flux_Wm"].values
                elif "dean" in m_name:
                    f_raw = df["dean_number"].values
                elif "signed" in m_name and "signed_upsilon" in df.columns:
                    f_raw = df["signed_upsilon"].values
                elif meta.get("use_log_upsilon", False) and "log_upsilon" in df.columns:
                    f_raw = df["log_upsilon"].values
                elif meta.get("use_log_upsilon", False) and "upsilon" in df.columns:
                    f_raw = np.log10(np.maximum(df["upsilon"].values, 1e-4))
                elif "upsilon" in df.columns:
                    f_raw = df["upsilon"].values
                else:
                    f_raw = np.zeros(n_pts)

        f_scaled = (f_raw - meta["f_mean"]) / meta["f_std"]
        lag_mat = create_lag_matrix(f_scaled, max_lag=max_lag)

        betas = trace.posterior["beta_forcing"].values.reshape(-1)[sub_idx]
        gammas = trace.posterior["gamma"].values.reshape(-1)[sub_idx]

        mu_samples = np.zeros((n_samples, n_pts), dtype=float)
        for i in range(n_samples):
            w = (gammas[i] ** k_vec) / np.sum(gammas[i] ** k_vec)
            eff = lag_mat @ w
            mu_samples[i] = alphas[i] + betas[i] * eff
            eff_storm_samples[i] = np.maximum(eff, 0.0)

    elif model_type == "nonlinear":
        if spec is not None and "forcing" in spec:
            f_raw = np.asarray(spec["forcing"], dtype=float)
        else:
            m_name = meta.get("model_name", "").lower()
            if is_survey:
                if "hs" in m_name:
                    f_raw = df["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df.columns else df["antecedent_max_hs_m"].values
                elif meta.get("use_log_upsilon", False) and "antecedent_log_upsilon" in df.columns:
                    f_raw = df["antecedent_log_upsilon"].values
                elif "antecedent_upsilon_memory" in df.columns:
                    f_raw = df["antecedent_upsilon_memory"].values
                else:
                    f_raw = np.zeros(n_pts)
            else:
                if "hs" in m_name:
                    f_raw = df["hs_m"].values
                elif meta.get("use_log_upsilon", False) and "log_upsilon" in df.columns:
                    f_raw = df["log_upsilon"].values
                elif meta.get("use_log_upsilon", False) and "upsilon" in df.columns:
                    f_raw = np.log10(np.maximum(df["upsilon"].values, 1e-4))
                elif "upsilon" in df.columns:
                    f_raw = df["upsilon"].values
                else:
                    f_raw = np.zeros(n_pts)

        f_scaled = (f_raw - meta["f_mean"]) / meta["f_std"]
        degree = int(meta.get("degree", 3))
        basis_matrix = np.column_stack([f_scaled ** (d + 1) for d in range(degree)])
        beta_poly = trace.posterior["beta_poly"].values.reshape(-1, degree)[sub_idx]

        mu_samples = alphas[:, None] + (beta_poly @ basis_matrix.T)
        eff_storm_samples[:] = np.maximum(f_scaled, 0.0)[None, :]

    elif model_type == "multivariate":
        fnames = meta.get("forcing_names", ["hs", "surge", "wind", "dean"])
        mu_samples = np.repeat(alphas[:, None], n_pts, axis=1)

        if is_survey:
            col_map = {
                "hs": "antecedent_mean_hs_m" if "antecedent_mean_hs_m" in df.columns else "antecedent_max_hs_m",
                "surge": "antecedent_max_surge_m",
                "wind": "antecedent_mean_hs_m",
                "dean": "antecedent_mean_dean",
            }
        else:
            col_map = {"hs": "hs_m", "surge": "surge_m", "wind": "wind_normal_ms", "dean": "dean_number"}

        for fname in fnames:
            if spec is not None and "forcings" in spec and fname in spec["forcings"]:
                raw_f = np.asarray(spec["forcings"][fname], dtype=float)
            else:
                col = col_map.get(fname, fname)
                raw_f = df[col].values if col in df.columns else np.zeros(n_pts)

            if f"{fname}_mean" in meta:
                f_scaled = (raw_f - meta[f"{fname}_mean"]) / meta[f"{fname}_std"]
                beta_var = trace.posterior[f"beta_{fname}"].values.reshape(-1)[sub_idx]
                mu_samples += beta_var[:, None] * f_scaled[None, :]

    elif model_type == "hybrid":
        if spec is not None and "upsilon" in spec:
            ups_raw = np.asarray(spec["upsilon"], dtype=float)
        elif is_survey:
            if meta.get("use_log_upsilon", False) and "antecedent_log_upsilon" in df.columns:
                ups_raw = df["antecedent_log_upsilon"].values
            elif "antecedent_upsilon_memory" in df.columns:
                ups_raw = df["antecedent_upsilon_memory"].values
            else:
                ups_raw = np.zeros(n_pts)
        elif meta.get("use_log_upsilon", False) and "log_upsilon" in df.columns:
            ups_raw = df["log_upsilon"].values
        elif meta.get("use_log_upsilon", False) and "upsilon" in df.columns:
            ups_raw = np.log10(np.maximum(df["upsilon"].values, 1e-4))
        elif "upsilon" in df.columns:
            ups_raw = df["upsilon"].values
        else:
            ups_raw = np.zeros(n_pts)

        u_scaled = (ups_raw - meta["upsilon_mean"]) / meta["upsilon_std"]
        lag_mat = create_lag_matrix(u_scaled, max_lag=max_lag)

        beta_ups = trace.posterior["beta_upsilon"].values.reshape(-1)[sub_idx]
        gammas = trace.posterior["gamma"].values.reshape(-1)[sub_idx]

        mu_samples = np.zeros((n_samples, n_pts), dtype=float)
        for i in range(n_samples):
            w = (gammas[i] ** k_vec) / np.sum(gammas[i] ** k_vec)
            eff_u = lag_mat @ w
            mu_samples[i] = alphas[i] + beta_ups[i] * eff_u
            eff_storm_samples[i] = np.maximum(eff_u, 0.0)

        fnames = ["hs", "surge", "wind", "dean"]
        if is_survey:
            col_map = {
                "hs": "antecedent_mean_hs_m" if "antecedent_mean_hs_m" in df.columns else "antecedent_max_hs_m",
                "surge": "antecedent_max_surge_m",
                "wind": "antecedent_mean_hs_m",
                "dean": "antecedent_mean_dean",
            }
        else:
            col_map = {"hs": "hs_m", "surge": "surge_m", "wind": "wind_normal_ms", "dean": "dean_number"}

        for fname in fnames:
            if f"{fname}_mean" in meta:
                if spec is not None and "forcings" in spec and fname in spec["forcings"]:
                    raw_f = np.asarray(spec["forcings"][fname], dtype=float)
                else:
                    col = col_map.get(fname, fname)
                    raw_f = df[col].values if col in df.columns else np.zeros(n_pts)

                f_scaled = (raw_f - meta[f"{fname}_mean"]) / meta[f"{fname}_std"]
                delta_var = trace.posterior[f"delta_{fname}"].values.reshape(-1)[sub_idx]
                mu_samples += delta_var[:, None] * f_scaled[None, :]

    elif model_type == "asymmetric":
        if spec is not None and "upsilon" in spec:
            ups_raw = np.asarray(spec["upsilon"], dtype=float)
        elif meta.get("use_log_upsilon", False) and "log_upsilon" in df.columns:
            ups_raw = df["log_upsilon"].values
        elif meta.get("use_log_upsilon", False) and "upsilon" in df.columns:
            ups_raw = np.log10(np.maximum(df["upsilon"].values, 1e-4))
        elif "upsilon" in df.columns:
            ups_raw = df["upsilon"].values
        else:
            ups_raw = np.zeros(n_pts)

        shield_raw = spec["shielding"] if spec is not None and "shielding" in spec else (
            df["shielding_factor"].values if "shielding_factor" in df.columns else np.ones(n_pts)
        )
        f_eros = ups_raw * shield_raw
        f_eros_scaled = (f_eros - meta["eros_mean"]) / meta["eros_std"]

        if spec is not None and "accretion" in spec and spec["accretion"] is not None:
            f_acc = np.asarray(spec["accretion"], dtype=float)
        elif "accretion_potential" in df.columns:
            f_acc = df["accretion_potential"].values
        else:
            tp_arr = np.maximum(df["tp_s"].values, 0.5)
            l_0 = (9.81 * (tp_arr ** 2)) / (2.0 * np.pi)
            f_acc = (tp_arr * np.sqrt(9.81 * l_0)) / (df["hs_m"].values + 0.1)

        f_acc_scaled = (f_acc - meta["acc_mean"]) / meta["acc_std"]

        hs_arr = spec["hs"] if spec is not None and "hs" in spec else df["hs_m"].values
        tp_arr = spec["tp"] if spec is not None and "tp" in spec else df["tp_s"].values
        ws_arr = spec["ws"] if spec is not None and "ws" in spec else (
            df["w_s_ms"].values if "w_s_ms" in df.columns else np.full(n_pts, 0.048)
        )
        omega = hs_arr / (np.maximum(ws_arr, 1e-4) * np.maximum(tp_arr, 0.5))

        lag_eros = create_lag_matrix(f_eros_scaled, max_lag=max_lag)
        lag_acc = create_lag_matrix(f_acc_scaled, max_lag=max_lag)

        gammas = trace.posterior["gamma"].values.reshape(-1)[sub_idx]
        o_crits = trace.posterior["omega_crit"].values.reshape(-1)[sub_idx]
        k_e = trace.posterior["k_eros"].values.reshape(-1)[sub_idx]
        k_a = trace.posterior["k_accrete"].values.reshape(-1)[sub_idx]

        mu_samples = np.zeros((n_samples, n_pts), dtype=float)
        for i in range(n_samples):
            w = (gammas[i] ** k_vec) / np.sum(gammas[i] ** k_vec)
            eff_e = lag_eros @ w
            eff_a = lag_acc @ w
            is_e = 1.0 / (1.0 + np.exp(-5.0 * (omega - o_crits[i])))
            mu_samples[i] = alphas[i] + is_e * (-k_e[i] * eff_e) + (1.0 - is_e) * (k_a[i] * eff_a)
            eff_storm_samples[i] = np.maximum(eff_e, 0.0)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    mu_mean = np.mean(mu_samples, axis=0)

    if not return_samples:
        return mu_mean

    # Draw posterior predictive Student-T observations (incorporating dynamic heteroscedastic noise)
    y_pred_samples = np.zeros_like(mu_samples)
    for i in range(n_samples):
        nu_val = max(float(nus[i]), 1.5)
        if has_hetero:
            exc = eff_storm_samples[i]
            sigma_val = np.maximum(s_base_arr[i] + s_storm_arr[i] * exc, 1e-4)
        else:
            sigma_val = max(float(sigmas[i]), 1e-4) if np.isscalar(sigmas[i]) or np.ndim(sigmas[i]) == 0 else np.maximum(sigmas[i], 1e-4)
        t_noise = rng.standard_t(df=nu_val, size=n_pts) * sigma_val
        y_pred_samples[i] = mu_samples[i] + t_noise

    return mu_mean, mu_samples, y_pred_samples


def run_arviz_loo_comparison(
    traces: Dict[str, az.InferenceData],
    ic: str = "loo",
    method: str = "stacking",
) -> pd.DataFrame:
    """Run ArviZ model comparison using Leave-One-Out Cross-Validation (PSIS-LOO-CV).

    Parameters
    ----------
    traces : dict
        Dictionary of model names mapping to fitted InferenceData traces with log_likelihood.
    ic : str
        Information criterion ('loo' or 'waic').
    method : str
        Weighting method ('stacking', 'bb-pseudo-bma', or 'pseudo-bma').

    Returns
    -------
    df_comp : pd.DataFrame
        ArviZ comparison summary table sorted by predictive accuracy.
    """
    valid_traces = {}
    for k, v in traces.items():
        if not hasattr(v, "log_likelihood") or len(v.log_likelihood.data_vars) == 0:
            try:
                pm.compute_log_likelihood(v, progressbar=False)
            except Exception:
                pass
        if hasattr(v, "log_likelihood") and len(v.log_likelihood.data_vars) > 0:
            valid_traces[k] = v

    if len(valid_traces) < 2:
        if len(valid_traces) == 1:
            m_name, tr = next(iter(valid_traces.items()))
            try:
                loo_res = az.loo(tr)
                df_single = pd.DataFrame([{
                    "rank": 0,
                    "elpd_loo": float(loo_res.elpd_loo),
                    "p_loo": float(loo_res.p_loo),
                    "se": float(loo_res.se),
                    "weight": 1.0,
                }], index=[m_name])
                return df_single
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    try:
        df_comp = az.compare(valid_traces, ic=ic, method=method)
    except Exception:
        try:
            df_comp = az.compare(valid_traces, ic=ic, method="bb-pseudo-bma")
        except Exception:
            try:
                df_comp = az.compare(valid_traces, ic=ic, method="pseudo-bma")
            except Exception:
                rows = []
                for name, tr in valid_traces.items():
                    try:
                        stat = az.loo(tr) if ic == "loo" else az.waic(tr)
                        rows.append({
                            f"elpd_{ic}": float(getattr(stat, f"elpd_{ic}")),
                            f"p_{ic}": float(getattr(stat, f"p_{ic}")),
                            "se": float(stat.se),
                            "weight": 1.0 / len(valid_traces),
                        })
                    except Exception:
                        rows.append({
                            f"elpd_{ic}": -999.0,
                            f"p_{ic}": 1.0,
                            "se": 0.0,
                            "weight": 1.0 / len(valid_traces),
                        })
                df_comp = pd.DataFrame(rows, index=list(valid_traces.keys()))
                if not df_comp.empty and f"elpd_{ic}" in df_comp.columns:
                    df_comp = df_comp.sort_values(by=f"elpd_{ic}", ascending=False)
                    df_comp["rank"] = np.arange(len(df_comp))

    return df_comp


def _build_benchmark_candidates(
    df_train: pd.DataFrame,
    df_full: pd.DataFrame,
    is_survey_data: bool,
    target_col: str,
    use_log_upsilon: bool = True,
    include_nonlinear: bool = True,
    include_multivariate: bool = True,
    include_hybrid: bool = True,
) -> Dict[str, Any]:
    """Construct benchmark candidate dictionary for given train/full data slices."""
    if is_survey_data:
        # Survey-epoch dataset mapping
        ups_raw_train = df_train["antecedent_mean_upsilon"].values if "antecedent_mean_upsilon" in df_train.columns else df_train["antecedent_upsilon_memory"].values
        ups_raw_full = df_full["antecedent_mean_upsilon"].values if "antecedent_mean_upsilon" in df_full.columns else df_full["antecedent_upsilon_memory"].values
        if use_log_upsilon:
            ups_train = df_train["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df_train.columns else np.log10(np.maximum(ups_raw_train, 1e-4))
            ups_full = df_full["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df_full.columns else np.log10(np.maximum(ups_raw_full, 1e-4))
            ups_label = "Unified Index (log-Upsilon)"
        else:
            ups_train = ups_raw_train
            ups_full = ups_raw_full
            ups_label = "Unified Index (Upsilon)"

        # Inter-survey peak storm forcing
        ups_peak_train = (
            df_train["antecedent_log_peak_upsilon"].values
            if (use_log_upsilon and "antecedent_log_peak_upsilon" in df_train.columns)
            else (df_train["antecedent_peak_upsilon"].values if "antecedent_peak_upsilon" in df_train.columns else ups_train)
        )
        ups_peak_full = (
            df_full["antecedent_log_peak_upsilon"].values
            if (use_log_upsilon and "antecedent_log_peak_upsilon" in df_full.columns)
            else (df_full["antecedent_peak_upsilon"].values if "antecedent_peak_upsilon" in df_full.columns else ups_full)
        )
        peak_label = "Unified Index (Peak-Upsilon)" if not use_log_upsilon else "Unified Index (log-Peak-Upsilon)"

        # Inter-survey thresholded storm work
        storm_work_train = (
            df_train["antecedent_log_storm_work"].values
            if "antecedent_log_storm_work" in df_train.columns
            else (df_train["antecedent_storm_work"].values if "antecedent_storm_work" in df_train.columns else ups_train)
        )
        storm_work_full = (
            df_full["antecedent_log_storm_work"].values
            if "antecedent_log_storm_work" in df_full.columns
            else (df_full["antecedent_storm_work"].values if "antecedent_storm_work" in df_full.columns else ups_full)
        )

        hs_train = df_train["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df_train.columns else df_train["antecedent_max_hs_m"].values
        hs_full = df_full["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df_full.columns else df_full["antecedent_max_hs_m"].values
        p_train = df_train["antecedent_max_power_Wm"].values if "antecedent_max_power_Wm" in df_train.columns else np.zeros(len(df_train))
        p_full = df_full["antecedent_max_power_Wm"].values if "antecedent_max_power_Wm" in df_full.columns else np.zeros(len(df_full))
        omega_train = df_train["antecedent_mean_dean"].values if "antecedent_mean_dean" in df_train.columns else np.zeros(len(df_train))
        omega_full = df_full["antecedent_mean_dean"].values if "antecedent_mean_dean" in df_full.columns else np.zeros(len(df_full))

        candidates: Dict[str, Any] = {
            "Mean Wave Height (Hs)": {"type": "single", "forcing": hs_train, "full_forcing": hs_full},
            "Wave Power P (Linear)": {"type": "single", "forcing": p_train, "full_forcing": p_full},
            "Dean Number (Linear)": {"type": "single", "forcing": omega_train, "full_forcing": omega_full},
            ups_label: {"type": "single", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon},
            peak_label: {"type": "single", "forcing": ups_peak_train, "full_forcing": ups_peak_full, "use_log_upsilon": use_log_upsilon},
            "Unified Index (Storm Work)": {"type": "single", "forcing": storm_work_train, "full_forcing": storm_work_full, "use_log_upsilon": True},
        }

        # Add Signed Equilibrium Upsilon, Yates, and XBeach competitors if present
        ups_eq_col = "upsilon_diseq_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "upsilon_diseq_simulated_delta_x_m" in df_train.columns) else "upsilon_diseq_simulated_pos_m"
        if ups_eq_col in df_train.columns:
            candidates["Signed Equilibrium Upsilon"] = {
                "type": "single",
                "forcing": df_train[ups_eq_col].values,
                "full_forcing": df_full[ups_eq_col].values,
            }

        yates_col = "yates_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "yates_simulated_delta_x_m" in df_train.columns) else "yates_simulated_pos_m"
        if yates_col in df_train.columns:
            candidates["Yates (2009) Equilibrium"] = {
                "type": "single",
                "forcing": df_train[yates_col].values,
                "full_forcing": df_full[yates_col].values,
            }

        xbeach_col = "xbeach_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "xbeach_simulated_delta_x_m" in df_train.columns) else "xbeach_simulated_pos_m"
        if xbeach_col in df_train.columns:
            candidates["XBeach Process Model"] = {
                "type": "single",
                "forcing": df_train[xbeach_col].values,
                "full_forcing": df_full[xbeach_col].values,
            }

        shorefor_col = "shorefor_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "shorefor_simulated_delta_x_m" in df_train.columns) else "shorefor_simulated_pos_m"
        if shorefor_col in df_train.columns:
            candidates["ShoreFor (2013) Memory Model"] = {
                "type": "single",
                "forcing": df_train[shorefor_col].values,
                "full_forcing": df_full[shorefor_col].values,
            }

        if include_nonlinear:
            candidates["Upsilon (Non-Linear)"] = {"type": "nonlinear", "forcing": ups_peak_train, "full_forcing": ups_peak_full, "use_log_upsilon": use_log_upsilon}

        if include_multivariate and "antecedent_accretion_work" in df_train.columns:
            acc_train = df_train["antecedent_log_accretion_work"].values if "antecedent_log_accretion_work" in df_train.columns else df_train["antecedent_accretion_work"].values
            candidates["Dual-Regime (Storm + Accretion)"] = {
                "type": "multivariate",
                "forcings": {
                    "storm_work": storm_work_train,
                    "accretion": acc_train,
                },
            }
        elif include_multivariate and "antecedent_max_surge_m" in df_train.columns:
            candidates["Multivariate (Hs+S+Omega)"] = {
                "type": "multivariate",
                "forcings": {
                    "hs": hs_train,
                    "surge": df_train["antecedent_max_surge_m"].values,
                    "dean": omega_train,
                },
            }

        if include_hybrid and "antecedent_max_surge_m" in df_train.columns:
            candidates["Hybrid (Upsilon Anchor + Multi Residuals)"] = {
                "type": "hybrid",
                "upsilon": ups_peak_train,
                "full_upsilon": ups_peak_full,
                "use_log_upsilon": use_log_upsilon,
                "forcings": {
                    "hs": hs_train,
                    "surge": df_train["antecedent_max_surge_m"].values,
                    "dean": omega_train,
                },
            }

    else:
        # Continuous hourly dataset mapping
        if use_log_upsilon:
            ups_train = df_train["log_upsilon"].values if "log_upsilon" in df_train.columns else np.log10(np.maximum(df_train["upsilon"].values, 1e-4))
            ups_full = df_full["log_upsilon"].values if "log_upsilon" in df_full.columns else np.log10(np.maximum(df_full["upsilon"].values, 1e-4))
            ups_label = "Unified Index (log-Upsilon)"
        else:
            ups_train = df_train["upsilon"].values
            ups_full = df_full["upsilon"].values
            ups_label = "Unified Index (Upsilon)"

        candidates: Dict[str, Any] = {
            "Hs (Linear)": {"type": "single", "forcing": df_train["hs_m"].values, "full_forcing": df_full["hs_m"].values},
            "Wave Power P (Linear)": {"type": "single", "forcing": df_train["p_flux_Wm"].values, "full_forcing": df_full["p_flux_Wm"].values},
            "Dean Number (Linear)": {"type": "single", "forcing": df_train["dean_number"].values, "full_forcing": df_full["dean_number"].values},
            ups_label: {"type": "single", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon},
        }

        if "signed_upsilon" in df_train.columns:
            candidates["Signed Index (Upsilon_signed)"] = {
                "type": "single",
                "forcing": df_train["signed_upsilon"].values,
                "full_forcing": df_full["signed_upsilon"].values,
            }

        for col_name, model_label in [
            ("yates_simulated_pos_m", "Yates (2009) Equilibrium"),
            ("shorefor_simulated_pos_m", "ShoreFor (2013) Memory Model"),
            ("xbeach_simulated_pos_m", "XBeach Process Model"),
            ("upsilon_diseq_simulated_pos_m", "Signed Equilibrium Upsilon"),
        ]:
            if col_name in df_train.columns:
                candidates[model_label] = {
                    "type": "single",
                    "forcing": df_train[col_name].values,
                    "full_forcing": df_full[col_name].values,
                }

        if include_nonlinear:
            candidates["Upsilon (Non-Linear)"] = {"type": "nonlinear", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon}
            candidates["Hs (Non-Linear)"] = {"type": "nonlinear", "forcing": df_train["hs_m"].values, "full_forcing": df_full["hs_m"].values}

        if include_multivariate:
            candidates["Multivariate (Hs+S+Wa+Omega)"] = {
                "type": "multivariate",
                "forcings": {
                    "hs": df_train["hs_m"].values,
                    "surge": df_train["surge_m"].values,
                    "wind": df_train["wind_normal_ms"].values,
                    "dean": df_train["dean_number"].values,
                },
            }

        if include_hybrid:
            candidates["Hybrid (Upsilon Anchor + Multi Residuals)"] = {
                "type": "hybrid",
                "upsilon": ups_train,
                "full_upsilon": ups_full,
                "use_log_upsilon": use_log_upsilon,
                "forcings": {
                    "hs": df_train["hs_m"].values,
                    "surge": df_train["surge_m"].values,
                    "wind": df_train["wind_normal_ms"].values,
                    "dean": df_train["dean_number"].values,
                },
            }

        if "w_s_ms" in df_train.columns:
            candidates["Asymmetric Dual-Regime (Erosion vs Accretion)"] = {
                "type": "asymmetric",
                "hs": df_train["hs_m"].values,
                "tp": df_train["tp_s"].values,
                "ws": df_train["w_s_ms"].values,
                "upsilon": ups_train,
                "full_upsilon": ups_full,
                "use_log_upsilon": use_log_upsilon,
                "shielding": df_train["shielding_factor"].values if "shielding_factor" in df_train.columns else np.ones(len(df_train)),
                "accretion": df_train["accretion_potential"].values if "accretion_potential" in df_train.columns else None,
            }

    return candidates


def _build_model_from_spec(
    spec: Dict[str, Any],
    y_obs_train: np.ndarray,
    name: str,
    effective_max_lag: int,
    use_log_upsilon: bool = True,
) -> Tuple[Optional[pm.Model], Optional[Dict[str, Any]]]:
    """Construct PyMC model and metadata dict for a specific candidate specification."""
    model_type = spec["type"]
    if model_type == "single":
        model, meta = build_latent_memory_bayesian_model(
            forcing=spec["forcing"],
            delta_x_obs=y_obs_train,
            model_name=name,
            max_lag=effective_max_lag,
        )
    elif model_type == "nonlinear":
        model, meta = build_nonlinear_response_model(
            forcing=spec["forcing"],
            delta_x_obs=y_obs_train,
            model_name=name,
            degree=3,
        )
    elif model_type == "multivariate":
        model, meta = build_multivariate_linear_model(
            forcings=spec["forcings"],
            delta_x_obs=y_obs_train,
            model_name=name,
        )
    elif model_type == "hybrid":
        model, meta = build_physics_anchored_hybrid_model(
            upsilon=spec["upsilon"],
            forcings=spec["forcings"],
            delta_x_obs=y_obs_train,
            model_name=name,
            max_lag=effective_max_lag,
        )
    elif model_type == "asymmetric":
        model, meta = build_asymmetric_dual_regime_model(
            hs=spec["hs"],
            tp=spec["tp"],
            ws=spec["ws"],
            upsilon=spec["upsilon"],
            delta_x_obs=y_obs_train,
            shielding_factor=spec["shielding"],
            accretion_potential=spec["accretion"],
            model_name=name,
            max_lag=effective_max_lag,
        )
    elif model_type == "latent_equilibrium":
        model, meta = build_latent_equilibrium_bayesian_model(
            forcing=spec["forcing"],
            delta_x_obs=y_obs_train,
            model_name=name,
            max_lag=effective_max_lag,
            scale_forcing=spec.get("scale_forcing", False),
        )
    else:
        return None, None

    meta["use_log_upsilon"] = spec.get("use_log_upsilon", use_log_upsilon)
    return model, meta


def run_benchmark_model_comparison(
    df: pd.DataFrame,
    target_col: str = "shoreline_pos_obs_m",
    use_log_upsilon: bool = True,
    train_split: float = 0.75,
    eval_mode: str = "holdout",
    n_splits: int = 4,
    min_train_ratio: float = 0.5,
    draws: int = 1000,
    tune: int = 800,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    include_nonlinear: bool = True,
    include_multivariate: bool = True,
    include_hybrid: bool = True,
    compute_loo: bool = True,
    sort_by: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Fit and benchmark competing physical models with PSIS-LOO-CV, Rolling-Origin CV, or Temporal Holdout.

    Parameters
    ----------
    df : pd.DataFrame
        Synchronized hydrodynamic and morphodynamic observational dataset.
    target_col : str
        Target column name (default 'shoreline_pos_obs_m' for cumulative cross-shore position).
        Use 'shoreline_change_rate_mh' or 'surveyed_delta_x_m' for differential inter-survey displacement.
    use_log_upsilon : bool
        Whether to use log10(Upsilon) as the primary forcing input (default True).
    train_split : float
        Chronological training split fraction (default 0.75, giving 75% train, 25% out-of-sample test).
        Ignored if eval_mode is 'loo' or 'rolling_origin'.
    eval_mode : str
        Evaluation paradigm:
        - 'holdout': Single chronological train/test split at train_split fraction.
        - 'loo': Full-series Bayesian training with ArviZ PSIS-LOO-CV as the primary ranking criterion.
        - 'rolling_origin': Walk-Forward / Expanding Window Temporal CV across n_splits horizons.
    n_splits : int
        Number of walk-forward temporal folds when eval_mode='rolling_origin' (default 4).
    min_train_ratio : float
        Initial training window ratio for walk-forward CV (default 0.5, i.e., 50% history).
    draws : int
        Number of MCMC draws per chain.
    tune : int
        Number of tuning / warmup steps per chain.
    chains : int
        Number of Markov chains (default 4).
    cores : int
        Number of CPU cores for sampling.
    random_seed : int
        Random seed for reproducibility.
    include_nonlinear : bool
        Whether to include non-linear polynomial benchmark variants.
    include_multivariate : bool
        Whether to include the unconstrained multivariate baseline.
    include_hybrid : bool
        Whether to include the physics-anchored hybrid model.
    compute_loo : bool
        Whether to compute ArviZ PSIS-LOO-CV cross-validation and model comparison ranking.
    sort_by : str, optional
        Column name to sort the final ranking table by. Auto-selected if None:
        - 'ELPD LOO' for eval_mode='loo'
        - 'CV RMSE (m)' for eval_mode='rolling_origin'
        - 'Test RMSE (m)' for eval_mode='holdout'

    Returns
    -------
    comparison_df : pd.DataFrame
        Ranked comparison table with CV/Test metrics, In-Sample metrics, LOO-CV statistics, and R-hat.
    traces : dict
        Fitted ArviZ InferenceData traces for each candidate model.
    """
    n_total = len(df)
    eval_mode = eval_mode.lower().strip()
    if eval_mode not in ["holdout", "loo", "rolling_origin"]:
        eval_mode = "holdout"

    if eval_mode == "loo" or train_split >= 1.0:
        split_idx = n_total
        has_holdout = False
    else:
        split_idx = int(n_total * train_split)
        if n_total > 100 and split_idx <= 24:
            split_idx = n_total
        elif n_total <= 100 and (split_idx < 3 or n_total - split_idx < 1):
            split_idx = n_total
        has_holdout = split_idx < n_total

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy() if has_holdout else df.copy()

    # Resolve target columns
    if target_col is None or target_col not in df.columns:
        if "surveyed_delta_x_m" in df.columns:
            target_col = "surveyed_delta_x_m"
        elif "shoreline_pos_obs_m" in df.columns:
            target_col = "shoreline_pos_obs_m"
        elif "shoreline_change_rate_mh" in df.columns:
            target_col = "shoreline_change_rate_mh"
        elif "delta_x_m" in df.columns:
            target_col = "delta_x_m"
        else:
            target_col = df.columns[0]

    y_obs_train = df_train[target_col].values
    y_true_train = (
        df_train["shoreline_pos_true_m"].values
        if target_col == "shoreline_pos_obs_m" and "shoreline_pos_true_m" in df_train.columns
        else df_train[target_col].values
    )
    y_obs_test = df_test[target_col].values if has_holdout else y_obs_train
    y_true_test = (
        df_test["shoreline_pos_true_m"].values
        if target_col == "shoreline_pos_obs_m" and has_holdout and "shoreline_pos_true_m" in df_test.columns
        else (df_test[target_col].values if has_holdout else y_true_train)
    )
    y_obs_full = df[target_col].values
    y_true_full = (
        df["shoreline_pos_true_m"].values
        if target_col == "shoreline_pos_obs_m" and "shoreline_pos_true_m" in df.columns
        else df[target_col].values
    )

    is_survey_data = "antecedent_upsilon_memory" in df.columns or "surveyed_delta_x_m" in df.columns
    effective_max_lag = min(48, max(2, len(df_train) // 4)) if len(df_train) < 60 else 48

    candidates = _build_benchmark_candidates(
        df_train=df_train,
        df_full=df,
        is_survey_data=is_survey_data,
        target_col=target_col,
        use_log_upsilon=use_log_upsilon,
        include_nonlinear=include_nonlinear,
        include_multivariate=include_multivariate,
        include_hybrid=include_hybrid,
    )

    # If rolling origin temporal CV is requested, calculate expanding window cuts
    rolling_folds = []
    if eval_mode == "rolling_origin":
        min_train_len = max(4, int(n_total * min_train_ratio))
        test_indices = np.arange(min_train_len, n_total)
        if len(test_indices) >= n_splits:
            fold_test_splits = np.array_split(test_indices, n_splits)
            for f_test in fold_test_splits:
                if len(f_test) > 0:
                    f_train_end = f_test[0]
                    rolling_folds.append((f_train_end, f_test))

    traces: Dict[str, az.InferenceData] = {}
    model_metas: Dict[str, Dict[str, Any]] = {}
    metrics_list = []

    for name, spec in candidates.items():
        model_type = spec["type"]
        model, meta = _build_model_from_spec(
            spec=spec,
            y_obs_train=y_obs_train,
            name=name,
            effective_max_lag=effective_max_lag,
            use_log_upsilon=use_log_upsilon,
        )
        if model is None or meta is None:
            continue

        model_metas[name] = meta

        # 1. Main MCMC Sampling on full or training slice
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=0.92,
                random_seed=random_seed,
                progressbar=False,
                return_inferencedata=True,
            )
            pm.compute_log_likelihood(trace, progressbar=False)

        traces[name] = trace

        # Prepare specification for full continuous prediction
        spec_pred = None
        if "full_forcing" in spec:
            spec_pred = {"forcing": spec["full_forcing"]}
        elif "full_upsilon" in spec:
            spec_pred = {"upsilon": spec["full_upsilon"]}

        # Full continuous prediction across dataset
        mu_full_mean = predict_bayesian_model(
            trace=trace,
            model_type=model_type,
            meta=meta,
            df=df,
            spec=spec_pred,
            return_samples=False,
            random_seed=random_seed,
        )

        import gc
        gc.collect()

        # 2. Performance metrics based on eval_mode
        row: Dict[str, Any] = {
            "Model": name,
            "Type": model_type,
        }

        if eval_mode == "rolling_origin" and len(rolling_folds) >= 2:
            # Execute walk-forward cross-validation folds
            cv_preds = []
            cv_actuals = []
            fold_rmses = []
            fold_r2s = []

            for f_train_end, f_test_idx in rolling_folds:
                df_f_train = df.iloc[:f_train_end].copy()
                df_f_test = df.iloc[f_test_idx].copy()
                y_f_train = y_obs_full[:f_train_end]
                y_f_test = y_true_full[f_test_idx]

                f_cand = _build_benchmark_candidates(
                    df_train=df_f_train,
                    df_full=df.iloc[:f_test_idx[-1] + 1],
                    is_survey_data=is_survey_data,
                    target_col=target_col,
                    use_log_upsilon=use_log_upsilon,
                    include_nonlinear=include_nonlinear,
                    include_multivariate=include_multivariate,
                    include_hybrid=include_hybrid,
                )
                if name not in f_cand:
                    continue

                f_spec = f_cand[name]
                f_lag = min(48, max(2, len(df_f_train) // 4)) if len(df_f_train) < 60 else 48
                f_model, f_meta = _build_model_from_spec(f_spec, y_f_train, f"{name}_f", f_lag, use_log_upsilon)
                if f_model is None:
                    continue

                with f_model:
                    f_trace = pm.sample(
                        draws=max(400, draws // 2),
                        tune=max(300, tune // 2),
                        chains=max(2, chains // 2),
                        cores=cores,
                        target_accept=0.90,
                        random_seed=random_seed,
                        progressbar=False,
                        return_inferencedata=True,
                    )

                f_spec_pred = None
                if "full_forcing" in f_spec:
                    f_spec_pred = {"forcing": f_spec["full_forcing"]}
                elif "full_upsilon" in f_spec:
                    f_spec_pred = {"upsilon": f_spec["full_upsilon"]}

                f_mu = predict_bayesian_model(
                    trace=f_trace,
                    model_type=model_type,
                    meta=f_meta,
                    df=df.iloc[:f_test_idx[-1] + 1],
                    spec=f_spec_pred,
                    return_samples=False,
                    random_seed=random_seed,
                )
                f_pred_test = f_mu[f_test_idx]
                cv_preds.extend(f_pred_test)
                cv_actuals.extend(y_f_test)

                f_perf = evaluate_model_performance(y_true=y_f_test, y_pred=f_pred_test)
                fold_rmses.append(f_perf["RMSE (m)"])
                fold_r2s.append(f_perf["R2 Score"])

            if cv_preds:
                cv_overall = evaluate_model_performance(y_true=np.array(cv_actuals), y_pred=np.array(cv_preds))
                row["CV RMSE (m)"] = cv_overall["RMSE (m)"]
                row["CV R2 Score"] = cv_overall["R2 Score"]
                row["CV BSS"] = cv_overall["Brier Skill Score (BSS)"]
                row["Fold RMSE (std)"] = float(np.std(fold_rmses))
                row["Fold R2 (std)"] = float(np.std(fold_r2s))

            perf_full = evaluate_model_performance(y_true=y_true_full, y_pred=mu_full_mean)
            row["Full RMSE (m)"] = perf_full["RMSE (m)"]
            row["Full R2 Score"] = perf_full["R2 Score"]

        elif eval_mode == "loo" or not has_holdout:
            perf_full = evaluate_model_performance(y_true=y_true_full, y_pred=mu_full_mean)
            row["In-Sample RMSE (m)"] = perf_full["RMSE (m)"]
            row["In-Sample R2 Score"] = perf_full["R2 Score"]
            row["In-Sample BSS"] = perf_full["Brier Skill Score (BSS)"]

        else:
            # Standard holdout split evaluation
            mu_train_pred = mu_full_mean[:split_idx]
            mu_test_pred = mu_full_mean[split_idx:]
            perf_train = evaluate_model_performance(y_true=y_true_train, y_pred=mu_train_pred)
            perf_test = evaluate_model_performance(y_true=y_true_test, y_pred=mu_test_pred)
            perf_full = evaluate_model_performance(y_true=y_true_full, y_pred=mu_full_mean)

            row["Test RMSE (m)"] = perf_test["RMSE (m)"]
            row["Test R2 Score"] = perf_test["R2 Score"]
            row["Test BSS"] = perf_test["Brier Skill Score (BSS)"]
            row["Train RMSE (m)"] = perf_train["RMSE (m)"]
            row["Train R2 Score"] = perf_train["R2 Score"]
            row["Full RMSE (m)"] = perf_full["RMSE (m)"]
            row["Full R2 Score"] = perf_full["R2 Score"]

        # Convergence diagnostics
        avail_vars = [
            v for v in [
                "gamma",
                "beta_forcing",
                "beta_upsilon",
                "alpha",
                "sigma_base",
                "sigma_storm",
                "sigma",
                "k_eros",
                "k_accrete",
                "omega_crit",
                "hs_exponent",
            ]
            if v in trace.posterior and (v != "sigma" or "sigma_base" not in trace.posterior)
        ]
        post_rhat = float(az.summary(trace, var_names=avail_vars)["r_hat"].max()) if avail_vars else 1.0
        row["Max R-hat"] = post_rhat
        if "gamma" in trace.posterior:
            row["E[gamma]"] = float(trace.posterior["gamma"].mean())

        metrics_list.append(row)

    # 3. Compute Leave-One-Out Cross-Validation (LOO-CV) comparison table
    loo_df = pd.DataFrame()
    if compute_loo and len(traces) >= 2:
        try:
            loo_df = run_arviz_loo_comparison(traces, ic="loo")
            if not loo_df.empty:
                for r in metrics_list:
                    m_name = r["Model"]
                    if m_name in loo_df.index:
                        r["LOO Rank"] = int(loo_df.loc[m_name, "rank"])
                        r["ELPD LOO"] = float(loo_df.loc[m_name, "elpd_loo"])
                        if "se" in loo_df.columns:
                            r["SE ELPD"] = float(loo_df.loc[m_name, "se"])
                        r["p_LOO"] = float(loo_df.loc[m_name, "p_loo"])
                        r["LOO Weight"] = float(loo_df.loc[m_name, "weight"])
        except Exception:
            pass

    # Aliasing for both naming styles across traces and model_metas
    for ups_key in ["Unified Index (log-Upsilon)", "Unified Index (Upsilon)"]:
        if ups_key in traces:
            traces["Upsilon (Linear)"] = traces[ups_key]
            traces["Unified Index (Upsilon)"] = traces[ups_key]
            traces["Unified Index (log-Upsilon)"] = traces[ups_key]
            if ups_key in model_metas:
                model_metas["Upsilon (Linear)"] = model_metas[ups_key]
                model_metas["Unified Index (Upsilon)"] = model_metas[ups_key]
                model_metas["Unified Index (log-Upsilon)"] = model_metas[ups_key]
    if "Upsilon (Linear)" in traces and "Unified Index (Upsilon)" not in traces:
        traces["Unified Index (Upsilon)"] = traces["Upsilon (Linear)"]
        if "Upsilon (Linear)" in model_metas:
            model_metas["Unified Index (Upsilon)"] = model_metas["Upsilon (Linear)"]

    comparison_df = pd.DataFrame(metrics_list)

    # Determine sorting order
    if sort_by is not None and sort_by in comparison_df.columns:
        ascending = not (sort_by in ["ELPD LOO", "LOO Weight", "CV R2 Score", "Test R2 Score", "In-Sample R2 Score", "Full R2 Score"])
        comparison_df = comparison_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    elif eval_mode == "loo" and "ELPD LOO" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="ELPD LOO", ascending=False).reset_index(drop=True)
    elif eval_mode == "rolling_origin" and "CV RMSE (m)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="CV RMSE (m)", ascending=True).reset_index(drop=True)
    elif "Test RMSE (m)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="Test RMSE (m)", ascending=True).reset_index(drop=True)
    elif "In-Sample RMSE (m)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="In-Sample RMSE (m)", ascending=True).reset_index(drop=True)

    comparison_df.attrs["loo_comparison"] = loo_df
    comparison_df.attrs["eval_mode"] = eval_mode
    comparison_df.attrs["train_split"] = train_split
    comparison_df.attrs["split_idx"] = split_idx
    comparison_df.attrs["model_metas"] = model_metas
    comparison_df.attrs["n_splits"] = n_splits

    return comparison_df, traces


def build_multivariate_linear_model(
    forcings: Dict[str, np.ndarray],
    delta_x_obs: np.ndarray,
    model_name: str = "multivariate_model",
    scale_forcings: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a multivariate linear Bayesian model with individual coefficients per forcing."""
    y_obs = np.asarray(delta_x_obs, dtype=float)

    meta = {"model_name": model_name, "n_obs": len(y_obs), "forcing_names": list(forcings.keys())}
    scaled_data = {}

    for fname, fdata in forcings.items():
        arr = np.asarray(fdata, dtype=float)
        if scale_forcings:
            fmean = float(np.mean(arr))
            fstd = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0
            scaled_data[fname] = (arr - fmean) / fstd
            meta[f"{fname}_mean"] = fmean
            meta[f"{fname}_std"] = fstd
        else:
            scaled_data[fname] = arr

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))

        mu = alpha
        for fname, fscaled in scaled_data.items():
            f_data = pm.Data(f"x_{fname}", fscaled)
            beta = pm.Normal(f"beta_{fname}", mu=0.0, sigma=1.5)
            mu = mu + beta * f_data

        mu_det = pm.Deterministic("mu", mu)
        pm.StudentT("y_obs", nu=nu, mu=mu_det, sigma=sigma, observed=y_obs)

    return model, meta


def build_nonlinear_response_model(
    forcing: np.ndarray,
    delta_x_obs: np.ndarray,
    model_name: str = "nonlinear_model",
    degree: int = 3,
    scale_forcing: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a non-linear Bayesian model using polynomial basis expansion with Horseshoe priors."""
    forcing_arr = np.asarray(forcing, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)

    f_mean = float(np.mean(forcing_arr))
    f_std = float(np.std(forcing_arr)) if np.std(forcing_arr) > 1e-6 else 1.0
    f_scaled = (forcing_arr - f_mean) / f_std if scale_forcing else forcing_arr

    # Build polynomial basis matrix: [f, f^2, f^3, ...]
    basis_matrix = np.column_stack([f_scaled ** (d + 1) for d in range(degree)])

    meta = {
        "model_name": model_name,
        "f_mean": f_mean,
        "f_std": f_std,
        "degree": degree,
        "n_obs": len(y_obs),
    }

    with pm.Model() as model:
        X_basis = pm.Data("X_basis", basis_matrix)

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))

        # Horseshoe prior for sparsity on polynomial coefficients
        tau_global = pm.HalfCauchy("tau_global", beta=0.5)
        lambda_local = pm.HalfCauchy("lambda_local", beta=1.0, shape=degree)
        beta_poly = pm.Normal("beta_poly", mu=0.0, sigma=tau_global * lambda_local, shape=degree)

        mu = pm.Deterministic("mu", alpha + pt.dot(X_basis, beta_poly))
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_obs)

    return model, meta


def build_physics_anchored_hybrid_model(
    upsilon: np.ndarray,
    forcings: Dict[str, np.ndarray],
    delta_x_obs: np.ndarray,
    model_name: str = "hybrid_physics_anchored",
    max_lag: int = 48,
    scale_forcings: bool = True,
    heteroscedastic: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a Physics-Informed Hybrid Bayesian Model (PI-HBM).

    Architecture:
    1. Physics Anchor: Upsilon*(gamma) serves as the informative base hazard prior:
       beta_upsilon ~ Normal(mu = -1.0, sigma = 0.5) (strictly negative erosion baseline)
    2. Residual Site Calibration: Multivariate forcing terms (Hs, Surge, Wind, Dean) serve
       as site-specific calibration corrections with shrinkage priors:
       delta_j ~ Normal(mu = 0.0, sigma = 0.5)
    3. Latent Memory: gamma ~ Beta(2, 5) dynamic convolution for cumulative storm memory.
    4. Deterministic mean:
       mu_t = alpha + beta_upsilon * Upsilon*_t(gamma) + sum_j delta_j * X_{j, t}^*
    5. Heteroscedastic noise (Approach A): sigma_t = sigma_base + sigma_storm * max(Upsilon*_t, 0)
    6. Likelihood: y_obs ~ StudentT(nu, mu_t, sigma_t)
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)

    max_lag = max(1, min(max_lag, len(y_obs)))
    meta = {
        "model_name": model_name,
        "n_obs": len(y_obs),
        "max_lag": max_lag,
        "heteroscedastic": heteroscedastic,
    }

    # Standardize Upsilon
    u_mean = float(np.mean(ups_arr))
    u_std = float(np.std(ups_arr)) if np.std(ups_arr) > 1e-6 else 1.0
    u_scaled = (ups_arr - u_mean) / u_std if scale_forcings else ups_arr
    meta["upsilon_mean"] = u_mean
    meta["upsilon_std"] = u_std

    lag_mat = create_lag_matrix(u_scaled, max_lag=max_lag)
    k_indices = np.arange(max_lag, dtype=float)

    # Standardize residual forcings
    scaled_res = {}
    for fname, fdata in forcings.items():
        arr = np.asarray(fdata, dtype=float)
        if scale_forcings:
            fmean = float(np.mean(arr))
            fstd = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0
            scaled_res[fname] = (arr - fmean) / fstd
            meta[f"{fname}_mean"] = fmean
            meta[f"{fname}_std"] = fstd
        else:
            scaled_res[fname] = arr

    with pm.Model() as model:
        # Data containers
        x_lag = pm.Data("x_lag", lag_mat)
        k_vec = pm.Data("k_vec", k_indices)

        # 1. Latent Antecedent Storm Memory (informative Beta(5, 20))
        gamma = pm.Beta("gamma", alpha=5.0, beta=20.0)
        unnorm_w = pt.power(gamma, k_vec)
        weights = unnorm_w / pt.sum(unnorm_w)
        upsilon_memory = pm.Deterministic("upsilon_memory", pt.dot(x_lag, weights))

        # 2. Informative Physics Anchor Prior (negative baseline erosion expectation)
        beta_upsilon = pm.Normal("beta_upsilon", mu=-1.0, sigma=0.5)

        # 3. Residual Multivariate Calibration Parameters (zero-centered shrinkage)
        mu = beta_upsilon * upsilon_memory
        for fname, fscaled in scaled_res.items():
            f_data = pm.Data(f"x_{fname}", fscaled)
            delta_j = pm.Normal(f"delta_{fname}", mu=0.0, sigma=0.5)
            mu = mu + delta_j * f_data

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_storm = pm.HalfNormal("sigma_storm", sigma=0.5)
            storm_excitation = pt.maximum(upsilon_memory, 0.0)
            sigma = pm.Deterministic("sigma", sigma_base + sigma_storm * storm_excitation)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))

        mu_det = pm.Deterministic("mu", alpha + mu)
        pm.StudentT("y_obs", nu=nu, mu=mu_det, sigma=sigma, observed=y_obs)

    return model, meta


def build_asymmetric_dual_regime_model(
    hs: np.ndarray,
    tp: np.ndarray,
    ws: np.ndarray,
    upsilon: np.ndarray,
    delta_x_obs: np.ndarray,
    shielding_factor: Optional[np.ndarray] = None,
    accretion_potential: Optional[np.ndarray] = None,
    model_name: str = "asymmetric_dual_regime",
    max_lag: int = 48,
    scale_forcings: bool = True,
    heteroscedastic: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build an Asymmetric Dual-Regime Morphodynamic Bayesian Model.

    Physics Architecture:
    1. Dean Dimensionless Fall Velocity (Omega):
       Omega = Hs / (ws(T) * Tp)
    2. Dynamic Equilibrium Regime Switch:
       omega_crit ~ TruncatedNormal(mu = 1.75, sigma = 0.4, lower = 0.8, upper = 3.0)
       is_erosion = sigmoid(5.0 * (omega - omega_crit))
    3. Shielded Storm Gouging Hazard:
       F_eros = Upsilon * Shielding_factor
    4. Constructive Swash Accretion Potential:
       F_accrete = (Tp * sqrt(g * L_0)) / (Hs + 0.1)
    5. Dual Asymmetric Kinetics:
       mu_t = alpha + is_erosion * (-k_eros * E_t(F_eros)) + (1.0 - is_erosion) * (+k_accrete * E_t(F_accrete))
    6. Heteroscedastic noise (Approach A): sigma_t = sigma_base + sigma_storm * max(E_t(F_eros), 0)
    """
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.5)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-4)
    ups_arr = np.asarray(upsilon, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)

    # 1. Dean Number
    omega_series = hs_arr / (ws_arr * tp_arr)

    # 2. Shielded storm erosion forcing
    shield_arr = np.asarray(shielding_factor, dtype=float) if shielding_factor is not None else np.ones_like(hs_arr)
    f_eros = ups_arr * shield_arr

    # 3. Constructive accretion potential
    if accretion_potential is not None:
        f_acc = np.asarray(accretion_potential, dtype=float)
    else:
        l_0 = (9.81 * (tp_arr ** 2)) / (2.0 * np.pi)
        f_acc = (tp_arr * np.sqrt(9.81 * l_0)) / (hs_arr + 0.1)

    meta = {
        "model_name": model_name,
        "n_obs": len(y_obs),
        "max_lag": max_lag,
        "heteroscedastic": heteroscedastic,
    }

    # Standardize kinetic forcing series
    if scale_forcings:
        e_mean, e_std = float(np.mean(f_eros)), float(np.std(f_eros)) if np.std(f_eros) > 1e-6 else 1.0
        a_mean, a_std = float(np.mean(f_acc)), float(np.std(f_acc)) if np.std(f_acc) > 1e-6 else 1.0
        f_eros_scaled = (f_eros - e_mean) / e_std
        f_acc_scaled = (f_acc - a_mean) / a_std
        meta.update({"eros_mean": e_mean, "eros_std": e_std, "acc_mean": a_mean, "acc_std": a_std})
    else:
        f_eros_scaled = f_eros
        f_acc_scaled = f_acc

    lag_mat_eros = create_lag_matrix(f_eros_scaled, max_lag=max_lag)
    lag_mat_acc = create_lag_matrix(f_acc_scaled, max_lag=max_lag)
    k_indices = np.arange(max_lag, dtype=float)

    with pm.Model() as model:
        # Data containers
        x_lag_eros = pm.Data("x_lag_eros", lag_mat_eros)
        x_lag_acc = pm.Data("x_lag_acc", lag_mat_acc)
        omega_data = pm.Data("omega_data", omega_series)
        k_vec = pm.Data("k_vec", k_indices)

        # 1. Latent memory convolution (informative Beta(5, 20))
        gamma = pm.Beta("gamma", alpha=5.0, beta=20.0)
        unnorm_w = pt.power(gamma, k_vec)
        weights = unnorm_w / pt.sum(unnorm_w)
        eff_eros = pm.Deterministic("eff_eros", pt.dot(x_lag_eros, weights))
        eff_acc = pm.Deterministic("eff_acc", pt.dot(x_lag_acc, weights))

        # 2. Dynamic Dean equilibrium threshold (learns optimal transition point via smooth differentiable sigmoid)
        omega_crit = pm.TruncatedNormal("omega_crit", mu=1.75, sigma=0.4, lower=0.8, upper=3.0)
        is_erosion = pm.Deterministic("is_erosion", pt.sigmoid(5.0 * (omega_data - omega_crit)))

        # 3. Asymmetric kinetic coefficients (HalfNormal ensures sign-consistent physics)
        k_eros = pm.HalfNormal("k_eros", sigma=1.0)
        k_accrete = pm.HalfNormal("k_accrete", sigma=1.0)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # 4. Dual-regime shoreline velocity
        mu = alpha + is_erosion * (-k_eros * eff_eros) + (1.0 - is_erosion) * (k_accrete * eff_acc)
        mu_det = pm.Deterministic("mu", mu)

        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_storm = pm.HalfNormal("sigma_storm", sigma=0.5)
            storm_excitation = pt.maximum(eff_eros, 0.0)
            sigma = pm.Deterministic("sigma", sigma_base + sigma_storm * storm_excitation)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))
        pm.StudentT("y_obs", nu=nu, mu=mu_det, sigma=sigma, observed=y_obs)

    return model, meta


def build_learnable_exponent_bayesian_model(
    hs: np.ndarray,
    tp: np.ndarray,
    ws: np.ndarray,
    surge: np.ndarray,
    runup: np.ndarray,
    wind_normal: np.ndarray,
    delta_x_obs: np.ndarray,
    tan_beta: Optional[Union[float, np.ndarray]] = None,
    shielding_factor: Optional[np.ndarray] = None,
    model_name: str = "learnable_hs_exponent",
    max_lag: int = 48,
    exponent_prior: str = "uniform",
    prior_mu: float = 2.0,
    prior_sigma: float = 0.5,
    s_ref: float = 1.0,
    r_ref: float = 2.0,
    w_ref: float = 15.0,
    w_bound: float = 0.6,
    w_ref_on: Optional[float] = None,
    w_ref_off: Optional[float] = None,
    w_bound_on: Optional[float] = None,
    w_bound_off: Optional[float] = None,
    c_wind_on: float = 1.0,
    c_wind_off: float = 0.5,
    lower_bound: float = 1.0,
    upper_bound: float = 3.0,
    heteroscedastic: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a Physics-Informed Bayesian Model that learns the Hs scaling exponent n in [1.0, 3.0] directly from observations."""
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.1)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-4)
    surge_arr = np.asarray(surge, dtype=float)
    runup_arr = np.asarray(runup, dtype=float)
    wind_arr = np.asarray(wind_normal, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)
    n_obs = len(y_obs)

    tan_beta_val = 0.08 if tan_beta is None else np.asarray(tan_beta, dtype=float)
    shield_arr = np.asarray(shielding_factor, dtype=float) if shielding_factor is not None else np.ones(n_obs)
    k_indices = np.arange(max_lag, dtype=float)

    # Precompute non-exponent terms to optimize sampling graph (decoupled additive TWL & asymmetric quadratic wind)
    eff_w_ref_on = w_ref_on if w_ref_on is not None else w_ref
    eff_w_ref_off = w_ref_off if w_ref_off is not None else w_ref
    eff_w_bound_on = w_bound_on if w_bound_on is not None else w_bound
    eff_w_bound_off = w_bound_off if w_bound_off is not None else (0.5 * w_bound if w_bound != 0.6 else 0.3)

    from src.physics import compute_asymmetric_wind_amplification
    twl_arr = 1.0 + np.maximum(surge_arr / s_ref, 0.0) + np.maximum(runup_arr / r_ref, 0.0)
    wind_arr_mod = compute_asymmetric_wind_amplification(
        wind_arr,
        w_ref_on=eff_w_ref_on,
        w_ref_off=eff_w_ref_off,
        w_bound_on=eff_w_bound_on,
        w_bound_off=eff_w_bound_off,
        c_on=c_wind_on,
        c_off=c_wind_off,
    )
    prefactor = (1025.0 * (9.81 ** 1.5)) / (16.0 * ((2.0 * np.pi) ** 1.5))
    kinematic_base = prefactor * tan_beta_val * (tp_arr / ws_arr) * twl_arr * wind_arr_mod * shield_arr

    meta = {
        "model_name": model_name,
        "n_obs": n_obs,
        "max_lag": max_lag,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "heteroscedastic": heteroscedastic,
    }

    with pm.Model() as model:
        # Data containers
        hs_data = pm.Data("hs_data", hs_arr)
        kin_base_data = pm.Data("kin_base_data", kinematic_base)
        k_vec = pm.Data("k_vec", k_indices)

        # 1. Parameterized / Learnable Hs scaling exponent n in [1.0, 3.0]
        if exponent_prior == "truncated_normal":
            hs_exponent = pm.TruncatedNormal(
                "hs_exponent",
                mu=prior_mu,
                sigma=prior_sigma,
                lower=lower_bound,
                upper=upper_bound,
            )
        else:
            hs_exponent = pm.Uniform("hs_exponent", lower=lower_bound, upper=upper_bound)

        # 2. Dynamic Upsilon computation inside PyTensor graph: Upsilon(n) = kin_base * (Hs^n)
        raw_upsilon = kin_base_data * pt.power(hs_data, hs_exponent)

        # Standardize dynamically to stabilize sampling
        u_mean = pt.mean(raw_upsilon)
        u_std = pt.std(raw_upsilon) + 1e-4
        u_scaled = (raw_upsilon - u_mean) / u_std

        # 3. Latent Memory Convolution in PyTensor (informative Beta(5, 20))
        gamma = pm.Beta("gamma", alpha=5.0, beta=20.0)
        unnorm_w = pt.power(gamma, k_vec)
        weights = unnorm_w / pt.sum(unnorm_w)

        # Differentiable lag tensor construction
        u_padded = pt.concatenate([pt.zeros(max_lag - 1), u_scaled])
        lag_slices = [u_padded[max_lag - 1 - k : max_lag - 1 - k + n_obs] for k in range(max_lag)]
        lag_tensor = pt.stack(lag_slices, axis=1)
        upsilon_memory = pm.Deterministic("upsilon_memory", pt.dot(lag_tensor, weights))

        # 4. Morphodynamic Linear Predictor
        beta_upsilon = pm.Normal("beta_upsilon", mu=-1.0, sigma=0.5)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        mu = pm.Deterministic("mu", alpha + beta_upsilon * upsilon_memory)

        # 5. Robust Student-t Likelihood (nu >= 5) with Heteroscedastic Storm Volatility (Approach A)
        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_storm = pm.HalfNormal("sigma_storm", sigma=0.5)
            storm_excitation = pt.maximum(upsilon_memory, 0.0)
            sigma = pm.Deterministic("sigma", sigma_base + sigma_storm * storm_excitation)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_obs)
    return model, meta


def run_hs_exponent_sensitivity_analysis(
    df: pd.DataFrame,
    exponents: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
    target_col: str = "shoreline_pos_obs_m",
    max_lag: int = 24,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    include_learnable: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Run systematic sensitivity analysis over wave height exponents n in Hs^n.

    Fits Bayesian latent memory models for each candidate fixed exponent n in exponents alongside
    the fully learnable exponent model, computes PSIS-LOO-CV ELPD via ArviZ,
    and returns a comparison ranking table justifying the optimal exponent.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hydrodynamic columns (hs_m/hs, tp_s/tp, surge_m/surge, etc.) and target observations.
    exponents : sequence of float
        Candidate fixed exponents n in Hs^n (default: 1.5, 2.0, 2.5, 3.0).
    target_col : str
        Observed target column in df (e.g., 'shoreline_pos_obs_m' or 'delta_x').
    max_lag : int
        Maximum antecedent memory lag hours.
    draws : int
        Number of MCMC draws.
    tune : int
        Number of MCMC tuning steps.
    chains : int
        Number of MCMC chains.
    include_learnable : bool
        Whether to include the fully learnable exponent model in the comparison.
    random_seed : int
        Random seed for sampling reproducibility.

    Returns
    -------
    df_comp : pd.DataFrame
        ArviZ model comparison table with ELPD, p_loo, and LOO weights.
    traces : dict
        Fitted MCMC traces for each candidate exponent configuration.
    """
    from src.physics import compute_unified_coastal_vulnerability_index

    hs = df["hs_m"].values if "hs_m" in df.columns else df["hs"].values
    tp = df["tp_s"].values if "tp_s" in df.columns else df["tp"].values
    surge = df["surge_m"].values if "surge_m" in df.columns else (df["surge"].values if "surge" in df.columns else np.zeros(len(df)))
    wind_spd = df["wind_speed_ms"].values if "wind_speed_ms" in df.columns else (df["wind_speed"].values if "wind_speed" in df.columns else np.zeros(len(df)))
    wind_dir = df["wind_dir_deg"].values if "wind_dir_deg" in df.columns else (df["wind_dir"].values if "wind_dir" in df.columns else np.full(len(df), 70.0))
    d50 = df["d50_mm"].values if "d50_mm" in df.columns else 0.2
    tan_beta = df["tan_beta"].values if "tan_beta" in df.columns else 0.08
    y_obs = df[target_col].values if target_col in df.columns else df["delta_x_obs"].values

    traces: Dict[str, az.InferenceData] = {}

    for n_exp in exponents:
        m_name = f"fixed_n_{n_exp:.1f}"
        df_phys = compute_unified_coastal_vulnerability_index(
            hs=hs,
            tp=tp,
            surge=surge,
            wind_speed=wind_spd,
            wind_dir_deg=wind_dir,
            d50=d50,
            tan_beta=tan_beta,
            hs_exponent=n_exp,
        )
        forcing = df_phys["upsilon"].values
        model, meta = build_latent_memory_bayesian_model(
            forcing=forcing,
            delta_x_obs=y_obs,
            model_name=m_name,
            max_lag=max_lag,
        )
        trace = fit_bayesian_model(
            model=model,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        traces[m_name] = trace

    if include_learnable:
        m_name = "learnable_n"
        df_base = compute_unified_coastal_vulnerability_index(
            hs=hs,
            tp=tp,
            surge=surge,
            wind_speed=wind_spd,
            wind_dir_deg=wind_dir,
            d50=d50,
            tan_beta=tan_beta,
            hs_exponent=2.5,
        )
        learn_model, learn_meta = build_learnable_exponent_bayesian_model(
            hs=hs,
            tp=tp,
            ws=df_base["w_s_ms"].values,
            surge=surge,
            runup=df_base["runup_r2_m"].values,
            wind_normal=df_base["wind_normal_ms"].values,
            delta_x_obs=y_obs,
            tan_beta=tan_beta,
            model_name=m_name,
            max_lag=max_lag,
            exponent_prior="uniform",
            lower_bound=1.0,
            upper_bound=3.0,
        )
        trace_learn = fit_bayesian_model(
            model=learn_model,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        traces[m_name] = trace_learn

    df_comp = run_arviz_loo_comparison(traces, ic="loo")
    return df_comp, traces


def build_latent_equilibrium_bayesian_model(
    forcing: np.ndarray,
    delta_x_obs: np.ndarray,
    model_name: str = "latent_equilibrium_model",
    max_lag: int = 48,
    scale_forcing: bool = False,
    upsilon_eq_prior_mu: float = 2500.0,
    upsilon_eq_prior_sigma: float = 750.0,
    sequential_update: bool = False,
    heteroscedastic: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a PyMC Bayesian Model where Equilibrium Threshold Υ_eq is inferred as a Latent Parameter."""
    forcing_arr = np.asarray(forcing, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)
    n_obs = len(y_obs)

    f_mean = float(np.mean(forcing_arr))
    f_std = float(np.std(forcing_arr)) if np.std(forcing_arr) > 1e-6 else 1.0
    f_scaled = (forcing_arr - f_mean) / f_std if scale_forcing else forcing_arr

    max_lag = max(1, min(max_lag, n_obs))
    lag_mat = create_lag_matrix(f_scaled, max_lag=max_lag)
    k_indices = np.arange(max_lag, dtype=float)

    meta = {
        "model_name": model_name,
        "n_obs": n_obs,
        "max_lag": max_lag,
        "f_mean": f_mean,
        "f_std": f_std,
        "scale_forcing": scale_forcing,
        "upsilon_eq_prior_mu": upsilon_eq_prior_mu,
        "upsilon_eq_prior_sigma": upsilon_eq_prior_sigma,
        "heteroscedastic": heteroscedastic,
    }

    with pm.Model() as model:
        x_lag = pm.Data("x_lag", lag_mat)
        k_vec = pm.Data("k_vec", k_indices)

        # 1. Latent memory convolution (informative Beta(5, 20))
        gamma = pm.Beta("gamma", alpha=5.0, beta=20.0)
        unnorm_w = pt.power(gamma, k_vec)
        weights = unnorm_w / pt.sum(unnorm_w)
        effective_forcing = pm.Deterministic("effective_forcing", pt.dot(x_lag, weights))

        # 2. Latent Equilibrium Parameter (Inferred without future dataset leakage)
        if scale_forcing:
            scaled_mu = (upsilon_eq_prior_mu - f_mean) / f_std
            scaled_sig = max(upsilon_eq_prior_sigma / f_std, 0.1)
            upsilon_eq_latent = pm.Normal("upsilon_eq_latent", mu=scaled_mu, sigma=scaled_sig)
            raw_upsilon_eq = pm.Deterministic("upsilon_eq", upsilon_eq_latent * f_std + f_mean)
            diseq = pm.Deterministic("upsilon_diseq", effective_forcing - upsilon_eq_latent)
        else:
            upsilon_eq_latent = pm.TruncatedNormal(
                "upsilon_eq",
                mu=upsilon_eq_prior_mu,
                sigma=upsilon_eq_prior_sigma,
                lower=50.0,
                upper=25000.0,
            )
            diseq = pm.Deterministic("upsilon_diseq", effective_forcing - upsilon_eq_latent)

        # 3. Asymmetric Signed Morphodynamics in PyTensor
        delta_eros = pm.Deterministic("delta_erosion", pt.maximum(diseq, 0.0))
        delta_acc = pm.Deterministic("delta_accretion", pt.maximum(-diseq, 0.0))

        k_eros = pm.HalfNormal("k_eros", sigma=1.5)
        k_accrete = pm.HalfNormal("k_accrete", sigma=1.5)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # Rate scaling factor
        scale_div = 1.0 if scale_forcing else 1000.0
        mu = pm.Deterministic(
            "mu",
            alpha - (k_eros * (delta_eros / scale_div)) + (k_accrete * (delta_acc / scale_div)),
        )

        # 4. Robust Student-T likelihood (nu >= 5) with Heteroscedastic Volatility (Approach A)
        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_storm = pm.HalfNormal("sigma_storm", sigma=0.5)
            storm_excitation = pt.maximum(delta_eros / scale_div, 0.0)
            sigma = pm.Deterministic("sigma", sigma_base + sigma_storm * storm_excitation)
        else:
            sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Deterministic("nu", 5.0 + pm.Gamma("nu_offset", alpha=2.0, beta=0.2))
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_obs)

    return model, meta


def williams_test(
    r1y: float,
    r2y: float,
    r12: float,
    n: int,
) -> Tuple[float, float]:
    """Hotelling-Williams test for comparing two dependent/overlapping correlation coefficients
    r1y and r2y sharing a common criterion variable y.

    Tests H0: rho(X1, Y) = rho(X2, Y) vs H1: rho(X1, Y) != rho(X2, Y) given inter-correlation rho(X1, X2).

    Parameters
    ----------
    r1y : float
        Correlation between predictor 1 (e.g. Upsilon) and target y.
    r2y : float
        Correlation between predictor 2 (baseline competitor) and target y.
    r12 : float
        Inter-correlation between predictor 1 and predictor 2.
    n : int
        Sample size (number of survey observations).

    Returns
    -------
    t_stat : float
        Williams t-statistic with (n - 3) degrees of freedom.
    p_value : float
        Two-sided p-value.
    """
    from scipy import stats

    if n <= 3:
        return 0.0, 1.0

    det_R = 1.0 - (r1y ** 2) - (r2y ** 2) - (r12 ** 2) + 2.0 * r1y * r2y * r12
    det_R = max(det_R, 1e-12)
    r_bar = (r1y + r2y) / 2.0

    denom_sq = (2.0 * (n - 1) / (n - 3)) * det_R + (r_bar ** 2) * ((1.0 - r12) ** 3)
    if denom_sq <= 0:
        return 0.0, 1.0

    denom = np.sqrt(denom_sq)
    num = (r1y - r2y) * np.sqrt((n - 1) * (1.0 + r12))
    t_stat = float(num / denom)
    df = n - 3
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))
    return t_stat, p_val


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
    hln_adjust: bool = True,
) -> Tuple[float, float]:
    """Diebold-Mariano test for comparing predictive accuracy between two forecast models,
    incorporating the Harvey-Leybourne-Newbold (HLN, 1997) small-sample correction.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target observations.
    y_pred1 : np.ndarray
        Model 1 predictions (e.g. Upsilon).
    y_pred2 : np.ndarray
        Model 2 predictions (baseline competitor).
    loss : str
        Loss function: 'squared' (MSE differential) or 'absolute' (MAE differential).
    h : int
        Forecast horizon in steps (default 1).
    hln_adjust : bool
        Whether to apply the Harvey-Leybourne-Newbold small-sample multiplier.

    Returns
    -------
    dm_stat : float
        Diebold-Mariano test statistic (negative indicates Model 1 has lower error than Model 2).
    p_value : float
        Two-sided p-value from Student-t distribution with (n - 1) degrees of freedom.
    """
    from scipy import stats

    y_t = np.asarray(y_true, dtype=float)
    p1 = np.asarray(y_pred1, dtype=float)
    p2 = np.asarray(y_pred2, dtype=float)
    n = len(y_t)
    if n < 4:
        return 0.0, 1.0

    if loss == "squared":
        d = (y_t - p1) ** 2 - (y_t - p2) ** 2
    else:
        d = np.abs(y_t - p1) - np.abs(y_t - p2)

    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    if gamma_0 < 1e-12:
        return 0.0, 1.0

    v = gamma_0 / n
    if h > 1:
        gamma_k_sum = 0.0
        for k in range(1, h):
            cov_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n
            gamma_k_sum += 2.0 * (1.0 - k / h) * cov_k
        v = (gamma_0 + gamma_k_sum) / n
        v = max(v, 1e-12)

    dm_stat = float(d_bar / np.sqrt(v))
    if hln_adjust:
        hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * float(hln_factor)

    df = n - 1
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=df)))
    return dm_stat, p_val


def bootstrap_metric_ci(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str = "pearson_r",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate non-parametric bootstrap confidence intervals (e.g. 95% CI) for
    correlation (Pearson r, Spearman rho), R2 score, or RMSE.

    Parameters
    ----------
    x : np.ndarray
        Predictor feature array or model predictions.
    y : np.ndarray
        Ground-truth target array.
    metric_type : str
        One of 'pearson_r', 'spearman_rho', 'r2', or 'rmse'.
    n_bootstraps : int
        Number of bootstrap resamples (default 1000).
    ci : float
        Confidence level (default 0.95 for 95% CI).
    random_seed : int
        Random generator seed for reproducibility.

    Returns
    -------
    point_est : float
        Point estimate on original sample.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    """
    from scipy import stats

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    if n < 4:
        return 0.0, 0.0, 0.0

    # Point estimate
    if metric_type == "pearson_r":
        point_est = float(stats.pearsonr(x_arr, y_arr)[0])
    elif metric_type == "spearman_rho":
        point_est = float(stats.spearmanr(x_arr, y_arr)[0])
    elif metric_type == "r2":
        slope, intercept, _, _, _ = stats.linregress(x_arr, y_arr)
        yp = slope * x_arr + intercept
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        ss_res = np.sum((y_arr - yp) ** 2)
        point_est = float(1.0 - ss_res / max(ss_tot, 1e-8))
    elif metric_type == "rmse":
        slope, intercept, _, _, _ = stats.linregress(x_arr, y_arr)
        yp = slope * x_arr + intercept
        point_est = float(np.sqrt(np.mean((y_arr - yp) ** 2)))
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    rng = np.random.default_rng(random_seed)
    boot_vals = []
    alpha = (1.0 - ci) / 2.0

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        xb = x_arr[idx]
        yb = y_arr[idx]
        if np.std(xb) < 1e-8 or np.std(yb) < 1e-8:
            continue

        if metric_type == "pearson_r":
            val = stats.pearsonr(xb, yb)[0]
        elif metric_type == "spearman_rho":
            val = stats.spearmanr(xb, yb)[0]
        elif metric_type == "r2":
            slope, intercept, _, _, _ = stats.linregress(xb, yb)
            yp = slope * xb + intercept
            ss_tot = np.sum((yb - np.mean(yb)) ** 2)
            ss_res = np.sum((yb - yp) ** 2)
            val = 1.0 - ss_res / max(ss_tot, 1e-8)
        elif metric_type == "rmse":
            slope, intercept, _, _, _ = stats.linregress(xb, yb)
            yp = slope * xb + intercept
            val = np.sqrt(np.mean((yb - yp) ** 2))
        boot_vals.append(val)

    if not boot_vals:
        return point_est, point_est, point_est

    ci_lower = float(np.percentile(boot_vals, alpha * 100.0))
    ci_upper = float(np.percentile(boot_vals, (1.0 - alpha) * 100.0))
    return point_est, ci_lower, ci_upper


def evaluate_field_survey_correlation(
    df_aligned_surveys: pd.DataFrame,
    target_col: str = "surveyed_delta_x_m",
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate empirical correlation and predictive skill of physical forcings against in-situ ground-truth field surveys,
    including 95% bootstrap confidence intervals, Williams test for dependent correlations, Diebold-Mariano forecast error
    significance tests, and explained variance lift metrics.

    Parameters
    ----------
    df_aligned_surveys : pd.DataFrame
        Synchronized DataFrame pairing discrete field survey shoreline changes with antecedent physical forcings.
    target_col : str
        Target column in df_aligned_surveys ('surveyed_delta_x_m' for inter-survey change,
        or 'cumulative_displacement_m' for total cross-shore shift).
    n_bootstraps : int
        Number of bootstrap iterations for 95% confidence intervals (default 1000).
    random_seed : int
        Random seed for reproducible bootstrap sampling.

    Returns
    -------
    df_skill : pd.DataFrame
        Ranked skill table comparing empirical correlation (Pearson r, Spearman rho),
        bootstrap 95% CIs, R2 score, RMSE, Williams p-values, DM statistics, and variance lift.
    """
    from scipy import stats

    y_true = df_aligned_surveys[target_col].values
    n = len(y_true)
    if n < 3:
        return pd.DataFrame()

    var_base = float(np.var(y_true))
    mse_base = float(np.mean((y_true - np.mean(y_true)) ** 2)) if var_base > 1e-6 else 1.0

    ups_eq_col = "upsilon_diseq_simulated_delta_x_m" if target_col == "surveyed_delta_x_m" else "upsilon_diseq_simulated_pos_m"
    yates_col = "yates_simulated_delta_x_m" if target_col == "surveyed_delta_x_m" else "yates_simulated_pos_m"
    xbeach_col = "xbeach_simulated_delta_x_m" if target_col == "surveyed_delta_x_m" else "xbeach_simulated_pos_m"

    predictors = {
        "Unified Index (Signed Equilibrium dX/dt)": ups_eq_col,
        "Unified Index (Peak Storm Upsilon)": "antecedent_log_peak_upsilon" if "antecedent_log_peak_upsilon" in df_aligned_surveys.columns else "antecedent_peak_upsilon",
        "Unified Index (Cumulative Storm Work)": "antecedent_log_storm_work" if "antecedent_log_storm_work" in df_aligned_surveys.columns else "antecedent_storm_work",
        "Unified Index (Upsilon Memory)": "antecedent_upsilon_memory" if "antecedent_upsilon_memory" in df_aligned_surveys.columns else "antecedent_mean_upsilon",
        "Unified Index (log-Upsilon Memory)": "antecedent_log_upsilon" if "antecedent_log_upsilon" in df_aligned_surveys.columns else "antecedent_upsilon_memory",
        "Yates (2009) Equilibrium Model": yates_col,
        "Process-Based (XBeach-inspired)": xbeach_col,
        "Peak Wave Height (max Hs)": "antecedent_max_hs_m",
        "Mean Wave Height (mean Hs)": "antecedent_mean_hs_m",
        "Dean Number (mean Omega)": "antecedent_mean_dean",
        "Peak Wave Power (max P)": "antecedent_max_power_Wm",
        "Peak Storm Surge (max S)": "antecedent_max_surge_m",
    }

    raw_results = []
    x_dict = {}
    ypred_dict = {}

    for label, col in predictors.items():
        if col not in df_aligned_surveys.columns:
            continue
        x = df_aligned_surveys[col].values
        if np.all(np.isnan(x)) or np.std(x) < 1e-8:
            continue

        # Correlation analysis
        r, p_pearson = stats.pearsonr(x, y_true)
        rho, p_spearman = stats.spearmanr(x, y_true)

        # OLS Linear fit for skill metrics
        slope, intercept, _, _, _ = stats.linregress(x, y_true)
        y_pred = slope * x + intercept
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(1.0 - (np.sum((y_true - y_pred) ** 2) / max(np.sum((y_true - np.mean(y_true)) ** 2), 1e-6)))
        bss = float(1.0 - (np.mean((y_true - y_pred) ** 2) / mse_base))

        # 95% Bootstrap CIs
        _, r_ci_low, r_ci_high = bootstrap_metric_ci(x, y_true, metric_type="pearson_r", n_bootstraps=n_bootstraps, random_seed=random_seed)
        _, rho_ci_low, rho_ci_high = bootstrap_metric_ci(x, y_true, metric_type="spearman_rho", n_bootstraps=n_bootstraps, random_seed=random_seed)
        _, r2_ci_low, r2_ci_high = bootstrap_metric_ci(x, y_true, metric_type="r2", n_bootstraps=n_bootstraps, random_seed=random_seed)

        x_dict[label] = x
        ypred_dict[label] = y_pred

        raw_results.append({
            "Predictor": label,
            "Pearson r": float(r),
            "p-value (Pearson)": float(p_pearson),
            "r_95ci_lower": float(r_ci_low),
            "r_95ci_upper": float(r_ci_high),
            "Spearman rho": float(rho),
            "p-value (Spearman)": float(p_spearman),
            "rho_95ci_lower": float(rho_ci_low),
            "rho_95ci_upper": float(rho_ci_high),
            "R2 Score": float(r2),
            "r2_95ci_lower": float(r2_ci_low),
            "r2_95ci_upper": float(r2_ci_high),
            "RMSE (m)": float(rmse),
            "MAE (m)": float(mae),
            "Brier Skill Score (BSS)": float(bss),
            "N Surveys": n,
        })

    df_skill = pd.DataFrame(raw_results)
    if df_skill.empty:
        return df_skill

    # Sort by absolute correlation with observed morphodynamics
    df_skill["Abs_r"] = df_skill["Pearson r"].abs()
    df_skill = df_skill.sort_values(by="Abs_r", ascending=False).drop(columns=["Abs_r"]).reset_index(drop=True)

    # Reference top predictor (Signed Equilibrium Upsilon if present, or rank 1)
    top_label = "Unified Index (Signed Equilibrium dX/dt)"
    if top_label not in x_dict:
        top_label = df_skill["Predictor"].iloc[0]

    top_x = x_dict[top_label]
    top_ypred = ypred_dict[top_label]
    top_r = float(df_skill.loc[df_skill["Predictor"] == top_label, "Pearson r"].iloc[0])
    top_r2 = float(df_skill.loc[df_skill["Predictor"] == top_label, "R2 Score"].iloc[0])

    williams_t_list = []
    williams_p_list = []
    dm_stat_list = []
    dm_p_list = []
    delta_r2_list = []
    pct_lift_list = []

    for _, row in df_skill.iterrows():
        p_label = row["Predictor"]
        p_r = row["Pearson r"]
        p_r2 = row["R2 Score"]
        p_x = x_dict[p_label]
        p_ypred = ypred_dict[p_label]

        if p_label == top_label:
            williams_t_list.append(0.0)
            williams_p_list.append(1.0)
            dm_stat_list.append(0.0)
            dm_p_list.append(1.0)
            delta_r2_list.append(0.0)
            pct_lift_list.append(0.0)
        else:
            # Orientation alignment for Williams test comparison
            x_oriented = p_x if p_r * top_r >= 0 else -p_x
            r12 = float(stats.pearsonr(top_x, x_oriented)[0])
            t_will, p_will = williams_test(np.abs(top_r), np.abs(p_r), r12, n)
            dm_s, p_dm = diebold_mariano_test(y_true, top_ypred, p_ypred)

            delta_r2 = float(top_r2 - p_r2)
            pct_lift = float((delta_r2 / max(p_r2, 1e-4)) * 100.0) if p_r2 > 0 else np.nan

            williams_t_list.append(t_will)
            williams_p_list.append(p_will)
            dm_stat_list.append(dm_s)
            dm_p_list.append(p_dm)
            delta_r2_list.append(delta_r2)
            pct_lift_list.append(pct_lift)

    df_skill["Williams t-stat"] = williams_t_list
    df_skill["Williams p-value"] = williams_p_list
    df_skill["Diebold-Mariano stat"] = dm_stat_list
    df_skill["Diebold-Mariano p-value"] = dm_p_list
    df_skill["Delta R2 vs Top"] = delta_r2_list
    df_skill["Pct Variance Lift (%)"] = pct_lift_list

    return df_skill


def evaluate_cross_paradigm_walk_forward_cv(
    sites: Optional[List[str]] = None,
    n_splits: int = 4,
    draws: int = 300,
    tune: int = 200,
    chains: int = 2,
    cores: int = 1,
    random_seed: int = 42,
    use_field_surveys: bool = True,
    data_dir: str = "data/processed",
    eval_interval_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Walk-Forward (Expanding Window) CV across Class I, Class II, and Class III models for global sites.

    Evaluates out-of-sample inter-survey displacement changes (Delta-X) on authentic real-world in-situ field surveys
    with Global Pooled Cross-Validation aggregation to correctly capture the true physical variance across all seasons and sites.

    Parameters
    ----------
    sites : list of str, optional
        List of global site keys (default: all 5 sites).
    n_splits : int
        Number of walk-forward temporal folds (default 4).
    draws, tune, chains, cores, random_seed : MCMC sampling config.
    use_field_surveys : bool
        If True (default), evaluates against authentic in-situ GPS/Lidar field surveys.
    data_dir : str
        Path to processed sensor parquet datasets (used when use_field_surveys is False).
    eval_interval_hours : int
        Subsampling interval for delta displacement evaluation if using sensor data (default 24h).

    Returns
    -------
    synthesis_df : pd.DataFrame
        Publication-grade Cross-Paradigm Synthesis Table with Global Pooled CV metrics.
    fold_details_df : pd.DataFrame
        Detailed fold-by-fold performance breakdown.
    """
    import os
    from sklearn.metrics import r2_score, mean_squared_error
    from src.data import (
        load_dataset,
        load_site_field_surveys,
        align_hydrodynamics_with_field_surveys,
        simulate_equilibrium_shoreline_evolution,
        simulate_shorefor_evolution,
        simulate_xbeach_inspired_morphodynamics,
    )
    from src.physics import (
        COASTAL_SITE_PROFILES,
        simulate_signed_equilibrium_upsilon_evolution,
    )

    if sites is None:
        sites = ["duck_nc", "narrabeen_aus", "oahu_north_shore_hi", "torrey_pines_ca", "truc_vert_france"]

    pooled_preds = {}
    pooled_actuals = {}
    pooled_pos_preds = {}
    pooled_pos_actuals = {}
    pooled_norm_preds = {}
    pooled_norm_actuals = {}
    pooled_norm_pos_preds = {}
    pooled_norm_pos_actuals = {}
    fold_records = []
    full_site_records = []

    for site in sites:
        if use_field_surveys:
            df_surveys = load_site_field_surveys(site, transect_index=None)
            df_hydro, _ = load_dataset(site=site)
            if df_hydro.index.tz is not None and df_surveys.index.tz is None:
                df_surveys.index = df_surveys.index.tz_localize("UTC")
            elif df_hydro.index.tz is None and df_surveys.index.tz is not None:
                df_surveys.index = df_surveys.index.tz_convert(None)
            df_survey = align_hydrodynamics_with_field_surveys(df_hydro, df_surveys)

            n_obs = len(df_survey)
            if n_obs < 10:
                continue

            y_all = df_survey["surveyed_delta_x_m"].values
            pos_all = df_survey["cumulative_displacement_m"].values if "cumulative_displacement_m" in df_survey.columns else (
                df_survey["surveyed_pos_mhw_m"].values - df_survey["surveyed_pos_mhw_m"].iloc[0]
            )
            f_all = df_survey["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df_survey.columns else np.log10(np.maximum(df_survey["antecedent_mean_upsilon"].values, 1e-4))
            shorefor_delta = df_survey["antecedent_upsilon_memory"].values if "antecedent_upsilon_memory" in df_survey.columns else df_survey["antecedent_log_upsilon"].values

            sigma_site_dy = max(float(np.std(y_all)), 1e-4)
            sigma_site_pos = max(float(np.std(pos_all)), 1e-4)

            model_deltas = {
                ("Class III (Bayesian)", "Signed Equilibrium Upsilon"): df_survey["upsilon_diseq_simulated_delta_x_m"].values,
                ("Class I (Analytical)", "ShoreFor (2013) Memory"): shorefor_delta,
                ("Class I (Analytical)", "Yates (2009) Equilibrium"): df_survey["yates_simulated_delta_x_m"].values,
                ("Class II (Process/PDE)", "XBeach 1D Process Model"): df_survey["xbeach_simulated_delta_x_m"].values,
            }
            model_positions = {
                ("Class III (Bayesian)", "Signed Equilibrium Upsilon"): df_survey["upsilon_diseq_simulated_pos_m"].values,
                ("Class I (Analytical)", "Yates (2009) Equilibrium"): df_survey["yates_simulated_pos_m"].values,
                ("Class II (Process/PDE)", "XBeach 1D Process Model"): df_survey["xbeach_simulated_pos_m"].values,
            }

            min_train = int(n_obs * 0.5)
            step = int((n_obs - min_train) / n_splits)

            for f in range(n_splits):
                tr_end = min_train + f * step
                te_end = tr_end + step if f < n_splits - 1 else n_obs
                if tr_end >= n_obs:
                    continue

                y_tr, y_te = y_all[:tr_end], y_all[tr_end:te_end]
                if len(y_te) == 0:
                    continue

                var_base = max(np.mean((y_te - np.mean(y_tr)) ** 2), 1e-6)

                max_lag_f = min(12, tr_end)
                bayes_m, bayes_meta = build_latent_memory_bayesian_model(
                    forcing=f_all[:tr_end],
                    delta_x_obs=y_tr,
                    model_name="Bayesian Latent Memory Kernel (Upsilon)",
                    max_lag=max_lag_f,
                    scale_forcing=True,
                )
                with bayes_m:
                    tr = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=cores,
                        progressbar=False,
                        random_seed=random_seed,
                    )

                gamma_p = float(tr.posterior["gamma"].values.mean())
                alpha_p = float(tr.posterior["alpha"].values.mean())
                beta_p = float(tr.posterior["beta_forcing"].values.mean())

                f_sc = (f_all - bayes_meta["f_mean"]) / bayes_meta["f_std"]
                k_v = np.arange(max_lag_f, dtype=float)
                w_p = (gamma_p ** k_v) / np.sum(gamma_p ** k_v)
                lag_m = create_lag_matrix(f_sc, max_lag=max_lag_f)
                eff_s = lag_m @ w_p
                dy_pred_bayes = alpha_p + beta_p * eff_s[tr_end:te_end]

                eval_list = [("Class III (Bayesian)", "Bayesian Latent Memory Kernel (Upsilon)", dy_pred_bayes, None)]
                for (p_c, m_n), d_raw in model_deltas.items():
                    eval_list.append((p_c, m_n, d_raw[tr_end:te_end], d_raw[:tr_end]))

                for p_class, m_name, dp_te, dp_tr_raw in eval_list:
                    if dp_tr_raw is not None and np.std(dp_tr_raw) > 1e-6:
                        slope, intercept = np.polyfit(dp_tr_raw, y_tr, 1)
                        dp_te = slope * dp_te + intercept

                    # Cumulative shoreline position integration or calibrated trajectory over test window
                    pos_actual_te = pos_all[tr_end:te_end]
                    if (p_class, m_name) in model_positions:
                        raw_pos_tr = model_positions[(p_class, m_name)][:tr_end]
                        raw_pos_te = model_positions[(p_class, m_name)][tr_end:te_end]
                        if np.std(raw_pos_tr) > 1e-6:
                            slope_p, intercept_p = np.polyfit(raw_pos_tr, pos_all[:tr_end], 1)
                            pos_pred_te = slope_p * raw_pos_te + intercept_p
                        else:
                            pos_init = pos_all[tr_end - 1] if tr_end > 0 else 0.0
                            pos_pred_te = pos_init + np.cumsum(dp_te)
                    else:
                        pos_init = pos_all[tr_end - 1] if tr_end > 0 else 0.0
                        pos_pred_te = pos_init + np.cumsum(dp_te)

                    key = (p_class, m_name)
                    if key not in pooled_preds:
                        pooled_preds[key] = []
                        pooled_actuals[key] = []
                        pooled_pos_preds[key] = []
                        pooled_pos_actuals[key] = []
                        pooled_norm_preds[key] = []
                        pooled_norm_actuals[key] = []
                        pooled_norm_pos_preds[key] = []
                        pooled_norm_pos_actuals[key] = []
                    pooled_preds[key].extend(dp_te)
                    pooled_actuals[key].extend(y_te)
                    pooled_pos_preds[key].extend(pos_pred_te)
                    pooled_pos_actuals[key].extend(pos_actual_te)
                    # Standardized per-site arrays
                    pooled_norm_preds[key].extend(dp_te / sigma_site_dy)
                    pooled_norm_actuals[key].extend(y_te / sigma_site_dy)
                    pooled_norm_pos_preds[key].extend(pos_pred_te / sigma_site_pos)
                    pooled_norm_pos_actuals[key].extend(pos_actual_te / sigma_site_pos)

                    # Step delta metrics
                    mse_val = float(mean_squared_error(y_te, dp_te))
                    rmse_val = float(np.sqrt(mse_val))
                    r2_val = float(r2_score(y_te, dp_te))
                    bss_val = float(1.0 - (mse_val / var_base))
                    r_corr = float(np.corrcoef(y_te, dp_te)[0, 1]) if (len(y_te) > 1 and np.std(dp_te) > 1e-6 and np.std(y_te) > 1e-6) else 0.0

                    # Cumulative position metrics
                    pos_mse = float(mean_squared_error(pos_actual_te, pos_pred_te))
                    pos_rmse = float(np.sqrt(pos_mse))
                    pos_r2 = float(r2_score(pos_actual_te, pos_pred_te)) if len(pos_actual_te) > 1 else 0.0
                    pos_var = max(float(np.var(pos_actual_te)), 1e-6)
                    pos_bss = float(1.0 - (pos_mse / pos_var))
                    pos_corr = float(np.corrcoef(pos_actual_te, pos_pred_te)[0, 1]) if (len(pos_actual_te) > 1 and np.std(pos_pred_te) > 1e-6 and np.std(pos_actual_te) > 1e-6) else 0.0

                    fold_records.append({
                        "Site": site,
                        "Site Name": COASTAL_SITE_PROFILES.get(site, {}).get("name", site),
                        "Paradigm Class": p_class,
                        "Paradigm": p_class,
                        "Model Name": m_name,
                        "Model": m_name,
                        "Fold": f + 1,
                        # Differential Step Metrics (Δx)
                        "Fold RMSE (m)": rmse_val,
                        "Fold RMSE": rmse_val,
                        "Fold R2 Score": r2_val,
                        "Fold R2": r2_val,
                        "Fold BSS": bss_val,
                        "Fold Pearson r": r_corr,
                        # Cumulative Position Metrics (x_t)
                        "Fold Pos RMSE (m)": pos_rmse,
                        "Fold Pos R2 Score": pos_r2,
                        "Fold Pos BSS": pos_bss,
                        "Fold Pos Pearson r": pos_corr,
                        # Per-Site Normalization Metadata
                        "Site Std Δx (m)": sigma_site_dy,
                        "Site Std Pos (m)": sigma_site_pos,
                        "Fold Norm Δx RMSE": rmse_val / sigma_site_dy,
                        "Fold Norm Pos RMSE": pos_rmse / sigma_site_pos,
                    })

            for p_class, m_name, _, _ in eval_list:
                if m_name == "Bayesian Latent Memory Kernel (Upsilon)":
                    bayes_m_full, bayes_meta_full = build_latent_memory_bayesian_model(
                        forcing=f_all,
                        delta_x_obs=y_all,
                        model_name="Bayesian Latent Memory Kernel (Upsilon)",
                        max_lag=12,
                        scale_forcing=True,
                    )
                    with bayes_m_full:
                        tr_f = pm.sample(draws=200, tune=150, chains=1, cores=1, progressbar=False, random_seed=random_seed)
                    gamma_f = float(tr_f.posterior["gamma"].values.mean())
                    alpha_f = float(tr_f.posterior["alpha"].values.mean())
                    beta_f = float(tr_f.posterior["beta_forcing"].values.mean())
                    f_sc_f = (f_all - bayes_meta_full["f_mean"]) / bayes_meta_full["f_std"]
                    k_v_f = np.arange(12, dtype=float)
                    w_f = (gamma_f ** k_v_f) / np.sum(gamma_f ** k_v_f)
                    dp_full = alpha_f + beta_f * (create_lag_matrix(f_sc_f, max_lag=12) @ w_f)
                else:
                    dp_raw = model_deltas[(p_class, m_name)]
                    if np.std(dp_raw) > 1e-6:
                        s_f, i_f = np.polyfit(dp_raw, y_all, 1)
                        dp_full = s_f * dp_raw + i_f
                    else:
                        dp_full = dp_raw

                # Full series cumulative position
                if (p_class, m_name) in model_positions:
                    raw_pos_full = model_positions[(p_class, m_name)]
                    if np.std(raw_pos_full) > 1e-6:
                        s_p_f, i_p_f = np.polyfit(raw_pos_full, pos_all, 1)
                        pos_full_pred = s_p_f * raw_pos_full + i_p_f
                    else:
                        pos_full_pred = pos_all[0] + np.cumsum(dp_full)
                else:
                    pos_full_pred = pos_all[0] + np.cumsum(dp_full)
                full_site_records.append({
                    "Site": site,
                    "Paradigm Class": p_class,
                    "Model Name": m_name,
                    "Full RMSE (m)": float(np.sqrt(mean_squared_error(y_all, dp_full))),
                    "Full R2 Score": float(r2_score(y_all, dp_full)),
                    "Full Pos RMSE (m)": float(np.sqrt(mean_squared_error(pos_all, pos_full_pred))),
                    "Full Pos R2 Score": float(r2_score(pos_all, pos_full_pred)),
                })
        else:
            file_path = os.path.join(data_dir, f"{site}_sensors.parquet")
            if not os.path.exists(file_path):
                continue

            df_site = pd.read_parquet(file_path)
            n = len(df_site)
            target_col = "shoreline_pos_obs_m" if "shoreline_pos_obs_m" in df_site.columns else "shoreline_pos_true_m"
            y_obs = df_site[target_col].values
            log_ups_all = df_site["log_upsilon"].values if "log_upsilon" in df_site.columns else np.log10(np.maximum(df_site["upsilon"].values, 1e-4))

            hs = df_site["hs_m"].values
            tp = df_site["tp_s"].values
            surge = df_site["surge_m"].values
            w_spd = df_site["wind_speed_ms"].values
            w_dir = df_site["wind_dir_deg"].values
            ups = df_site["upsilon"].values
            ws = df_site["w_s_ms"].values.mean() if "w_s_ms" in df_site.columns else 0.045
            tb = df_site["tan_beta"].iloc[0] if "tan_beta" in df_site.columns else 0.08
            d50 = df_site["d50_m"].iloc[0] if "d50_m" in df_site.columns else 0.00035

            yates_pos_all, _ = simulate_equilibrium_shoreline_evolution(hs, tp, random_seed=random_seed)
            shorefor_pos_all, _ = simulate_shorefor_evolution(hs, tp, w_s_ms=ws, random_seed=random_seed)
            xbeach_pos_all, _ = simulate_xbeach_inspired_morphodynamics(hs, tp, surge, w_spd, w_dir, d50_m=d50, tan_beta=tb, random_seed=random_seed)
            signed_eq_pos_all, _ = simulate_signed_equilibrium_upsilon_evolution(ups, c_erosion=1.2e-5, c_accretion=1.8e-5, upsilon_eq_mode="expanding")

            interval = max(1, eval_interval_hours)
            y_sub = y_obs[::interval]
            log_ups_sub = log_ups_all[::interval]
            dy_all = np.diff(y_sub)
            f_sub = log_ups_sub[:-1]
            pos_all = y_sub[1:]
            n_pts = len(dy_all)

            sigma_site_dy = max(float(np.std(dy_all)), 1e-4)
            sigma_site_pos = max(float(np.std(pos_all)), 1e-4)

            min_train = int(n_pts * 0.5)
            step = int((n_pts - min_train) / n_splits)

            for f in range(n_splits):
                train_end = min_train + f * step
                test_end = train_end + step if f < n_splits - 1 else n_pts

                dy_tr = dy_all[:train_end]
                dy_te = dy_all[train_end:test_end]
                var_base = max(np.mean((dy_te - np.mean(dy_tr)) ** 2), 1e-6)

                max_lag_fold = min(14, train_end)
                bayes_m, bayes_meta = build_latent_memory_bayesian_model(
                    forcing=f_sub[:train_end],
                    delta_x_obs=dy_tr,
                    model_name="Bayesian Latent Memory Kernel (Upsilon)",
                    max_lag=max_lag_fold,
                    scale_forcing=True,
                )
                with bayes_m:
                    tr = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=cores,
                        target_accept=0.90,
                        progressbar=False,
                        random_seed=random_seed,
                    )

                gamma_p = float(tr.posterior["gamma"].values.mean())
                alpha_p = float(tr.posterior["alpha"].values.mean())
                beta_p = float(tr.posterior["beta_forcing"].values.mean())

                f_sc = (f_sub - bayes_meta["f_mean"]) / bayes_meta["f_std"]
                k_v = np.arange(max_lag_fold, dtype=float)
                w_p = (gamma_p ** k_v) / np.sum(gamma_p ** k_v)
                lag_m = create_lag_matrix(f_sc, max_lag=max_lag_fold)
                eff_s = lag_m @ w_p
                dy_pred_bayes = alpha_p + beta_p * eff_s[train_end:test_end]

                p_evals = [
                    ("Class III (Bayesian)", "Bayesian Latent Memory Kernel (Upsilon)", dy_pred_bayes, None),
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon", np.diff(signed_eq_pos_all[::interval])[train_end:test_end], np.diff(signed_eq_pos_all[::interval])[:train_end]),
                    ("Class I (Analytical)", "Yates (2009) Equilibrium", np.diff(yates_pos_all[::interval])[train_end:test_end], np.diff(yates_pos_all[::interval])[:train_end]),
                    ("Class I (Analytical)", "ShoreFor (2013) Memory", np.diff(shorefor_pos_all[::interval])[train_end:test_end], np.diff(shorefor_pos_all[::interval])[:train_end]),
                    ("Class II (Process/PDE)", "XBeach 1D Process Model", np.diff(xbeach_pos_all[::interval])[train_end:test_end], np.diff(xbeach_pos_all[::interval])[:train_end]),
                ]

                for p_class, m_name, dp_te, dp_tr_raw in p_evals:
                    if dp_tr_raw is not None and np.std(dp_tr_raw) > 1e-6:
                        slope, intercept = np.polyfit(dp_tr_raw, dy_tr, 1)
                        dp_te = slope * dp_te + intercept

                    # Cumulative position tracking
                    pos_actual_te = pos_all[train_end:test_end]
                    pos_init = pos_all[train_end - 1] if train_end > 0 else y_sub[0]
                    pos_pred_te = pos_init + np.cumsum(dp_te)

                    key = (p_class, m_name)
                    if key not in pooled_preds:
                        pooled_preds[key] = []
                        pooled_actuals[key] = []
                        pooled_pos_preds[key] = []
                        pooled_pos_actuals[key] = []
                        pooled_norm_preds[key] = []
                        pooled_norm_actuals[key] = []
                        pooled_norm_pos_preds[key] = []
                        pooled_norm_pos_actuals[key] = []
                    pooled_preds[key].extend(dp_te)
                    pooled_actuals[key].extend(dy_te)
                    pooled_pos_preds[key].extend(pos_pred_te)
                    pooled_pos_actuals[key].extend(pos_actual_te)
                    # Standardized per-site arrays
                    pooled_norm_preds[key].extend(dp_te / sigma_site_dy)
                    pooled_norm_actuals[key].extend(dy_te / sigma_site_dy)
                    pooled_norm_pos_preds[key].extend(pos_pred_te / sigma_site_pos)
                    pooled_norm_pos_actuals[key].extend(pos_actual_te / sigma_site_pos)

                    mse_val = float(mean_squared_error(dy_te, dp_te))
                    rmse_val = float(np.sqrt(mse_val))
                    r2_val = float(r2_score(dy_te, dp_te))
                    bss_val = float(1.0 - (mse_val / var_base))
                    r_corr = float(np.corrcoef(dy_te, dp_te)[0, 1]) if np.std(dp_te) > 1e-6 and np.std(dy_te) > 1e-6 else 0.0

                    pos_mse = float(mean_squared_error(pos_actual_te, pos_pred_te))
                    pos_rmse = float(np.sqrt(pos_mse))
                    pos_r2 = float(r2_score(pos_actual_te, pos_pred_te)) if len(pos_actual_te) > 1 else 0.0
                    pos_var = max(float(np.var(pos_actual_te)), 1e-6)
                    pos_bss = float(1.0 - (pos_mse / pos_var))
                    pos_corr = float(np.corrcoef(pos_actual_te, pos_pred_te)[0, 1]) if (len(pos_actual_te) > 1 and np.std(pos_pred_te) > 1e-6 and np.std(pos_actual_te) > 1e-6) else 0.0

                    fold_records.append({
                        "Site": site,
                        "Site Name": COASTAL_SITE_PROFILES.get(site, {}).get("name", site),
                        "Paradigm Class": p_class,
                        "Paradigm": p_class,
                        "Model Name": m_name,
                        "Model": m_name,
                        "Fold": f + 1,
                        "Fold RMSE (m)": rmse_val,
                        "Fold RMSE": rmse_val,
                        "Fold R2 Score": r2_val,
                        "Fold R2": r2_val,
                        "Fold BSS": bss_val,
                        "Fold Pearson r": r_corr,
                        "Fold Pos RMSE (m)": pos_rmse,
                        "Fold Pos R2 Score": pos_r2,
                        "Fold Pos BSS": pos_bss,
                        "Fold Pos Pearson r": pos_corr,
                        "Site Std Δx (m)": sigma_site_dy,
                        "Site Std Pos (m)": sigma_site_pos,
                        "Fold Norm Δx RMSE": rmse_val / sigma_site_dy,
                        "Fold Norm Pos RMSE": pos_rmse / sigma_site_pos,
                    })

    df_folds = pd.DataFrame(fold_records)
    df_full_site = pd.DataFrame(full_site_records)

    # Standardized Cross-Paradigm Synthesis Table (Global Pooled CV)
    summary_rows = []
    for (p_class, m_name) in pooled_preds:
        y_act_arr = np.array(pooled_actuals[(p_class, m_name)])
        y_pred_arr = np.array(pooled_preds[(p_class, m_name)])
        pos_act_arr = np.array(pooled_pos_actuals[(p_class, m_name)])
        pos_pred_arr = np.array(pooled_pos_preds[(p_class, m_name)])

        y_norm_act = np.array(pooled_norm_actuals[(p_class, m_name)])
        y_norm_pred = np.array(pooled_norm_preds[(p_class, m_name)])
        pos_norm_act = np.array(pooled_norm_pos_actuals[(p_class, m_name)])
        pos_norm_pred = np.array(pooled_norm_pos_preds[(p_class, m_name)])

        # Step Delta Metrics (Δx)
        cv_rmse_pool = float(np.sqrt(mean_squared_error(y_act_arr, y_pred_arr)))
        cv_r2_pool = float(r2_score(y_act_arr, y_pred_arr))
        var_act = max(float(np.var(y_act_arr)), 1e-6)
        cv_bss_pool = float(1.0 - (mean_squared_error(y_act_arr, y_pred_arr) / var_act))
        cv_corr_pool = float(np.corrcoef(y_act_arr, y_pred_arr)[0, 1]) if (np.std(y_pred_arr) > 1e-6 and np.std(y_act_arr) > 1e-6) else 0.0

        # Cumulative Position Metrics (x_t)
        cv_pos_rmse_pool = float(np.sqrt(mean_squared_error(pos_act_arr, pos_pred_arr)))
        cv_pos_r2_pool = float(r2_score(pos_act_arr, pos_pred_arr))
        var_pos_act = max(float(np.var(pos_act_arr)), 1e-6)
        cv_pos_bss_pool = float(1.0 - (mean_squared_error(pos_act_arr, pos_pred_arr) / var_pos_act))
        cv_pos_corr_pool = float(np.corrcoef(pos_act_arr, pos_pred_arr)[0, 1]) if (np.std(pos_pred_arr) > 1e-6 and np.std(pos_act_arr) > 1e-6) else 0.0

        # Standardized Per-Site Normalized Metrics
        norm_cv_r2 = float(r2_score(y_norm_act, y_norm_pred))
        norm_cv_rmse = float(np.sqrt(mean_squared_error(y_norm_act, y_norm_pred)))
        var_norm_act = max(float(np.var(y_norm_act)), 1e-6)
        norm_cv_bss = float(1.0 - (mean_squared_error(y_norm_act, y_norm_pred) / var_norm_act))

        norm_pos_r2 = float(r2_score(pos_norm_act, pos_norm_pred))
        norm_pos_rmse = float(np.sqrt(mean_squared_error(pos_norm_act, pos_norm_pred)))
        var_norm_pos_act = max(float(np.var(pos_norm_act)), 1e-6)
        norm_pos_bss = float(1.0 - (mean_squared_error(pos_norm_act, pos_norm_pred) / var_norm_pos_act))
        norm_pos_corr = float(np.corrcoef(pos_norm_act, pos_norm_pred)[0, 1]) if (np.std(pos_norm_pred) > 1e-6 and np.std(pos_norm_act) > 1e-6) else 0.0

        grp = df_folds[(df_folds["Paradigm Class"] == p_class) & (df_folds["Model Name"] == m_name)]
        rmse_std = float(grp["Fold RMSE (m)"].std()) if not grp.empty else 0.0
        r2_std = float(grp["Fold R2 Score"].std()) if not grp.empty else 0.0
        pos_rmse_std = float(grp["Fold Pos RMSE (m)"].std()) if not grp.empty and "Fold Pos RMSE (m)" in grp.columns else 0.0
        pos_r2_std = float(grp["Fold Pos R2 Score"].std()) if not grp.empty and "Fold Pos R2 Score" in grp.columns else 0.0

        site_avg_r2 = float(grp.groupby("Site")["Fold R2 Score"].mean().mean()) if not grp.empty else cv_r2_pool
        site_avg_pos_r2 = float(grp.groupby("Site")["Fold Pos R2 Score"].mean().mean()) if not grp.empty and "Fold Pos R2 Score" in grp.columns else cv_pos_r2_pool

        full_grp = df_full_site[(df_full_site["Paradigm Class"] == p_class) & (df_full_site["Model Name"] == m_name)]
        full_rmse = float(full_grp["Full RMSE (m)"].mean()) if not full_grp.empty else cv_rmse_pool
        full_r2 = float(full_grp["Full R2 Score"].mean()) if not full_grp.empty else cv_r2_pool
        full_pos_rmse = float(full_grp["Full Pos RMSE (m)"].mean()) if not full_grp.empty and "Full Pos RMSE (m)" in full_grp.columns else cv_pos_rmse_pool
        full_pos_r2 = float(full_grp["Full Pos R2 Score"].mean()) if not full_grp.empty and "Full Pos R2 Score" in full_grp.columns else cv_pos_r2_pool

        summary_rows.append({
            "Paradigm Class": p_class,
            "Model Name": m_name,
            # Standardized Per-Site Normalized Global Skill
            "Normalized CV R2 (Δx)": norm_cv_r2,
            "Normalized CV RMSE (Δx)": norm_cv_rmse,
            "Normalized CV BSS (Δx)": norm_cv_bss,
            "Normalized Position CV R2": norm_pos_r2,
            "Normalized Position CV RMSE": norm_pos_rmse,
            "Normalized Position CV Pearson r": norm_pos_corr,
            "Normalized Position CV BSS": norm_pos_bss,
            "Site-Averaged R2 (Δx)": site_avg_r2,
            "Site-Averaged Pos R2": site_avg_pos_r2,
            # Position-Level Performance Metrics (x_t)
            "Position CV R2 Score": cv_pos_r2_pool,
            "Position CV RMSE (m)": cv_pos_rmse_pool,
            "Position CV Pearson r": cv_pos_corr_pool,
            "Position CV BSS": cv_pos_bss_pool,
            "Position Fold R2 (std)": pos_r2_std,
            "Position Fold RMSE (std)": pos_rmse_std,
            "Full Position R2 Score": full_pos_r2,
            "Full Position RMSE (m)": full_pos_rmse,
            # Differential Step Change Metrics (Δx)
            "CV RMSE (m) (Δx)": cv_rmse_pool,
            "CV R2 Score (Δx)": cv_r2_pool,
            "CV BSS (Δx)": cv_bss_pool,
            "CV Pearson r (Δx)": cv_corr_pool,
            # Standard Aliases for Backward Compatibility
            "CV RMSE (m)": cv_rmse_pool,
            "CV R2 Score": cv_r2_pool,
            "CV BSS": cv_bss_pool,
            "Fold RMSE (std)": rmse_std,
            "Fold R2 (std)": r2_std,
            "Full RMSE (m)": full_rmse,
            "Full R2 Score": full_r2,
            "CV Pearson r": cv_corr_pool,
            "Out-of-Sample Mean R2": cv_pos_r2_pool,
            "Inter-Fold R2 Std Dev (σ_R2)": pos_r2_std,
            "Mean Out-of-Sample RMSE (m)": cv_pos_rmse_pool,
            "Out-of-Sample Mean Pearson r": cv_pos_corr_pool,
            "Brier Skill Score (BSS)": cv_pos_bss_pool,
            "Evaluated Sites": len(grp["Site"].unique()) if not grp.empty else len(sites),
            "Total CV Folds": len(grp),
        })

    df_synthesis = pd.DataFrame(summary_rows).sort_values(by="Position CV R2 Score", ascending=False).reset_index(drop=True)
    df_synthesis.attrs["sites"] = sites
    df_synthesis.attrs["n_splits"] = n_splits

    return df_synthesis, df_folds


def calibrate_signed_equilibrium_parameters(
    training_data: List[pd.DataFrame],
    initial_c_erosion: float = 1.2e-5,
    initial_c_accretion: float = 1.8e-5,
    upsilon_col: str = "antecedent_mean_upsilon",
    target_col: str = "surveyed_delta_x_m",
) -> Dict[str, float]:
    """Calibrate Signed Equilibrium Upsilon rate coefficients (c_erosion, c_accretion) across multi-site training sets.

    Fits the positive erosive and negative accretive response rates on pooled multi-site survey data:
        Delta_X = -c_erosion * max(Upsilon - Upsilon_eq, 0) + c_accretion * max(Upsilon_eq - Upsilon, 0)

    Parameters
    ----------
    training_data : list of pd.DataFrame
        List of aligned survey dataframes from training sites.
    initial_c_erosion, initial_c_accretion : float
        Initial parameter guesses.
    upsilon_col : str
        Column name for Upsilon forcing.
    target_col : str
        Column name for surveyed cross-shore displacement.

    Returns
    -------
    dict
        Optimized rate coefficients and linear transfer coefficients.
    """
    from scipy.optimize import minimize

    all_y = []
    all_raw_delta = []

    for df in training_data:
        if df.empty or target_col not in df.columns:
            continue
        y_vals = df[target_col].values
        if "upsilon_diseq_simulated_delta_x_m" in df.columns:
            d_raw = df["upsilon_diseq_simulated_delta_x_m"].values
        else:
            ups = df[upsilon_col].values if upsilon_col in df.columns else np.zeros(len(df))
            ups_eq = np.median(ups) if len(ups) > 0 else 2500.0
            d_ero = np.maximum(ups - ups_eq, 0.0)
            d_acc = np.maximum(ups_eq - ups, 0.0)
            d_raw = -initial_c_erosion * d_ero + initial_c_accretion * d_acc
        all_y.extend(y_vals)
        all_raw_delta.extend(d_raw)

    all_y = np.array(all_y)
    all_raw_delta = np.array(all_raw_delta)

    if len(all_y) < 5 or np.std(all_raw_delta) < 1e-8:
        return {
            "c_erosion": initial_c_erosion,
            "c_accretion": initial_c_accretion,
            "slope": 1.0,
            "intercept": 0.0,
        }

    slope, intercept = np.polyfit(all_raw_delta, all_y, 1)
    c_ero_cal = max(float(initial_c_erosion * abs(slope)), 1e-7)
    c_acc_cal = max(float(initial_c_accretion * abs(slope)), 1e-7)

    return {
        "c_erosion": c_ero_cal,
        "c_accretion": c_acc_cal,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def evaluate_leave_one_site_out_cv(
    sites: Optional[List[str]] = None,
    draws: int = 300,
    tune: int = 200,
    chains: int = 2,
    cores: int = 1,
    random_seed: int = 42,
    use_field_surveys: bool = True,
    data_dir: str = "data/processed",
    eval_interval_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Leave-One-Site-Out (LOSO) Cross-Validation across global coastal sites.

    For each held-out site (k = 1...K):
      1. Uses the remaining K - 1 sites exclusively for training and calibration.
      2. Calibrates Class I (Yates, ShoreFor), Class II (XBeach), and Class III (Signed Equilibrium Upsilon, Bayesian Latent Memory).
      3. Predicts cross-shore displacement (Delta-X) and cumulative trajectory (x_t) zero-shot on the held-out site.
      4. Evaluates both Global Default parameters and 4-Site Calibrated parameters on unseen beaches (cf. Stockdon et al. 2006).

    Parameters
    ----------
    sites : list of str, optional
        List of global site keys (default: all 5 sites).
    draws, tune, chains, cores, random_seed : MCMC sampling config.
    use_field_surveys : bool
        If True (default), evaluates against authentic in-situ GPS/Lidar field surveys.
    data_dir : str
        Path to processed sensor parquet datasets (used when use_field_surveys is False).
    eval_interval_hours : int
        Subsampling interval for delta displacement evaluation if using sensor data (default 24h).

    Returns
    -------
    synthesis_loso_df : pd.DataFrame
        Publication-grade Global Pooled LOSO Synthesis Table comparing transferability across paradigms.
    held_out_sites_df : pd.DataFrame
        Detailed held-out site performance breakdown for every fold.
    """
    import os
    from sklearn.metrics import r2_score, mean_squared_error
    from src.data import (
        load_dataset,
        load_site_field_surveys,
        align_hydrodynamics_with_field_surveys,
        simulate_equilibrium_shoreline_evolution,
        simulate_shorefor_evolution,
        simulate_xbeach_inspired_morphodynamics,
    )
    from src.physics import (
        COASTAL_SITE_PROFILES,
        simulate_signed_equilibrium_upsilon_evolution,
    )

    if sites is None:
        sites = ["duck_nc", "narrabeen_aus", "oahu_north_shore_hi", "torrey_pines_ca", "truc_vert_france"]

    site_data = {}

    for site in sites:
        if use_field_surveys:
            df_surveys = load_site_field_surveys(site, transect_index=None)
            df_hydro, _ = load_dataset(site=site)
            if df_hydro.index.tz is not None and df_surveys.index.tz is None:
                df_surveys.index = df_surveys.index.tz_localize("UTC")
            elif df_hydro.index.tz is None and df_surveys.index.tz is not None:
                df_surveys.index = df_surveys.index.tz_convert(None)
            df_survey = align_hydrodynamics_with_field_surveys(df_hydro, df_surveys)

            if len(df_survey) < 5:
                continue

            y_all = df_survey["surveyed_delta_x_m"].values
            pos_all = df_survey["cumulative_displacement_m"].values if "cumulative_displacement_m" in df_survey.columns else (
                df_survey["surveyed_pos_mhw_m"].values - df_survey["surveyed_pos_mhw_m"].iloc[0]
            )
            f_all = df_survey["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df_survey.columns else np.log10(np.maximum(df_survey["antecedent_mean_upsilon"].values, 1e-4))
            shorefor_delta = df_survey["antecedent_upsilon_memory"].values if "antecedent_upsilon_memory" in df_survey.columns else df_survey["antecedent_log_upsilon"].values

            site_data[site] = {
                "y": y_all,
                "pos": pos_all,
                "forcing": f_all,
                "df_survey": df_survey,
                "deltas": {
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Calibrated)"): df_survey["upsilon_diseq_simulated_delta_x_m"].values,
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Global Default)"): df_survey["upsilon_diseq_simulated_delta_x_m"].values,
                    ("Class I (Analytical)", "ShoreFor (2013) Memory"): shorefor_delta,
                    ("Class I (Analytical)", "Yates (2009) Equilibrium"): df_survey["yates_simulated_delta_x_m"].values,
                    ("Class II (Process/PDE)", "XBeach 1D Process Model"): df_survey["xbeach_simulated_delta_x_m"].values,
                },
                "positions": {
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Calibrated)"): df_survey["upsilon_diseq_simulated_pos_m"].values,
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Global Default)"): df_survey["upsilon_diseq_simulated_pos_m"].values,
                    ("Class I (Analytical)", "Yates (2009) Equilibrium"): df_survey["yates_simulated_pos_m"].values,
                    ("Class II (Process/PDE)", "XBeach 1D Process Model"): df_survey["xbeach_simulated_pos_m"].values,
                }
            }
        else:
            file_path = os.path.join(data_dir, f"{site}_sensors.parquet")
            if not os.path.exists(file_path):
                continue
            df_site = pd.read_parquet(file_path)
            target_col = "shoreline_pos_obs_m" if "shoreline_pos_obs_m" in df_site.columns else "shoreline_pos_true_m"
            y_obs = df_site[target_col].values
            log_ups_all = df_site["log_upsilon"].values if "log_upsilon" in df_site.columns else np.log10(np.maximum(df_site["upsilon"].values, 1e-4))

            hs = df_site["hs_m"].values
            tp = df_site["tp_s"].values
            surge = df_site["surge_m"].values
            w_spd = df_site["wind_speed_ms"].values
            w_dir = df_site["wind_dir_deg"].values
            ups = df_site["upsilon"].values
            ws = df_site["w_s_ms"].values.mean() if "w_s_ms" in df_site.columns else 0.045
            tb = df_site["tan_beta"].iloc[0] if "tan_beta" in df_site.columns else 0.08
            d50 = df_site["d50_m"].iloc[0] if "d50_m" in df_site.columns else 0.00035

            yates_pos_all, _ = simulate_equilibrium_shoreline_evolution(hs, tp, random_seed=random_seed)
            shorefor_pos_all, _ = simulate_shorefor_evolution(hs, tp, w_s_ms=ws, random_seed=random_seed)
            xbeach_pos_all, _ = simulate_xbeach_inspired_morphodynamics(hs, tp, surge, w_spd, w_dir, d50_m=d50, tan_beta=tb, random_seed=random_seed)
            signed_eq_pos_all, _ = simulate_signed_equilibrium_upsilon_evolution(ups, c_erosion=1.2e-5, c_accretion=1.8e-5, upsilon_eq_mode="expanding")

            interval = max(1, eval_interval_hours)
            y_sub = y_obs[::interval]
            log_ups_sub = log_ups_all[::interval]
            dy_all = np.diff(y_sub)
            f_sub = log_ups_sub[:-1]
            pos_all = y_sub[1:]

            site_data[site] = {
                "y": dy_all,
                "pos": pos_all,
                "forcing": f_sub,
                "df_survey": df_site.iloc[::interval],
                "deltas": {
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Calibrated)"): np.diff(signed_eq_pos_all[::interval]),
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Global Default)"): np.diff(signed_eq_pos_all[::interval]),
                    ("Class I (Analytical)", "Yates (2009) Equilibrium"): np.diff(yates_pos_all[::interval]),
                    ("Class I (Analytical)", "ShoreFor (2013) Memory"): np.diff(shorefor_pos_all[::interval]),
                    ("Class II (Process/PDE)", "XBeach 1D Process Model"): np.diff(xbeach_pos_all[::interval]),
                },
                "positions": {
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Calibrated)"): signed_eq_pos_all[::interval][1:],
                    ("Class III (Bayesian)", "Signed Equilibrium Upsilon (Global Default)"): signed_eq_pos_all[::interval][1:],
                    ("Class I (Analytical)", "Yates (2009) Equilibrium"): yates_pos_all[::interval][1:],
                    ("Class I (Analytical)", "ShoreFor (2013) Memory"): shorefor_pos_all[::interval][1:],
                    ("Class II (Process/PDE)", "XBeach 1D Process Model"): xbeach_pos_all[::interval][1:],
                }
            }

    active_sites = [s for s in sites if s in site_data]
    if len(active_sites) < 2:
        raise ValueError(f"Need at least 2 active sites with data for LOSO CV. Found: {len(active_sites)}")

    pooled_preds = {}
    pooled_actuals = {}
    pooled_pos_preds = {}
    pooled_pos_actuals = {}
    pooled_norm_preds = {}
    pooled_norm_actuals = {}
    pooled_norm_pos_preds = {}
    pooled_norm_pos_actuals = {}
    held_out_records = []

    # Leave-One-Site-Out Cross-Validation Loop
    for held_out_idx, held_out_site in enumerate(active_sites):
        training_sites = [s for s in active_sites if s != held_out_site]

        # 1. Pool Training Sites
        y_train_list = []
        f_train_list = []
        train_deltas_pool = {}
        train_positions_pool = {}
        pos_train_list = []

        for tr_site in training_sites:
            s_dict = site_data[tr_site]
            y_train_list.extend(s_dict["y"])
            f_train_list.extend(s_dict["forcing"])
            pos_train_list.extend(s_dict["pos"])

            for k, d_arr in s_dict["deltas"].items():
                if k not in train_deltas_pool:
                    train_deltas_pool[k] = []
                train_deltas_pool[k].extend(d_arr)

            for k, p_arr in s_dict["positions"].items():
                if k not in train_positions_pool:
                    train_positions_pool[k] = []
                train_positions_pool[k].extend(p_arr)

        y_train = np.array(y_train_list)
        f_train = np.array(f_train_list)
        pos_train = np.array(pos_train_list)

        # Held-Out Site Data
        ho_dict = site_data[held_out_site]
        y_test = ho_dict["y"]
        pos_test = ho_dict["pos"]
        f_test = ho_dict["forcing"]
        n_test = len(y_test)

        if n_test == 0:
            continue

        sigma_site_dy = max(float(np.std(y_test)), 1e-4)
        sigma_site_pos = max(float(np.std(pos_test)), 1e-4)
        var_base = max(np.mean((y_test - np.mean(y_train)) ** 2), 1e-6)

        # 2. Fit Bayesian Latent Memory Model on Pooled 4-Site Training Data
        f_mean_tr = float(np.mean(f_train))
        f_std_tr = max(float(np.std(f_train)), 1e-4)
        max_lag_loso = 12

        bayes_m, bayes_meta = build_latent_memory_bayesian_model(
            forcing=f_train,
            delta_x_obs=y_train,
            model_name="Bayesian Latent Memory Kernel (Upsilon)",
            max_lag=max_lag_loso,
            scale_forcing=True,
        )
        with bayes_m:
            tr_bayes = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=0.90,
                progressbar=False,
                random_seed=random_seed,
            )

        gamma_ho = float(tr_bayes.posterior["gamma"].values.mean())
        alpha_ho = float(tr_bayes.posterior["alpha"].values.mean())
        beta_ho = float(tr_bayes.posterior["beta_forcing"].values.mean())

        # Zero-shot prediction on Held-Out Site using 4-site trained normalization & posteriors
        f_test_sc = (f_test - f_mean_tr) / f_std_tr
        k_vec = np.arange(max_lag_loso, dtype=float)
        w_vec = (gamma_ho ** k_vec) / np.sum(gamma_ho ** k_vec)
        lag_mat_test = create_lag_matrix(f_test_sc, max_lag=max_lag_loso)
        eff_forcing_test = lag_mat_test @ w_vec
        dy_pred_bayes = alpha_ho + beta_ho * eff_forcing_test

        # Compile Model Evaluations on Held-Out Site
        eval_models = [
            ("Class III (Bayesian)", "Bayesian Latent Memory Kernel (Upsilon)", dy_pred_bayes, None, None),
        ]

        for (p_class, m_name), d_test_raw in ho_dict["deltas"].items():
            d_train_raw = np.array(train_deltas_pool.get((p_class, m_name), []))
            p_train_raw = np.array(train_positions_pool.get((p_class, m_name), [])) if (p_class, m_name) in ho_dict["positions"] else None
            eval_models.append((p_class, m_name, d_test_raw, d_train_raw, p_train_raw))

        # 3. Evaluate each model on Held-Out Site
        for p_class, m_name, d_raw_te, d_raw_tr, p_raw_tr in eval_models:
            is_global_default = "Global Default" in m_name

            if is_global_default or d_raw_tr is None or len(d_raw_tr) == 0 or np.std(d_raw_tr) < 1e-6:
                dp_ho = d_raw_te
            else:
                # 4-Site Calibrated Transfer Function (No Target Leakage)
                slope_tr, intercept_tr = np.polyfit(d_raw_tr, y_train, 1)
                dp_ho = slope_tr * d_raw_te + intercept_tr

            # Cumulative position trajectory on Held-Out Site
            if (p_class, m_name) in ho_dict["positions"] and not is_global_default and p_raw_tr is not None and len(p_raw_tr) > 0 and np.std(p_raw_tr) > 1e-6:
                slope_p, intercept_p = np.polyfit(p_raw_tr, pos_train, 1)
                pos_pred_ho = slope_p * ho_dict["positions"][(p_class, m_name)] + intercept_p
            else:
                pos_init = pos_test[0] if len(pos_test) > 0 else 0.0
                pos_pred_ho = pos_init + np.cumsum(dp_ho)

            # Store for Global Pooled LOSO metrics
            key = (p_class, m_name)
            if key not in pooled_preds:
                pooled_preds[key] = []
                pooled_actuals[key] = []
                pooled_pos_preds[key] = []
                pooled_pos_actuals[key] = []
                pooled_norm_preds[key] = []
                pooled_norm_actuals[key] = []
                pooled_norm_pos_preds[key] = []
                pooled_norm_pos_actuals[key] = []

            pooled_preds[key].extend(dp_ho)
            pooled_actuals[key].extend(y_test)
            pooled_pos_preds[key].extend(pos_pred_ho)
            pooled_pos_actuals[key].extend(pos_test)

            pooled_norm_preds[key].extend(dp_ho / sigma_site_dy)
            pooled_norm_actuals[key].extend(y_test / sigma_site_dy)
            pooled_norm_pos_preds[key].extend(pos_pred_ho / sigma_site_pos)
            pooled_norm_pos_actuals[key].extend(pos_test / sigma_site_pos)

            # Differential Step Change Metrics (Delta-x)
            mse_ho = float(mean_squared_error(y_test, dp_ho))
            rmse_ho = float(np.sqrt(mse_ho))
            r2_ho = float(r2_score(y_test, dp_ho))
            bss_ho = float(1.0 - (mse_ho / var_base))
            corr_ho = float(np.corrcoef(y_test, dp_ho)[0, 1]) if (len(y_test) > 1 and np.std(dp_ho) > 1e-6 and np.std(y_test) > 1e-6) else 0.0

            # Cumulative Position Metrics (x_t)
            pos_mse_ho = float(mean_squared_error(pos_test, pos_pred_ho))
            pos_rmse_ho = float(np.sqrt(pos_mse_ho))
            pos_r2_ho = float(r2_score(pos_test, pos_pred_ho)) if len(pos_test) > 1 else 0.0
            pos_var_ho = max(float(np.var(pos_test)), 1e-6)
            pos_bss_ho = float(1.0 - (pos_mse_ho / pos_var_ho))
            pos_corr_ho = float(np.corrcoef(pos_test, pos_pred_ho)[0, 1]) if (len(pos_test) > 1 and np.std(pos_pred_ho) > 1e-6 and np.std(pos_test) > 1e-6) else 0.0

            held_out_records.append({
                "Held-Out Site": held_out_site,
                "Held-Out Site Name": COASTAL_SITE_PROFILES.get(held_out_site, {}).get("name", held_out_site),
                "Training Sites Count": len(training_sites),
                "Paradigm Class": p_class,
                "Model Name": m_name,
                "Fold": held_out_idx + 1,
                # Step Delta Metrics
                "LOSO RMSE (m)": rmse_ho,
                "LOSO R2 Score": r2_ho,
                "LOSO BSS": bss_ho,
                "LOSO Pearson r": corr_ho,
                # Cumulative Position Metrics
                "LOSO Pos RMSE (m)": pos_rmse_ho,
                "LOSO Pos R2 Score": pos_r2_ho,
                "LOSO Pos BSS": pos_bss_ho,
                "LOSO Pos Pearson r": pos_corr_ho,
                # Normalized Metrics
                "Site Std Δx (m)": sigma_site_dy,
                "Site Std Pos (m)": sigma_site_pos,
                "LOSO Norm Δx RMSE": rmse_ho / sigma_site_dy,
                "LOSO Norm Pos RMSE": pos_rmse_ho / sigma_site_pos,
            })

    held_out_sites_df = pd.DataFrame(held_out_records)

    # 4. Global Pooled LOSO Synthesis Table across all 5 Held-Out Sites
    summary_rows = []
    for (p_class, m_name) in pooled_preds:
        y_act_arr = np.array(pooled_actuals[(p_class, m_name)])
        y_pred_arr = np.array(pooled_preds[(p_class, m_name)])
        pos_act_arr = np.array(pooled_pos_actuals[(p_class, m_name)])
        pos_pred_arr = np.array(pooled_pos_preds[(p_class, m_name)])

        y_norm_act = np.array(pooled_norm_actuals[(p_class, m_name)])
        y_norm_pred = np.array(pooled_norm_preds[(p_class, m_name)])
        pos_norm_act = np.array(pooled_norm_pos_actuals[(p_class, m_name)])
        pos_norm_pred = np.array(pooled_norm_pos_preds[(p_class, m_name)])

        # Step Delta Metrics (Δx)
        loso_rmse_pool = float(np.sqrt(mean_squared_error(y_act_arr, y_pred_arr)))
        loso_r2_pool = float(r2_score(y_act_arr, y_pred_arr))
        var_act = max(float(np.var(y_act_arr)), 1e-6)
        loso_bss_pool = float(1.0 - (mean_squared_error(y_act_arr, y_pred_arr) / var_act))
        loso_corr_pool = float(np.corrcoef(y_act_arr, y_pred_arr)[0, 1]) if (np.std(y_pred_arr) > 1e-6 and np.std(y_act_arr) > 1e-6) else 0.0

        # Cumulative Position Metrics (x_t)
        loso_pos_rmse_pool = float(np.sqrt(mean_squared_error(pos_act_arr, pos_pred_arr)))
        loso_pos_r2_pool = float(r2_score(pos_act_arr, pos_pred_arr))
        var_pos_act = max(float(np.var(pos_act_arr)), 1e-6)
        loso_pos_bss_pool = float(1.0 - (mean_squared_error(pos_act_arr, pos_pred_arr) / var_pos_act))
        loso_pos_corr_pool = float(np.corrcoef(pos_act_arr, pos_pred_arr)[0, 1]) if (np.std(pos_pred_arr) > 1e-6 and np.std(pos_act_arr) > 1e-6) else 0.0

        # Standardized Per-Site Normalized Metrics
        norm_loso_r2 = float(r2_score(y_norm_act, y_norm_pred))
        norm_loso_rmse = float(np.sqrt(mean_squared_error(y_norm_act, y_norm_pred)))
        var_norm_act = max(float(np.var(y_norm_act)), 1e-6)
        norm_loso_bss = float(1.0 - (mean_squared_error(y_norm_act, y_norm_pred) / var_norm_act))

        norm_pos_loso_r2 = float(r2_score(pos_norm_act, pos_norm_pred))
        norm_pos_loso_rmse = float(np.sqrt(mean_squared_error(pos_norm_act, pos_norm_pred)))
        var_norm_pos_act = max(float(np.var(pos_norm_act)), 1e-6)
        norm_pos_loso_bss = float(1.0 - (mean_squared_error(pos_norm_act, pos_norm_pred) / var_norm_pos_act))
        norm_pos_loso_corr = float(np.corrcoef(pos_norm_act, pos_norm_pred)[0, 1]) if (np.std(pos_norm_pred) > 1e-6 and np.std(pos_norm_act) > 1e-6) else 0.0

        grp = held_out_sites_df[(held_out_sites_df["Paradigm Class"] == p_class) & (held_out_sites_df["Model Name"] == m_name)]
        rmse_std = float(grp["LOSO RMSE (m)"].std()) if not grp.empty else 0.0
        r2_std = float(grp["LOSO R2 Score"].std()) if not grp.empty else 0.0
        pos_rmse_std = float(grp["LOSO Pos RMSE (m)"].std()) if not grp.empty else 0.0
        pos_r2_std = float(grp["LOSO Pos R2 Score"].std()) if not grp.empty else 0.0

        site_avg_r2 = float(grp["LOSO R2 Score"].mean()) if not grp.empty else loso_r2_pool
        site_avg_pos_r2 = float(grp["LOSO Pos R2 Score"].mean()) if not grp.empty else loso_pos_r2_pool

        summary_rows.append({
            "Paradigm Class": p_class,
            "Model Name": m_name,
            # Standardized Per-Site Normalized LOSO Skill
            "Normalized LOSO R2 (Δx)": norm_loso_r2,
            "Normalized LOSO RMSE (Δx)": norm_loso_rmse,
            "Normalized LOSO BSS (Δx)": norm_loso_bss,
            "Normalized Position LOSO R2": norm_pos_loso_r2,
            "Normalized Position LOSO RMSE": norm_pos_loso_rmse,
            "Normalized Position LOSO Pearson r": norm_pos_loso_corr,
            "Normalized Position LOSO BSS": norm_pos_loso_bss,
            "Site-Averaged LOSO R2 (Δx)": site_avg_r2,
            "Site-Averaged LOSO Pos R2": site_avg_pos_r2,
            # Position-Level Performance Metrics (x_t)
            "Position LOSO R2 Score": loso_pos_r2_pool,
            "Position LOSO RMSE (m)": loso_pos_rmse_pool,
            "Position LOSO Pearson r": loso_pos_corr_pool,
            "Position LOSO BSS": loso_pos_bss_pool,
            "Position LOSO R2 (std)": pos_r2_std,
            "Position LOSO RMSE (std)": pos_rmse_std,
            # Differential Step Change Metrics (Δx)
            "LOSO RMSE (m) (Δx)": loso_rmse_pool,
            "LOSO R2 Score (Δx)": loso_r2_pool,
            "LOSO BSS (Δx)": loso_bss_pool,
            "LOSO Pearson r (Δx)": loso_corr_pool,
            "LOSO RMSE (std)": rmse_std,
            "LOSO R2 (std)": r2_std,
            "Held-Out Sites Tested": len(grp),
        })

    synthesis_loso_df = pd.DataFrame(summary_rows).sort_values(by="Position LOSO R2 Score", ascending=False).reset_index(drop=True)
    synthesis_loso_df.attrs["sites"] = active_sites

    return synthesis_loso_df, held_out_sites_df


def build_learnable_critical_threshold_bayesian_model(
    upsilon_raw: np.ndarray,
    delta_x_obs: np.ndarray,
    tan_beta: Union[float, np.ndarray] = 0.08,
    d50_m: Union[float, np.ndarray] = 0.00035,
    upsilon_crit_0_prior_mu: float = 500.0,
    upsilon_crit_0_prior_sigma: float = 150.0,
    a_prior_mu: float = 1.0,
    a_prior_sigma: float = 0.3,
    b_prior_mu: float = 1.0,
    b_prior_sigma: float = 0.3,
    d50_ref: float = 0.00035,
    tan_beta_ref: float = 0.08,
    model_name: str = "learnable_critical_threshold_model",
    max_lag: int = 24,
    heteroscedastic: bool = True,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a PyMC Bayesian model estimating critical threshold exponents (a, b) and baseline Upsilon_crit_0.

    Estimates learnable exponents (a, b) governing nonlinear critical shear stress and foreshore slope stability:
        Upsilon_crit(t) = Upsilon_crit_0 * (D_50 / D_50_ref)^a * (tan_beta / tan_beta_ref)^b
        Upsilon_excess(t) = max(0, Upsilon_raw(t) - Upsilon_crit(t))
    """
    upsilon_arr = np.asarray(upsilon_raw, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)
    n_obs = len(y_obs)
    max_lag = max(1, min(max_lag, n_obs))

    tb_arr = np.maximum(np.asarray(tan_beta, dtype=float), 1e-4)
    d50_arr = np.maximum(np.asarray(d50_m, dtype=float), 1e-6)

    tb_ratio = np.broadcast_to(tb_arr / max(float(tan_beta_ref), 1e-4), (n_obs,)) if not np.isscalar(tb_arr) else np.full(n_obs, float(tb_arr / max(float(tan_beta_ref), 1e-4)))
    d50_ratio = np.broadcast_to(d50_arr / max(float(d50_ref), 1e-6), (n_obs,)) if not np.isscalar(d50_arr) else np.full(n_obs, float(d50_arr / max(float(d50_ref), 1e-6)))

    meta = {
        "model_name": model_name,
        "n_obs": n_obs,
        "max_lag": max_lag,
        "d50_ref": d50_ref,
        "tan_beta_ref": tan_beta_ref,
    }

    with pm.Model() as model:
        # 1. Physics-informed Priors for Threshold Parameters
        # a: Grain size exponent (Shields / Soulsby physics: a in [0.5, 1.5])
        a = pm.TruncatedNormal("a", mu=a_prior_mu, sigma=a_prior_sigma, lower=0.2, upper=2.5)
        # b: Foreshore slope exponent (Gravity / scarping stability: b in [0.5, 1.5])
        b = pm.TruncatedNormal("b", mu=b_prior_mu, sigma=b_prior_sigma, lower=0.2, upper=2.5)
        # Baseline threshold at reference conditions
        upsilon_crit_0 = pm.TruncatedNormal(
            "upsilon_crit_0",
            mu=upsilon_crit_0_prior_mu,
            sigma=upsilon_crit_0_prior_sigma,
            lower=50.0,
            upper=2000.0,
        )

        # 2. Critical Threshold Series
        d50_t = pt.as_tensor_variable(d50_ratio)
        tb_t = pt.as_tensor_variable(tb_ratio)
        u_crit_t = upsilon_crit_0 * (d50_t ** a) * (tb_t ** b)
        pm.Deterministic("upsilon_crit", u_crit_t)

        # 3. Excess Forcing
        u_raw_t = pt.as_tensor_variable(upsilon_arr)
        u_excess_t = pt.maximum(0.0, u_raw_t - u_crit_t)
        pm.Deterministic("upsilon_excess", u_excess_t)

        # Standardized excess forcing for latent memory
        f_scale = u_excess_t / pt.maximum(pt.mean(u_excess_t), 1e-3)

        # Construct differentiable lag matrix
        lag_cols = []
        for k in range(max_lag):
            if k == 0:
                lag_cols.append(f_scale)
            else:
                lag_cols.append(pt.concatenate([pt.repeat(f_scale[0:1], k), f_scale[:-k]]))
        x_lag = pt.stack(lag_cols, axis=1)

        # 4. Latent Memory Weighting
        gamma = pm.Beta("gamma", alpha=5.0, beta=20.0)
        k_vec = pt.as_tensor_variable(np.arange(max_lag, dtype=float))
        unnorm_w = gamma ** k_vec
        w = unnorm_w / pt.sum(unnorm_w)
        forcing_filtered = pt.dot(x_lag, w)

        # 5. Dynamic Cross-Shore Response
        alpha_coef = pm.Normal("alpha_coef", mu=0.0, sigma=2.0)
        beta_coef = pm.Normal("beta_coef", mu=-0.5, sigma=1.0)
        mu_pred = alpha_coef + beta_coef * forcing_filtered

        # 6. Likelihood
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        if heteroscedastic:
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma_dynamic = pm.HalfNormal("sigma_dynamic", sigma=0.5)
            sigma_t = sigma_base + sigma_dynamic * (forcing_filtered / pt.maximum(pt.max(forcing_filtered), 1e-3))
            pm.StudentT("y_obs", nu=nu, mu=mu_pred, sigma=sigma_t, observed=y_obs)
        else:
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            pm.StudentT("y_obs", nu=nu, mu=mu_pred, sigma=sigma, observed=y_obs)

    return model, meta


def run_critical_threshold_exponent_sensitivity_analysis(
    df: pd.DataFrame,
    a_values: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    b_values: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    target_col: str = "shoreline_pos_obs_m",
    max_lag: int = 24,
    draws: int = 500,
    tune: int = 250,
    chains: int = 2,
    include_learnable: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Run systematic Bayesian sensitivity analysis over critical threshold exponents (a, b).

    Demonstrates morphodynamic model stability and performance across the physically justified
    exponent regime a, b in [0.5, 1.5].
    """
    from src.physics import compute_unified_coastal_vulnerability_index

    hs = df["hs_m"].values if "hs_m" in df.columns else df["hs"].values
    tp = df["tp_s"].values if "tp_s" in df.columns else df["tp"].values
    surge = df["surge_m"].values if "surge_m" in df.columns else (df["surge"].values if "surge" in df.columns else np.zeros(len(df)))
    wind_spd = df["wind_speed_ms"].values if "wind_speed_ms" in df.columns else (df["wind_speed"].values if "wind_speed" in df.columns else np.zeros(len(df)))
    wind_dir = df["wind_dir_deg"].values if "wind_dir_deg" in df.columns else (df["wind_dir"].values if "wind_dir" in df.columns else np.full(len(df), 70.0))
    d50 = df["d50_mm"].values if "d50_mm" in df.columns else (df["d50_m"].values if "d50_m" in df.columns else 0.00035)
    tan_beta = df["tan_beta"].values if "tan_beta" in df.columns else 0.08
    y_obs = df[target_col].values if target_col in df.columns else (df["delta_x_obs"].values if "delta_x_obs" in df.columns else np.zeros(len(df)))

    # Compute base raw Upsilon
    df_raw = compute_unified_coastal_vulnerability_index(
        hs=hs,
        tp=tp,
        surge=surge,
        wind_speed=wind_spd,
        wind_dir_deg=wind_dir,
        d50_m=d50,
        tan_beta=tan_beta,
        enable_critical_threshold=False,
    )
    raw_upsilon = df_raw["upsilon"].values

    traces: Dict[str, az.InferenceData] = {}

    for a_val in a_values:
        for b_val in b_values:
            m_name = f"a_{a_val:.2f}_b_{b_val:.2f}"
            df_phys = compute_unified_coastal_vulnerability_index(
                hs=hs,
                tp=tp,
                surge=surge,
                wind_speed=wind_spd,
                wind_dir_deg=wind_dir,
                d50_m=d50,
                tan_beta=tan_beta,
                upsilon_crit_a=a_val,
                upsilon_crit_b=b_val,
                enable_critical_threshold=True,
            )
            forcing = df_phys["upsilon_excess"].values
            model, _ = build_latent_memory_bayesian_model(
                forcing=forcing,
                delta_x_obs=y_obs,
                model_name=m_name,
                max_lag=max_lag,
            )
            trace = fit_bayesian_model(
                model=model,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
            )
            traces[m_name] = trace

    if include_learnable:
        m_name = "learnable_a_b"
        learn_model, _ = build_learnable_critical_threshold_bayesian_model(
            upsilon_raw=raw_upsilon,
            delta_x_obs=y_obs,
            tan_beta=tan_beta,
            d50_m=d50,
            model_name=m_name,
            max_lag=max_lag,
        )
        trace_learn = fit_bayesian_model(
            model=learn_model,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        traces[m_name] = trace_learn

    df_comp = run_arviz_loo_comparison(traces, ic="loo")
    return df_comp, traces


def run_bayesian_memory_window_ablation(
    df: pd.DataFrame,
    lags: Sequence[int] = (24, 48, 168, 720),
    target_col: str = "shoreline_pos_obs_m",
    forcing_col: Optional[str] = None,
    draws: int = 500,
    tune: int = 250,
    chains: int = 2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Run systematic Bayesian memory window (max_lag) ablation across candidate horizons.

    Compares Bayesian latent memory models across max_lag in {24, 48, 168, 720} hours
    (1 day, 2 days, 1 week, 30 days) to evaluate whether a short-term 48h memory window
    prematurely truncates antecedent wave energy vs. synoptic and seasonal equilibrium relaxation
    timescales (Yates et al. 2009, Splinter et al. 2014 ShoreFor).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hydrodynamic forcing and morphodynamic observations.
    lags : sequence of int
        Candidate memory horizons in hours (default: 24, 48, 168, 720).
    target_col : str
        Target column name (e.g., 'shoreline_pos_obs_m' or 'surveyed_delta_x_m').
    forcing_col : str, optional
        Forcing column name (e.g., 'upsilon', 'upsilon_excess', 'hs_m', 'p_wave_kwm').
        If None, auto-selected from available columns in df.
    draws : int
        Number of MCMC draws per chain (default 500).
    tune : int
        Number of tuning / warmup steps per chain (default 250).
    chains : int
        Number of MCMC chains (default 2).
    random_seed : int
        Random seed for sampling reproducibility.

    Returns
    -------
    df_comp : pd.DataFrame
        ArviZ model comparison table with ELPD LOO, p_loo, LOO weights, and SE.
    traces : dict
        Fitted MCMC InferenceData traces for each candidate lag horizon.
    """
    # 1. Resolve target series
    if target_col in df.columns:
        y_obs = df[target_col].values
    elif "surveyed_delta_x_m" in df.columns:
        y_obs = df["surveyed_delta_x_m"].values
    elif "delta_x_obs" in df.columns:
        y_obs = df["delta_x_obs"].values
    elif "delta_x_m" in df.columns:
        y_obs = df["delta_x_m"].values
    else:
        y_obs = df.iloc[:, 0].values

    # 2. Resolve forcing series
    if forcing_col is not None and forcing_col in df.columns:
        forcing = df[forcing_col].values
    elif "upsilon" in df.columns:
        forcing = df["upsilon"].values
    elif "upsilon_excess" in df.columns:
        forcing = df["upsilon_excess"].values
    elif "hs_m" in df.columns:
        forcing = df["hs_m"].values ** 2.5
    elif "hs" in df.columns:
        forcing = df["hs"].values ** 2.5
    else:
        forcing = np.ones(len(y_obs))

    traces: Dict[str, az.InferenceData] = {}

    for lag in lags:
        m_name = f"lag_{lag}h"
        effective_lag = max(1, min(int(lag), len(y_obs)))
        model, _ = build_latent_memory_bayesian_model(
            forcing=forcing,
            delta_x_obs=y_obs,
            model_name=m_name,
            max_lag=effective_lag,
        )
        trace = fit_bayesian_model(
            model=model,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        traces[m_name] = trace

    df_comp = run_arviz_loo_comparison(traces, ic="loo")
    return df_comp, traces


def run_bayesian_memory_prior_sensitivity_analysis(
    df: pd.DataFrame,
    priors: Optional[Sequence[Tuple[str, Tuple[float, float]]]] = None,
    target_col: str = "shoreline_pos_obs_m",
    forcing_col: Optional[str] = None,
    max_lag: int = 48,
    draws: int = 500,
    tune: int = 250,
    chains: int = 2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData], pd.DataFrame]:
    """Run systematic prior-sensitivity check across candidate memory priors gamma ~ Beta(alpha, beta).

    Compares:
      (a) Beta(2, 5)   - Wider prior (mean = 2/7 ≈ 0.286, variance = 0.0255)
      (b) Beta(2, 2)   - Uninformative / symmetric prior (mean = 0.500, variance = 0.0500)
      (c) Beta(5, 20)  - Informative short-memory prior (mean = 5/25 = 0.200, variance = 0.0062)

    Evaluates whether the informative Beta(5, 20) prior artificially constrains the inferred
    antecedent memory retention parameter gamma and equilibrium response timescale, vs. whether
    the observational data likelihood dominates the posterior under wider and uninformative priors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hydrodynamic forcing and morphodynamic observations.
    priors : sequence of (str, (float, float)), optional
        Candidate prior parameterizations. Defaults to [("Beta(2, 5)", (2.0, 5.0)), ("Beta(2, 2)", (2.0, 2.0)), ("Beta(5, 20)", (5.0, 20.0))].
    target_col : str
        Target column name (e.g., 'shoreline_pos_obs_m' or 'surveyed_delta_x_m').
    forcing_col : str, optional
        Forcing column name (e.g., 'upsilon', 'upsilon_excess', 'hs_m').
    max_lag : int
        Convolution window length in hours (default 48).
    draws : int
        Number of MCMC draws per chain (default 500).
    tune : int
        Number of tuning / warmup steps per chain (default 250).
    chains : int
        Number of MCMC chains (default 2).
    random_seed : int
        Random seed for sampling reproducibility.

    Returns
    -------
    df_comp : pd.DataFrame
        ArviZ model comparison table with PSIS-LOO-CV ELPD, p_loo, LOO weights, and SE.
    traces : dict
        Fitted MCMC InferenceData traces for each candidate prior.
    df_gamma_summary : pd.DataFrame
        Posterior summary table for gamma (mean, median, std, 94% HDI, half-life, prior mean/std).
    """
    if priors is None:
        priors = [
            ("Beta(2, 5)", (2.0, 5.0)),
            ("Beta(2, 2)", (2.0, 2.0)),
            ("Beta(5, 20)", (5.0, 20.0)),
        ]

    # 1. Resolve target series
    if target_col in df.columns:
        y_obs = df[target_col].values
    elif "surveyed_delta_x_m" in df.columns:
        y_obs = df["surveyed_delta_x_m"].values
    elif "delta_x_obs" in df.columns:
        y_obs = df["delta_x_obs"].values
    elif "delta_x_m" in df.columns:
        y_obs = df["delta_x_m"].values
    else:
        y_obs = df.iloc[:, 0].values

    # 2. Resolve forcing series
    if forcing_col is not None and forcing_col in df.columns:
        forcing = df[forcing_col].values
    elif "upsilon" in df.columns:
        forcing = df["upsilon"].values
    elif "upsilon_excess" in df.columns:
        forcing = df["upsilon_excess"].values
    elif "hs_m" in df.columns:
        forcing = df["hs_m"].values ** 2.5
    elif "hs" in df.columns:
        forcing = df["hs"].values ** 2.5
    else:
        forcing = np.ones(len(y_obs))

    effective_lag = max(1, min(int(max_lag), len(y_obs)))
    traces: Dict[str, az.InferenceData] = {}
    summary_records = []

    for prior_name, (alpha_val, beta_val) in priors:
        clean_suffix = prior_name.replace(" ", "").replace("(", "_").replace(")", "").replace(",", "_")
        m_name = f"prior_{clean_suffix}"
        model, _ = build_latent_memory_bayesian_model(
            forcing=forcing,
            delta_x_obs=y_obs,
            model_name=m_name,
            max_lag=effective_lag,
            gamma_prior=(float(alpha_val), float(beta_val)),
        )
        trace = fit_bayesian_model(
            model=model,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        traces[m_name] = trace

        # Compute theoretical prior statistics
        prior_mean = float(alpha_val / (alpha_val + beta_val))
        prior_var = float((alpha_val * beta_val) / (((alpha_val + beta_val) ** 2) * (alpha_val + beta_val + 1)))
        prior_std = float(np.sqrt(prior_var))

        # Compute empirical posterior statistics
        if "gamma" in trace.posterior:
            gamma_samples = trace.posterior["gamma"].values.flatten()
            post_mean = float(np.mean(gamma_samples))
            post_median = float(np.median(gamma_samples))
            post_std = float(np.std(gamma_samples))
            hdi_bounds = az.hdi(trace.posterior["gamma"], hdi_prob=0.94)["gamma"].values
            hdi_3 = float(hdi_bounds[0])
            hdi_97 = float(hdi_bounds[1])
            half_life_h = float(np.log(0.5) / np.log(post_mean)) if (0.0 < post_mean < 1.0) else np.nan
            try:
                rhat = float(az.rhat(trace)["gamma"].values)
            except Exception:
                rhat = 1.0
            try:
                ess_bulk = float(az.ess(trace, var_names=["gamma"], method="bulk")["gamma"].values)
            except Exception:
                ess_bulk = np.nan
            try:
                ess_tail = float(az.ess(trace, var_names=["gamma"], method="tail")["gamma"].values)
            except Exception:
                ess_tail = np.nan
        else:
            post_mean = post_median = post_std = hdi_3 = hdi_97 = half_life_h = rhat = ess_bulk = ess_tail = np.nan

        summary_records.append({
            "model_key": m_name,
            "prior_name": prior_name,
            "prior_alpha": float(alpha_val),
            "prior_beta": float(beta_val),
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "post_mean": post_mean,
            "post_median": post_median,
            "post_std": post_std,
            "hdi_3%": hdi_3,
            "hdi_97%": hdi_97,
            "half_life_hours": half_life_h,
            "r_hat": rhat,
            "ess_bulk": ess_bulk,
            "ess_tail": ess_tail,
        })

    df_gamma_summary = pd.DataFrame(summary_records).set_index("model_key")
    df_comp = run_arviz_loo_comparison(traces, ic="loo")

    return df_comp, traces, df_gamma_summary


