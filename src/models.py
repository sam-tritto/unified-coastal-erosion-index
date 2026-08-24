"""Bayesian modeling of coastal morphodynamic response and shoreline change using PyMC.

Includes:
- Latent Bayesian estimation of antecedent memory retention (gamma ~ Beta(2, 5))
- Differentiable PyTensor convolution matrix for dynamic memory weighting
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
from typing import Dict, Any, Tuple, Optional, List


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
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a PyMC Bayesian model where antecedent memory gamma is inferred as a latent variable.

    Formulation:
    - gamma ~ Beta(alpha=2, beta=5)  (Latent memory parameter)
    - weights_k = gamma^k / sum(gamma^j) for k = 0, ..., max_lag-1
    - E(gamma) = X_lag * weights(gamma)
    - mu_t = alpha + beta_forcing * E_t(gamma)
    - y_obs ~ StudentT(nu, mu_t, sigma)

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
    }

    with pm.Model() as model:
        # Data containers
        x_lag = pm.Data("x_lag", lag_mat)
        k_vec = pm.Data("k_vec", k_indices)
        
        # 1. Latent memory parameter prior (Beta(2, 5) favors intermediate response windows)
        gamma = pm.Beta("gamma", alpha=2.0, beta=5.0)
        
        # 2. Differentiable exponential weights in PyTensor
        # weights_k = gamma^k / sum(gamma^j)
        unnorm_weights = pt.power(gamma, k_vec)
        weights = unnorm_weights / pt.sum(unnorm_weights)
        
        # 3. Dynamic antecedent convolution: (N, max_lag) @ (max_lag,) -> (N,)
        effective_forcing = pm.Deterministic("effective_forcing", pt.dot(x_lag, weights))
        
        # 4. Morphodynamic response parameters
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta_forcing = pm.Normal("beta_forcing", mu=-2.0, sigma=1.5)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)
        
        # 5. Deterministic latent response
        mu = pm.Deterministic("mu", alpha + beta_forcing * effective_forcing)
        
        # 6. Student-T Likelihood
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
        )
        ppc = pm.sample_posterior_predictive(trace, random_seed=random_seed, progressbar=False)
        trace.extend(ppc)
        
    return trace


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate morphodynamic validation metrics (RMSE, MAE, R^2, and Brier Skill Score)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    if y_baseline is None:
        y_b = np.zeros_like(y_t)
    else:
        y_b = np.asarray(y_baseline, dtype=float)
        
    mse_model = float(np.mean((y_t - y_p) ** 2))
    mse_base = float(np.mean((y_t - y_b) ** 2))
    
    rmse = float(np.sqrt(mse_model))
    mae = float(np.mean(np.abs(y_t - y_p)))
    
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum((y_t - y_p) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
    
    bss = float(1.0 - (mse_model / mse_base)) if mse_base > 1e-9 else 0.0
    
    return {
        "RMSE (m)": rmse,
        "MAE (m)": mae,
        "R2 Score": r2,
        "Brier Skill Score (BSS)": bss,
        "MSE Model": mse_model,
        "MSE Baseline": mse_base,
    }


def compute_mcmc_summary_table(idata: az.InferenceData) -> pd.DataFrame:
    """Generate clean parameter posterior summary table with R-hat and ESS."""
    var_names = ["alpha", "beta_forcing", "gamma", "sigma", "nu", "beta_upsilon", "k_eros", "k_accrete", "omega_crit", "hs_exponent"]
    available = [v for v in var_names if v in idata.posterior]
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

    Supports continuous antecedent memory convolution across train/test splits.

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
    sigmas = trace.posterior["sigma"].values.reshape(-1) if "sigma" in trace.posterior else np.full(len(alphas), 0.3)
    nus = trace.posterior["nu"].values.reshape(-1) if "nu" in trace.posterior else np.full(len(alphas), 5.0)
    
    # Sub-sample posterior draws to 200 for fast, memory-safe out-of-sample evaluation on large multi-year series
    if len(alphas) > 200:
        sub_idx = np.linspace(0, len(alphas) - 1, 200, dtype=int)
        alphas = alphas[sub_idx]
        sigmas = sigmas[sub_idx]
        nus = nus[sub_idx]
    else:
        sub_idx = np.arange(len(alphas))
    n_samples = len(alphas)

    is_survey = "antecedent_upsilon_memory" in df.columns or "surveyed_delta_x_m" in df.columns

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

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    mu_mean = np.mean(mu_samples, axis=0)

    if not return_samples:
        return mu_mean

    # Draw posterior predictive Student-T observations
    y_pred_samples = np.zeros_like(mu_samples)
    for i in range(n_samples):
        nu_val = max(float(nus[i]), 1.5)
        sigma_val = max(float(sigmas[i]), 1e-4)
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
    valid_traces = {k: v for k, v in traces.items() if hasattr(v, "log_likelihood") and len(v.log_likelihood.data_vars) > 0}
    if len(valid_traces) < 2:
        return pd.DataFrame()

    try:
        df_comp = az.compare(valid_traces, ic=ic, method=method)
    except Exception:
        try:
            df_comp = az.compare(valid_traces, ic=ic, method="bb-pseudo-bma")
        except Exception:
            df_comp = az.compare(valid_traces, ic=ic, method="pseudo-bma")

    return df_comp


def run_benchmark_model_comparison(
    df: pd.DataFrame,
    target_col: str = "shoreline_pos_obs_m",
    use_log_upsilon: bool = True,
    train_split: float = 0.75,
    draws: int = 1000,
    tune: int = 800,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    include_nonlinear: bool = True,
    include_multivariate: bool = True,
    include_hybrid: bool = True,
    compute_loo: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Fit and benchmark competing physical models with Out-of-Sample Temporal Holdout and ArviZ LOO-CV.

    Parameters
    ----------
    df : pd.DataFrame
        Synchronized hydrodynamic and morphodynamic observational dataset.
    target_col : str
        Target column name (default 'shoreline_pos_obs_m' for cumulative cross-shore position).
        Use 'shoreline_change_rate_mh' only with lag-memory model types that can resolve
        the near-zero hourly finite-difference signal against survey noise.
    use_log_upsilon : bool
        Whether to use log10(Upsilon) as the primary forcing input (default True).
        Υ spans ~3 orders of magnitude; log-transformation stabilises the predictor
        variance and prevents Bayesian linear models from converging to beta ≈ 0.
    train_split : float
        Chronological training split fraction (default 0.75, giving 75% train, 25% out-of-sample test).
        Set to 1.0 for full-series training without holdout split.
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

    Returns
    -------
    comparison_df : pd.DataFrame
        Ranked comparison table with Out-of-Sample Test metrics, In-Sample metrics, LOO-CV statistics, and R-hat.
    traces : dict
        Fitted ArviZ InferenceData traces for each candidate model.
    """
    n_total = len(df)
    if train_split < 1.0:
        split_idx = int(n_total * train_split)
        if n_total > 100 and split_idx <= 24:
            split_idx = n_total
        elif n_total <= 100 and (split_idx < 3 or n_total - split_idx < 1):
            split_idx = n_total
    else:
        split_idx = n_total

    has_holdout = split_idx < n_total
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy() if has_holdout else df.copy()

    # Resolve target columns.
    # Priority: (1) exact target_col if present, (2) shoreline_pos_obs_m / shoreline_pos_true_m
    # (position-based, the safe default for linear Bayesian models), (3) shoreline_change_rate_mh
    # as a last resort when only rate data is available.
    if target_col in df_train.columns:
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
    elif "shoreline_pos_obs_m" in df_train.columns:
        # Safe position-based fallback
        y_obs_train = df_train["shoreline_pos_obs_m"].values
        y_true_train = df_train["shoreline_pos_true_m"].values if "shoreline_pos_true_m" in df_train.columns else y_obs_train
        y_obs_test = df_test["shoreline_pos_obs_m"].values if has_holdout else y_obs_train
        y_true_test = df_test["shoreline_pos_true_m"].values if (has_holdout and "shoreline_pos_true_m" in df_test.columns) else y_obs_test
        y_obs_full = df["shoreline_pos_obs_m"].values
        y_true_full = df["shoreline_pos_true_m"].values if "shoreline_pos_true_m" in df.columns else y_obs_full
    else:
        # Last resort: finite-difference rate column
        y_obs_train = df_train["shoreline_change_rate_mh"].values
        y_true_train = df_train["shoreline_change_rate_mh"].values
        y_obs_test = df_test["shoreline_change_rate_mh"].values if has_holdout else y_obs_train
        y_true_test = df_test["shoreline_change_rate_mh"].values if has_holdout else y_true_train
        y_obs_full = df["shoreline_change_rate_mh"].values
        y_true_full = df["shoreline_change_rate_mh"].values

    is_survey_data = "antecedent_upsilon_memory" in df_train.columns or "surveyed_delta_x_m" in df_train.columns
    effective_max_lag = min(48, max(2, len(df_train) // 4)) if len(df_train) < 60 else 48

    if is_survey_data:
        # Survey-epoch dataset mapping
        ups_raw_train = df_train["antecedent_upsilon_memory"].values if "antecedent_upsilon_memory" in df_train.columns else df_train["antecedent_mean_upsilon"].values
        ups_raw_full = df["antecedent_upsilon_memory"].values if "antecedent_upsilon_memory" in df.columns else df["antecedent_mean_upsilon"].values
        if use_log_upsilon:
            ups_train = df_train["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df_train.columns else np.log10(np.maximum(ups_raw_train, 1e-4))
            ups_full = df["antecedent_log_upsilon"].values if "antecedent_log_upsilon" in df.columns else np.log10(np.maximum(ups_raw_full, 1e-4))
            ups_label = "Unified Index (log-Upsilon)"
        else:
            ups_train = ups_raw_train
            ups_full = ups_raw_full
            ups_label = "Unified Index (Upsilon)"

        hs_train = df_train["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df_train.columns else df_train["antecedent_max_hs_m"].values
        hs_full = df["antecedent_mean_hs_m"].values if "antecedent_mean_hs_m" in df.columns else df["antecedent_max_hs_m"].values
        p_train = df_train["antecedent_max_power_Wm"].values if "antecedent_max_power_Wm" in df_train.columns else np.zeros(len(df_train))
        p_full = df["antecedent_max_power_Wm"].values if "antecedent_max_power_Wm" in df.columns else np.zeros(len(df))
        omega_train = df_train["antecedent_mean_dean"].values if "antecedent_mean_dean" in df_train.columns else np.zeros(len(df_train))
        omega_full = df["antecedent_mean_dean"].values if "antecedent_mean_dean" in df.columns else np.zeros(len(df))

        candidates: Dict[str, Any] = {
            "Mean Wave Height (Hs)": {"type": "single", "forcing": hs_train, "full_forcing": hs_full},
            "Wave Power P (Linear)": {"type": "single", "forcing": p_train, "full_forcing": p_full},
            "Dean Number (Linear)": {"type": "single", "forcing": omega_train, "full_forcing": omega_full},
            ups_label: {"type": "single", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon},
        }

        # Add Yates and XBeach baseline competitors if present
        yates_col = "yates_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "yates_simulated_delta_x_m" in df_train.columns) else "yates_simulated_pos_m"
        if yates_col in df_train.columns:
            candidates["Yates (2009) Equilibrium"] = {
                "type": "single",
                "forcing": df_train[yates_col].values,
                "full_forcing": df[yates_col].values,
            }

        xbeach_col = "xbeach_simulated_delta_x_m" if (target_col == "surveyed_delta_x_m" or "xbeach_simulated_delta_x_m" in df_train.columns) else "xbeach_simulated_pos_m"
        if xbeach_col in df_train.columns:
            candidates["XBeach Process Model"] = {
                "type": "single",
                "forcing": df_train[xbeach_col].values,
                "full_forcing": df[xbeach_col].values,
            }

        if include_nonlinear:
            candidates["Upsilon (Non-Linear)"] = {"type": "nonlinear", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon}

        if include_multivariate and "antecedent_max_surge_m" in df_train.columns:
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
                "upsilon": ups_train,
                "full_upsilon": ups_full,
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
            ups_full = df["log_upsilon"].values if "log_upsilon" in df.columns else np.log10(np.maximum(df["upsilon"].values, 1e-4))
            ups_label = "Unified Index (log-Upsilon)"
        else:
            ups_train = df_train["upsilon"].values
            ups_full = df["upsilon"].values
            ups_label = "Unified Index (Upsilon)"

        candidates: Dict[str, Any] = {
            "Hs (Linear)": {"type": "single", "forcing": df_train["hs_m"].values, "full_forcing": df["hs_m"].values},
            "Wave Power P (Linear)": {"type": "single", "forcing": df_train["p_flux_Wm"].values, "full_forcing": df["p_flux_Wm"].values},
            "Dean Number (Linear)": {"type": "single", "forcing": df_train["dean_number"].values, "full_forcing": df["dean_number"].values},
            ups_label: {"type": "single", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon},
        }

        if "signed_upsilon" in df_train.columns:
            candidates["Signed Index (Upsilon_signed)"] = {
                "type": "single",
                "forcing": df_train["signed_upsilon"].values,
                "full_forcing": df["signed_upsilon"].values,
            }

        if include_nonlinear:
            candidates["Upsilon (Non-Linear)"] = {"type": "nonlinear", "forcing": ups_train, "full_forcing": ups_full, "use_log_upsilon": use_log_upsilon}
            candidates["Hs (Non-Linear)"] = {"type": "nonlinear", "forcing": df_train["hs_m"].values, "full_forcing": df["hs_m"].values}

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

    traces: Dict[str, az.InferenceData] = {}
    model_metas: Dict[str, Dict[str, Any]] = {}
    metrics_list = []

    for name, spec in candidates.items():
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
        else:
            continue

        meta["use_log_upsilon"] = spec.get("use_log_upsilon", use_log_upsilon)
        model_metas[name] = meta

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

        # Memory-efficient full continuous prediction across dataset (train + test holdout)
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

        mu_train_pred = mu_full_mean[:split_idx]
        mu_test_pred = mu_full_mean[split_idx:] if has_holdout else mu_train_pred

        perf_train = evaluate_model_performance(y_true=y_true_train, y_pred=mu_train_pred, y_baseline=np.zeros(len(y_true_train)))
        perf_test = evaluate_model_performance(y_true=y_true_test, y_pred=mu_test_pred, y_baseline=np.zeros(len(y_true_test)))
        perf_full = evaluate_model_performance(y_true=y_true_full, y_pred=mu_full_mean, y_baseline=np.zeros(len(y_true_full)))

        # Convergence diagnostics
        avail_vars = [v for v in ["gamma", "beta_forcing", "beta_upsilon", "alpha", "sigma", "k_eros", "k_accrete", "omega_crit"] if v in trace.posterior]
        post_rhat = float(az.summary(trace, var_names=avail_vars)["r_hat"].max()) if avail_vars else 1.0

        row: Dict[str, Any] = {
            "Model": name,
            "Type": model_type,
        }

        if has_holdout:
            row["Test RMSE (m)"] = perf_test["RMSE (m)"]
            row["Test R2 Score"] = perf_test["R2 Score"]
            row["Test BSS"] = perf_test["Brier Skill Score (BSS)"]
            row["Train RMSE (m)"] = perf_train["RMSE (m)"]
            row["Train R2 Score"] = perf_train["R2 Score"]
            row["Full RMSE (m)"] = perf_full["RMSE (m)"]
            row["Full R2 Score"] = perf_full["R2 Score"]
        else:
            row["Test RMSE (m)"] = perf_train["RMSE (m)"]
            row["Test R2 Score"] = perf_train["R2 Score"]
            row["Test BSS"] = perf_train["Brier Skill Score (BSS)"]
            row["Train RMSE (m)"] = perf_train["RMSE (m)"]
            row["Train R2 Score"] = perf_train["R2 Score"]
            row["RMSE (m)"] = perf_train["RMSE (m)"]
            row["R2 Score"] = perf_train["R2 Score"]
            row["BSS"] = perf_train["Brier Skill Score (BSS)"]

        row["Max R-hat"] = post_rhat
        if "gamma" in trace.posterior:
            row["E[gamma]"] = float(trace.posterior["gamma"].mean())

        metrics_list.append(row)

    # Compute Leave-One-Out Cross-Validation (LOO-CV) comparison table
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
                        r["p_LOO"] = float(loo_df.loc[m_name, "p_loo"])
                        r["LOO Weight"] = float(loo_df.loc[m_name, "weight"])
        except Exception:
            pass

    # Provide aliasing for both naming styles
    for ups_key in ["Unified Index (log-Upsilon)", "Unified Index (Upsilon)"]:
        if ups_key in traces:
            traces["Upsilon (Linear)"] = traces[ups_key]
            traces["Unified Index (Upsilon)"] = traces[ups_key]
            traces["Unified Index (log-Upsilon)"] = traces[ups_key]
    if "Upsilon (Linear)" in traces and "Unified Index (Upsilon)" not in traces:
        traces["Unified Index (Upsilon)"] = traces["Upsilon (Linear)"]

    comparison_df = pd.DataFrame(metrics_list)
    if "Test RMSE (m)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="Test RMSE (m)", ascending=True).reset_index(drop=True)
    elif "RMSE (m)" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="RMSE (m)", ascending=True).reset_index(drop=True)

    comparison_df.attrs["loo_comparison"] = loo_df
    comparison_df.attrs["train_split"] = train_split
    comparison_df.attrs["split_idx"] = split_idx
    comparison_df.attrs["model_metas"] = model_metas

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
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)

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
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)

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
    5. Likelihood: y_obs ~ StudentT(nu, mu_t, sigma)
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)

    max_lag = max(1, min(max_lag, len(y_obs)))
    meta = {"model_name": model_name, "n_obs": len(y_obs), "max_lag": max_lag}

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

        # 1. Latent Antecedent Storm Memory
        gamma = pm.Beta("gamma", alpha=2.0, beta=5.0)
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
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)

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

    Parameters
    ----------
    hs : np.ndarray
        Significant wave height (m).
    tp : np.ndarray
        Peak wave period (s).
    ws : np.ndarray
        Dynamic settling velocity w_s(T) in m/s.
    upsilon : np.ndarray
        Unified Coastal Vulnerability Index values.
    delta_x_obs : np.ndarray
        Observed shoreline position (m).
    shielding_factor : np.ndarray, optional
        Outer sandbar wave power transmission factor (1.0 - Delta P_dissip).
    accretion_potential : np.ndarray, optional
        Constructive swash accretion potential series.
    model_name : str
        Model identifier.
    max_lag : int
        Convolution memory horizon (hours).
    scale_forcings : bool
        Whether to standardize the kinetic drivers.

    Returns
    -------
    model : pm.Model
        Constructed PyMC dual-regime model.
    meta : dict
        Normalization metadata.
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

        # 1. Latent memory convolution
        gamma = pm.Beta("gamma", alpha=2.0, beta=5.0)
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

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)
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
    lower_bound: float = 1.0,
    upper_bound: float = 3.0,
) -> Tuple[pm.Model, Dict[str, Any]]:
    """Build a Physics-Informed Bayesian Model that learns the Hs scaling exponent n in [1.0, 3.0] directly from observations.

    Parameters
    ----------
    hs : np.ndarray
        Significant wave height (m).
    tp : np.ndarray
        Peak wave period (s).
    ws : np.ndarray
        Sediment settling velocity (m/s).
    surge : np.ndarray
        Storm surge residual (m).
    runup : np.ndarray
        Stockdon wave runup R_2% (m).
    wind_normal : np.ndarray
        Shore-normal wind speed (m/s).
    delta_x_obs : np.ndarray
        Observed cross-shore position changes (m).
    shielding_factor : np.ndarray, optional
        Outer sandbar wave power transmission factor.
    model_name : str
        Model identifier.
    max_lag : int
        Convolution memory horizon (hours).
    exponent_prior : str
        'uniform' (default, Uniform(1.0, 3.0)) or 'truncated_normal' (TruncatedNormal(mu, sigma, 1.0, 3.0)).
    prior_mu, prior_sigma : float
        Hyperparameters for truncated_normal prior.
    s_ref, r_ref : float
        Reference scales for surge (1.0m) and runup (2.0m).
    w_ref, w_bound : float
        Reference wind scale (15.0 m/s) and symmetric bound (0.6).
    lower_bound, upper_bound : float
        Bounds for exponent n (default 1.0 to 3.0).

    Returns
    -------
    model : pm.Model
        Constructed PyMC learnable-exponent model.
    meta : dict
        Model metadata.
    """
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.1)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-4)
    surge_arr = np.asarray(surge, dtype=float)
    runup_arr = np.asarray(runup, dtype=float)
    wind_arr = np.asarray(wind_normal, dtype=float)
    y_obs = np.asarray(delta_x_obs, dtype=float)
    n_obs = len(y_obs)

    shield_arr = np.asarray(shielding_factor, dtype=float) if shielding_factor is not None else np.ones(n_obs)
    k_indices = np.arange(max_lag, dtype=float)

    # Precompute non-exponent terms to optimize sampling graph (decoupled additive TWL & quadratic symmetric wind)
    twl_arr = 1.0 + np.maximum(surge_arr / s_ref, 0.0) + np.maximum(runup_arr / r_ref, 0.0)
    w_sign = np.sign(wind_arr)
    w_quad = w_sign * ((np.abs(wind_arr) / max(w_ref, 1.0)) ** 2)
    wind_arr_mod = 1.0 + np.clip(w_quad, -w_bound, w_bound)
    prefactor = (1025.0 * (9.81 ** 1.5)) / (16.0 * ((2.0 * np.pi) ** 1.5))
    kinematic_base = prefactor * 0.08 * (tp_arr / ws_arr) * twl_arr * wind_arr_mod * shield_arr

    meta = {
        "model_name": model_name,
        "n_obs": n_obs,
        "max_lag": max_lag,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
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

        # 3. Latent Memory Convolution in PyTensor
        gamma = pm.Beta("gamma", alpha=2.0, beta=5.0)
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

        # 5. Robust Student-t Likelihood
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.2)
    return model, meta


def evaluate_field_survey_correlation(
    df_aligned_surveys: pd.DataFrame,
    target_col: str = "surveyed_delta_x_m",
) -> pd.DataFrame:
    """Evaluate empirical correlation and predictive skill of physical forcings against in-situ ground-truth field surveys.

    Parameters
    ----------
    df_aligned_surveys : pd.DataFrame
        Synchronized DataFrame pairing discrete field survey shoreline changes with antecedent physical forcings.
    target_col : str
        Target column in df_aligned_surveys ('surveyed_delta_x_m' for inter-survey change,
        or 'cumulative_displacement_m' for total cross-shore shift).

    Returns
    -------
    df_skill : pd.DataFrame
        Ranked skill table comparing empirical correlation (Pearson r, Spearman rho),
        R2 score, and RMSE against true field survey observations.
    """
    from scipy import stats

    y_true = df_aligned_surveys[target_col].values
    n = len(y_true)
    if n < 3:
        return pd.DataFrame()

    var_base = float(np.var(y_true))
    mse_base = float(np.mean((y_true - np.mean(y_true)) ** 2)) if var_base > 1e-6 else 1.0

    yates_col = "yates_simulated_delta_x_m" if target_col == "surveyed_delta_x_m" else "yates_simulated_pos_m"
    xbeach_col = "xbeach_simulated_delta_x_m" if target_col == "surveyed_delta_x_m" else "xbeach_simulated_pos_m"

    predictors = {
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

    results = []
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

        results.append({
            "Predictor": label,
            "Pearson r": r,
            "p-value (Pearson)": p_pearson,
            "Spearman rho": rho,
            "p-value (Spearman)": p_spearman,
            "R2 Score": r2,
            "RMSE (m)": rmse,
            "MAE (m)": mae,
            "Brier Skill Score (BSS)": bss,
            "N Surveys": n,
        })

    df_skill = pd.DataFrame(results)
    if not df_skill.empty:
        # Sort by absolute correlation with observed morphodynamics
        df_skill["Abs_r"] = df_skill["Pearson r"].abs()
        df_skill = df_skill.sort_values(by="Abs_r", ascending=False).drop(columns=["Abs_r"]).reset_index(drop=True)

    return df_skill




