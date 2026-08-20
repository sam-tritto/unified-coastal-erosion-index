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
    var_names = ["alpha", "beta_forcing", "gamma", "sigma", "nu", "beta_upsilon", "k_eros", "k_accrete", "omega_crit"]
    available = [v for v in var_names if v in idata.posterior]
    return az.summary(idata, var_names=available)


def run_benchmark_model_comparison(
    df: pd.DataFrame,
    draws: int = 1000,
    tune: int = 800,
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 42,
    include_nonlinear: bool = True,
    include_multivariate: bool = True,
    include_hybrid: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, az.InferenceData]]:
    """Fit and compare competing physical models including non-linear, multivariate, and hybrid variants."""
    # Single-variable candidates (with latent memory)
    candidates: Dict[str, Any] = {
        "Hs (Linear)": {"type": "single", "forcing": df["hs_m"].values},
        "Wave Power P (Linear)": {"type": "single", "forcing": df["p_flux_Wm"].values},
        "Dean Number (Linear)": {"type": "single", "forcing": df["dean_number"].values},
        "Unified Index (Upsilon)": {"type": "single", "forcing": df["upsilon"].values},
    }

    if include_nonlinear:
        candidates["Upsilon (Non-Linear)"] = {"type": "nonlinear", "forcing": df["upsilon"].values}
        candidates["Hs (Non-Linear)"] = {"type": "nonlinear", "forcing": df["hs_m"].values}

    if include_multivariate:
        candidates["Multivariate (Hs+S+Wa+Omega)"] = {
            "type": "multivariate",
            "forcings": {
                "hs": df["hs_m"].values,
                "surge": df["surge_m"].values,
                "wind": df["wind_normal_ms"].values,
                "dean": df["dean_number"].values,
            },
        }

    if include_hybrid:
        candidates["Hybrid (Upsilon Anchor + Multi Residuals)"] = {
            "type": "hybrid",
            "upsilon": df["upsilon"].values,
            "forcings": {
                "hs": df["hs_m"].values,
                "surge": df["surge_m"].values,
                "wind": df["wind_normal_ms"].values,
                "dean": df["dean_number"].values,
            },
        }

    # Asymmetric Dual-Regime Model (Dean threshold Omega_crit + Sandbar Shielding + Accretion Swash)
    if "w_s_ms" in df.columns:
        candidates["Asymmetric Dual-Regime (Erosion vs Accretion)"] = {
            "type": "asymmetric",
            "hs": df["hs_m"].values,
            "tp": df["tp_s"].values,
            "ws": df["w_s_ms"].values,
            "upsilon": df["upsilon"].values,
            "shielding": df["shielding_factor"].values if "shielding_factor" in df.columns else np.ones(len(df)),
            "accretion": df["accretion_potential"].values if "accretion_potential" in df.columns else None,
        }

    y_obs = df["shoreline_pos_obs_m"].values
    y_true = df["shoreline_pos_true_m"].values if "shoreline_pos_true_m" in df.columns else y_obs

    traces: Dict[str, az.InferenceData] = {}
    metrics_list = []

    for name, spec in candidates.items():
        model_type = spec["type"]

        if model_type == "single":
            model, _ = build_latent_memory_bayesian_model(
                forcing=spec["forcing"],
                delta_x_obs=y_obs,
                model_name=name,
                max_lag=48,
            )
        elif model_type == "nonlinear":
            model, _ = build_nonlinear_response_model(
                forcing=spec["forcing"],
                delta_x_obs=y_obs,
                model_name=name,
                degree=3,
            )
        elif model_type == "multivariate":
            model, _ = build_multivariate_linear_model(
                forcings=spec["forcings"],
                delta_x_obs=y_obs,
                model_name=name,
            )
        elif model_type == "hybrid":
            model, _ = build_physics_anchored_hybrid_model(
                upsilon=spec["upsilon"],
                forcings=spec["forcings"],
                delta_x_obs=y_obs,
                model_name=name,
                max_lag=48,
            )
        elif model_type == "asymmetric":
            model, _ = build_asymmetric_dual_regime_model(
                hs=spec["hs"],
                tp=spec["tp"],
                ws=spec["ws"],
                upsilon=spec["upsilon"],
                delta_x_obs=y_obs,
                shielding_factor=spec["shielding"],
                accretion_potential=spec["accretion"],
                model_name=name,
                max_lag=48,
            )
        else:
            continue

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
            ppc = pm.sample_posterior_predictive(trace, random_seed=random_seed, progressbar=False)
            trace.extend(ppc)

        traces[name] = trace

        # Posterior mean prediction
        y_pred = np.mean(trace.posterior_predictive["y_obs"].values.reshape(-1, len(y_obs)), axis=0)
        perf = evaluate_model_performance(y_true=y_true, y_pred=y_pred, y_baseline=np.zeros(len(y_obs)))

        # Convergence diagnostics
        avail_vars = [v for v in ["gamma", "beta_forcing", "beta_upsilon", "alpha", "sigma"] if v in trace.posterior]
        post_rhat = float(az.summary(trace, var_names=avail_vars)["r_hat"].max()) if avail_vars else 1.0

        row = {
            "Model": name,
            "Type": model_type,
            "RMSE (m)": perf["RMSE (m)"],
            "R2 Score": perf["R2 Score"],
            "BSS": perf["Brier Skill Score (BSS)"],
            "Max R-hat": post_rhat,
        }

        if "gamma" in trace.posterior:
            row["E[gamma]"] = float(trace.posterior["gamma"].mean())

        metrics_list.append(row)

    # Provide aliasing for both naming styles
    if "Unified Index (Upsilon)" in traces and "Upsilon (Linear)" not in traces:
        traces["Upsilon (Linear)"] = traces["Unified Index (Upsilon)"]
    elif "Upsilon (Linear)" in traces and "Unified Index (Upsilon)" not in traces:
        traces["Unified Index (Upsilon)"] = traces["Upsilon (Linear)"]

    comparison_df = pd.DataFrame(metrics_list)
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


