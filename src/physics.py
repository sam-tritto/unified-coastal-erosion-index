"""Physics formulations and derived hydrodynamic/morphodynamic indices.

Includes:
- Soulsby (1997) sediment settling velocity (w_s)
- Deep-water wave dynamics (wavelength L_0, wave energy flux P)
- Stockdon et al. (2006) wave runup (R_2%) and Total Water Level (TWL)
- Iribarren number (ξ / surf similarity parameter) & breaker regimes
- Dean number / dimensionless fall velocity (Ω)
- Projected shore-normal wind forcing (W_a)
- Unified Coastal Vulnerability Index (Υ) with non-collapsing boundary formulations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional

# Fundamental Physical Constants
G_CONST: float = 9.81              # Gravitational acceleration (m/s^2)
TAU_CONST: float = 2.0 * np.pi      # 2*pi
RHO_WATER: float = 1025.0          # Seawater density (kg/m^3)
RHO_SEDIMENT: float = 2650.0       # Quartz sediment density (kg/m^3)
NU_SEAWATER: float = 1.05e-6       # Kinematic viscosity of seawater at ~20C (m^2/s)
DUCK_SHORE_NORMAL_DEG: float = 70.0 # USACE FRF Duck, NC shore-normal azimuth (deg True)


def compute_kinematic_viscosity_seawater(
    temp_c: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute temperature-dependent kinematic viscosity of seawater (m^2/s).

    Empirical formulation:
    nu(T) = (1.79e-6) / (1 + 0.03368 * T + 0.000221 * T^2)

    Parameters
    ----------
    temp_c : float or np.ndarray
        Water temperature in degrees Celsius (e.g. 4C to 30C).

    Returns
    -------
    nu : float or np.ndarray
        Kinematic viscosity in m^2/s (~1.55e-6 at 5C to 0.95e-6 at 25C).
    """
    t_arr = np.maximum(np.asarray(temp_c, dtype=float), 0.0)
    denominator = 1.0 + 0.03368 * t_arr + 0.000221 * (t_arr ** 2)
    nu = 1.79e-6 / denominator
    if np.isscalar(temp_c):
        return float(nu)
    return nu


def compute_settling_velocity_soulsby(
    d50_m: Union[float, np.ndarray] = 0.00035,
    water_temp_c: Optional[Union[float, np.ndarray]] = None,
    rho_water: float = RHO_WATER,
    rho_sed: float = RHO_SEDIMENT,
    nu: float = NU_SEAWATER,
    g: float = G_CONST,
) -> Union[float, np.ndarray]:
    """Compute sediment settling velocity (w_s) using the Soulsby (1997) formulation with dynamic temperature.

    Parameters
    ----------
    d50_m : float or np.ndarray
        Median grain diameter (D_50) in meters (e.g. 0.00035 m for 0.35 mm sand).
    water_temp_c : float or np.ndarray, optional
        Water temperature in Celsius. If provided, dynamically updates kinematic viscosity nu(T).
    rho_water : float
        Fluid density in kg/m^3 (default 1025 kg/m^3).
    rho_sed : float
        Sediment grain density in kg/m^3 (default 2650 kg/m^3).
    nu : float
        Default kinematic viscosity in m^2/s (used if water_temp_c is None).
    g : float
        Gravitational acceleration in m/s^2 (default 9.81 m/s^2).

    Returns
    -------
    w_s : float or np.ndarray
        Settling velocity in m/s (~0.038 m/s in cold winter vs ~0.052 m/s in warm summer).
    """
    s = rho_sed / rho_water
    d50_arr = np.asarray(d50_m, dtype=float)
    
    if water_temp_c is not None:
        nu_eff = compute_kinematic_viscosity_seawater(water_temp_c)
    else:
        nu_eff = nu
    
    # Dimensionless grain size d_*
    d_star = d50_arr * ((g * (s - 1.0)) / (nu_eff ** 2)) ** (1.0 / 3.0)
    
    # Soulsby (1997) formula
    bracket = np.sqrt(10.36 ** 2 + 1.049 * (d_star ** 3)) - 10.36
    w_s = (nu_eff / d50_arr) * bracket
    
    if np.isscalar(d50_m) and (water_temp_c is None or np.isscalar(water_temp_c)):
        return float(w_s)
    return w_s


def compute_deepwater_wavelength(
    tp: Union[float, np.ndarray],
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute deep-water wavelength L_0 = (g * Tp^2) / (2*pi)."""
    tp_arr = np.asarray(tp, dtype=float)
    return (g * (tp_arr ** 2)) / tau


def compute_stockdon_wave_runup(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    tan_beta: Union[float, np.ndarray] = 0.08,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute the 2% exceedance wave runup elevation (R_2%) using Stockdon et al. (2006).

    R_2% = 1.1 * [ 0.35 * tan(beta) * sqrt(Hs * L0) + 0.5 * sqrt(Hs * L0 * (0.563 * tan(beta)^2 + 0.004)) ]

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak wave period (s).
    tan_beta : float or np.ndarray
        Foreshore beach slope.

    Returns
    -------
    r2 : float or np.ndarray
        2% wave runup elevation above still water level (m).
    """
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.1)
    tan_beta_arr = np.asarray(tan_beta, dtype=float)
    
    l0 = compute_deepwater_wavelength(tp_arr, g=g, tau=tau)
    hl = hs_arr * l0
    
    setup = 0.35 * tan_beta_arr * np.sqrt(hl)
    swash = 0.5 * np.sqrt(hl * (0.563 * (tan_beta_arr ** 2) + 0.004))
    r2 = 1.1 * (setup + swash)
    return r2


def compute_iribarren_number(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    tan_beta: Union[float, np.ndarray] = 0.08,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute Iribarren number (surf similarity parameter ξ).

    ξ = tan(alpha) / sqrt(Hs / L_0) = tan(alpha) * sqrt(g / tau) * (Tp / sqrt(Hs))
    """
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 1e-4)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 1e-2)
    tan_beta_arr = np.asarray(tan_beta, dtype=float)
    
    return tan_beta_arr * np.sqrt(g / tau) * (tp_arr / np.sqrt(hs_arr))


def classify_breaker_regime(xi: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """Classify breaking wave regime from Iribarren number ξ.

    - Spilling: ξ < 0.5 (Dissipative, gradual wave collapse)
    - Plunging: 0.5 <= ξ < 3.3 (Intermediate, violent curling jet)
    - Surging/Collapsing: ξ >= 3.3 (Reflective, low breaking, steep shore)
    """
    xi_arr = np.asarray(xi, dtype=float)
    
    if xi_arr.ndim == 0:
        val = float(xi_arr)
        if val < 0.5:
            return "Spilling"
        elif val < 3.3:
            return "Plunging"
        else:
            return "Surging"
            
    conditions = [
        xi_arr < 0.5,
        (xi_arr >= 0.5) & (xi_arr < 3.3),
        xi_arr >= 3.3
    ]
    choices = ["Spilling", "Plunging", "Surging"]
    return np.select(conditions, choices, default="Plunging")


def compute_wave_energy_flux(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute deep-water wave energy flux / wave power P (W/m).

    P = (rho * g^2 / (16 * tau)) * Hs^2 * Tp
    """
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.asarray(tp, dtype=float)
    coeff = (rho * (g ** 2)) / (16.0 * tau)
    return coeff * (hs_arr ** 2) * tp_arr


def compute_dean_number(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    ws: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute Dean number / dimensionless fall velocity Ω = Hs / (w_s * Tp)."""
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 1e-3)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-4)
    return hs_arr / (ws_arr * tp_arr)


def compute_shore_normal_wind(
    wind_speed: Union[float, np.ndarray],
    wind_dir_deg: Union[float, np.ndarray],
    shore_normal_deg: float = DUCK_SHORE_NORMAL_DEG,
) -> Union[float, np.ndarray]:
    """Project wind vector onto the shore-normal unit vector.

    W_a = |U| * cos(theta_wind - theta_shore_normal)
    """
    speed_arr = np.asarray(wind_speed, dtype=float)
    dir_arr = np.asarray(wind_dir_deg, dtype=float)
    
    angle_rad = np.radians(dir_arr - shore_normal_deg)
    w_a = speed_arr * np.cos(angle_rad)
    return w_a


def compute_cross_array_dissipation_ratio(
    p_outer_Wm: Union[float, np.ndarray],
    p_inner_Wm: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute offshore sandbar wave energy dissipation ratio and beach shielding factor.

    Delta P_dissip = clip((P_outer - P_inner) / (P_outer + 1e-4), 0.0, 1.0)
    Shielding = 1.0 - Delta P_dissip

    When waves break on the offshore sandbar, Delta P_dissip is high (0.6 - 0.9)
    and shielding is high (attenuated energy reaching the beach). When bars breach or
    waves pass unattenuated, Delta P_dissip approaches 0.0 (direct beach gouging).

    Parameters
    ----------
    p_outer_Wm : float or np.ndarray
        Outer buoy wave energy flux / power (e.g. NDBC 44100 at 26m depth) in W/m.
    p_inner_Wm : float or np.ndarray
        Inner buoy wave energy flux / power (e.g. NDBC 44056 at 8m depth) in W/m.

    Returns
    -------
    delta_p_dissip : float or np.ndarray
        Fraction of wave power dissipated across the nearshore profile (0 to 1).
    shielding_factor : float or np.ndarray
        Transmission factor reaching the beach face (1.0 - delta_p_dissip).
    """
    p_out_arr = np.maximum(np.asarray(p_outer_Wm, dtype=float), 0.0)
    p_in_arr = np.maximum(np.asarray(p_inner_Wm, dtype=float), 0.0)
    
    diff = p_out_arr - p_in_arr
    delta_p = np.clip(diff / (p_out_arr + 1e-4), 0.0, 1.0)
    shielding = 1.0 - delta_p
    
    if np.isscalar(p_outer_Wm) and np.isscalar(p_inner_Wm):
        return float(delta_p), float(shielding)
    return delta_p, shielding


def compute_constructive_accretion_potential(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute constructive swash accretion potential for quiet berm welding.

    Upsilon_acc = (Tp * sqrt(g * L_0)) / (Hs + 0.1)

    Driven by long-period low-steepness swell that pushes sand onshore.
    """
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.05)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.5)
    l_0 = compute_deepwater_wavelength(tp_arr, g=g, tau=tau)
    upsilon_acc = (tp_arr * np.sqrt(g * l_0)) / (hs_arr + 0.1)
    if np.isscalar(hs) and np.isscalar(tp):
        return float(upsilon_acc)
    return upsilon_acc


def compute_unified_coastal_vulnerability_index(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    surge: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    wind_dir_deg: Union[float, np.ndarray],
    water_temp_c: Optional[Union[float, np.ndarray]] = None,
    p_outer_Wm: Optional[Union[float, np.ndarray]] = None,
    d50_m: float = 0.00035,
    tan_beta: float = 0.08,
    shore_normal_deg: float = DUCK_SHORE_NORMAL_DEG,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> pd.DataFrame:
    """Compute the Unified Coastal Vulnerability Index (Υ) with dynamic viscosity, settling velocity, and sandbar shielding.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak wave period (s).
    surge : float or np.ndarray
        Storm surge residual elevation S = Observed - Tide (m).
    wind_speed : float or np.ndarray
        Wind speed |U| in m/s.
    wind_dir_deg : float or np.ndarray
        Wind direction in degrees True.
    water_temp_c : float or np.ndarray, optional
        Observed seawater temperature in Celsius.
    p_outer_Wm : float or np.ndarray, optional
        Outer array wave power (W/m) to calculate sandbar dissipation ratio.
    d50_m : float
        Median grain size (default 0.00035 m).
    tan_beta : float
        Beach slope (default 0.08).
    shore_normal_deg : float
        Shore-normal orientation (default 70.0 deg for Duck, NC).
    rho : float
        Water density (1025 kg/m^3).
    g : float
        Gravitational acceleration (9.81 m/s^2).
    tau : float
        2*pi.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with hydrodynamics, dynamic settling kinetics, wave dissipation, runup, and Υ.
    """
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.asarray(tp, dtype=float)
    surge_arr = np.asarray(surge, dtype=float)
    
    # 1. Dynamic temperature-dependent settling velocity
    ws = compute_settling_velocity_soulsby(d50_m=d50_m, water_temp_c=water_temp_c, rho_water=rho, g=g)
    
    # 2. Projected shore-normal wind forcing
    w_a = compute_shore_normal_wind(wind_speed, wind_dir_deg, shore_normal_deg)
    
    # 3. Wave dynamics, power, and runup
    l_0 = compute_deepwater_wavelength(tp_arr, g=g, tau=tau)
    r_2 = compute_stockdon_wave_runup(hs_arr, tp_arr, tan_beta=tan_beta, g=g, tau=tau)
    xi = compute_iribarren_number(hs_arr, tp_arr, tan_beta=tan_beta, g=g, tau=tau)
    regimes = classify_breaker_regime(xi)
    p_flux = compute_wave_energy_flux(hs_arr, tp_arr, rho=rho, g=g, tau=tau)
    dean = compute_dean_number(hs_arr, tp_arr, ws=ws)
    accretion_pot = compute_constructive_accretion_potential(hs_arr, tp_arr, g=g, tau=tau)
    
    # 4. Cross-array wave dissipation & sandbar shielding
    if p_outer_Wm is not None:
        delta_p, shielding = compute_cross_array_dissipation_ratio(p_outer_Wm, p_flux)
    else:
        delta_p = np.zeros_like(hs_arr)
        shielding = np.ones_like(hs_arr)
    
    # 5. Prefactor: (rho * g^1.5) / (16 * tau^1.5)
    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    
    # Wave energy & sediment kinetic core: (Hs^2.5 * Tp) / ws(T)
    wave_sed_core = ( (np.maximum(hs_arr, 0.01) ** 2.5) * np.maximum(tp_arr, 0.1) ) / ws
    
    # Total Water Level / Surge amplification factor (non-collapsing)
    twl_amplification = 1.0 + np.maximum((surge_arr + r_2) / (hs_arr + 0.1), 0.0)
    
    # Wind amplification factor
    wind_amplification = 1.0 + np.maximum(w_a / 10.0, -0.3)
    
    # Robust Unified Coastal Vulnerability Index
    upsilon = prefactor * tan_beta * wave_sed_core * twl_amplification * wind_amplification
    
    # Shielded Upsilon (accounting for outer sandbar energy dissipation)
    upsilon_shielded = upsilon * shielding
    
    # Raw closed-form synthesis for comparison
    upsilon_raw = prefactor * tan_beta * ((np.maximum(hs_arr, 0.01) ** 2.5) * tp_arr * np.maximum(surge_arr, 0.01) / ws) * np.maximum(w_a, 0.01)
    log_upsilon = np.log10(np.maximum(upsilon, 1e-4))
    
    df_out = pd.DataFrame({
        "hs_m": hs_arr,
        "tp_s": tp_arr,
        "surge_m": surge_arr,
        "runup_r2_m": r_2,
        "total_water_level_m": surge_arr + r_2,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir_deg,
        "wind_normal_ms": w_a,
        "w_s_ms": ws,
        "l_0_m": l_0,
        "xi_iribarren": xi,
        "breaker_regime": regimes,
        "p_flux_Wm": p_flux,
        "dean_number": dean,
        "delta_p_dissip": delta_p,
        "shielding_factor": shielding,
        "accretion_potential": accretion_pot,
        "upsilon": upsilon,
        "upsilon_shielded": upsilon_shielded,
        "upsilon_raw": upsilon_raw,
        "log_upsilon": log_upsilon,
    })
    if water_temp_c is not None:
        df_out["water_temp_c"] = water_temp_c
    
    return df_out


def compute_jonswap_wind_wave_hindcast(
    wind_speed_ms: Union[float, np.ndarray],
    wind_dir_deg: Union[float, np.ndarray],
    fetch_km: float = 120.0,
    shore_normal_deg: float = DUCK_SHORE_NORMAL_DEG,
    background_swell_hs: float = 0.45,
    background_swell_tp: float = 8.5,
    g: float = G_CONST,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Empirical wind-wave hindcasting via USACE CEM / JONSWAP growth formulas.

    Used as an authoritative physical fallback when buoy telemetry drops during storms.
    Combines fetch-limited wind-sea growth with background Atlantic swell.

    Parameters
    ----------
    wind_speed_ms : float or np.ndarray
        Wind speed U_10 in m/s.
    wind_dir_deg : float or np.ndarray
        Wind direction in degrees True.
    fetch_km : float
        Effective offshore/onshore fetch in kilometers (default 120.0 km).
    shore_normal_deg : float
        Shore-normal orientation (default 70 deg for Duck, NC).
    background_swell_hs : float
        Baseline ocean swell significant wave height (m).
    background_swell_tp : float
        Baseline ocean swell peak period (s).

    Returns
    -------
    hs_hindcast : float or np.ndarray
        Estimated significant wave height (m).
    tp_hindcast : float or np.ndarray
        Estimated peak wave period (s).
    """
    speed_arr = np.maximum(np.asarray(wind_speed_ms, dtype=float), 0.1)
    dir_arr = np.asarray(wind_dir_deg, dtype=float)

    # Relative angle to shore-normal
    angle_diff = np.radians((dir_arr - shore_normal_deg + 180.0) % 360.0 - 180.0)
    # Directional fetch modulation: maximum for onshore winds (cos > 0)
    onshore_factor = np.maximum(np.cos(angle_diff), 0.15)
    effective_fetch_m = fetch_km * 1000.0 * onshore_factor

    # Dimensionless fetch: X_tilde = (g * F) / U^2
    x_tilde = np.maximum((g * effective_fetch_m) / (speed_arr ** 2), 1.0)

    # Standard CEM / JONSWAP fetch-limited wave height: H_tilde = min(0.0016 * X_tilde^0.5, 0.2433)
    h_tilde = np.minimum(0.0016 * np.sqrt(x_tilde), 0.2433)
    hs_windsea = h_tilde * (speed_arr ** 2) / g

    # Standard CEM / JONSWAP peak period: T_tilde = min(0.2857 * X_tilde^(1/3), 8.134)
    t_tilde = np.minimum(0.2857 * (x_tilde ** (1.0 / 3.0)), 8.134)
    tp_windsea = t_tilde * speed_arr / g

    # Combine wind-sea energy with background swell: Hs_tot = sqrt(Hs_windsea^2 + Hs_swell^2)
    hs_tot = np.sqrt(hs_windsea ** 2 + background_swell_hs ** 2)
    # Period weighting: energy-weighted or dominant peak
    tp_tot = np.where(hs_windsea > background_swell_hs, tp_windsea, background_swell_tp)
    tp_tot = np.maximum(tp_tot, 3.5)

    if np.isscalar(wind_speed_ms) and np.isscalar(wind_dir_deg):
        return float(hs_tot), float(tp_tot)
    return hs_tot, tp_tot

