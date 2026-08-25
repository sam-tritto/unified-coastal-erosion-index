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
from typing import Union, Tuple, Optional, Dict, Any

# Fundamental Physical Constants
G_CONST: float = 9.81              # Gravitational acceleration (m/s^2)
TAU_CONST: float = 2.0 * np.pi      # 2*pi
RHO_WATER: float = 1025.0          # Seawater density (kg/m^3)
RHO_SEDIMENT: float = 2650.0       # Quartz sediment density (kg/m^3)
NU_SEAWATER: float = 1.05e-6       # Kinematic viscosity of seawater at ~20C (m^2/s)
DUCK_SHORE_NORMAL_DEG: float = 70.0 # USACE FRF Duck, NC shore-normal azimuth (deg True)

# Global Coastal Site Geomorphology Profiles Database
COASTAL_SITE_PROFILES: Dict[str, Dict[str, Any]] = {
    "duck_nc": {
        "name": "USACE FRF Duck, NC, USA",
        "shore_normal_deg": 70.0,
        "tan_beta": 0.08,
        "d50_m": 0.00035,        # 0.35 mm medium quartz sand
        "berm_crest_m": 2.5,
        "z_crest_m": 2.5,
        "min_slope": 0.035,
        "max_slope": 0.12,
        "tide_range_m": 1.0,     # Microtidal
        "shelter_factor": 1.0,
        "is_reef": False,
        "reef_flat_width_m": 0.0,
        "z_reef_m": -1.0,
        "c_f": 0.08,
        "gamma_b": 0.78,
        "default_transects": ["FRF_0058", "FRF_0109"],
        "coops_station_id": "8651370",
        "primary_buoy_id": "44056",
        "outer_buoy_id": "44100",
        "fallback_buoys": ["44100", "44014", "41025"],
    },
    "narrabeen_aus": {
        "name": "Narrabeen-Collaroy, NSW, Australia",
        "shore_normal_deg": 85.0,
        "tan_beta": 0.09,
        "d50_m": 0.00030,        # 0.30 mm medium marine sand
        "berm_crest_m": 3.0,
        "z_crest_m": 3.0,
        "min_slope": 0.040,
        "max_slope": 0.13,
        "tide_range_m": 1.2,     # Micro-mesotidal
        "shelter_factor": 0.85,
        "is_reef": False,
        "reef_flat_width_m": 0.0,
        "z_reef_m": -1.0,
        "c_f": 0.08,
        "gamma_b": 0.78,
        "default_transects": ["PF2", "PF4"],
    },
    "torrey_pines_ca": {
        "name": "Torrey Pines Beach, CA, USA",
        "shore_normal_deg": 265.0,
        "tan_beta": 0.04,
        "d50_m": 0.00020,        # 0.20 mm fine sand
        "berm_crest_m": 2.0,
        "z_crest_m": 2.0,
        "min_slope": 0.025,
        "max_slope": 0.07,
        "tide_range_m": 1.5,     # Mixed semi-diurnal
        "shelter_factor": 1.0,
        "is_reef": False,
        "reef_flat_width_m": 0.0,
        "z_reef_m": -1.0,
        "c_f": 0.08,
        "gamma_b": 0.78,
        "default_transects": ["TP_NORTH", "TP_SOUTH"],
        "coops_station_id": "9410230",
        "primary_buoy_id": "46254",
        "outer_buoy_id": "46225",
        "fallback_buoys": ["46225", "46232", "46086"],
    },
    "gold_coast_aus": {
        "name": "Gold Coast (Surfers Paradise), QLD, Australia",
        "shore_normal_deg": 65.0,
        "tan_beta": 0.05,
        "d50_m": 0.00022,        # 0.22 mm fine quartz sand
        "berm_crest_m": 2.2,
        "z_crest_m": 2.2,
        "min_slope": 0.030,
        "max_slope": 0.08,
        "tide_range_m": 1.4,
        "shelter_factor": 1.0,
        "is_reef": False,
        "reef_flat_width_m": 0.0,
        "z_reef_m": -1.0,
        "c_f": 0.08,
        "gamma_b": 0.78,
        "default_transects": ["GC_NORTH", "GC_SOUTH"],
    },
    "truc_vert_france": {
        "name": "Truc Vert Beach, Gironde, France",
        "shore_normal_deg": 275.0,
        "tan_beta": 0.06,
        "d50_m": 0.00038,        # 0.38 mm medium-coarse sand
        "berm_crest_m": 3.5,
        "z_crest_m": 3.5,
        "min_slope": 0.035,
        "max_slope": 0.10,
        "tide_range_m": 4.5,     # Macrotidal
        "shelter_factor": 1.0,
        "is_reef": False,
        "reef_flat_width_m": 0.0,
        "z_reef_m": -1.0,
        "c_f": 0.08,
        "gamma_b": 0.78,
        "default_transects": ["TV_MAIN", "TV_NORTH"],
    },
    "oahu_north_shore_hi": {
        "name": "Oahu's North Shore (Waimea / Sunset), HI, USA",
        "shore_normal_deg": 325.0,
        "tan_beta": 0.09,
        "d50_m": 0.00045,        # 0.45 mm medium-coarse carbonate sand
        "berm_crest_m": 3.5,
        "z_crest_m": 3.5,
        "min_slope": 0.045,
        "max_slope": 0.14,
        "tide_range_m": 0.8,     # Microtidal mixed semi-diurnal/diurnal
        "shelter_factor": 1.0,
        "is_reef": True,
        "reef_flat_width_m": 150.0, # 150m reef flat width
        "z_reef_m": -1.0,        # Reef crest elevation relative to MSL (-1.0m)
        "c_f": 0.08,             # Coral friction coefficient
        "gamma_b": 0.78,         # Depth-limited theoretical breaker index
        "default_transects": ["NS_SUNSET", "NS_WAIMEA"],
        "coops_station_id": "1612480",
        "primary_buoy_id": "51201",
        "outer_buoy_id": "51001",
        "fallback_buoys": ["51001", "51101", "51202", "51212"],
    },
    "waikiki_hi": {
        "name": "Waikiki Beach (Canoes / Queens), Oahu, HI, USA",
        "shore_normal_deg": 210.0,
        "tan_beta": 0.06,
        "d50_m": 0.00030,        # 0.30 mm fine-medium carbonate sand
        "berm_crest_m": 2.2,
        "z_crest_m": 2.2,
        "min_slope": 0.035,
        "max_slope": 0.09,
        "tide_range_m": 0.7,     # Microtidal
        "shelter_factor": 0.80,
        "is_reef": True,
        "reef_flat_width_m": 250.0, # 250m wide shallow coral reef flat
        "z_reef_m": -0.8,        # Reef crest elevation relative to MSL (-0.8m)
        "c_f": 0.10,             # High rugosity coral bed friction
        "gamma_b": 0.78,
        "default_transects": ["WK_CANOES", "WK_QUEENS"],
        "coops_station_id": "1612340",
        "primary_buoy_id": "51202",
        "outer_buoy_id": "51201",
        "fallback_buoys": ["51201", "51001"],
    },
}


def compute_dynamic_foreshore_slope(
    hs: Union[float, np.ndarray],
    delta_x: Optional[Union[float, np.ndarray]] = None,
    base_slope: float = 0.08,
    min_slope: float = 0.035,
    max_slope: float = 0.12,
    memory_hours: int = 48,
) -> Union[float, np.ndarray]:
    """Compute dynamic time-varying foreshore beach slope tan(beta)(t).

    Reflects the morphodynamic feedback where storm wave scour flattens the intertidal
    beach profile toward dissipative conditions (min_slope ~ 0.035-0.05), while post-storm
    fair-weather berm welding steepens the foreshore toward reflective conditions (max_slope ~ 0.10-0.12).

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height time series (m).
    delta_x : float or np.ndarray, optional
        Observed shoreline cross-shore displacement (m). If provided, modulates slope with beach width.
    base_slope : float
        Mean baseline foreshore slope (default 0.08).
    min_slope, max_slope : float
        Physical bounds on intertidal slope.
    memory_hours : int
        Antecedent wave memory window for slope adjustment.

    Returns
    -------
    tan_beta_dyn : float or np.ndarray
        Dynamic foreshore slope time series.
    """
    hs_arr = np.asarray(hs, dtype=float)
    is_scalar = np.isscalar(hs)
    hs_vec = np.atleast_1d(hs_arr)
    n = len(hs_vec)

    # 1. Exponential moving average of wave energy
    decay_w = np.exp(-np.arange(memory_hours) / (memory_hours / 3.0))
    decay_w /= np.sum(decay_w)
    padded_hs = np.pad(hs_vec, (memory_hours - 1, 0), mode="edge")
    hs_mean = np.convolve(padded_hs, decay_w, mode="valid")[:n]

    # 2. Wave-steepness flattening response: high storm Hs -> dissipative flattening
    # Normalizing around Hs_ref ~ 1.5m
    slope_mod = np.clip(1.0 - 0.35 * (hs_mean - 1.2) / 2.5, 0.45, 1.45)
    slope_dyn = base_slope * slope_mod

    # 3. Optional shoreline displacement coupling: eroded beach -> flatter profile
    if delta_x is not None:
        dx_vec = np.atleast_1d(np.asarray(delta_x, dtype=float))
        dx_mod = np.clip(1.0 + 0.04 * dx_vec, 0.70, 1.30)
        slope_dyn = slope_dyn * dx_mod

    slope_clipped = np.clip(slope_dyn, min_slope, max_slope)
    return float(slope_clipped[0]) if is_scalar else slope_clipped


def compute_dynamic_grain_size(
    hs: Union[float, np.ndarray],
    base_d50_m: float = 0.00035,
    coarsening_factor: float = 1.25,
    min_d50_m: float = 0.00020,
    max_d50_m: float = 0.00060,
) -> Union[float, np.ndarray]:
    """Compute dynamic storm-coarsened lag grain size D_50(t) via wave winnowing.

    During storms, energetic wave turbulence winnows fine particles offshore into the bar,
    leaving a coarser lag deposit on the active foreshore.
    """
    hs_arr = np.asarray(hs, dtype=float)
    is_scalar = np.isscalar(hs)
    hs_vec = np.atleast_1d(hs_arr)

    # Coarsening activated for storm waves Hs > 2.0m
    storm_intensity = np.clip((hs_vec - 1.5) / 3.0, 0.0, 1.0)
    d50_dyn = base_d50_m * (1.0 + (coarsening_factor - 1.0) * storm_intensity)
    d50_clipped = np.clip(d50_dyn, min_d50_m, max_d50_m)

    return float(d50_clipped[0]) if is_scalar else d50_clipped


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


def compute_tidal_smearing_factor(
    tide_range_m: Union[float, np.ndarray],
    hs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute tidal dissipation and intertidal energy smearing factor Psi_tide.

    Psi_tide = 1 / (1 + TR / (2 * Hs))

    When tidal range TR is large (macrotidal), breaking wave energy is smeared across
    a wide intertidal sweep (W = TR / tan(beta)), diluting resident wave work at the dune toe.
    When TR approaches 0 (microtidal limit), Psi_tide -> 1.0 (concentrated berm attack).

    Parameters
    ----------
    tide_range_m : float or np.ndarray
        Mean spring or diurnal tidal range (m).
    hs : float or np.ndarray
        Significant wave height (m).

    Returns
    -------
    psi_tide : float or np.ndarray
        Dimensionless tidal smearing factor in (0, 1].
    """
    tr_arr = np.maximum(np.asarray(tide_range_m, dtype=float), 0.0)
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.05)
    psi = 1.0 / (1.0 + (tr_arr / (2.0 * hs_arr)))
    if np.isscalar(tide_range_m) and np.isscalar(hs):
        return float(psi)
    return psi


def compute_directional_wave_factor(
    wave_dir_deg: Union[float, np.ndarray],
    shore_normal_deg: float,
    m: float = 2.0,
) -> Union[float, np.ndarray]:
    """Compute the rectified shore-normal directional wave factor Psi_dir.

    Psi_dir = max(0, cos(theta_wave - theta_shore))^m

    Shore-normal incidence (theta_rel = 0) yields Psi_dir = 1.0.
    Oblique waves (e.g. |theta_rel| = 45 deg, m=2) scale to 0.50, matching the shore-normal
    radiation stress S_xx ~ E * cos^2(theta) that drives cross-shore setup and sediment transport.
    Offshore / shadowed waves (|theta_rel| >= 90 deg) are rectified to 0.0 to prevent non-physical accretion.

    Parameters
    ----------
    wave_dir_deg : float or np.ndarray
        Wave approach direction (degrees True, coming from).
    shore_normal_deg : float
        Shore-normal orientation azimuth (degrees True, pointing offshore).
    m : float
        Directional exponent (default 2.0 for radiation stress S_xx).

    Returns
    -------
    psi_dir : float or np.ndarray
        Dimensionless directional alignment factor in [0, 1].
    """
    dir_arr = np.asarray(wave_dir_deg, dtype=float)
    angle_rad = np.radians(dir_arr - float(shore_normal_deg))
    cos_proj = np.maximum(0.0, np.cos(angle_rad))
    psi = cos_proj ** float(m)
    if np.isscalar(wave_dir_deg):
        return float(psi)
    return psi


def compute_longshore_transport_divergence(
    wave_dir_deg: Union[float, np.ndarray],
    shore_normal_deg: float,
) -> Union[float, np.ndarray]:
    """Compute alongshore transport / rotational potential Phi_longshore.

    Phi_longshore = sin(2 * (theta_wave - theta_shore))

    On embayed or rotational beaches (e.g., Narrabeen), cross-shore 1D erosion is coupled
    with alongshore sediment transport gradients Q_ls ~ Hs^2.5 * sin(2 * theta_rel).

    Parameters
    ----------
    wave_dir_deg : float or np.ndarray
        Wave approach direction (degrees True).
    shore_normal_deg : float
        Shore-normal azimuth (degrees True).

    Returns
    -------
    phi_longshore : float or np.ndarray
        Alongshore transport divergence / rotational factor in [-1, 1].
    """
    dir_arr = np.asarray(wave_dir_deg, dtype=float)
    angle_rad = np.radians(dir_arr - float(shore_normal_deg))
    phi = np.sin(2.0 * angle_rad)
    if np.isscalar(wave_dir_deg):
        return float(phi)
    return phi


def compute_reef_dissipation_modulator(
    hs: Union[float, np.ndarray],
    tide: Optional[Union[float, np.ndarray]] = None,
    surge: Optional[Union[float, np.ndarray]] = None,
    z_reef_m: float = -1.0,
    reef_flat_width_m: float = 150.0,
    c_f: float = 0.08,
    gamma_b: float = 0.78,
    is_reef: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute the Reef Dissipation Modulator (Psi_reef) for reef-fronted coastal archetypes.

    Coral reefs act as physical low-pass filters, attenuating deep-water wave energy before it reaches
    the shoreline through a two-step dissipation process:
    1. Crest Clipping (Depth-Limited Breaking):
       h_r(t) = max(0, Tide(t) + S(t) - Z_reef)
       K_c = min(1.0, (gamma_b * h_r(t)) / H_0)
    2. Flat Friction (Coral Drag Decay across Reef Flat):
       K_f = exp(- (c_f * W_r) / h_r(t))

    Composite Modulator & Shoreline Wave Scaling:
       Psi_reef = K_c * K_f in [0, 1]
       H_shore = H_0 * Psi_reef

    Parameters
    ----------
    hs : float or np.ndarray
        Deep-water significant wave height H_0 (m).
    tide : float or np.ndarray, optional
        Astronomical tide elevation relative to MSL (m). Defaults to 0.0m if omitted.
    surge : float or np.ndarray, optional
        Storm surge residual elevation relative to MSL (m). Defaults to 0.0m if omitted.
    z_reef_m : float
        Elevation of the reef crest relative to MSL (m). E.g. -1.0m for crest 1m below MSL.
    reef_flat_width_m : float
        Reef flat width W_r in meters (default 150m). If 0, no flat friction occurs.
    c_f : float
        Coral bed friction coefficient (default 0.08).
    gamma_b : float
        Theoretical depth-limited breaker index (default 0.78).
    is_reef : bool
        Whether the site possesses a fronting coral reef barrier. If False, returns unattenuated (Psi_reef = 1.0).

    Returns
    -------
    psi_reef : float or np.ndarray
        Reef transmission / dissipation modulator in [0, 1].
    h_r : float or np.ndarray
        Instantaneous water depth over the reef crest / flat (m).
    k_c : float or np.ndarray
        Crest depth-limited breaking transmission coefficient in [0, 1].
    k_f : float or np.ndarray
        Reef-flat bottom friction attenuation coefficient in [0, 1].
    hs_shore : float or np.ndarray
        Transmitted shoreline wave height H_shore = H_0 * Psi_reef (m).
    """
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.0)
    is_scalar = np.isscalar(hs)

    if not is_reef or reef_flat_width_m < 0.0:
        psi_reef = np.ones_like(hs_arr)
        h_r = np.zeros_like(hs_arr)
        k_c = np.ones_like(hs_arr)
        k_f = np.ones_like(hs_arr)
        hs_shore = hs_arr.copy()
        if is_scalar:
            return float(psi_reef), float(h_r), float(k_c), float(k_f), float(hs_shore)
        return psi_reef, h_r, k_c, k_f, hs_shore

    tide_arr = np.asarray(tide, dtype=float) if tide is not None else np.zeros_like(hs_arr)
    surge_arr = np.asarray(surge, dtype=float) if surge is not None else np.zeros_like(hs_arr)

    # 1. Instantaneous Reef Depth hr(t) = max(0, Tide(t) + S(t) - Zreef)
    water_level = tide_arr + surge_arr
    h_r = np.maximum(0.0, water_level - float(z_reef_m))

    # 2. Crest Clipping (Depth-Limited Breaking): Kc = min(1.0, (gamma_b * hr) / H0)
    # If hr <= 0, no water over reef -> Kc = 0.0
    safe_hs = np.maximum(hs_arr, 1e-4)
    k_c = np.where(h_r <= 1e-5, 0.0, np.minimum(1.0, (float(gamma_b) * h_r) / safe_hs))

    # 3. Flat Friction (The Coral Drag): Kf = exp(- (c_f * Wr) / hr)
    # If Wr == 0, Kf = 1.0. If hr <= 0, Kf = 0.0.
    if float(reef_flat_width_m) <= 0.0:
        k_f = np.where(h_r <= 1e-5, 0.0, 1.0)
    else:
        k_f = np.where(h_r <= 1e-5, 0.0, np.exp(- (float(c_f) * float(reef_flat_width_m)) / np.maximum(h_r, 1e-4)))

    # Composite Modulator: Psi_reef = Kc * Kf
    psi_reef = np.clip(k_c * k_f, 0.0, 1.0)
    hs_shore = hs_arr * psi_reef

    if is_scalar:
        return float(psi_reef), float(h_r), float(k_c), float(k_f), float(hs_shore)
    return psi_reef, h_r, k_c, k_f, hs_shore


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


def compute_signed_coastal_vulnerability_index(
    upsilon: Union[float, np.ndarray],
    accretion_potential: Union[float, np.ndarray],
    dean_number: Union[float, np.ndarray],
    omega_crit: float = 2.5,
    transition_width: float = 0.4,
    use_log_scale: bool = True,
) -> Union[float, np.ndarray]:
    """Compute the Signed Net Coastal Vulnerability & Accretion Index (Upsilon_signed).

    Provides a physically grounded, bipolar morphodynamic forcing channel where:
    - Positive values (+Upsilon_signed > 0) signify constructive onshore swash accretion and fair-weather berm recovery.
    - Negative values (-Upsilon_signed < 0) signify destructive offshore storm erosion and beach scarping.

    Transition dynamics:
    - Governed by the Dean dimensionless fall velocity Omega = Hs / (w_s * Tp) relative to the critical equilibrium threshold omega_crit ~ 2.0-2.5.
    - Differentiable sigmoid weighting ensures smooth transitions between erosive and accretive swash regimes.

    Parameters
    ----------
    upsilon : float or np.ndarray
        Unified Coastal Vulnerability Index (positive storm erosive hazard).
    accretion_potential : float or np.ndarray
        Constructive swash accretion potential Upsilon_acc = (Tp * sqrt(g * L_0)) / (Hs + 0.1).
    dean_number : float or np.ndarray
        Dean number / dimensionless fall velocity Omega.
    omega_crit : float
        Critical Dean equilibrium threshold (default 2.5).
    transition_width : float
        Sigmoid transition smoothing parameter (default 0.4).
    use_log_scale : bool
        Whether to compute signed response in log-normalized space (default True).

    Returns
    -------
    signed_upsilon : float or np.ndarray
        Signed morphodynamic index series (+ = Accretion, - = Erosion).
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    acc_arr = np.asarray(accretion_potential, dtype=float)
    dean_arr = np.asarray(dean_number, dtype=float)

    is_erosion = 1.0 / (1.0 + np.exp(-(dean_arr - omega_crit) / max(transition_width, 0.01)))
    is_accretion = 1.0 - is_erosion

    if use_log_scale:
        log_ups = np.log10(np.maximum(ups_arr, 1e-4))
        log_acc = np.log10(np.maximum(acc_arr, 1e-2))

        ups_min, ups_max = np.percentile(log_ups, [5, 95])
        acc_min, acc_max = np.percentile(log_acc, [5, 95])

        ups_norm = np.clip((log_ups - ups_min) / max(ups_max - ups_min, 1e-4), 0.0, 5.0)
        acc_norm = np.clip((log_acc - acc_min) / max(acc_max - acc_min, 1e-4), 0.0, 5.0)

        signed_ups = is_accretion * (0.5 + acc_norm) - is_erosion * (0.5 + ups_norm)
    else:
        signed_ups = is_accretion * acc_arr - is_erosion * (ups_arr / 1000.0)

    if np.isscalar(upsilon) and np.isscalar(accretion_potential) and np.isscalar(dean_number):
        return float(signed_ups)
    return signed_ups


def compute_unified_coastal_vulnerability_index(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    surge: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    wind_dir_deg: Union[float, np.ndarray],
    wave_dir_deg: Optional[Union[float, np.ndarray]] = None,
    tide: Optional[Union[float, np.ndarray]] = None,
    tide_range_m: Optional[Union[float, np.ndarray]] = None,
    water_temp_c: Optional[Union[float, np.ndarray]] = None,
    p_outer_Wm: Optional[Union[float, np.ndarray]] = None,
    is_reef: Optional[bool] = None,
    z_reef_m: Optional[float] = None,
    reef_flat_width_m: Optional[float] = None,
    c_f: Optional[float] = None,
    gamma_b: Optional[float] = None,
    hs_exponent: float = 2.5,
    s_ref: float = 1.0,
    r_ref: float = 2.0,
    w_ref: float = 15.0,
    w_bound: float = 0.6,
    d50_m: Optional[Union[float, np.ndarray]] = None,
    tan_beta: Optional[Union[float, np.ndarray]] = None,
    shore_normal_deg: Optional[float] = None,
    site: Optional[str] = None,
    dynamic_slope: bool = False,
    dynamic_grain_size: bool = False,
    delta_x: Optional[Union[float, np.ndarray]] = None,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> pd.DataFrame:
    """Compute the Unified Coastal Vulnerability Index (Υ) with continuous dimensionless modulators and site parameterization.

    Theoretical & Dimensional Formulation
    -------------------------------------
    The Unified Coastal Vulnerability Index (Υ) is a semi-empirical, physics-informed composite index
    designed to synthesize wave power flux, dynamic sediment settling kinetics, foreshore beach slope,
    empirical wave height scaling, total water level (TWL), tidal energy dilution, directional radiation stress,
    and reef-fronted dissipation:

        Υ = C_0 * tan(beta) * [ (H_shore^n * Tp) / ws(T) ] * TWL_amp * Psi_tide * Psi_dir * Wind_amp

    Where:
        H_shore = H_0 * Psi_reef

    Dimensional Breakdown:
    - Prefactor C_0 = (rho * g^1.5) / (16 * (2*pi)^1.5) has physical dimensions [kg / (m^0.5 * s^3)].
    - Foreshore slope tan(beta) is dimensionless [-].
    - Sediment-wave core [ (H_shore^n * Tp) / ws(T) ] has dimensions [m^(n-1) * s^2].
    - At the baseline exponent n = 2.5, the core dimensions are [m^1.5 * s^2], giving Υ net dimensions
      of [kg / s] (kinetic mass forcing rate per second).
    - Continuous modulators (TWL_amp, Psi_tide, Psi_dir, Psi_reef, Wind_amp) are dimensionless scaling factors [-].

    Continuous Dimensionless Scaling Factors:
    1. Tidal Smearing Factor: Psi_tide = 1 / (1 + TR / (2 * Hs))
       Dilutes wave work across a broad intertidal sweep on macrotidal beaches (Truc Vert) while
       concentrating berm attack on microtidal coasts (Duck, NC).
    2. Rectified Shore-Normal Directional Alignment: Psi_dir = max(0, cos(theta_wave - theta_shore))^2
       Matches shore-normal radiation stress S_xx ~ E * cos^2(theta_rel) that drives cross-shore gouging.
    3. Alongshore Rotational Divergence: Phi_longshore = sin(2 * (theta_wave - theta_shore))
       Captures alongshore transport gradients and beach rotation on embayed coasts (Narrabeen).
    4. Reef Dissipation Modulator: Psi_reef = K_c * K_f = min(1, gamma_b * h_r / H_0) * exp(-c_f * W_r / h_r)
       Dynamically models coral reefs as physical low-pass filters via crest clipping and flat friction (Oahu / Waikiki).
    5. Decoupled TWL Amplification: TWL_amp = 1.0 + max(S / s_ref, 0) + max(R_2% / r_ref, 0)
    6. Wind Amplification: Wind_amp = 1.0 + clip(sgn(W_a) * (|W_a| / w_ref)^2, -w_bound, w_bound)

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
    wave_dir_deg : float or np.ndarray, optional
        Wave approach direction in degrees True. Defaults to shore-normal orientation.
    tide : float or np.ndarray, optional
        Astronomical tide water level relative to MSL (m). Used for instantaneous reef depth h_r(t).
    tide_range_m : float or np.ndarray, optional
        Tidal range (m). If omitted, inferred from site profile or set to 0.0 (microtidal limit).
    water_temp_c : float or np.ndarray, optional
        Observed seawater temperature in Celsius.
    p_outer_Wm : float or np.ndarray, optional
        Outer array wave power (W/m) to calculate sandbar dissipation ratio.
    is_reef : bool, optional
        Whether the coast has a fronting coral reef barrier (default from site profile).
    z_reef_m : float, optional
        Elevation of reef crest relative to MSL (m).
    reef_flat_width_m : float, optional
        Width of reef flat in meters (default from site profile).
    c_f : float, optional
        Coral bed friction coefficient (default 0.08).
    gamma_b : float, optional
        Theoretical depth-limited breaker index (default 0.78).
    hs_exponent : float
        Nonlinear wave height scaling exponent n in Hs^n (default 2.5, configurable/learnable n in [1.0, 3.0]).
    s_ref, r_ref : float
        Reference scales for surge (1.0m) and runup (2.0m).
    w_ref, w_bound : float
        Reference wind speed (15.0 m/s) and symmetric modulation bound (0.6).
    d50_m : float or np.ndarray, optional
        Median grain size in meters (default 0.00035m or from site profile). Can be a time series.
    tan_beta : float or np.ndarray, optional
        Foreshore beach slope (default 0.08 or from site profile). Can be a dynamic time series.
    shore_normal_deg : float, optional
        Shore-normal orientation (default 70.0 deg for Duck, NC or from site profile).
    site : str, optional
        Named site key from COASTAL_SITE_PROFILES ('duck_nc', 'narrabeen_aus', 'torrey_pines_ca', 'gold_coast_aus', 'truc_vert_france', 'oahu_north_shore_hi', 'waikiki_hi').
    dynamic_slope : bool
        If True, dynamically adjusts foreshore slope tan(beta)(t) based on storm wave energy and profile feedback.
    dynamic_grain_size : bool
        If True, dynamically coarsens active foreshore grain size D_50(t) during storms via wave winnowing.
    delta_x : float or np.ndarray, optional
        Observed cross-shore position changes used to couple dynamic slope with beach width.
    rho : float
        Water density (1025 kg/m^3).
    g : float
        Gravitational acceleration (9.81 m/s^2).
    tau : float
        2*pi.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with hydrodynamics, dynamic settling kinetics, wave dissipation, reef attenuation, runup, slope, grain size, modulators, and Υ.
    """
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.asarray(tp, dtype=float)
    surge_arr = np.asarray(surge, dtype=float)
    n_exp = float(hs_exponent)
    n_pts = len(np.atleast_1d(hs_arr))

    # 1. Resolve site profile parameters if specified
    site_cfg: Dict[str, Any] = {}
    if site is not None and site.lower() in COASTAL_SITE_PROFILES:
        site_cfg = COASTAL_SITE_PROFILES[site.lower()]

    eff_shore_normal = shore_normal_deg if shore_normal_deg is not None else site_cfg.get("shore_normal_deg", DUCK_SHORE_NORMAL_DEG)
    base_tan_beta = tan_beta if tan_beta is not None else site_cfg.get("tan_beta", 0.08)
    base_d50 = d50_m if d50_m is not None else site_cfg.get("d50_m", 0.00035)
    base_tr = tide_range_m if tide_range_m is not None else site_cfg.get("tide_range_m", 0.0)

    site_is_reef = is_reef if is_reef is not None else site_cfg.get("is_reef", False)
    site_z_reef = z_reef_m if z_reef_m is not None else site_cfg.get("z_reef_m", -1.0)
    site_wr = reef_flat_width_m if reef_flat_width_m is not None else site_cfg.get("reef_flat_width_m", (150.0 if site_is_reef else 0.0))
    site_cf = c_f if c_f is not None else site_cfg.get("c_f", 0.08)
    site_gammab = gamma_b if gamma_b is not None else site_cfg.get("gamma_b", 0.78)

    tr_arr = np.broadcast_to(np.asarray(base_tr, dtype=float), hs_arr.shape) if not np.isscalar(base_tr) else np.full_like(hs_arr, float(base_tr))

    if wave_dir_deg is not None:
        w_dir_arr = np.broadcast_to(np.asarray(wave_dir_deg, dtype=float), hs_arr.shape) if not np.isscalar(wave_dir_deg) else np.full_like(hs_arr, float(wave_dir_deg))
    else:
        # Default to shore-normal wave incidence (Psi_dir = 1.0, Phi_ls = 0.0) when wave direction is unspecified
        w_dir_arr = np.full_like(hs_arr, float(eff_shore_normal))

    # 2. Dynamic geomorphological evolution
    if dynamic_slope and np.isscalar(base_tan_beta):
        min_sl = site_cfg.get("min_slope", 0.035)
        max_sl = site_cfg.get("max_slope", 0.12)
        tan_beta_eff = compute_dynamic_foreshore_slope(hs_arr, delta_x=delta_x, base_slope=float(base_tan_beta), min_slope=min_sl, max_slope=max_sl)
    else:
        tan_beta_eff = base_tan_beta

    if dynamic_grain_size and np.isscalar(base_d50):
        d50_eff = compute_dynamic_grain_size(hs_arr, base_d50_m=float(base_d50))
    else:
        d50_eff = base_d50

    tan_beta_arr = np.broadcast_to(np.asarray(tan_beta_eff, dtype=float), hs_arr.shape) if not np.isscalar(tan_beta_eff) else np.full_like(hs_arr, float(tan_beta_eff))
    d50_arr = np.broadcast_to(np.asarray(d50_eff, dtype=float), hs_arr.shape) if not np.isscalar(d50_eff) else np.full_like(hs_arr, float(d50_eff))

    # 3. Dynamic temperature-dependent settling velocity
    ws = compute_settling_velocity_soulsby(d50_m=d50_arr, water_temp_c=water_temp_c, rho_water=rho, g=g)
    
    # 4. Projected shore-normal wind forcing
    w_a = compute_shore_normal_wind(wind_speed, wind_dir_deg, eff_shore_normal)
    
    # 5. Reef Dissipation Modulator (Psi_reef) for reef-fronted coastlines
    psi_reef, h_reef, k_c, k_f, hs_shore = compute_reef_dissipation_modulator(
        hs=hs_arr,
        tide=tide,
        surge=surge_arr,
        z_reef_m=site_z_reef,
        reef_flat_width_m=site_wr,
        c_f=site_cf,
        gamma_b=site_gammab,
        is_reef=site_is_reef,
    )
    hs_eff = hs_shore if site_is_reef else hs_arr

    # 6. Wave dynamics, power, runup, and breaker regimes using transmitted shoreline wave height
    l_0 = compute_deepwater_wavelength(tp_arr, g=g, tau=tau)
    r_2 = compute_stockdon_wave_runup(hs_eff, tp_arr, tan_beta=tan_beta_arr, g=g, tau=tau)
    xi = compute_iribarren_number(hs_eff, tp_arr, tan_beta=tan_beta_arr, g=g, tau=tau)
    regimes = classify_breaker_regime(xi)
    p_flux = compute_wave_energy_flux(hs_eff, tp_arr, rho=rho, g=g, tau=tau)
    dean = compute_dean_number(hs_eff, tp_arr, ws=ws)
    accretion_pot = compute_constructive_accretion_potential(hs_eff, tp_arr, g=g, tau=tau)
    
    # 7. Cross-array wave dissipation & sandbar shielding
    if p_outer_Wm is not None:
        delta_p, shielding = compute_cross_array_dissipation_ratio(p_outer_Wm, p_flux)
    else:
        delta_p = np.zeros_like(hs_arr)
        shielding = np.ones_like(hs_arr)
    
    # 8. Continuous Dimensionless Physical Modulators
    psi_tide = compute_tidal_smearing_factor(tr_arr, hs_arr)
    psi_dir = compute_directional_wave_factor(w_dir_arr, eff_shore_normal, m=2.0)
    phi_longshore = compute_longshore_transport_divergence(w_dir_arr, eff_shore_normal)

    # 9. Prefactor & Wave-Sediment Kinetic Core: C_0 * (H_shore^n * Tp) / ws(T)
    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    wave_sed_core = ( (np.maximum(hs_eff, 0.01) ** n_exp) * np.maximum(tp_arr, 0.1) ) / ws
    
    # 10. Decoupled Non-Dimensional TWL Amplification Factor
    twl_amplification = (
        1.0
        + np.maximum(surge_arr / max(s_ref, 1e-3), 0.0)
        + np.maximum(r_2 / max(r_ref, 1e-3), 0.0)
    )
    
    # 11. Projected Wind Stress Amplification Factor
    w_sign = np.sign(w_a)
    w_quad = w_sign * ((np.abs(w_a) / max(w_ref, 1.0)) ** 2)
    wind_amplification = 1.0 + np.clip(w_quad, -w_bound, w_bound)
    
    # 12. Unified Coastal Vulnerability Index (Upsilon_unified)
    upsilon = prefactor * tan_beta_arr * wave_sed_core * twl_amplification * psi_tide * psi_dir * wind_amplification
    
    # Shielded Upsilon (accounting for outer sandbar energy dissipation)
    upsilon_shielded = upsilon * shielding
    
    # Raw closed-form synthesis for comparison
    upsilon_raw = prefactor * tan_beta_arr * ((np.maximum(hs_eff, 0.01) ** n_exp) * tp_arr * np.maximum(surge_arr, 0.01) / ws) * np.maximum(w_a, 0.01)
    log_upsilon = np.log10(np.maximum(upsilon, 1e-4))
    
    # Signed Bipolar Accretion & Vulnerability Index (+ = Accretion, - = Erosion)
    signed_upsilon = compute_signed_coastal_vulnerability_index(
        upsilon=upsilon,
        accretion_potential=accretion_pot,
        dean_number=dean,
    )
    
    df_out = pd.DataFrame({
        "hs_m": hs_arr,
        "tp_s": tp_arr,
        "surge_m": surge_arr,
        "runup_r2_m": r_2,
        "total_water_level_m": surge_arr + r_2,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir_deg,
        "wind_normal_ms": w_a,
        "wave_dir_deg": w_dir_arr,
        "tide_range_m": tr_arr,
        "psi_tide": psi_tide,
        "psi_dir": psi_dir,
        "phi_longshore": phi_longshore,
        "psi_reef": psi_reef,
        "h_reef_m": h_reef,
        "k_c_crest": k_c,
        "k_f_friction": k_f,
        "hs_shore_m": hs_shore,
        "w_s_ms": ws,
        "l_0_m": l_0,
        "xi_iribarren": xi,
        "breaker_regime": regimes,
        "p_flux_Wm": p_flux,
        "dean_number": dean,
        "delta_p_dissip": delta_p,
        "shielding_factor": shielding,
        "accretion_potential": accretion_pot,
        "signed_upsilon": signed_upsilon,
        "hs_exponent": np.full_like(hs_arr, n_exp),
        "tan_beta": tan_beta_arr,
        "d50_m": d50_arr,
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

