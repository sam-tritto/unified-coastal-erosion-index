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
        "name": "Duck, NC, USA",
        "latitude": 36.183,
        "longitude": -75.750,
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
        "is_embayment": False,   # Straight open barrier beach
        "l_beach_m": 1000.0,     # FRF experimental zone alongshore span (m)
        "y_center_m": 500.0,     # FRF central research pier coordinate (m)
        "alpha_rot": 0.0,        # No headland-bounded rotation
        "default_transects": ["FRF_0058", "FRF_0109", "FRF_m091"],
        "transect_coords_m": {"FRF_0058": 580.0, "FRF_0109": 1090.0, "FRF_m091": -910.0},
        "coops_station_id": "8651370",
        "primary_buoy_id": "44056",
        "outer_buoy_id": "44100",
        "fallback_buoys": ["44100", "44014", "41025"],
    },
    "narrabeen_aus": {
        "name": "Narrabeen-Collaroy, NSW, Australia",
        "latitude": -33.720,
        "longitude": 151.310,
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
        "is_embayment": True,    # Classic headland-bounded rotational embayment
        "l_beach_m": 3600.0,     # 3.6 km embayment length from Collaroy to North Narrabeen
        "y_center_m": 1800.0,    # Embayment center fulcrum (PF2)
        "alpha_rot": 12.0,       # 12.0 m rotational sensitivity during East Coast Low storms
        "default_transects": ["PF2", "PF4", "PF1"],
        "transect_coords_m": {"PF1": 3400.0, "PF2": 1800.0, "PF4": 400.0, "PF6": 200.0, "PF8": 50.0},
    },
    "torrey_pines_ca": {
        "name": "Torrey Pines Beach, CA, USA",
        "latitude": 32.900,
        "longitude": -117.250,
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
        "is_embayment": True,    # Embayed cliff-backed pocket beach
        "l_beach_m": 4800.0,     # 4.8 km compartment length
        "y_center_m": 2400.0,    # Central fulcrum
        "alpha_rot": 6.0,        # Moderate rotational response under swell obliquity
        "default_transects": ["TP_NORTH", "TP_SOUTH"],
        "transect_coords_m": {"TP_NORTH": 4200.0, "TP_SOUTH": 1200.0},
        "coops_station_id": "9410230",
        "primary_buoy_id": "46254",
        "outer_buoy_id": "46225",
        "fallback_buoys": ["46225", "46232", "46086"],
    },
    "gold_coast_aus": {
        "name": "Gold Coast (Surfers Paradise), QLD, Australia",
        "latitude": -28.000,
        "longitude": 153.430,
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
        "is_embayment": False,   # Semi-open northward littoral drift system
        "l_beach_m": 6000.0,     # 6.0 km compartment length
        "y_center_m": 3000.0,
        "alpha_rot": 3.0,
        "default_transects": ["GC_NORTH", "GC_SOUTH"],
        "transect_coords_m": {"GC_NORTH": 5000.0, "GC_SOUTH": 1500.0},
    },
    "truc_vert_france": {
        "name": "Truc Vert Beach, Gironde, France",
        "latitude": 44.730,
        "longitude": -1.250,
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
        "is_embayment": False,   # Open continuous macrotidal sandy barrier coast
        "l_beach_m": 2000.0,     # 2.0 km research grid span
        "y_center_m": 1000.0,
        "alpha_rot": 0.0,        # Pure cross-shore bar dynamics
        "default_transects": ["TV_MAIN", "TV_NORTH"],
        "transect_coords_m": {"TV_MAIN": 1000.0, "TV_NORTH": 1800.0},
    },
    "oahu_north_shore_hi": {
        "name": "Oahu's North Shore (Waimea / Sunset), HI, USA",
        "latitude": 21.630,
        "longitude": -158.060,
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
        "l_d_ref_m": 25.0,       # Characteristic shallow-water reef damping scale (m)
        "is_embayment": True,    # Pocket embayments bounded by basalt rocky points
        "l_beach_m": 7200.0,     # 7.2 km coastal segment length
        "y_center_m": 3600.0,    # Embayment fulcrum
        "alpha_rot": 8.0,        # Alongshore rotation between headlands under NW vs NE swell
        "default_transects": ["NS_SUNSET", "NS_WAIMEA"],
        "transect_coords_m": {"NS_SUNSET": 6200.0, "NS_WAIMEA": 1200.0},
        "coops_station_id": "1612480",
        "primary_buoy_id": "51201",
        "outer_buoy_id": "51001",
        "fallback_buoys": ["51001", "51101", "51202", "51212"],
    },
    "waikiki_hi": {
        "name": "Waikiki Beach (Canoes / Queens), Oahu, HI, USA",
        "latitude": 21.275,
        "longitude": -157.825,
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
        "l_d_ref_m": 25.0,       # Characteristic damping scale (m)
        "is_embayment": True,    # Urban pocket beach bounded by groynes and reef channels
        "l_beach_m": 3200.0,     # 3.2 km compartment length
        "y_center_m": 1600.0,    # Center coordinate
        "alpha_rot": 5.0,        # Controlled embayment rotation
        "default_transects": ["WK_CANOES", "WK_QUEENS"],
        "transect_coords_m": {"WK_CANOES": 2100.0, "WK_QUEENS": 1100.0},
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


def compute_asymmetric_wind_amplification(
    w_a: Union[float, np.ndarray],
    w_ref_on: float = 15.0,
    w_ref_off: float = 15.0,
    w_bound_on: float = 0.6,
    w_bound_off: float = 0.3,
    c_on: float = 1.0,
    c_off: float = 0.5,
) -> Union[float, np.ndarray]:
    """Compute asymmetric wind stress amplification factor for shore-normal wind velocity W_a.

    Physical Asymmetry Formulation:
    - Onshore winds (W_a >= 0): Enhance wind set-up, overtop dunes via surface shear,
      and drive cross-shore momentum flux into swash zone:
      Wind_amp = 1.0 + min(c_on * (W_a / w_ref_on)^2, w_bound_on)
    - Offshore winds (W_a < 0): Oppose incoming wave crests, flattening wave steepness
      and damping set-up, with attenuated impact:
      Wind_amp = 1.0 - min(c_off * (|W_a| / w_ref_off)^2, w_bound_off)

    Parameters
    ----------
    w_a : float or np.ndarray
        Projected shore-normal wind velocity (m/s), positive onshore, negative offshore.
    w_ref_on : float
        Reference wind speed for onshore wind scaling (m/s, default 15.0).
    w_ref_off : float
        Reference wind speed for offshore wind scaling (m/s, default 15.0).
    w_bound_on : float
        Upper modulation bound for onshore wind amplification (default 0.6).
    w_bound_off : float
        Lower modulation bound for offshore wind damping (default 0.3).
    c_on : float
        Scaling gain coefficient for onshore wind (default 1.0).
    c_off : float
        Scaling gain coefficient for offshore wind (default 0.5).

    Returns
    -------
    wind_amp : float or np.ndarray
        Dimensionless wind amplification factor.
    """
    is_scalar = np.isscalar(w_a)
    w_arr = np.asarray(w_a, dtype=float)

    mod_on = np.minimum(
        c_on * ((np.maximum(w_arr, 0.0) / max(w_ref_on, 1.0)) ** 2),
        w_bound_on,
    )
    mod_off = np.minimum(
        c_off * ((np.maximum(-w_arr, 0.0) / max(w_ref_off, 1.0)) ** 2),
        w_bound_off,
    )

    wind_amp = 1.0 + mod_on - mod_off
    return float(wind_amp) if is_scalar else wind_amp


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
    psi = np.clip(1.0 / (1.0 + (tr_arr / (2.0 * hs_arr))), 0.0, 1.0)
    if np.isscalar(tide_range_m) and np.isscalar(hs):
        return float(psi)
    return psi


def compute_instantaneous_tide_phase_factor(
    tide: Union[float, np.ndarray],
    tide_range_m: Union[float, np.ndarray],
    msl_m: float = 0.0,
    p: float = 1.0,
) -> Union[float, np.ndarray]:
    """Compute instantaneous high-tide wave energy gating factor Psi_tide_phase (Psi_phase).

    Psi_phase(t) = [ (eta_tide(t) - MSL) / (TR / 2) + 1 ]^p

    Weights wave power by instantaneous water level relative to the berm crest / MSL.
    At low tide (eta_tide = -TR/2), Psi_phase -> 0.0, eliminating false-positive erosion
    signals when large waves break far offshore across a wide intertidal sweep (e.g. Truc Vert).
    At high tide (eta_tide = +TR/2), Psi_phase -> 2^p, concentrating wave attack against
    the upper beach and dune toe.
    At MSL (eta_tide = 0.0) or microtidal limit (TR -> 0), Psi_phase -> 1.0.

    Parameters
    ----------
    tide : float or np.ndarray
        Instantaneous astronomical tide elevation or water level relative to MSL (m).
    tide_range_m : float or np.ndarray
        Tidal range (m).
    msl_m : float, default 0.0
        Mean Sea Level datum reference offset (m).
    p : float, default 1.0
        Tidal phase gating exponent.

    Returns
    -------
    psi_phase : float or np.ndarray
        Dimensionless tidal phase gating factor in [0, 2^p].
    """
    tide_arr = np.asarray(tide, dtype=float)
    tr_arr = np.maximum(np.asarray(tide_range_m, dtype=float), 0.0)

    # In microtidal limit (TR -> 0), tidal variation is negligible -> psi_phase = 1.0
    semi_amp = tr_arr / 2.0
    is_micro = semi_amp <= 1e-4

    safe_semi = np.where(is_micro, 1.0, semi_amp)
    norm_tide = np.where(is_micro, 0.0, (tide_arr - float(msl_m)) / safe_semi)

    # Base bracket: [ (eta - MSL)/(TR/2) + 1 ] clipped at 0.0, with safe upper ceiling [0, 10.0]
    base_bracket = np.maximum(0.0, norm_tide + 1.0)
    psi = np.clip(base_bracket ** float(p), 0.0, 10.0)

    if np.isscalar(tide) and np.isscalar(tide_range_m):
        return float(psi)
    return psi


compute_tide_phase_factor = compute_instantaneous_tide_phase_factor
compute_tide_phase_gating = compute_instantaneous_tide_phase_factor


def compute_tidal_inundation_factor(
    tide: Union[float, np.ndarray],
    mhw: float = 0.0,
    hs: Union[float, np.ndarray] = 1.0,
    a_min: float = 0.1,
    a_max: float = 10.0,
) -> Union[float, np.ndarray]:
    """Compute exponential tidal inundation modulator with strict numerical clipping.

    Psi_tide(eta, Hs) = clip(exp((eta - MHW) / max(0.5, Hs)), a_min=0.1, a_max=10.0)

    Captures vertical water level excursion relative to Mean High Water (MHW) while
    preventing denominator collapse via max(0.5, Hs) and bounding exponential magnification
    via safe numerical clamping [a_min, a_max] during extreme hurricane surges or calm spring tides.

    Parameters
    ----------
    tide : float or np.ndarray
        Instantaneous astronomical tide elevation or water level eta (m).
    mhw : float, default 0.0
        Mean High Water datum elevation relative to MSL/reference (m).
    hs : float or np.ndarray, default 1.0
        Significant wave height (m).
    a_min : float, default 0.1
        Minimum numerical floor preventing complete attenuation.
    a_max : float, default 10.0
        Maximum numerical ceiling preventing unbounded exponential explosion.

    Returns
    -------
    psi_tide : float or np.ndarray
        Saturated tidal inundation modulator in [a_min, a_max].
    """
    eta_arr = np.asarray(tide, dtype=float)
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.5)
    exponent = np.clip((eta_arr - float(mhw)) / hs_arr, -10.0, 10.0)
    psi = np.clip(np.exp(exponent), float(a_min), float(a_max))

    if np.isscalar(tide) and np.isscalar(hs):
        return float(psi)
    return psi


compute_tidal_inundation_modulator = compute_tidal_inundation_factor



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


def compute_coupled_2d_embayment_displacement(
    delta_x_cross_shore: Union[float, np.ndarray],
    phi_longshore: Union[float, np.ndarray],
    y_alongshore_m: Union[float, np.ndarray],
    y_center_m: float = 1800.0,
    l_beach_m: float = 3600.0,
    alpha_rot: float = 12.0,
) -> Union[float, np.ndarray]:
    """Compute coupled 2D shoreline displacement for embayed and rotational coastlines.

    Governing Coupled 2D Proxy Formulation:
        Delta_X_net = Delta_X_cross_shore(Upsilon) + alpha_rot * Phi_longshore * ((y - y_center) / L_beach)

    Where:
        Phi_longshore = sin(2 * (theta_wave - theta_shore)) in [-1, 1]
        (y - y_center) / L_beach in [-0.5, 0.5] represents the normalized alongshore transect position.

    Physical & Morphodynamic Mechanisms:
    - Cross-Shore Term Delta_X_cross_shore(Upsilon): Uniform cross-shore volumetric cut/fill
      driven by storm wave power, surge, runup, and sediment settling kinetics.
    - Alongshore Transport Gradient Term alpha_rot * Phi_longshore * ((y - y_center) / L_beach):
      Captures alongshore transport gradients and headland-bounded beach rotation (e.g. Narrabeen).
      Oblique East Coast Low (ECL) storms driving waves from the southeast (Phi_longshore > 0)
      induce counter-clockwise beach rotation: sand is eroded from the southern updrift end (PF4)
      and transported alongshore to accrete at the northern downdrift headland (PF1) during the exact same storm.
    - Central Embayment Fulcrum (y = y_center): Rotational displacement is exactly 0,
      so the central profile experiences pure cross-shore morphodynamics.

    Parameters
    ----------
    delta_x_cross_shore : float or np.ndarray
        Cross-shore shoreline displacement (m) predicted from Upsilon / equilibrium morphodynamics.
    phi_longshore : float or np.ndarray
        Alongshore transport divergence / rotational potential in [-1, 1].
    y_alongshore_m : float or np.ndarray
        Alongshore coordinate of the target transect/profile in meters [m] (0 <= y <= L_beach).
    y_center_m : float
        Central fulcrum / pivot coordinate of the embayment in meters (default 1800.0 m for Narrabeen).
    l_beach_m : float
        Total length of the embayed beach compartment in meters (default 3600.0 m for Narrabeen).
    alpha_rot : float
        Rotational mobility scaling coefficient in meters (default 12.0 m).

    Returns
    -------
    delta_x_net : float or np.ndarray
        Net coupled 2D shoreline displacement (m) accounting for cross-shore erosion and alongshore rotation.
    """
    dx_cross = np.asarray(delta_x_cross_shore, dtype=float)
    phi_ls = np.asarray(phi_longshore, dtype=float)
    y_arr = np.asarray(y_alongshore_m, dtype=float)

    l_eff = max(float(l_beach_m), 1.0)
    norm_y = (y_arr - float(y_center_m)) / l_eff
    rot_offset = float(alpha_rot) * phi_ls * norm_y
    dx_net = dx_cross + rot_offset

    if np.isscalar(delta_x_cross_shore) and np.isscalar(phi_longshore) and np.isscalar(y_alongshore_m):
        return float(dx_net)
    return dx_net


# Physical aliases for Alongshore Transport Gradient & Beach Rotation
compute_alongshore_transport_gradient_rotation = compute_coupled_2d_embayment_displacement
compute_embayed_beach_rotation = compute_coupled_2d_embayment_displacement


def compute_reef_dissipation_modulator(
    hs: Union[float, np.ndarray],
    tide: Optional[Union[float, np.ndarray]] = None,
    surge: Optional[Union[float, np.ndarray]] = None,
    z_reef_m: float = -1.0,
    reef_flat_width_m: float = 150.0,
    c_f: float = 0.08,
    gamma_b: float = 0.78,
    l_d_ref_m: float = 25.0,
    is_reef: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute the Reef Dissipation Modulator (Psi_reef) for reef-fronted coastal archetypes.

    Coral reefs act as physical low-pass filters, attenuating deep-water wave energy before it reaches
    the shoreline through a two-step dissipation process (Lowe et al. 2005, Ferrario et al. 2014):
    1. Crest Clipping (Depth-Limited Breaking):
       h_r(t) = max(0, Tide(t) + S(t) - Z_reef)
       K_c = min(1.0, (gamma_b * h_r(t)) / H_0)
    2. Flat Friction (Coral Drag Decay across Reef Flat):
       K_f = exp(- (c_f * W_r) / (L_d * h_r(t)))

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
    l_d_ref_m : float
        Characteristic shallow-water reef damping length scale (default 25.0m).
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

    # 3. Flat Friction (Coral Drag Decay across Shallow Reef Flat): Kf = exp(- (c_f * Wr) / (L_d * hr))
    # If Wr == 0, Kf = 1.0. If hr <= 0, Kf = 0.0.
    if float(reef_flat_width_m) <= 0.0:
        k_f = np.where(h_r <= 1e-5, 0.0, 1.0)
    else:
        l_d = max(float(l_d_ref_m), 1.0)
        k_f = np.where(h_r <= 1e-5, 0.0, np.exp(- (float(c_f) * float(reef_flat_width_m)) / (l_d * np.maximum(h_r, 1e-4))))

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


def compute_expanding_window_equilibrium(
    upsilon: Union[float, np.ndarray, pd.Series],
    window_hours: Optional[int] = None,
    min_periods: int = 24,
    initial_baseline: float = 2500.0,
    alpha_ema: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Compute non-leaking, strictly causal equilibrium threshold series Υ_eq(t).

    Machine Learning / Target Leakage Prevention (Suggestion 7):
    Calculating a static Υ_eq from the mean wave climate of the entire dataset causes target leakage
    where the training model implicitly "sees" future storm and calm periods.
    This function computes causal equilibrium strictly from antecedent historical forcing (t' <= t) via:
    1. Expanding-Window Cumulative Mean (when window_hours is None and alpha_ema is None):
       Υ_eq(t) = (1 / (t + 1)) * sum_{i=0}^t Υ(i)
       (with smooth Bayesian prior warm-up blending for t < min_periods)
    2. Trailing Moving Average (when window_hours is specified, e.g., trailing 3-year mean):
       Υ_eq(t) = mean(Υ[max(0, t - window_hours + 1) : t + 1])
    3. Causal Exponential Moving Average (when alpha_ema is specified):
       Υ_eq(t) = alpha * Υ(t) + (1 - alpha) * Υ_eq(t - 1)

    Parameters
    ----------
    upsilon : float, np.ndarray, or pd.Series
        Instantaneous Unified Coastal Vulnerability Index series.
    window_hours : int, optional
        Trailing window length in hours (e.g. trailing 3-year mean = 3 * 365.25 * 24 ~ 26298 hours).
        If None, computes full expanding cumulative mean.
    min_periods : int
        Minimum number of antecedent observations before relying solely on empirical window mean.
        Before min_periods, empirical mean is smoothly blended with initial_baseline. Default is 24.
    initial_baseline : float
        Prior/baseline equilibrium value before sufficient antecedent observations accumulate (default 2500.0).
    alpha_ema : float, optional
        If provided (0 < alpha_ema <= 1), computes recursive causal EMA rather than expanding/trailing mean.

    Returns
    -------
    upsilon_eq : float or np.ndarray
        Strictly causal, non-leaking equilibrium threshold series.
    """
    if np.isscalar(upsilon):
        return float(initial_baseline) if initial_baseline is not None else float(upsilon)

    ups_arr = np.asarray(upsilon, dtype=float)
    n = len(ups_arr)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(initial_baseline if initial_baseline is not None else ups_arr[0])], dtype=float)

    if alpha_ema is not None:
        alpha = float(np.clip(alpha_ema, 1e-6, 1.0))
        eq_arr = np.zeros(n, dtype=float)
        eq_arr[0] = float(initial_baseline) if initial_baseline is not None else ups_arr[0]
        for t in range(1, n):
            eq_arr[t] = alpha * ups_arr[t] + (1.0 - alpha) * eq_arr[t - 1]
        return eq_arr

    counts = np.arange(1, n + 1, dtype=float)

    if window_hours is None or window_hours >= n:
        # Expanding cumulative mean
        cumsum = np.cumsum(ups_arr)
        raw_eq = cumsum / counts
    else:
        # Trailing moving average of width W
        w = max(1, int(window_hours))
        padded = np.pad(ups_arr, (1, 0), mode="constant", constant_values=0.0)
        cumsum = np.cumsum(padded)
        window_sum = cumsum[1:] - cumsum[np.maximum(0, np.arange(n) - w + 1)]
        window_count = np.minimum(counts, float(w))
        raw_eq = window_sum / window_count

    # Warm-up blending with initial baseline to prevent wild variance on initial steps
    if min_periods > 1 and initial_baseline is not None:
        blend_weight = np.clip(counts / float(min_periods), 0.0, 1.0)
        eq_arr = blend_weight * raw_eq + (1.0 - blend_weight) * float(initial_baseline)
    else:
        eq_arr = raw_eq

    return eq_arr


def compute_upsilon_disequilibrium(
    upsilon: Union[float, np.ndarray],
    upsilon_eq: Union[float, np.ndarray, str] = 2500.0,
    window_hours: Optional[int] = None,
    min_periods: int = 24,
    initial_baseline: float = 2500.0,
) -> Union[float, np.ndarray]:
    """Compute Upsilon Disequilibrium (Upsilon_diseq = Upsilon - Upsilon_eq) without data leakage.

    Analogous to Yates et al. (2009) equilibrium wave energy disequilibrium (Delta E = E - E_eq),
    Upsilon Disequilibrium measures the instantaneous departure of kinetic coastal forcing
    from the geomorphic equilibrium baseline Upsilon_eq:

        Upsilon_diseq = Upsilon - Upsilon_eq

    - Positive (Upsilon_diseq > 0): Destructive storm state where wave-sediment-surge forcing
      exceeds equilibrium capacity, driving offshore sandbar migration and dune scarping.
    - Negative (Upsilon_diseq < 0): Constructive fair-weather recovery state where hydrodynamic
      forcing is subdued, allowing swash berm welding and shoreline accretion.

    Parameters
    ----------
    upsilon : float or np.ndarray
        Instantaneous Unified Coastal Vulnerability Index.
    upsilon_eq : float, np.ndarray, or str
        Equilibrium baseline Upsilon threshold (default 2500.0), array series, or mode string
        ('expanding', 'trailing', 'static').
    window_hours : int, optional
        Trailing window size in hours if upsilon_eq is 'trailing'.
    min_periods : int
        Minimum warm-up periods for expanding window.
    initial_baseline : float
        Prior baseline for warm-up periods.

    Returns
    -------
    upsilon_diseq : float or np.ndarray
        Dimensionless/kinetic forcing disequilibrium series.
    """
    if isinstance(upsilon_eq, str):
        mode = upsilon_eq.lower()
        if mode in ["expanding", "expanding_window"]:
            eq_val = compute_expanding_window_equilibrium(
                upsilon, window_hours=None, min_periods=min_periods, initial_baseline=initial_baseline
            )
        elif mode in ["trailing", "trailing_3yr", "trailing_window", "rolling"]:
            w_h = window_hours if window_hours is not None else (3 * 365 * 24)
            eq_val = compute_expanding_window_equilibrium(
                upsilon, window_hours=w_h, min_periods=min_periods, initial_baseline=initial_baseline
            )
        else:
            eq_val = initial_baseline
    else:
        eq_val = upsilon_eq

    ups_arr = np.asarray(upsilon, dtype=float)
    eq_arr = np.asarray(eq_val, dtype=float)
    diseq = ups_arr - eq_arr
    if np.isscalar(upsilon) and np.isscalar(eq_val):
        return float(diseq)
    return diseq


def compute_signed_equilibrium_rate(
    upsilon: Union[float, np.ndarray],
    upsilon_eq: Union[float, np.ndarray, str] = 2500.0,
    c_erosion: float = 1.2e-5,
    c_accretion: float = 1.8e-5,
    window_hours: Optional[int] = None,
    min_periods: int = 24,
    initial_baseline: float = 2500.0,
) -> Union[float, np.ndarray]:
    """Compute instantaneous shoreline cross-shore displacement rate dX/dt using the Signed Equilibrium Formulation.

    Differential Formulation:
        dX/dt = -C_erosion * max(Upsilon - Upsilon_eq, 0) + C_accretion * max(Upsilon_eq - Upsilon, 0)

    This formulation resolves both asymmetric storm-driven gouging (-dX/dt) and post-storm berm recovery (+dX/dt)
    without arbitrary regime switching and without lookahead data leakage.

    Parameters
    ----------
    upsilon : float or np.ndarray
        Instantaneous Unified Coastal Vulnerability Index.
    upsilon_eq : float, np.ndarray, or str
        Equilibrium baseline Upsilon threshold (default 2500.0), series, or mode string ('expanding', 'trailing').
    c_erosion : float
        Erosive rate coefficient (default 1.2e-5).
    c_accretion : float
        Accretive rate coefficient (default 1.8e-5).
    window_hours : int, optional
        Trailing window size in hours if upsilon_eq is 'trailing'.
    min_periods : int
        Minimum warm-up periods for expanding window.
    initial_baseline : float
        Prior baseline for warm-up periods.

    Returns
    -------
    dx_dt : float or np.ndarray
        Instantaneous shoreline velocity in meters per hour (or per time step).
    """
    if isinstance(upsilon_eq, str):
        mode = upsilon_eq.lower()
        if mode in ["expanding", "expanding_window"]:
            eq_val = compute_expanding_window_equilibrium(
                upsilon, window_hours=None, min_periods=min_periods, initial_baseline=initial_baseline
            )
        elif mode in ["trailing", "trailing_3yr", "trailing_window", "rolling"]:
            w_h = window_hours if window_hours is not None else (3 * 365 * 24)
            eq_val = compute_expanding_window_equilibrium(
                upsilon, window_hours=w_h, min_periods=min_periods, initial_baseline=initial_baseline
            )
        else:
            eq_val = initial_baseline
    else:
        eq_val = upsilon_eq

    ups_arr = np.asarray(upsilon, dtype=float)
    eq_arr = np.asarray(eq_val, dtype=float)

    delta_erosion = np.maximum(ups_arr - eq_arr, 0.0)
    delta_accretion = np.maximum(eq_arr - ups_arr, 0.0)

    rate = -float(c_erosion) * delta_erosion + float(c_accretion) * delta_accretion
    if np.isscalar(upsilon) and np.isscalar(eq_val):
        return float(rate)
    return rate


def simulate_signed_equilibrium_upsilon_evolution(
    upsilon: np.ndarray,
    c_erosion: float = 1.2e-5,
    c_accretion: float = 1.8e-5,
    upsilon_eq: Optional[Union[float, np.ndarray, str]] = 2500.0,
    upsilon_eq_mode: str = "auto",
    window_hours: Optional[int] = None,
    min_periods: int = 24,
    a_eq: float = 0.0,
    b_eq: float = 2500.0,
    dynamic_eq: bool = False,
    initial_x: float = 0.0,
    x_bounds: Tuple[float, float] = (-25.0, 15.0),
    damping_mode: str = "asymptotic_tanh",
    taper_widths: Tuple[float, float] = (8.0, 5.0),
    random_seed: int = 42,
    add_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate continuous shoreline morphodynamics driven by the Signed Equilibrium Upsilon Formulation.

    Governing Differential Equation:
        dX/dt = -C_erosion * max(Upsilon(t) - Upsilon_eq(t), 0) + C_accretion * max(Upsilon_eq(t) - Upsilon(t), 0)

    Where equilibrium threshold Upsilon_eq can be static, dynamic beach-state coupled, expanding-window,
    or trailing-window moving average (preventing target data leakage).

    Parameters
    ----------
    upsilon : np.ndarray
        Time series of Unified Coastal Vulnerability Index.
    c_erosion : float
        Erosive rate scaling constant.
    c_accretion : float
        Accretive rate scaling constant.
    upsilon_eq : float, np.ndarray, or str, optional
        Baseline equilibrium threshold value, array, or mode ('expanding', 'trailing', 'static').
    upsilon_eq_mode : str
        Equilibrium computation mode ('auto', 'static', 'expanding', 'trailing', 'dynamic').
    window_hours : int, optional
        Trailing window size in hours for trailing mode.
    min_periods : int
        Warm-up periods for expanding mode.
    a_eq : float
        Slope of dynamic equilibrium threshold against cross-shore displacement X.
    b_eq : float
        Intercept of dynamic equilibrium threshold.
    dynamic_eq : bool
        Whether to couple Upsilon_eq dynamically with beach displacement X(t).
    initial_x : float
        Initial shoreline position offset (m).
    x_bounds : tuple of float
        (min_x, max_x) clipping limits in meters.
    damping_mode : str, default 'asymptotic_tanh'
        Boundary damping mode ('asymptotic_tanh', 'exponential', or 'hard').
        Asymptotic damping smoothly tapers displacement rates representing subaerial
        berm accommodation capacity limits (max_x) and cliff/scarp resistance (min_x),
        eliminating artificial flat-top saturation plateaus.
    taper_widths : tuple of float, default (8.0, 5.0)
        Buffer widths (delta_erosion, delta_accretion) in meters for asymptotic taper.
    random_seed : int
        Seed for survey observation noise.
    add_noise : bool
        Whether to add simulated survey measurement noise (sigma = 0.15m).

    Returns
    -------
    x_pos : np.ndarray
        Simulated deterministic shoreline cross-shore displacement trajectory (m).
    x_obs : np.ndarray
        Simulated observed shoreline position trajectory (m).
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    n = len(ups_arr)
    x_pos = np.zeros(n, dtype=float)
    curr_x = float(initial_x)
    x_pos[0] = curr_x

    min_x, max_x = x_bounds

    init_base = float(upsilon_eq) if (isinstance(upsilon_eq, (int, float)) and b_eq == 2500.0) else b_eq

    # Precompute causal equilibrium array if expanding/trailing mode
    eq_series: Optional[np.ndarray] = None
    if isinstance(upsilon_eq, np.ndarray) and len(upsilon_eq) == n:
        eq_series = upsilon_eq
    elif upsilon_eq_mode in ["expanding", "expanding_window"] or (isinstance(upsilon_eq, str) and upsilon_eq == "expanding"):
        eq_series = compute_expanding_window_equilibrium(ups_arr, window_hours=None, min_periods=min_periods, initial_baseline=init_base)
    elif upsilon_eq_mode in ["trailing", "trailing_3yr", "trailing_window"] or (isinstance(upsilon_eq, str) and "trailing" in upsilon_eq):
        w_h = window_hours if window_hours is not None else (3 * 365 * 24)
        eq_series = compute_expanding_window_equilibrium(ups_arr, window_hours=w_h, min_periods=min_periods, initial_baseline=init_base)

    for t in range(1, n):
        u = ups_arr[t]
        if dynamic_eq:
            u_eq = a_eq * curr_x + b_eq
        elif eq_series is not None:
            u_eq = float(eq_series[t])
        else:
            u_eq = float(upsilon_eq) if (upsilon_eq is not None and not isinstance(upsilon_eq, str)) else float(b_eq)

        d_ero = max(u - u_eq, 0.0)
        d_acc = max(u_eq - u, 0.0)

        rate = -c_erosion * d_ero + c_accretion * d_acc
        if damping_mode in ["asymptotic", "asymptotic_tanh", "tanh"]:
            delta_ero, delta_acc = taper_widths
            if rate > 0:
                headroom = max_x - curr_x
                if headroom <= 0:
                    rate = 0.0
                elif headroom < delta_acc:
                    h = headroom / max(delta_acc, 0.1)
                    rate = min(rate * (np.tanh(h * 2.0) ** 2.0), headroom * np.tanh(h))
            elif rate < 0:
                footroom = curr_x - min_x
                if footroom <= 0:
                    rate = 0.0
                elif footroom < delta_ero:
                    f = footroom / max(delta_ero, 0.1)
                    rate = max(rate * (np.tanh(f * 2.0) ** 2.0), -footroom * np.tanh(f))
        elif damping_mode in ["exponential", "exp"]:
            delta_ero, delta_acc = taper_widths
            if rate > 0:
                headroom = max_x - curr_x
                if headroom <= 0:
                    rate = 0.0
                elif headroom < delta_acc:
                    h = headroom / max(delta_acc, 0.1)
                    taper = (1.0 - np.exp(-3.0 * h)) ** 2.0
                    rate = min(rate * taper, headroom * (1.0 - np.exp(-h)))
            elif rate < 0:
                footroom = curr_x - min_x
                if footroom <= 0:
                    rate = 0.0
                elif footroom < delta_ero:
                    f = footroom / max(delta_ero, 0.1)
                    taper = (1.0 - np.exp(-3.0 * f)) ** 2.0
                    rate = max(rate * taper, -footroom * (1.0 - np.exp(-f)))

        curr_x = np.clip(curr_x + rate, min_x, max_x)
        x_pos[t] = curr_x

    if add_noise:
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0.0, 0.15, n)
        x_obs = x_pos + noise
    else:
        x_obs = x_pos.copy()

    return x_pos, x_obs


def simulate_normalized_equilibrium_upsilon_evolution(
    upsilon: Optional[np.ndarray] = None,
    delta_upsilon_star: Optional[np.ndarray] = None,
    c_erosion_star: float = 0.02,
    c_accretion_star: float = 0.01,
    upsilon_eq: Optional[Union[float, np.ndarray, str]] = None,
    upsilon_crit: Optional[Union[float, np.ndarray]] = None,
    upsilon_eq_mode: str = "auto",
    window_hours: Optional[int] = None,
    min_periods: int = 24,
    initial_x: float = 0.0,
    x_bounds: Tuple[float, float] = (-25.0, 15.0),
    damping_mode: str = "asymptotic_tanh",
    taper_widths: Tuple[float, float] = (8.0, 5.0),
    random_seed: int = 42,
    add_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate continuous shoreline morphodynamics driven by Dimensionless Normalized Upsilon Disequilibrium (ΔΥ*).

    Governing Differential Equation:
        dX/dt = -c_erosion* * max(Delta_Upsilon*(t), 0) + c_accretion* * max(-Delta_Upsilon*(t), 0)

    where dimensionless site-normalized disequilibrium is:
        Delta_Upsilon*(t) = (Upsilon(t) - Upsilon_eq(t)) / Upsilon_crit

    Eliminates cross-site order-of-magnitude scaling collapse (e.g. Duck vs Truc Vert),
    yielding scale-universal displacement rates (m/hr per unit normalized disequilibrium).

    Parameters
    ----------
    upsilon : np.ndarray, optional
        Time series of Unified Coastal Vulnerability Index (if delta_upsilon_star not directly provided).
    delta_upsilon_star : np.ndarray, optional
        Direct pre-computed dimensionless disequilibrium series. If provided, overrides upsilon calculation.
    c_erosion_star : float, default 0.02
        Dimensionless erosive rate scaling constant [m/hr per unit ΔΥ*].
    c_accretion_star : float, default 0.01
        Dimensionless accretive rate scaling constant [m/hr per unit ΔΥ*].
    upsilon_eq : float, np.ndarray, or str, optional
        Equilibrium threshold value or mode ('median', 'expanding', 'trailing', 'static').
        If None and upsilon provided, defaults to median(upsilon) or causal expanding mean.
    upsilon_crit : float or np.ndarray, optional
        Critical storm threshold scaling scale [kg/s].
        If None and upsilon provided, defaults to max(1.0, P90(upsilon) - median(upsilon)).
    upsilon_eq_mode : str, default 'auto'
        Equilibrium computation mode ('auto', 'median', 'static', 'expanding', 'trailing').
    window_hours : int, optional
        Trailing window size in hours for trailing mode.
    min_periods : int, default 24
        Warm-up periods for expanding mode.
    initial_x : float, default 0.0
        Initial shoreline position offset (m).
    x_bounds : tuple of float, default (-25.0, 15.0)
        (min_x, max_x) clipping limits in meters.
    damping_mode : str, default 'asymptotic_tanh'
        Boundary damping mode ('asymptotic_tanh', 'exponential', or 'hard').
        Asymptotic damping smoothly tapers displacement rates representing subaerial
        berm accommodation capacity limits (max_x) and cliff/scarp resistance (min_x),
        eliminating artificial flat-top saturation plateaus.
    taper_widths : tuple of float, default (8.0, 5.0)
        Buffer widths (delta_erosion, delta_accretion) in meters for asymptotic taper.
    random_seed : int, default 42
        Seed for observation noise.
    add_noise : bool, default False
        Whether to add simulated survey measurement noise (sigma = 0.15m).

    Returns
    -------
    x_pos : np.ndarray
        Simulated deterministic shoreline cross-shore displacement trajectory (m).
    x_obs : np.ndarray
        Simulated observed shoreline position trajectory (m).
    """
    if delta_upsilon_star is not None:
        du_star = np.asarray(delta_upsilon_star, dtype=float)
        n = len(du_star)
    elif upsilon is not None:
        ups_arr = np.asarray(upsilon, dtype=float)
        n = len(ups_arr)

        # Determine upsilon_eq
        if isinstance(upsilon_eq, np.ndarray) and len(upsilon_eq) == n:
            eq_series = upsilon_eq
        elif upsilon_eq_mode in ["expanding", "expanding_window"] or (isinstance(upsilon_eq, str) and upsilon_eq == "expanding"):
            init_base = float(np.median(ups_arr)) if len(ups_arr) > 0 else 2500.0
            eq_series = compute_expanding_window_equilibrium(ups_arr, window_hours=None, min_periods=min_periods, initial_baseline=init_base)
        elif upsilon_eq_mode in ["trailing", "trailing_window"] or (isinstance(upsilon_eq, str) and "trailing" in upsilon_eq):
            w_h = window_hours if window_hours is not None else (3 * 365 * 24)
            init_base = float(np.median(ups_arr)) if len(ups_arr) > 0 else 2500.0
            eq_series = compute_expanding_window_equilibrium(ups_arr, window_hours=w_h, min_periods=min_periods, initial_baseline=init_base)
        elif isinstance(upsilon_eq, (int, float)):
            eq_series = np.full(n, float(upsilon_eq))
        else:
            # Default: median of series
            u_med = float(np.median(ups_arr)) if len(ups_arr) > 0 else 1500.0
            eq_series = np.full(n, u_med)

        # Determine upsilon_crit
        if upsilon_crit is not None and not isinstance(upsilon_crit, str):
            crit_val = np.maximum(np.asarray(upsilon_crit, dtype=float), 1.0)
        else:
            u_med_val = float(np.median(ups_arr)) if len(ups_arr) > 0 else 1500.0
            u_p90 = float(np.percentile(ups_arr, 90)) if len(ups_arr) > 0 else 10000.0
            crit_val = max(1.0, u_p90 - u_med_val)

        du_star = (ups_arr - eq_series) / crit_val
    else:
        raise ValueError("Either 'delta_upsilon_star' or 'upsilon' must be provided.")

    x_pos = np.zeros(n, dtype=float)
    curr_x = float(initial_x)
    x_pos[0] = curr_x
    min_x, max_x = x_bounds

    for t in range(1, n):
        d_val = du_star[t]
        d_ero = max(d_val, 0.0)
        d_acc = max(-d_val, 0.0)

        rate = -c_erosion_star * d_ero + c_accretion_star * d_acc
        if damping_mode in ["asymptotic", "asymptotic_tanh", "tanh"]:
            delta_ero, delta_acc = taper_widths
            if rate > 0:
                headroom = max_x - curr_x
                if headroom <= 0:
                    rate = 0.0
                elif headroom < delta_acc:
                    h = headroom / max(delta_acc, 0.1)
                    rate = min(rate * (np.tanh(h * 2.0) ** 2.0), headroom * np.tanh(h))
            elif rate < 0:
                footroom = curr_x - min_x
                if footroom <= 0:
                    rate = 0.0
                elif footroom < delta_ero:
                    f = footroom / max(delta_ero, 0.1)
                    rate = max(rate * (np.tanh(f * 2.0) ** 2.0), -footroom * np.tanh(f))
        elif damping_mode in ["exponential", "exp"]:
            delta_ero, delta_acc = taper_widths
            if rate > 0:
                headroom = max_x - curr_x
                if headroom <= 0:
                    rate = 0.0
                elif headroom < delta_acc:
                    h = headroom / max(delta_acc, 0.1)
                    taper = (1.0 - np.exp(-3.0 * h)) ** 2.0
                    rate = min(rate * taper, headroom * (1.0 - np.exp(-h)))
            elif rate < 0:
                footroom = curr_x - min_x
                if footroom <= 0:
                    rate = 0.0
                elif footroom < delta_ero:
                    f = footroom / max(delta_ero, 0.1)
                    taper = (1.0 - np.exp(-3.0 * f)) ** 2.0
                    rate = max(rate * taper, -footroom * (1.0 - np.exp(-f)))

        curr_x = np.clip(curr_x + rate, min_x, max_x)
        x_pos[t] = curr_x

    if add_noise:
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0.0, 0.15, n)
        x_obs = x_pos + noise
    else:
        x_obs = x_pos.copy()

    return x_pos, x_obs


def simulate_coupled_2d_embayment_evolution(
    upsilon: np.ndarray,
    phi_longshore: np.ndarray,
    y_alongshore_m: float,
    y_center_m: float = 1800.0,
    l_beach_m: float = 3600.0,
    alpha_rot: float = 12.0,
    c_erosion: float = 1.2e-5,
    c_accretion: float = 1.8e-5,
    upsilon_eq: Optional[float] = 2500.0,
    a_eq: float = 0.0,
    b_eq: float = 2500.0,
    dynamic_eq: bool = False,
    initial_x: float = 0.0,
    x_bounds: Tuple[float, float] = (-35.0, 25.0),
    random_seed: int = 42,
    add_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate continuous coupled 2D shoreline morphodynamics for embayed and rotational coastlines.

    Couples the signed equilibrium Upsilon cross-shore morphodynamics with the instantaneous
    alongshore transport gradient rotation over time:

        Delta_X_net(y, t) = X_cross(t) - X_cross(0) + alpha_rot * Phi_longshore(t) * ((y - y_center) / L_beach)

    Parameters
    ----------
    upsilon : np.ndarray
        Time series of Unified Coastal Vulnerability Index.
    phi_longshore : np.ndarray
        Time series of alongshore transport potential Phi_longshore = sin(2 * (theta_wave - theta_shore)).
    y_alongshore_m : float
        Alongshore coordinate of the transect in meters [m].
    y_center_m : float
        Central fulcrum coordinate of the embayment (m).
    l_beach_m : float
        Total length of the embayed beach compartment (m).
    alpha_rot : float
        Rotational mobility coefficient (m).
    c_erosion, c_accretion : float
        Cross-shore morphodynamic rate constants.
    upsilon_eq : float, optional
        Static baseline equilibrium threshold.
    a_eq, b_eq : float
        Dynamic equilibrium parameters.
    dynamic_eq : bool
        Whether to dynamically couple Upsilon_eq with beach displacement.
    initial_x : float
        Initial shoreline position offset (m).
    x_bounds : tuple of float
        (min_x, max_x) clipping limits in meters.
    random_seed : int
        Seed for survey observation noise.
    add_noise : bool
        Whether to add simulated survey measurement noise (sigma = 0.15m).

    Returns
    -------
    x_net : np.ndarray
        Deterministic coupled 2D shoreline displacement trajectory (m).
    x_obs : np.ndarray
        Observed coupled 2D shoreline trajectory with noise (m).
    """
    # 1. Simulate 1D cross-shore equilibrium evolution
    x_cross, _ = simulate_signed_equilibrium_upsilon_evolution(
        upsilon=upsilon,
        c_erosion=c_erosion,
        c_accretion=c_accretion,
        upsilon_eq=upsilon_eq,
        a_eq=a_eq,
        b_eq=b_eq,
        dynamic_eq=dynamic_eq,
        initial_x=initial_x,
        x_bounds=x_bounds,
        random_seed=random_seed,
        add_noise=False,
    )

    # 2. Couple with 2D alongshore transport gradient rotation
    phi_arr = np.asarray(phi_longshore, dtype=float)
    dx_cross = x_cross - float(initial_x)
    dx_rot = compute_coupled_2d_embayment_displacement(
        delta_x_cross_shore=dx_cross,
        phi_longshore=phi_arr,
        y_alongshore_m=float(y_alongshore_m),
        y_center_m=float(y_center_m),
        l_beach_m=float(l_beach_m),
        alpha_rot=float(alpha_rot),
    )
    x_net = float(initial_x) + dx_rot
    min_x, max_x = x_bounds
    x_net = np.clip(x_net, min_x, max_x)

    if add_noise:
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0.0, 0.15, len(x_net))
        x_obs = x_net + noise
    else:
        x_obs = x_net.copy()

    return x_net, x_obs


def compute_wave_steepness(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    g: float = G_CONST,
    tau: float = TAU_CONST,
    s_crit: float = 0.025,
    gamma: float = 0.5,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute deep-water wave steepness (s_0 = H_s / L_0) and non-dimensional steepness modulator Psi_steep.

    Physical Formulation:
    ---------------------
    Deep-water wavelength:
        L_0 = (g * T_p^2) / (2 * pi)
    Wave steepness:
        s_0 = H_s / L_0 = (2 * pi * H_s) / (g * T_p^2)

    Morphodynamic Scaling:
    ----------------------
    Wave steepness controls whether incoming wave energy drives constructive berm accretion (long swell, s_0 < 0.025)
    or erosive bar cutting and dune gouging (short steep storm seas, s_0 > 0.025).
    The dimensionless steepness modulator scales as:
        Psi_steep = clip( (s_0 / s_crit)^gamma, 0.2, 2.5 )
    where s_crit = 0.025 is the classical empirical threshold and gamma = 0.5 is the sublinear scaling exponent.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray
        Peak wave period (s).
    g : float
        Gravitational acceleration (9.81 m/s^2).
    tau : float
        2 * pi.
    s_crit : float
        Critical threshold between swell and erosive seas (default 0.025).
    gamma : float
        Steepness scaling exponent (default 0.5).

    Returns
    -------
    s_0 : float or np.ndarray
        Dimensionless wave steepness [-].
    psi_steep : float or np.ndarray
        Bounded continuous steepness modulator [-].
    """
    hs_arr = np.asarray(hs, dtype=float)
    tp_arr = np.asarray(tp, dtype=float)

    l_0 = (g * (np.maximum(tp_arr, 0.1) ** 2)) / tau
    s_0 = np.maximum(hs_arr, 0.01) / np.maximum(l_0, 1.0)
    psi_steep = np.clip((s_0 / max(float(s_crit), 1e-4)) ** float(gamma), 0.2, 2.5)

    if np.isscalar(hs) and np.isscalar(tp):
        return float(s_0), float(psi_steep)
    return s_0, psi_steep


def compute_radiation_stress_tensor(
    hs: Union[float, np.ndarray],
    tp: Optional[Union[float, np.ndarray]] = None,
    wave_dir_deg: Optional[Union[float, np.ndarray]] = None,
    shore_normal_deg: float = DUCK_SHORE_NORMAL_DEG,
    rho: float = RHO_WATER,
    g: float = G_CONST,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute exact radiation stress tensor components (S_xx, S_xy) and normalized stress ratio Psi_Sxx.

    Theoretical Formulation (Longuet-Higgins & Stewart, 1964):
    ---------------------------------------------------------
    Wave energy density per unit horizontal area:
        E = (1 / 8) * rho * g * H_s^2   [J / m^2]
    Relative wave incidence angle to shore-normal azimuth:
        theta_rel = (wave_dir_deg - shore_normal_deg) * (pi / 180)
    In deep water (group velocity c_g / c = n = 1/2):
        S_xx = E * (n * (1 + cos^2(theta_rel)) - 1/2) = E * (1/2) * cos^2(theta_rel)   [N / m]
        S_xy = E * (n / 2) * sin(2 * theta_rel) = E * (1/4) * sin(2 * theta_rel)       [N / m]

    The normalized shore-normal radiation stress ratio:
        Psi_Sxx = S_xx / E = (1/2) * cos^2(theta_rel)
    which directly quantifies the excess cross-shore momentum flux driving wave set-up and beach gouging.

    Parameters
    ----------
    hs : float or np.ndarray
        Significant wave height (m).
    tp : float or np.ndarray, optional
        Peak wave period (s).
    wave_dir_deg : float or np.ndarray, optional
        Wave direction in degrees True.
    shore_normal_deg : float
        Shore-normal orientation azimuth (deg True).
    rho : float
        Seawater density (1025 kg/m^3).
    g : float
        Gravitational acceleration (9.81 m/s^2).

    Returns
    -------
    s_xx : float or np.ndarray
        Shore-normal cross-shore radiation stress (N/m).
    s_xy : float or np.ndarray
        Longshore shear radiation stress driving littoral currents (N/m).
    psi_sxx : float or np.ndarray
        Dimensionless radiation stress ratio S_xx / E [-].
    e_energy : float or np.ndarray
        Total wave energy density E (J/m^2).
    """
    hs_arr = np.asarray(hs, dtype=float)
    if wave_dir_deg is not None:
        w_dir = np.asarray(wave_dir_deg, dtype=float)
    else:
        w_dir = np.full_like(hs_arr, float(shore_normal_deg))

    theta_rel = np.radians(w_dir - float(shore_normal_deg))
    e_energy = (1.0 / 8.0) * rho * g * (np.maximum(hs_arr, 0.0) ** 2)

    s_xx = e_energy * 0.5 * (np.cos(theta_rel) ** 2)
    s_xy = e_energy * 0.25 * np.sin(2.0 * theta_rel)
    psi_sxx = 0.5 * (np.cos(theta_rel) ** 2)

    if np.isscalar(hs) and (wave_dir_deg is None or np.isscalar(wave_dir_deg)):
        return float(s_xx), float(s_xy), float(psi_sxx), float(e_energy)
    return s_xx, s_xy, psi_sxx, e_energy


def compute_sallenger_impact_ratio(
    twl: Union[float, np.ndarray],
    berm_crest_m: Union[float, np.ndarray] = 2.5,
    hs: Optional[Union[float, np.ndarray]] = None,
    alpha_impact: float = 0.5,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """Compute Sallenger (2000) storm impact scale regime classification and collision/overwash ratio Psi_impact.

    Theoretical Formulation (Sallenger, 2000):
    -----------------------------------------
    The Sallenger storm impact scale establishes 4 discrete regimes based on Total Water Level (TWL = Tide + Surge + R_2%):
    1. Swash Regime (TWL < z_berm):
       Wave runup is confined to the foreshore; typical mild-weather or minor storm conditions.
       Psi_impact = 1.0 (baseline).
    2. Collision Regime (z_berm <= TWL < z_dune):
       Wave runup directly collides with the base of the berm or dune toe, triggering scarping, slumping, and rapid gouging.
       Psi_impact = 1.0 + alpha_impact * (TWL - z_berm) / max(H_s, 0.5)
    3. Overwash Regime (TWL >= z_dune, or severe TWL > z_berm + 1.2m):
       Wave runup crests over the berm/dune, transporting sediment landward into washover fans.

    Parameters
    ----------
    twl : float or np.ndarray
        Total Water Level elevation above vertical datum (m).
    berm_crest_m : float or np.ndarray
        Elevation of berm/dune toe crest above vertical datum (m).
    hs : float or np.ndarray, optional
        Significant wave height (m) used for scale normalization.
    alpha_impact : float
        Collision scaling multiplier (default 0.5).

    Returns
    -------
    psi_impact : float or np.ndarray
        Continuous non-negative impact amplification factor (>= 1.0) [-].
    regimes : np.ndarray of str
        Regime string array ('swash', 'collision', 'overwash').
    """
    twl_arr = np.asarray(twl, dtype=float)
    z_berm_arr = np.broadcast_to(np.asarray(berm_crest_m, dtype=float), twl_arr.shape) if not np.isscalar(berm_crest_m) else np.full_like(twl_arr, float(berm_crest_m))

    if hs is not None:
        hs_scale = np.maximum(np.asarray(hs, dtype=float), 0.5)
    else:
        hs_scale = np.ones_like(twl_arr)

    freeboard_excess = np.maximum(0.0, twl_arr - z_berm_arr)
    psi_impact = 1.0 + float(alpha_impact) * (freeboard_excess / hs_scale)

    regimes = np.full(twl_arr.shape, "swash", dtype=object)
    regimes[twl_arr >= z_berm_arr] = "collision"
    regimes[twl_arr >= (z_berm_arr + 1.2)] = "overwash"

    if np.isscalar(twl):
        return float(psi_impact.item()), regimes.item()
    return psi_impact, regimes


def compute_multiscale_memory_decay(
    upsilon: Union[float, np.ndarray],
    tau_days: Tuple[int, ...] = (7, 30, 90, 365),
    dt_hours: float = 1.0,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute multi-scale exponential memory decay convolutions for Upsilon across multiple physical timescales.

    Theoretical Formulation:
    ------------------------
    Beach morphology responds to past storm forcing over multiple decaying timescales:
        tau in {7 days (synoptic storm relaxation), 30 days (monthly spring-neap envelope),
                90 days (seasonal bar transition), 365 days (interannual carryover)}.

    The antecedent exponentially weighted moving average is:
        Upsilon_decay_tau(t) = integral_0^infty Upsilon(t - s) * exp(-s / tau) / tau ds
    and the corresponding non-equilibrium anomaly is:
        Delta_Upsilon_tau(t) = Upsilon(t) - Upsilon_decay_tau(t)

    Parameters
    ----------
    upsilon : float or np.ndarray
        Hourly Unified Coastal Vulnerability Index series.
    tau_days : tuple of int
        Decay timescales in days (default: 7, 30, 90, 365).
    dt_hours : float
        Sampling interval in hours (default: 1.0).

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'upsilon_decay_{tau}d': Convolved antecedent memory series for each tau.
        - 'upsilon_diseq_{tau}d': Instantaneous disequilibrium pulse relative to tau memory.
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    out_dict: Dict[str, Union[float, np.ndarray]] = {}

    if ups_arr.ndim == 0 or len(ups_arr) <= 1:
        scalar_val = float(ups_arr.item()) if ups_arr.ndim == 0 else float(ups_arr[0])
        for tau in tau_days:
            out_dict[f"upsilon_decay_{tau}d"] = scalar_val
            out_dict[f"upsilon_diseq_{tau}d"] = 0.0
        return out_dict

    s_ups = pd.Series(ups_arr)
    for tau in tau_days:
        halflife_hours = float(tau) * 24.0 / float(dt_hours)
        decay_series = s_ups.ewm(halflife=halflife_hours, min_periods=1).mean().values
        diseq_series = ups_arr - decay_series
        out_dict[f"upsilon_decay_{tau}d"] = decay_series
        out_dict[f"upsilon_diseq_{tau}d"] = diseq_series

    return out_dict


def compute_shields_critical_shear_stress_soulsby(
    d50_m: Union[float, np.ndarray] = 0.00035,
    water_temp_c: Optional[Union[float, np.ndarray]] = None,
    rho_water: float = RHO_WATER,
    rho_sed: float = RHO_SEDIMENT,
    nu: float = NU_SEAWATER,
    g: float = G_CONST,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute critical dimensionless Shields parameter (theta_cr) and bed shear stress (tau_cr, N/m^2).

    Uses the Soulsby (1997) nonlinear formulation for the threshold of incipient sediment motion
    as a function of dimensionless grain size D_*:

        D_* = D_50 * [ g * (s - 1) / nu^2 ]^(1/3)
        theta_cr(D_*) = 0.30 / (1 + 1.2 * D_*) + 0.055 * [ 1 - exp(-0.020 * D_*) ]
        tau_cr = theta_cr * g * (rho_sed - rho_water) * D_50

    Physics & Nonlinearity:
    - Critical shear stress does NOT scale strictly linearly with D_50 across all regimes.
    - theta_cr decreases from ~0.24 for fine silt/sand (D_* < 1) to ~0.03 for medium-coarse sand
      (D_* ~ 10-20), before asymptotically leveling off to ~0.055 for coarse gravel.
    - As a result, the effective power-law scaling exponent a = d(ln tau_cr) / d(ln D_50) varies
      between ~0.5 and 1.3 across typical coastal grain size regimes (0.1 mm - 2.0 mm).

    Parameters
    ----------
    d50_m : float or np.ndarray
        Median sediment grain diameter in meters (e.g. 0.00035 m).
    water_temp_c : float or np.ndarray, optional
        Water temperature in Celsius for dynamic viscosity nu(T).
    rho_water : float
        Fluid density in kg/m^3 (default 1025.0).
    rho_sed : float
        Sediment density in kg/m^3 (default 2650.0).
    nu : float
        Kinematic viscosity in m^2/s (default 1.05e-6).
    g : float
        Gravitational acceleration in m/s^2 (default 9.81).

    Returns
    -------
    theta_cr : float or np.ndarray
        Dimensionless critical Shields parameter [-].
    tau_cr : float or np.ndarray
        Critical bed shear stress in Pascals [N/m^2].
    """
    s = float(rho_sed) / float(rho_water)
    d50_arr = np.maximum(np.asarray(d50_m, dtype=float), 1e-6)

    if water_temp_c is not None:
        nu_eff = compute_kinematic_viscosity_seawater(water_temp_c)
    else:
        nu_eff = nu

    # Dimensionless grain size D_*
    d_star = d50_arr * ((g * (s - 1.0)) / (nu_eff ** 2)) ** (1.0 / 3.0)

    # Soulsby (1997) critical Shields parameter
    theta_cr = (0.30 / (1.0 + 1.2 * d_star)) + 0.055 * (1.0 - np.exp(-0.020 * d_star))

    # Dimensional critical bed shear stress (N/m^2)
    tau_cr = theta_cr * g * (rho_sed - rho_water) * d50_arr

    if np.isscalar(d50_m) and (water_temp_c is None or np.isscalar(water_temp_c)):
        return float(theta_cr), float(tau_cr)
    return theta_cr, tau_cr


def compute_critical_shear_threshold(
    tan_beta: Union[float, np.ndarray],
    d50_m: Union[float, np.ndarray],
    upsilon_crit_0: float = 500.0,
    d50_ref: float = 0.00035,
    tan_beta_ref: float = 0.08,
    a: float = 1.0,
    b: float = 1.0,
    mode: str = "power_law",
) -> Union[float, np.ndarray]:
    """Compute the Critical Threshold Shear Activation threshold Upsilon_crit(tan(beta), D_50).

    Beaches with steep dunes/slopes or coarser sand resist morphodynamic change until
    a critical hydrodynamic forcing threshold is crossed:

        Upsilon_crit(tan(beta), D_50) = Upsilon_crit_0 * (D_50 / D_50_ref)^a * (tan(beta) / tan(beta)_ref)^b

    Physics & Geomorphic Rationale:
    - Coarser sand (larger D_50) increases critical shear stress for incipient sediment motion
      (Shields 1936, Soulsby 1997), requiring higher turbulent wave power to dislodge grains.
    - In classical linear assumptions, a = 1.0 and b = 1.0. However, critical shear stress scales
      nonlinearly with grain size (Shields parameter theta_cr is a function of D_* / Re*),
      producing effective scaling exponents a in [0.5, 1.5] across differing hydrodynamic regimes.
    - Steeper foreshore/dune slopes (larger tan(beta)) exhibit higher gravitational stability
      and angle-of-repose thresholds before upper-beach scarping and dune face retreat initiate (b in [0.5, 1.5]).
    - Finer sand and flatter dissipative profiles mobilize at lower energetic thresholds.

    Parameters
    ----------
    tan_beta : float or np.ndarray
        Foreshore beach slope [-].
    d50_m : float or np.ndarray
        Median sediment grain size in meters [m].
    upsilon_crit_0 : float
        Baseline critical shear threshold at reference conditions (default 500.0).
    d50_ref : float
        Reference median grain size (default 0.00035 m = 0.35 mm for Duck, NC).
    tan_beta_ref : float
        Reference foreshore beach slope (default 0.08 for Duck, NC).
    a : float
        Grain size scaling exponent (default 1.0; physically justified range [0.5, 1.5]).
    b : float
        Foreshore slope scaling exponent (default 1.0; physically justified range [0.5, 1.5]).
    mode : str
        Threshold formulation mode: 'power_law' (default, parameterized by exponents a, b) or
        'soulsby_shields' (exact Soulsby 1997 critical shear stress ratio modulated by slope).

    Returns
    -------
    upsilon_crit : float or np.ndarray
        Critical shear activation threshold in Upsilon units.
    """
    tb_arr = np.maximum(np.asarray(tan_beta, dtype=float), 1e-4)
    d50_arr = np.maximum(np.asarray(d50_m, dtype=float), 1e-6)

    if mode == "soulsby_shields":
        _, tau_cr_local = compute_shields_critical_shear_stress_soulsby(d50_m=d50_arr)
        _, tau_cr_ref = compute_shields_critical_shear_stress_soulsby(d50_m=float(d50_ref))
        grain_factor = (tau_cr_local / max(float(tau_cr_ref), 1e-6))
        slope_factor = (tb_arr / max(float(tan_beta_ref), 1e-4)) ** float(b)
        crit = float(upsilon_crit_0) * grain_factor * slope_factor
    else:
        grain_factor = (d50_arr / max(float(d50_ref), 1e-6)) ** float(a)
        slope_factor = (tb_arr / max(float(tan_beta_ref), 1e-4)) ** float(b)
        crit = float(upsilon_crit_0) * grain_factor * slope_factor

    if np.isscalar(tan_beta) and np.isscalar(d50_m):
        return float(crit)
    return crit


def sweep_critical_threshold_sensitivity(
    upsilon: Union[float, np.ndarray],
    tan_beta: Union[float, np.ndarray] = 0.08,
    d50_m: Union[float, np.ndarray] = 0.00035,
    a_values: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    b_values: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    upsilon_crit_0: float = 500.0,
    d50_ref: float = 0.00035,
    tan_beta_ref: float = 0.08,
) -> pd.DataFrame:
    """Conduct a systematic sensitivity sweep on critical threshold exponents (a, b).

    Evaluates the morphodynamic threshold behavior across the full physically admissible
    sub-linear to super-linear exponent space a, b in [0.5, 1.5].

    Parameters
    ----------
    upsilon : float or np.ndarray
        Instantaneous Unified Coastal Vulnerability Index series.
    tan_beta : float or np.ndarray
        Foreshore beach slope.
    d50_m : float or np.ndarray
        Median grain diameter (m).
    a_values : sequence of float
        Grid of grain size scaling exponents a (default: 0.5, 0.75, 1.0, 1.25, 1.5).
    b_values : sequence of float
        Grid of slope scaling exponents b (default: 0.5, 0.75, 1.0, 1.25, 1.5).
    upsilon_crit_0 : float
        Baseline critical shear threshold (default 500.0).
    d50_ref, tan_beta_ref : float
        Reference scaling constants.

    Returns
    -------
    df_sweep : pd.DataFrame
        Sensitivity grid with (a, b, mean_upsilon_crit, median_upsilon_crit,
        activation_fraction, mean_excess_upsilon, max_excess_upsilon).
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    records = []

    for a_val in a_values:
        for b_val in b_values:
            u_crit = compute_critical_shear_threshold(
                tan_beta=tan_beta,
                d50_m=d50_m,
                upsilon_crit_0=upsilon_crit_0,
                d50_ref=d50_ref,
                tan_beta_ref=tan_beta_ref,
                a=a_val,
                b=b_val,
            )
            u_excess = compute_excess_upsilon(
                upsilon=ups_arr,
                upsilon_crit=u_crit,
            )
            u_crit_arr = np.broadcast_to(np.asarray(u_crit, dtype=float), ups_arr.shape)
            u_exc_arr = np.asarray(u_excess, dtype=float)
            
            act_frac = float(np.mean(u_exc_arr > 0.0))
            records.append({
                "a": float(a_val),
                "b": float(b_val),
                "mean_upsilon_crit": float(np.mean(u_crit_arr)),
                "median_upsilon_crit": float(np.median(u_crit_arr)),
                "activation_fraction": act_frac,
                "mean_excess_upsilon": float(np.mean(u_exc_arr)),
                "max_excess_upsilon": float(np.max(u_exc_arr)),
            })

    return pd.DataFrame(records)


def compute_excess_upsilon(
    upsilon: Union[float, np.ndarray],
    upsilon_crit: Optional[Union[float, np.ndarray]] = None,
    tan_beta: Optional[Union[float, np.ndarray]] = None,
    d50_m: Optional[Union[float, np.ndarray]] = None,
    upsilon_crit_0: float = 500.0,
    d50_ref: float = 0.00035,
    tan_beta_ref: float = 0.08,
    a: float = 1.0,
    b: float = 1.0,
    mode: str = "power_law",
) -> Union[float, np.ndarray]:
    """Compute the Excess Vulnerability Index (Upsilon_excess = max(0, Upsilon - Upsilon_crit)).

    Formulation:
        Upsilon_excess = max(0, Upsilon - Upsilon_crit(tan(beta), D_50))

    Impact:
    - Eliminates background hydrodynamic noise during calm/fair-weather seasons.
    - Sharpens storm event peak detection and non-linear morphodynamic damage thresholds.

    Parameters
    ----------
    upsilon : float or np.ndarray
        Instantaneous Unified Coastal Vulnerability Index (Upsilon).
    upsilon_crit : float or np.ndarray, optional
        Pre-computed critical threshold. If omitted, computed from tan_beta and d50_m.
    tan_beta : float or np.ndarray, optional
        Foreshore beach slope [-] (used if upsilon_crit is not provided).
    d50_m : float or np.ndarray, optional
        Median sediment grain size [m] (used if upsilon_crit is not provided).
    upsilon_crit_0 : float
        Baseline critical threshold (default 500.0).
    d50_ref, tan_beta_ref : float
        Reference scales for grain size (0.00035 m) and slope (0.08).
    a, b : float
        Scaling exponents for grain size and slope (default 1.0, 1.0; range [0.5, 1.5]).
    mode : str
        Critical threshold calculation mode ('power_law' or 'soulsby_shields').

    Returns
    -------
    upsilon_excess : float or np.ndarray
        Excess vulnerability forcing exceeding the critical morphodynamic threshold.
    """
    ups_arr = np.asarray(upsilon, dtype=float)
    if upsilon_crit is None:
        if tan_beta is None or d50_m is None:
            tb = 0.08 if tan_beta is None else tan_beta
            d50 = 0.00035 if d50_m is None else d50_m
            crit_arr = compute_critical_shear_threshold(
                tb, d50, upsilon_crit_0=upsilon_crit_0, d50_ref=d50_ref, tan_beta_ref=tan_beta_ref, a=a, b=b, mode=mode
            )
        else:
            crit_arr = compute_critical_shear_threshold(
                tan_beta, d50_m, upsilon_crit_0=upsilon_crit_0, d50_ref=d50_ref, tan_beta_ref=tan_beta_ref, a=a, b=b, mode=mode
            )
    else:
        crit_arr = np.asarray(upsilon_crit, dtype=float)

    excess = np.maximum(0.0, ups_arr - crit_arr)
    if np.isscalar(upsilon) and (upsilon_crit is None or np.isscalar(upsilon_crit)):
        return float(excess)
    return excess


# Physical aliases for Critical Threshold Shear Activation
compute_critical_threshold_shear_activation = compute_excess_upsilon
compute_excess_vulnerability_index = compute_excess_upsilon



def compute_tier1_base_upsilon(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    surge: Union[float, np.ndarray],
    ws: Union[float, np.ndarray],
    tan_beta: Union[float, np.ndarray] = 0.08,
    hs_exponent: float = 2.5,
    s_ref: float = 1.0,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute Tier 1 (Base Upsilon) using wave power flux core and surge residual elevation."""
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.1)
    surge_arr = np.maximum(np.asarray(surge, dtype=float), 0.0)
    tan_beta_arr = np.asarray(tan_beta, dtype=float)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-5)

    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    wave_sed_core = ((hs_arr ** hs_exponent) * tp_arr) / ws_arr
    twl_surge = 1.0 + (surge_arr / max(s_ref, 1e-3))
    return prefactor * tan_beta_arr * wave_sed_core * twl_surge


def compute_tier2_infragravity_upsilon(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    surge: Union[float, np.ndarray],
    ws: Union[float, np.ndarray],
    tan_beta: Union[float, np.ndarray] = 0.08,
    hs_exponent: float = 2.5,
    s_ref: float = 1.0,
    r_ref: float = 2.0,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute Tier 2 (Infragravity Upsilon) adding Stockdon et al. (2006) IG runup proxy (R_IG = 0.06 * sqrt(H0 * L0))."""
    hs_arr = np.maximum(np.asarray(hs, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp, dtype=float), 0.1)
    surge_arr = np.maximum(np.asarray(surge, dtype=float), 0.0)
    tan_beta_arr = np.asarray(tan_beta, dtype=float)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-5)

    l0 = compute_deepwater_wavelength(tp_arr, g=g, tau=tau)
    r_ig = 0.06 * np.sqrt(hs_arr * l0)

    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    wave_sed_core = ((hs_arr ** hs_exponent) * tp_arr) / ws_arr
    twl_ig = 1.0 + (surge_arr / max(s_ref, 1e-3)) + (r_ig / max(r_ref, 1e-3))
    upsilon_tier2 = prefactor * tan_beta_arr * wave_sed_core * twl_ig
    return upsilon_tier2, r_ig


def compute_tier3_alongshore_upsilon(
    upsilon_tier2: Union[float, np.ndarray],
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    wave_dir_deg: Union[float, np.ndarray],
    shore_normal_deg: float = 70.0,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute Tier 3 (Alongshore Radiation Stress Upsilon) adding 1D wave power gradient leakage penalty P_l = 0.5 * P_flux * sin(2 * alpha_b)."""
    p_flux = compute_wave_energy_flux(hs, tp, rho=rho, g=g, tau=tau)
    alpha_b = np.radians(np.asarray(wave_dir_deg, dtype=float) - float(shore_normal_deg))
    p_l = 0.5 * p_flux * np.sin(2.0 * alpha_b)
    leakage_penalty = 1.0 + 0.5 * np.abs(np.sin(2.0 * alpha_b))
    upsilon_tier3 = np.asarray(upsilon_tier2, dtype=float) * leakage_penalty
    return upsilon_tier3, p_l


def compute_tier4_directional_spread_upsilon(
    upsilon_tier3: Union[float, np.ndarray],
    sigma_theta_deg: Union[float, np.ndarray],
    sigma_ref_deg: float = 30.0,
    focus_gamma: float = 0.5,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute Tier 4 (Directional Focusing Upsilon) applying groundswell collimation multiplier K_focus = (sigma_ref / max(sigma_theta, 1.0))^gamma."""
    sigma_arr = np.maximum(np.asarray(sigma_theta_deg, dtype=float), 1.0)
    k_focus = (float(sigma_ref_deg) / sigma_arr) ** float(focus_gamma)
    upsilon_tier4 = np.asarray(upsilon_tier3, dtype=float) * k_focus
    return upsilon_tier4, k_focus


def compute_site_reference_flux(
    tan_beta: Union[float, np.ndarray],
    hs_mean: Union[float, np.ndarray],
    tp_mean: Union[float, np.ndarray],
    ws: Union[float, np.ndarray],
    hs_exponent: float = 2.5,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
) -> Union[float, np.ndarray]:
    """Compute site-specific reference kinetic forcing flux Upsilon_ref [kg/s].

    Theoretical Formulation:
        Upsilon_ref = C_0 * tan(beta_site) * [ (Hs_mean^n * Tp_mean) / ws_site ]

    Parameters
    ----------
    tan_beta : float or np.ndarray
        Foreshore beach slope [-].
    hs_mean : float or np.ndarray
        Site climatological or series mean significant wave height [m].
    tp_mean : float or np.ndarray
        Site climatological or series mean peak wave period [s].
    ws : float or np.ndarray
        Site mean sediment settling velocity [m/s].
    hs_exponent : float
        Nonlinear wave height scaling exponent (default 2.5).
    rho : float
        Seawater density [kg/m^3] (default 1025.0).
    g : float
        Gravitational acceleration [m/s^2] (default 9.81).
    tau : float
        2*pi.

    Returns
    -------
    upsilon_ref : float or np.ndarray
        Site reference forcing flux [kg/s].
    """
    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    tan_beta_arr = np.asarray(tan_beta, dtype=float)
    hs_arr = np.maximum(np.asarray(hs_mean, dtype=float), 0.01)
    tp_arr = np.maximum(np.asarray(tp_mean, dtype=float), 0.1)
    ws_arr = np.maximum(np.asarray(ws, dtype=float), 1e-5)

    return prefactor * tan_beta_arr * ((hs_arr ** float(hs_exponent)) * tp_arr) / ws_arr


def compute_dimensionless_upsilon(
    upsilon: Union[float, np.ndarray],
    upsilon_ref: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Normalize dimensional Upsilon [kg/s] by site reference flux Upsilon_ref [kg/s] to yield dimensionless relative forcing Upsilon* [-].

    Theoretical Formulation:
        Upsilon* = Upsilon / Upsilon_ref

    Parameters
    ----------
    upsilon : float or np.ndarray
        Dimensional Unified Coastal Vulnerability Index [kg/s].
    upsilon_ref : float or np.ndarray
        Site reference forcing flux [kg/s].

    Returns
    -------
    upsilon_star : float or np.ndarray
        Dimensionless relative forcing index [-].
    """
    ref = np.maximum(np.asarray(upsilon_ref, dtype=float), 1e-6)
    return np.asarray(upsilon, dtype=float) / ref


def compute_direction_aware_upsilon_disequilibrium(
    upsilon: Union[float, np.ndarray],
    upsilon_ref: Union[float, np.ndarray],
    upsilon_crit: Optional[Union[float, np.ndarray]] = None,
    target: str = "erosion",
    sigma: float = 0.35,
    recovery_mode: str = "capacity_deficit",
) -> Union[float, np.ndarray]:
    """Compute Direction-Aware Signed Upsilon Disequilibrium Score S_Upsilon(t).

    Theoretical Formulation:
    - Erosive / Storm Queries (target = 'erosion'):
        S_Upsilon(t) = ReLU(Upsilon*(t) - 1.0)
    - Constructive Swash / Accretion Queries (target = 'recovery'):
        S_Upsilon(t) = ReLU(1.0 - Upsilon*(t))  [if recovery_mode == 'capacity_deficit']
        or ReLU((Upsilon_crit - Upsilon(t)) / Upsilon_ref) [if recovery_mode == 'critical_deficit']
        or max(ReLU(1.0 - Upsilon*), ReLU((Upsilon_crit - Upsilon) / Upsilon_ref)) [if recovery_mode == 'combined']
    - Equilibrium Background Queries (target = 'equilibrium'):
        S_Upsilon(t) = exp(- (Upsilon*(t) - 1.0)^2 / (2 * sigma^2))

    Parameters
    ----------
    upsilon : float or np.ndarray
        Dimensional Unified Coastal Vulnerability Index Upsilon [kg/s].
    upsilon_ref : float or np.ndarray
        Site reference kinetic forcing flux Upsilon_ref [kg/s].
    upsilon_crit : float or np.ndarray, optional
        Critical threshold shear activation flux Upsilon_crit [kg/s].
    target : str, default 'erosion'
        Target intent: 'erosion', 'recovery', or 'equilibrium'.
    sigma : float, default 0.35
        Gaussian scale parameter for equilibrium background centering.
    recovery_mode : str, default 'capacity_deficit'
        Mode for recovery calculation: 'capacity_deficit' (ReLU(1.0 - Upsilon*)),
        'critical_deficit' (ReLU((Upsilon_crit - Upsilon) / Upsilon_ref)),
        or 'combined' (maximum of both).

    Returns
    -------
    s_upsilon : float or np.ndarray
        Direction-aware disequilibrium score S_Upsilon(t).
    """
    u_arr = np.asarray(upsilon, dtype=float)
    ref_arr = np.maximum(np.asarray(upsilon_ref, dtype=float), 1e-6)
    u_star = u_arr / ref_arr

    target_clean = str(target).lower().strip()
    if target_clean in ["recovery", "accretion", "accretive", "swash", "berm", "calm"]:
        if recovery_mode in ["critical_deficit", "crit"] and upsilon_crit is not None:
            crit_arr = np.asarray(upsilon_crit, dtype=float)
            score = np.maximum(0.0, (crit_arr - u_arr) / ref_arr)
        elif recovery_mode == "combined" and upsilon_crit is not None:
            crit_arr = np.asarray(upsilon_crit, dtype=float)
            score_cap = np.maximum(0.0, 1.0 - u_star)
            score_crit = np.maximum(0.0, (crit_arr - u_arr) / ref_arr)
            score = np.maximum(score_cap, score_crit)
        else:
            score = np.maximum(0.0, 1.0 - u_star)
    elif target_clean in ["equilibrium", "steady_state", "stable", "background", "neutral"]:
        score = np.exp(-((u_star - 1.0) ** 2) / (2.0 * (sigma ** 2)))
    else:
        # Default: erosion / storm / peak
        score = np.maximum(0.0, u_star - 1.0)

    if np.isscalar(upsilon) and np.isscalar(upsilon_ref):
        return float(score)
    return score



def compute_unified_coastal_vulnerability_index(
    hs: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
    surge: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    wind_dir_deg: Union[float, np.ndarray],
    wave_dir_deg: Optional[Union[float, np.ndarray]] = None,
    directional_spread_deg: Optional[Union[float, np.ndarray]] = None,
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
    tide_phase_exponent: float = 1.0,
    enable_tide_phase_gating: bool = True,
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
    d50_m: Optional[Union[float, np.ndarray]] = None,
    d50: Optional[Union[float, np.ndarray]] = None,
    tan_beta: Optional[Union[float, np.ndarray]] = None,
    shore_normal_deg: Optional[float] = None,
    site: Optional[str] = None,
    dynamic_slope: bool = False,
    dynamic_grain_size: bool = False,
    delta_x: Optional[Union[float, np.ndarray]] = None,
    upsilon_crit_0: float = 500.0,
    d50_ref: float = 0.00035,
    tan_beta_ref: float = 0.08,
    upsilon_crit_a: float = 1.0,
    upsilon_crit_b: float = 1.0,
    upsilon_crit_mode: str = "power_law",
    upsilon_crit: Optional[Union[float, np.ndarray]] = None,
    enable_critical_threshold: bool = True,
    y_alongshore_m: Optional[Union[float, np.ndarray]] = None,
    y_center_m: Optional[float] = None,
    l_beach_m: Optional[float] = None,
    alpha_rot: Optional[float] = None,
    upsilon_eq_mode: str = "expanding",
    upsilon_eq_val: Optional[Union[float, np.ndarray]] = None,
    eq_window_hours: Optional[int] = None,
    eq_min_periods: int = 24,
    eq_initial_baseline: float = 2500.0,
    rho: float = RHO_WATER,
    g: float = G_CONST,
    tau: float = TAU_CONST,
    enable_steepness_scaling: bool = False,
    enable_sallenger_impact: bool = False,
    s_crit: float = 0.025,
    steepness_gamma: float = 0.5,
    alpha_impact: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute the Unified Coastal Vulnerability Index (Υ) with continuous dimensionless modulators and site parameterization.

    Theoretical & Dimensional Formulation
    -------------------------------------
    The Unified Coastal Vulnerability Index (Υ) is a semi-empirical, physics-informed composite index
    designed to synthesize wave power flux, dynamic sediment settling kinetics, foreshore beach slope,
    empirical wave height scaling, total water level (TWL), tidal energy dilution, instantaneous high-tide gating,
    directional radiation stress, reef-fronted dissipation, and critical shear activation:

        Υ = C_0 * tan(beta) * [ (H_shore^n * Tp) / ws(T) ] * TWL_amp * Psi_tide * Psi_phase * Psi_dir * Wind_amp
        Υ_excess = max(0, Υ - Υ_crit(tan(beta), D_50))

    Where:
        H_shore = H_0 * Psi_reef
        Υ_crit(tan(beta), D_50) = Υ_crit,0 * (D_50 / D_50,ref)^a * (tan(beta) / tan(beta)_ref)^b

    Dimensional Breakdown:
    - Prefactor C_0 = (rho * g^1.5) / (16 * (2*pi)^1.5) has physical dimensions [kg / (m^1.5 * s^3)].
    - Foreshore slope tan(beta) is dimensionless [-].
    - Sediment-wave core [ (H_shore^n * Tp) / ws(T) ] has dimensions [m^(n-1) * s^2].
    - At the baseline exponent n = 2.5, the core dimensions are [m^1.5 * s^2], giving Υ net dimensions
      of [kg / s] (kinetic sediment forcing mass flux rate per second).
    - Continuous modulators (TWL_amp, Psi_tide, Psi_phase, Psi_dir, Psi_reef, Wind_amp) are dimensionless scaling factors [-].

    Dimensionless Relative Forcing Index (Upsilon*):
    - Normalizing dimensional Upsilon by site reference flux Upsilon_ref produces scale-invariant relative forcing:
        Upsilon_ref = C_0 * tan(beta_site) * [ (Hs_mean^n * Tp_mean) / ws_site ]
        Upsilon* = Upsilon / Upsilon_ref
      where Upsilon* = 1.0 represents the mean background kinetic carrying capacity for any global archetype.

    Continuous Dimensionless Scaling Factors & Morphodynamic Regimes:
    1. Critical Threshold Shear Activation: Upsilon_excess = max(0, Upsilon - Upsilon_crit(tan(beta), D_50))
       Beaches with steep dunes or coarser sand resist morphodynamic change until a critical threshold is crossed,
       eliminating background noise during calm seasons and sharpening storm event detection.
    2. Tidal Smearing Factor: Psi_tide = 1 / (1 + TR / (2 * Hs))
       Dilutes wave work across a broad intertidal sweep on macrotidal beaches (Truc Vert) while
       concentrating berm attack on microtidal coasts (Duck, NC).
    3. Instantaneous High-Tide Wave Energy Gating: Psi_phase(t) = [ (eta_tide(t) - MSL) / (TR / 2) + 1 ]^p
       Weights hourly wave energy flux by instantaneous tidal phase relative to MSL/berm crest.
       Eliminates false-positive erosion signals when huge waves occur at low tide on macrotidal beaches (Truc Vert),
       while concentrating attack at high tide.
    4. Rectified Shore-Normal Directional Alignment: Psi_dir = max(0, cos(theta_wave - theta_shore))^2
       Matches shore-normal radiation stress S_xx ~ E * cos^2(theta_rel) that drives cross-shore gouging.
    5. Alongshore Rotational Divergence: Phi_longshore = sin(2 * (theta_wave - theta_shore))
       Captures alongshore transport gradients and beach rotation on embayed coasts (Narrabeen).
    6. Reef Dissipation Modulator: Psi_reef = K_c * K_f = min(1, gamma_b * h_r / H_0) * exp(-c_f * W_r / h_r)
       Dynamically models coral reefs as physical low-pass filters via crest clipping and flat friction (Oahu / Waikiki).
    7. Decoupled TWL Amplification: TWL_amp = 1.0 + max(S / s_ref, 0) + max(R_2% / r_ref, 0)
    8. Asymmetric Wind Stress Amplification:
       - Onshore (W_a >= 0): Wind_amp = 1.0 + min(c_on * (W_a / w_ref_on)^2, w_bound_on)
       - Offshore (W_a < 0): Wind_amp = 1.0 - min(c_off * (|W_a| / w_ref_off)^2, w_bound_off)
       Onshore winds enhance set-up and overtopping, while offshore winds primarily flatten wave steepness with attenuated impact.

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
        Astronomical tide water level relative to MSL (m). Used for instantaneous tide-phase gating and reef depth h_r(t).
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
    tide_phase_exponent : float
        Nonlinear exponent p for instantaneous high-tide gating Psi_phase = [ (eta - MSL)/(TR/2) + 1 ]^p (default 1.0).
    enable_tide_phase_gating : bool
        Whether to modulate Upsilon by instantaneous tide phase Psi_phase when tide series is provided (default True).
    s_ref, r_ref : float
        Reference scales for surge (1.0m) and runup (2.0m).
    w_ref, w_bound : float
        Reference wind speed (15.0 m/s) and symmetric modulation bound fallback (0.6).
    w_ref_on, w_ref_off : float, optional
        Onshore and offshore reference wind speeds (default w_ref = 15.0 m/s).
    w_bound_on, w_bound_off : float, optional
        Onshore and offshore wind modulation bounds (default w_bound = 0.6 and 0.3).
    c_wind_on, c_wind_off : float
        Onshore and offshore scaling gain coefficients (default 1.0 and 0.5).
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
    upsilon_crit_0 : float
        Baseline critical shear threshold at reference conditions (default 500.0).
    d50_ref : float
        Reference median grain size (default 0.00035 m).
    tan_beta_ref : float
        Reference foreshore beach slope (default 0.08).
    upsilon_crit : float or np.ndarray, optional
        Explicit critical threshold series. If None, calculated via compute_critical_shear_threshold.
    enable_critical_threshold : bool
        Whether to compute Upsilon_excess above the critical activation threshold (default True).
    rho : float
        Water density (1025 kg/m^3).
    g : float
        Gravitational acceleration (9.81 m/s^2).
    tau : float
        2*pi.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with hydrodynamics, dynamic settling kinetics, wave dissipation, reef attenuation, runup, slope, grain size, modulators, Upsilon, Upsilon_crit, and Upsilon_excess.
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
    
    resolved_d50 = d50_m if d50_m is not None else (d50 if d50 is not None else kwargs.get("d50_mm", None))
    if resolved_d50 is not None:
        resolved_d50_val = np.asarray(resolved_d50, dtype=float)
        # If user passed mm e.g. 0.35 mm vs 0.00035 m
        if np.all(resolved_d50_val > 0.05):
            resolved_d50_m = resolved_d50_val * 1e-3
        else:
            resolved_d50_m = resolved_d50_val
    else:
        resolved_d50_m = None

    base_d50 = resolved_d50_m if resolved_d50_m is not None else site_cfg.get("d50_m", 0.00035)
    base_tr = tide_range_m if tide_range_m is not None else site_cfg.get("tide_range_m", 0.0)

    crit_a = float(kwargs.get("a", upsilon_crit_a))
    crit_b = float(kwargs.get("b", upsilon_crit_b))

    site_is_reef = is_reef if is_reef is not None else site_cfg.get("is_reef", False)
    site_z_reef = z_reef_m if z_reef_m is not None else site_cfg.get("z_reef_m", -1.0)
    site_wr = reef_flat_width_m if reef_flat_width_m is not None else site_cfg.get("reef_flat_width_m", (150.0 if site_is_reef else 0.0))
    site_cf = c_f if c_f is not None else site_cfg.get("c_f", 0.08)
    site_gammab = gamma_b if gamma_b is not None else site_cfg.get("gamma_b", 0.78)

    tr_arr = np.broadcast_to(np.asarray(base_tr, dtype=float), hs_arr.shape) if not np.isscalar(base_tr) else np.full_like(hs_arr, float(base_tr))
    tide_arr = np.broadcast_to(np.asarray(tide, dtype=float), hs_arr.shape) if (tide is not None and not np.isscalar(tide)) else (np.full_like(hs_arr, float(tide)) if tide is not None else np.zeros_like(hs_arr))

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
    site_ld = site_cfg.get("l_d_ref_m", 25.0)
    psi_reef, h_reef, k_c, k_f, hs_shore = compute_reef_dissipation_modulator(
        hs=hs_arr,
        tide=tide_arr,
        surge=surge_arr,
        z_reef_m=site_z_reef,
        reef_flat_width_m=site_wr,
        c_f=site_cf,
        gamma_b=site_gammab,
        l_d_ref_m=site_ld,
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

    # 6b. Wave steepness scaling & exact radiation stress tensor
    s_0, psi_steep = compute_wave_steepness(hs_eff, tp_arr, g=g, tau=tau, s_crit=s_crit, gamma=steepness_gamma)
    s_xx, s_xy, psi_sxx, e_energy = compute_radiation_stress_tensor(
        hs=hs_eff, tp=tp_arr, wave_dir_deg=w_dir_arr, shore_normal_deg=eff_shore_normal, rho=rho, g=g
    )

    # 6c. Sallenger storm impact scale (Collision / Overwash)
    site_berm = site_cfg.get("berm_crest_m", 2.5)
    twl_elev = tide_arr + surge_arr + r_2
    psi_impact, impact_regimes = compute_sallenger_impact_ratio(
        twl=twl_elev, berm_crest_m=site_berm, hs=hs_eff, alpha_impact=alpha_impact
    )

    # 7. Cross-array wave dissipation & sandbar / reef shielding
    if p_outer_Wm is not None:
        delta_p, shielding = compute_cross_array_dissipation_ratio(p_outer_Wm, p_flux)
    elif site_is_reef:
        # Deepwater incident wave power vs transmitted shore wave power across reef flat
        p_deep = compute_wave_energy_flux(hs_arr, tp_arr, rho=rho, g=g, tau=tau)
        delta_p, shielding = compute_cross_array_dissipation_ratio(p_deep, p_flux)
    else:
        delta_p = np.zeros_like(hs_arr)
        shielding = np.ones_like(hs_arr)
    
    # 8. Continuous Dimensionless Physical Modulators
    psi_tide = compute_tidal_smearing_factor(tr_arr, hs_arr)
    if enable_tide_phase_gating and tide is not None:
        psi_phase = compute_instantaneous_tide_phase_factor(
            tide=tide_arr,
            tide_range_m=tr_arr,
            msl_m=0.0,
            p=tide_phase_exponent,
        )
    else:
        psi_phase = np.ones_like(hs_arr) if not np.isscalar(hs) else 1.0

    psi_dir = compute_directional_wave_factor(w_dir_arr, eff_shore_normal, m=2.0)
    phi_longshore = compute_longshore_transport_divergence(w_dir_arr, eff_shore_normal)

    # Alongshore Transport Gradient & Embayment Rotation Parameters
    site_is_embayment = site_cfg.get("is_embayment", False)
    site_l_beach = l_beach_m if l_beach_m is not None else site_cfg.get("l_beach_m", 3600.0)
    site_y_center = y_center_m if y_center_m is not None else site_cfg.get("y_center_m", 1800.0)
    site_alpha_rot = alpha_rot if alpha_rot is not None else site_cfg.get("alpha_rot", (12.0 if site_is_embayment else 0.0))

    if y_alongshore_m is not None:
        y_coord_arr = np.broadcast_to(np.asarray(y_alongshore_m, dtype=float), hs_arr.shape) if not np.isscalar(y_alongshore_m) else np.full_like(hs_arr, float(y_alongshore_m))
        phi_rot_gradient = (y_coord_arr - float(site_y_center)) / max(float(site_l_beach), 1.0)
        delta_x_rot_proxy = float(site_alpha_rot) * phi_longshore * phi_rot_gradient
    else:
        y_coord_arr = np.full_like(hs_arr, float(site_y_center))
        phi_rot_gradient = np.zeros_like(hs_arr)
        delta_x_rot_proxy = np.zeros_like(hs_arr)

    # 9. Prefactor & Wave-Sediment Kinetic Core: C_0 * (H_shore^n * Tp) / ws(T)
    prefactor = (rho * (g ** 1.5)) / (16.0 * (tau ** 1.5))
    wave_sed_core = ( (np.maximum(hs_eff, 0.01) ** n_exp) * np.maximum(tp_arr, 0.1) ) / ws
    
    # 10. Decoupled Non-Dimensional TWL Amplification Factor (with safe upper ceiling)
    twl_amplification = np.clip(
        1.0
        + np.maximum(surge_arr / max(s_ref, 1e-3), 0.0)
        + np.maximum(r_2 / max(r_ref, 1e-3), 0.0),
        1.0,
        20.0
    )
    
    # 11. Projected Asymmetric Wind Stress Amplification Factor
    eff_w_ref_on = w_ref_on if w_ref_on is not None else w_ref
    eff_w_ref_off = w_ref_off if w_ref_off is not None else w_ref
    eff_w_bound_on = w_bound_on if w_bound_on is not None else w_bound
    eff_w_bound_off = w_bound_off if w_bound_off is not None else (0.5 * w_bound if w_bound != 0.6 else 0.3)

    wind_amplification = compute_asymmetric_wind_amplification(
        w_a=w_a,
        w_ref_on=eff_w_ref_on,
        w_ref_off=eff_w_ref_off,
        w_bound_on=eff_w_bound_on,
        w_bound_off=eff_w_bound_off,
        c_on=c_wind_on,
        c_off=c_wind_off,
    )
    
    # 12. Unified Coastal Vulnerability Index (Upsilon_unified)
    upsilon_base = prefactor * tan_beta_arr * wave_sed_core * twl_amplification * psi_tide * psi_phase * psi_dir * wind_amplification
    upsilon = upsilon_base.copy()
    if enable_steepness_scaling:
        upsilon = upsilon * psi_steep
    if enable_sallenger_impact:
        upsilon = upsilon * psi_impact
    upsilon_full = upsilon_base * psi_steep * psi_impact

    # Multi-scale antecedent memory decay convolutions
    multiscale_decay = compute_multiscale_memory_decay(upsilon, tau_days=(7, 30, 90, 365))
    
    # 13. Critical Threshold Shear Activation (Upsilon_excess = max(0, Upsilon - Upsilon_crit))
    if upsilon_crit is not None:
        u_crit_arr = np.broadcast_to(np.asarray(upsilon_crit, dtype=float), hs_arr.shape) if not np.isscalar(upsilon_crit) else np.full_like(hs_arr, float(upsilon_crit))
    else:
        u_crit_arr = compute_critical_shear_threshold(
            tan_beta=tan_beta_arr,
            d50_m=d50_arr,
            upsilon_crit_0=upsilon_crit_0,
            d50_ref=d50_ref,
            tan_beta_ref=tan_beta_ref,
            a=crit_a,
            b=crit_b,
            mode=upsilon_crit_mode,
        )
        if np.isscalar(u_crit_arr):
            u_crit_arr = np.full_like(hs_arr, float(u_crit_arr))

    if enable_critical_threshold:
        upsilon_excess = np.maximum(0.0, upsilon - u_crit_arr)
    else:
        upsilon_excess = upsilon.copy()

    log_upsilon_excess = np.log10(np.maximum(upsilon_excess, 1e-4))

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

    # Signed Equilibrium Formulation (Upsilon_diseq = Upsilon - Upsilon_eq) - Suggestion 7 (Preventing Data Leakage)
    if upsilon_eq_val is not None:
        eq_series = upsilon_eq_val
    elif upsilon_eq_mode in ["expanding", "expanding_window"]:
        eq_series = compute_expanding_window_equilibrium(
            upsilon,
            window_hours=None,
            min_periods=eq_min_periods,
            initial_baseline=eq_initial_baseline,
        )
    elif upsilon_eq_mode in ["trailing", "trailing_3yr", "trailing_window", "rolling"]:
        w_h = eq_window_hours if eq_window_hours is not None else (3 * 365 * 24)
        eq_series = compute_expanding_window_equilibrium(
            upsilon,
            window_hours=w_h,
            min_periods=eq_min_periods,
            initial_baseline=eq_initial_baseline,
        )
    else:
        eq_series = eq_initial_baseline

    upsilon_diseq = compute_upsilon_disequilibrium(upsilon, upsilon_eq=eq_series)
    signed_eq_rate = compute_signed_equilibrium_rate(upsilon, upsilon_eq=eq_series)
    
    # Calculate explicit Tier 1 to 4 Physical Complexity Progression
    u_t1 = compute_tier1_base_upsilon(
        hs=hs_eff,
        tp=tp_arr,
        surge=surge_arr,
        ws=ws,
        tan_beta=tan_beta_arr,
        hs_exponent=n_exp,
        s_ref=s_ref,
        rho=rho,
        g=g,
        tau=tau,
    )

    u_t2, r_ig = compute_tier2_infragravity_upsilon(
        hs=hs_eff,
        tp=tp_arr,
        surge=surge_arr,
        ws=ws,
        tan_beta=tan_beta_arr,
        hs_exponent=n_exp,
        s_ref=s_ref,
        r_ref=r_ref,
        rho=rho,
        g=g,
        tau=tau,
    )

    u_t3, p_l = compute_tier3_alongshore_upsilon(
        upsilon_tier2=u_t2,
        hs=hs_eff,
        tp=tp_arr,
        wave_dir_deg=w_dir_arr,
        shore_normal_deg=eff_shore_normal,
        rho=rho,
        g=g,
        tau=tau,
    )

    if directional_spread_deg is not None:
        sigma_theta_arr = np.broadcast_to(np.asarray(directional_spread_deg, dtype=float), hs_arr.shape) if not np.isscalar(directional_spread_deg) else np.full_like(hs_arr, float(directional_spread_deg))
        proxy_sigma = np.maximum(5.0, 35.0 * np.exp(-(tp_arr - 4.0) / 8.0) + 8.0)
        sigma_theta_arr = np.where(np.isnan(sigma_theta_arr), proxy_sigma, sigma_theta_arr)
    else:
        # Physical proxy for wave spreading angle (deg) based on peak period Tp: long swell ~8-15 deg, short wind sea ~35-45 deg
        sigma_theta_arr = np.maximum(5.0, 35.0 * np.exp(-(tp_arr - 4.0) / 8.0) + 8.0)

    u_t4, k_focus = compute_tier4_directional_spread_upsilon(
        upsilon_tier3=u_t3,
        sigma_theta_deg=sigma_theta_arr,
    )

    # 14. Site Reference Kinetic Flux & Dimensionless Relative Forcing Index (Upsilon*)
    hs_ref_val = float(np.nanmean(hs_eff))
    tp_ref_val = float(np.nanmean(tp_arr))
    tan_beta_ref_val = float(np.nanmean(tan_beta_arr))
    ws_ref_val = float(np.nanmean(ws))
    u_ref_scalar = compute_site_reference_flux(
        tan_beta=tan_beta_ref_val,
        hs_mean=hs_ref_val,
        tp_mean=tp_ref_val,
        ws=ws_ref_val,
        hs_exponent=n_exp,
        rho=rho,
        g=g,
        tau=tau,
    )
    u_ref_arr = np.full_like(hs_arr, float(u_ref_scalar))
    upsilon_star = compute_dimensionless_upsilon(upsilon, u_ref_arr)
    upsilon_star_excess = compute_dimensionless_upsilon(upsilon_excess, u_ref_arr)
    upsilon_star_crit = compute_dimensionless_upsilon(u_crit_arr, u_ref_arr)

    df_out = pd.DataFrame({
        "hs_m": hs_arr,
        "tp_s": tp_arr,
        "surge_m": surge_arr,
        "runup_r2_m": r_2,
        "runup_r_ig_m": r_ig,
        "total_water_level_m": surge_arr + r_2,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir_deg,
        "wind_normal_ms": w_a,
        "wave_dir_deg": w_dir_arr,
        "directional_spread_deg": sigma_theta_arr,
        "k_focus": k_focus,
        "tide_range_m": tr_arr,
        "tide_m": tide_arr,
        "psi_tide": psi_tide,
        "psi_phase": psi_phase,
        "psi_tide_phase": psi_phase,
        "psi_dir": psi_dir,
        "phi_longshore": phi_longshore,
        "p_l_alongshore_Wm": p_l,
        "y_alongshore_m": y_coord_arr,
        "phi_rot_gradient": phi_rot_gradient,
        "delta_x_rot_proxy": delta_x_rot_proxy,
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
        "upsilon_tier1": u_t1,
        "upsilon_tier2": u_t2,
        "upsilon_tier3": u_t3,
        "upsilon_tier4": u_t4,
        "upsilon_eq": eq_series if isinstance(eq_series, np.ndarray) else np.full_like(hs_arr, float(eq_series)),
        "upsilon_diseq": upsilon_diseq,
        "signed_equilibrium_rate": signed_eq_rate,
        "hs_exponent": np.full_like(hs_arr, n_exp),
        "tan_beta": tan_beta_arr,
        "d50_m": d50_arr,
        "upsilon_crit": u_crit_arr,
        "upsilon": upsilon,
        "upsilon_excess": upsilon_excess,
        "log_upsilon_excess": log_upsilon_excess,
        "upsilon_shielded": upsilon_shielded,
        "upsilon_raw": upsilon_raw,
        "log_upsilon": log_upsilon,
        "upsilon_ref": u_ref_arr,
        "upsilon_star": upsilon_star,
        "upsilon_star_excess": upsilon_star_excess,
        "upsilon_star_crit": upsilon_star_crit,
        "wave_steepness_s0": s_0,
        "psi_steep": psi_steep,
        "s_xx_Wm": s_xx,
        "s_xy_Wm": s_xy,
        "psi_sxx": psi_sxx,
        "e_energy_Jm2": e_energy,
        "psi_impact": psi_impact,
        "impact_regime": impact_regimes,
        "upsilon_full": upsilon_full,
        "upsilon_decay_7d": multiscale_decay["upsilon_decay_7d"],
        "upsilon_decay_30d": multiscale_decay["upsilon_decay_30d"],
        "upsilon_decay_90d": multiscale_decay["upsilon_decay_90d"],
        "upsilon_decay_365d": multiscale_decay["upsilon_decay_365d"],
        "upsilon_diseq_7d": multiscale_decay["upsilon_diseq_7d"],
        "upsilon_diseq_30d": multiscale_decay["upsilon_diseq_30d"],
        "upsilon_diseq_90d": multiscale_decay["upsilon_diseq_90d"],
        "upsilon_diseq_365d": multiscale_decay["upsilon_diseq_365d"],
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

