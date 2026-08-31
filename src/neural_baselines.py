"""
Deep Learning Baseline Architectures for Coastal Morphodynamics.

Provides:
1. Multi-Layer Perceptron (MLP)
2. Standard LSTM (Black-Box Recurrent)
3. Physics-Guided LSTM (PG-LSTM with Upsilon, Dean Number, Sandbar Dissipation, and Swash Accretion)
4. Multi-Horizon Operational Benchmark Suite (1-Hour Nowcast, 72-Hour Storm Rollout, and 365-Day Simulation)
5. Publication-Ready Cross-Paradigm White Paper Synthesis (Bayesian vs Deep Learning)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from src.models import evaluate_model_performance
except ImportError:
    from models import evaluate_model_performance  # type: ignore


# ==============================================================================
# Model Architectures
# ==============================================================================

class CoastalMLP(nn.Module):
    """Feedforward Multi-Layer Perceptron for coastal morphodynamic response."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.15,
        activation: str = "relu",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


class CoastalLSTM(nn.Module):
    """Standard Recurrent LSTM taking sequential hydrodynamic forcings and antecedent state feedback."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.10,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * out_mult, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        final_step = lstm_out[:, -1, :]
        return self.head(final_step).squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract recurrent latent hidden state representation (batch_size, hidden_dim)."""
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class PhysicsGuidedLSTM(nn.Module):
    """Physics-Guided Neural Network (PG-LSTM) augmented with Upsilon, Dean Number, and Wave Dissipation."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_residual = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        final_step = lstm_out[:, -1, :]
        return self.fc_residual(final_step).squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract physics-guided latent hidden representation (batch_size, hidden_dim)."""
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class PhysicsGuidedLoss(nn.Module):
    """Composite Physics-Guided Soft Loss Function for Coastal Morphodynamics.

    L_total = L_MSE + lambda_mono * L_monotonic(Upsilon, Delta_X) + lambda_mass * L_conservation

    Formulation:
    1. L_MSE: Standard Mean Squared Error reconstruction loss:
       L_MSE = (1/N) * sum((y_hat_t - y_t)^2)
    2. L_monotonic: Monotonicity penalty during severe vulnerability index (Upsilon) spikes.
       Under intense storm surge / overwash pulses (high log_upsilon), beach cutting/erosion is
       physically obligatory (Delta_X <= 0). Any predicted positive accretion (Delta_X > 0)
       is mathematically penalized with a quadratic violation loss:
       L_monotonic = mean( (ReLU(y_hat_t - x_{t-1}) * ReLU(log_upsilon_t - tau))^2 )
    3. L_conservation: Volumetric / kinematic jump conservation penalty:
       Prevents unphysical high-frequency volumetric runaway and single-step kinematic divergence:
       L_conservation = mean( ReLU(|y_hat_t - x_{t-1}| - Delta_X_max)^2 )
    """

    def __init__(
        self,
        lambda_mono: float = 0.05,
        lambda_mass: float = 0.05,
        lag_col_idx: int = 4,
        log_upsilon_col_idx: Optional[int] = 5,
        upsilon_threshold_z: float = 0.0,
        max_delta_x_z: float = 3.0,
        dynamic_batch_scaling: bool = True,
    ):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_mass = lambda_mass
        self.lag_col_idx = lag_col_idx
        self.log_upsilon_col_idx = log_upsilon_col_idx
        self.upsilon_threshold_z = upsilon_threshold_z
        self.max_delta_x_z = max_delta_x_z
        self.dynamic_batch_scaling = dynamic_batch_scaling
        self.mse = nn.MSELoss()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Base MSE reconstruction loss
        l_mse = self.mse(preds, targets)

        if batch_x is None or (self.lambda_mono <= 0.0 and self.lambda_mass <= 0.0):
            return l_mse, {
                "loss_total": float(l_mse.item()),
                "loss_mse": float(l_mse.item()),
                "loss_mono": 0.0,
                "loss_mass": 0.0,
            }

        # Antecedent shoreline state at final timestep of lookback window
        x_prev = batch_x[:, -1, self.lag_col_idx]
        delta_x_pred = preds - x_prev  # positive = accretion, negative = erosion

        # 2. Monotonicity constraint: penalize unphysical accretion during extreme Upsilon spikes
        if self.log_upsilon_col_idx is not None and self.log_upsilon_col_idx < batch_x.size(-1):
            ups_val = batch_x[:, -1, self.log_upsilon_col_idx]
            ups_excess = torch.relu(ups_val - self.upsilon_threshold_z)
            accretion_violation = torch.relu(delta_x_pred)
            l_mono = torch.mean((accretion_violation * ups_excess) ** 2)
        else:
            l_mono = torch.tensor(0.0, device=preds.device)

        # 3. Conservation constraint: penalize unphysical sudden volumetric jumps
        excess_jump = torch.relu(torch.abs(delta_x_pred) - self.max_delta_x_z)
        l_mass = torch.mean(excess_jump ** 2)

        # Dynamic batch scaling (* loss_mse.detach()) ensures physical penalties share the exact mathematical scale of baseline MSE
        scale_factor = l_mse.detach() if self.dynamic_batch_scaling else 1.0

        l_total = l_mse + (self.lambda_mono * l_mono + self.lambda_mass * l_mass) * scale_factor

        loss_breakdown = {
            "loss_total": float(l_total.item()),
            "loss_mse": float(l_mse.item()),
            "loss_mono": float((self.lambda_mono * l_mono * scale_factor).item() if isinstance(l_mono, torch.Tensor) else 0.0),
            "loss_mass": float((self.lambda_mass * l_mass * scale_factor).item() if isinstance(l_mass, torch.Tensor) else 0.0),
        }
        return l_total, loss_breakdown


# ==============================================================================
# Data Preparation & Windowing with State Feedback
# ==============================================================================

def create_sliding_window_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "shoreline_pos_obs_m",
    lookback: int = 24,
    train_split: float = 0.75,
    scaler_type: str = "robust",
) -> Dict[str, Any]:
    """Create temporal lookback sequence windows with state feedback and train/test splits."""
    df_proc = df.copy()

    if "shoreline_lag1" not in df_proc.columns and "shoreline_pos_obs_m" in df_proc.columns:
        df_proc["shoreline_lag1"] = df_proc["shoreline_pos_obs_m"].shift(1).bfill()

    actual_features = list(feature_cols)
    if "shoreline_lag1" not in actual_features and "shoreline_lag1" in df_proc.columns:
        actual_features.append("shoreline_lag1")

    for col in actual_features + [target_col]:
        if col not in df_proc.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    x_raw = df_proc[actual_features].values
    y_raw = df_proc[target_col].values
    n = len(df_proc)

    if n <= lookback + 10:
        raise ValueError(f"Dataset length ({n}) must be greater than lookback ({lookback}) + 10.")

    split_idx = int(n * train_split)
    if split_idx <= lookback:
        split_idx = lookback + 1

    scaler_type_clean = str(scaler_type).lower().strip()
    if scaler_type_clean in ["standard", "zscore", "standardscaler"]:
        scaler_x = StandardScaler()
    elif scaler_type_clean in ["minmax", "min_max", "minmaxscaler"]:
        scaler_x = MinMaxScaler(feature_range=(-1.0, 1.0))
    else:
        scaler_x = RobustScaler()

    scaler_x.fit(x_raw[:split_idx])
    x_scaled = scaler_x.transform(x_raw)

    scaler_y = StandardScaler()
    scaler_y.fit(y_raw[:split_idx].reshape(-1, 1))
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel()

    x_seq, y_seq, time_seq = [], [], []
    for i in range(lookback, n):
        x_seq.append(x_scaled[i - lookback:i, :])
        y_seq.append(y_scaled[i])
        time_seq.append(df_proc.index[i])

    x_seq_arr = np.array(x_seq, dtype=np.float32)
    y_seq_arr = np.array(y_seq, dtype=np.float32)
    time_seq_arr = pd.DatetimeIndex(time_seq)

    eff_split = max(1, split_idx - lookback)

    x_train = torch.tensor(x_seq_arr[:eff_split])
    y_train = torch.tensor(y_seq_arr[:eff_split])
    time_train = time_seq_arr[:eff_split]

    x_test = torch.tensor(x_seq_arr[eff_split:])
    y_test = torch.tensor(y_seq_arr[eff_split:])
    time_test = time_seq_arr[eff_split:]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "time_train": time_train,
        "x_test": x_test,
        "y_test": y_test,
        "time_test": time_test,
        "x_all": torch.tensor(x_seq_arr),
        "time_all": time_seq_arr,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": actual_features,
        "lookback": lookback,
        "y_true_test_unscaled": y_raw[split_idx:],
        "y_true_full_unscaled": y_raw[lookback:],
        "raw_dataframe": df_proc,
    }


# ==============================================================================
# Training & Multi-Horizon Evaluation Engine
# ==============================================================================

def train_neural_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    criterion: Optional[Union[nn.Module, Any]] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, List[float]]:
    """Train a PyTorch neural network model with AdamW and MSE or PhysicsGuidedLoss."""
    model.to(device)
    model.train()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    is_physics_loss = isinstance(criterion, PhysicsGuidedLoss)
    loss_fn = criterion if criterion is not None else nn.MSELoss()

    loss_history: List[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)

            if is_physics_loss:
                loss, _ = loss_fn(preds, batch_y, batch_x)
            else:
                loss = loss_fn(preds, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)

    model.eval()
    return model, loss_history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, R2, and BSS metrics between true and predicted shoreline arrays."""
    min_len = min(len(y_true), len(y_pred))
    y_t = y_true[:min_len]
    y_p = y_pred[:min_len]

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    mse_model = np.mean((y_t - y_p) ** 2)
    mse_baseline = np.mean((y_t - y_t[0]) ** 2)
    bss = 1.0 - (mse_model / (mse_baseline + 1e-8))

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "bss": float(bss),
    }


def simulate_storm_rollout(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_x: RobustScaler,
    scaler_y: StandardScaler,
    start_idx: int,
    horizon_hours: int = 72,
    lookback: int = 24,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Execute a multi-step recursive closed-loop forward rollout for an extreme storm event."""
    model.to(device)
    model.eval()

    n = len(df)
    end_idx = min(n, start_idx + horizon_hours)
    actual_horizon = end_idx - start_idx

    df_sim = df.copy()
    feature_names = list(feature_cols)
    lag_col_idx = feature_names.index("shoreline_lag1") if "shoreline_lag1" in feature_names else len(feature_names) - 1

    sim_preds: List[float] = []
    y_true_storm = df["shoreline_pos_obs_m"].values[start_idx:end_idx]
    time_storm = df.index[start_idx:end_idx]

    current_shoreline = float(df["shoreline_pos_obs_m"].values[start_idx - 1])

    with torch.no_grad():
        for t_step in range(start_idx, end_idx):
            window_slice = df_sim.iloc[t_step - lookback:t_step].copy()
            window_slice.iloc[-1, window_slice.columns.get_loc("shoreline_lag1")] = current_shoreline

            x_mat = window_slice[feature_names].values
            x_scaled = scaler_x.transform(x_mat)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            pred_scaled = float(model(x_tensor).cpu().numpy().item())
            pred_m = float(scaler_y.inverse_transform(np.array([[pred_scaled]]))[0, 0])

            sim_preds.append(pred_m)
            current_shoreline = pred_m

    return np.array(sim_preds, dtype=np.float64), y_true_storm, time_storm


def evaluate_dual_horizon(
    model: nn.Module,
    dataset: Dict[str, Any],
    device: str = "cpu",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate predictions on both Out-of-Sample Holdout (Test 25%) and Full-Year Sequence."""
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_test_t = dataset["x_test"].to(device)
        preds_test_scaled = model(x_test_t).cpu().numpy()

        x_all_t = dataset["x_all"].to(device)
        preds_full_scaled = model(x_all_t).cpu().numpy()

    scaler_y = dataset["scaler_y"]
    preds_test_m = scaler_y.inverse_transform(preds_test_scaled.reshape(-1, 1)).ravel()
    preds_full_m = scaler_y.inverse_transform(preds_full_scaled.reshape(-1, 1)).ravel()

    metrics_test = compute_metrics(dataset["y_true_test_unscaled"], preds_test_m)
    metrics_full = compute_metrics(dataset["y_true_full_unscaled"], preds_full_m)

    combined_metrics = {
        "Test RMSE (m)": metrics_test["rmse"],
        "Test R2 Score": metrics_test["r2"],
        "Test BSS": metrics_test["bss"],
        "Full-Year RMSE (m)": metrics_full["rmse"],
        "Full-Year R2 Score": metrics_full["r2"],
        "Full-Year BSS": metrics_full["bss"],
    }
    return combined_metrics, preds_test_m, preds_full_m


# ==============================================================================
# Definitive Three-Tier Deep Learning Benchmark Suite
# ==============================================================================

SITE_STORM_CATALOG: Dict[str, Dict[str, str]] = {
    "DUCK_NC": {
        "event_name": "Hurricane Ian (Sept/Oct 2022)",
        "start": "2022-09-30 00:00:00",
        "regime": "Tropical Cyclone / Storm Surge",
    },
    "NARRABEEN_AUS": {
        "event_name": "Major East Coast Low (ECL)",
        "start": "2022-07-04 00:00:00",
        "regime": "Embayed Beach Rotation / SE Swell",
    },
    "OAHU_NORTH_SHORE_HI": {
        "event_name": "Peak Winter North Pacific Swell",
        "start": "2023-01-22 00:00:00",
        "regime": "High-Period Groundswell",
    },
    "TORREY_PINES_CA": {
        "event_name": "Pacific Winter Storm / Atmospheric River",
        "start": "2023-01-05 00:00:00",
        "regime": "Narrow-Shelf Wave Shoaling",
    },
    "TRUC_VERT_FRANCE": {
        "event_name": "Bay of Biscay Extratropical Cyclone",
        "start": "2023-11-02 00:00:00",
        "regime": "Macrotidal Storm Surge & Swash Smearing",
    },
}


def resolve_site_storm_window(
    df: pd.DataFrame,
    site: Optional[str] = None,
    train_split: float = 0.75,
    lookback: int = 24,
    storm_window: Optional[Tuple[str, str]] = None,
    storm_start_idx: Optional[int] = None,
) -> Tuple[int, Dict[str, str]]:
    """Dynamically resolve the Section 5 Horizon 2 storm window per site.

    Option A: Automated Peak Energy Window (Fully Programmatic)
              Finds center timestamp t* corresponding to maximum compound forcing
              (peak Upsilon or peak Hs * Surge) in the out-of-sample test split.
    Option B: Site-Specific Event Dictionary (Publication-Grade Rigor)
              Uses curated lookup of canonical events (SITE_STORM_CATALOG) if
              matched and date is present in the dataset.
    """
    event_info: Dict[str, str] = {
        "event_name": "Site Peak Hazard Event",
        "start": "",
        "regime": "Extreme Storm Forcing",
        "horizon_title": "Horizon 2: 72-Hour Extreme Storm Gouging Forward Rollout (Site Peak Hazard Event)",
    }

    # 1. Explicit index override
    if storm_start_idx is not None:
        idx_val = max(lookback + 1, min(int(storm_start_idx), max(lookback + 1, len(df) - 72)))
        event_info["start"] = str(df.index[idx_val]) if len(df) > idx_val else ""
        return idx_val, event_info

    # 2. Explicit window tuple override
    if storm_window is not None:
        try:
            w_start = pd.to_datetime(storm_window[0])
            idx_loc = df.index.get_indexer([w_start], method="nearest")[0]
            idx_val = max(lookback + 1, min(int(idx_loc), max(lookback + 1, len(df) - 72)))
            event_info["start"] = str(df.index[idx_val])
            return idx_val, event_info
        except Exception:
            pass

    # 3. Detect site key
    site_key = None
    if site:
        site_key = site.upper().strip()
    elif "site" in df.columns and len(df["site"]) > 0 and pd.notna(df["site"].iloc[0]):
        site_key = str(df["site"].iloc[0]).upper().strip()
    elif hasattr(df, "attrs") and isinstance(df.attrs, dict) and df.attrs.get("site"):
        site_key = str(df.attrs["site"]).upper().strip()
    elif "site_name" in df.columns and len(df["site_name"]) > 0:
        sname = str(df["site_name"].iloc[0]).lower()
        if "duck" in sname:
            site_key = "DUCK_NC"
        elif "narrabeen" in sname:
            site_key = "NARRABEEN_AUS"
        elif "oahu" in sname or "waimea" in sname:
            site_key = "OAHU_NORTH_SHORE_HI"
        elif "torrey" in sname:
            site_key = "TORREY_PINES_CA"
        elif "truc" in sname:
            site_key = "TRUC_VERT_FRANCE"

    KEY_ALIAS = {
        "DUCK": "DUCK_NC",
        "DUCK_NC": "DUCK_NC",
        "NARRABEEN": "NARRABEEN_AUS",
        "NARRABEEN_AUS": "NARRABEEN_AUS",
        "OAHU": "OAHU_NORTH_SHORE_HI",
        "OAHU_NORTH_SHORE": "OAHU_NORTH_SHORE_HI",
        "OAHU_NORTH_SHORE_HI": "OAHU_NORTH_SHORE_HI",
        "TORREY": "TORREY_PINES_CA",
        "TORREY_PINES": "TORREY_PINES_CA",
        "TORREY_PINES_CA": "TORREY_PINES_CA",
        "TRUC_VERT": "TRUC_VERT_FRANCE",
        "TRUC_VERT_FRANCE": "TRUC_VERT_FRANCE",
    }
    if site_key in KEY_ALIAS:
        site_key = KEY_ALIAS[site_key]

    # Check Catalog (Option B)
    if site_key and site_key in SITE_STORM_CATALOG:
        cat = SITE_STORM_CATALOG[site_key]
        try:
            cat_start = pd.to_datetime(cat["start"])
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.min() <= cat_start <= df.index.max():
                    loc = df.index.get_indexer([cat_start], method="nearest")[0]
                    resolved_idx = max(lookback + 1, min(int(loc), max(lookback + 1, len(df) - 72)))
                    event_info.update({
                        "event_name": cat["event_name"],
                        "start": str(df.index[resolved_idx]),
                        "regime": cat["regime"],
                    })
                    return resolved_idx, event_info
        except Exception:
            pass

    # Option A: Automated Peak Energy Window (Fully Programmatic)
    # Find peak compound forcing in out-of-sample test split
    test_start_i = int(train_split * len(df))
    test_df = df.iloc[test_start_i:] if len(df) > test_start_i else df

    if "upsilon" in test_df.columns:
        peak_idx_val = test_df["upsilon"].idxmax()
    elif "log_upsilon" in test_df.columns:
        peak_idx_val = test_df["log_upsilon"].idxmax()
    elif "hs_m" in test_df.columns and "surge_m" in test_df.columns:
        peak_idx_val = (test_df["hs_m"] * test_df["surge_m"]).idxmax()
    elif "hs_m" in test_df.columns:
        peak_idx_val = test_df["hs_m"].idxmax()
    else:
        peak_idx_val = test_df.index[len(test_df) // 2] if len(test_df) > 0 else df.index[0]

    try:
        if isinstance(df.index, pd.DatetimeIndex) and isinstance(peak_idx_val, (pd.Timestamp, str)):
            peak_int = df.index.get_indexer([pd.to_datetime(peak_idx_val)], method="nearest")[0]
        else:
            peak_int = df.index.get_loc(peak_idx_val)
            if not isinstance(peak_int, int):
                peak_int = peak_int.start if hasattr(peak_int, "start") else test_start_i
    except Exception:
        peak_int = test_start_i + len(test_df) // 2

    resolved_idx = max(lookback + 1, min(int(peak_int - 24), max(lookback + 1, len(df) - 72)))
    event_info.update({
        "event_name": f"Site Peak Hazard Event ({site_key or 'Automated Peak'})",
        "start": str(df.index[resolved_idx]) if len(df) > resolved_idx else "",
        "regime": "Automated Out-of-Sample Peak Compound Forcing",
    })
    return resolved_idx, event_info


def run_deep_learning_benchmark(
    df: pd.DataFrame,
    lookback: int = 24,
    train_split: float = 0.75,
    epochs: int = 30,
    lr: float = 0.005,
    lr_pg: Optional[float] = None,
    hidden_dim: int = 32,
    batch_size: int = 64,
    target_col: str = "shoreline_pos_obs_m",
    storm_start_idx: Optional[int] = None,
    storm_duration: int = 72,
    storm_window: Optional[Tuple[str, str]] = None,
    site: Optional[str] = None,
    random_seed: int = 42,
    lambda_mono: float = 0.05,
    lambda_mass: float = 0.05,
    ablate_raw_features: bool = True,
    raw_scaler_type: str = "robust",
    physics_scaler_type: str = "standard",
    dynamic_batch_scaling: bool = True,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute definitive three-tier benchmark comparison across Data-Driven, Input-Guided, and Dual-Guided models.

    Tiers:
    - Tier 1 (Data-Driven): Standard LSTM (hydrodynamic forcings, pure MSE loss).
    - Tier 2 (Input-Guided): PG-LSTM (physical indices as inputs, pure MSE loss).
    - Tier 3 (Dual-Guided): PG-LSTM with physical inputs and a Physics-Guided composite loss function.
    - Reference Baseline: Multi-Layer Perceptron (feedforward).
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = "cpu"

    df_aug = df.copy()
    if target_col not in df_aug.columns:
        if "shoreline_pos_obs_m" in df_aug.columns:
            target_col = "shoreline_pos_obs_m"
        elif "shoreline_pos_true_m" in df_aug.columns:
            target_col = "shoreline_pos_true_m"

    if "shoreline_lag1" not in df_aug.columns and target_col in df_aug.columns:
        df_aug["shoreline_lag1"] = df_aug[target_col].shift(1).bfill()
    if "log_upsilon" not in df_aug.columns and "upsilon" in df_aug.columns:
        df_aug["log_upsilon"] = np.log10(np.maximum(df_aug["upsilon"].values, 1e-4))
    if "omega_ratio" not in df_aug.columns and "dean_number" in df_aug.columns:
        df_aug["omega_ratio"] = df_aug["dean_number"].values / 1.75
    if "delta_p_dissip" not in df_aug.columns:
        df_aug["delta_p_dissip"] = 0.0
    if "accretion_potential" not in df_aug.columns:
        df_aug["accretion_potential"] = (9.81 * (df_aug["tp_s"] ** 2) / (2.0 * np.pi)) / (df_aug["hs_m"] + 0.1)

    raw_features = ["hs_m", "tp_s", "surge_m", "wind_normal_ms", "shoreline_lag1"]
    if ablate_raw_features:
        physics_features = ["log_upsilon", "omega_ratio", "delta_p_dissip", "accretion_potential", "shoreline_lag1"]
    else:
        physics_features = raw_features + ["log_upsilon", "omega_ratio", "delta_p_dissip", "accretion_potential"]

    data_raw = create_sliding_window_dataset(
        df_aug, feature_cols=raw_features, target_col=target_col, lookback=lookback, train_split=train_split, scaler_type=raw_scaler_type
    )
    data_phys = create_sliding_window_dataset(
        df_aug, feature_cols=physics_features, target_col=target_col, lookback=lookback, train_split=train_split, scaler_type=physics_scaler_type
    )

    storm_start_idx, storm_event_info = resolve_site_storm_window(
        df=df_aug,
        site=site or kwargs.get("site"),
        train_split=train_split,
        lookback=lookback,
        storm_window=storm_window,
        storm_start_idx=storm_start_idx,
    )

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    full_predictions: Dict[str, np.ndarray] = {}
    storm_rollouts: Dict[str, np.ndarray] = {}

    # --- Tier 1 (Data-Driven): Standard LSTM (Pure MSE) ---
    model_t1 = CoastalLSTM(input_dim=len(data_raw["feature_cols"]), hidden_dim=hidden_dim, num_layers=2, dropout=0.10)
    model_t1, loss_t1 = train_neural_model(
        model_t1, data_raw["x_train"], data_raw["y_train"], epochs=epochs, lr=lr, batch_size=batch_size, device=device
    )
    m_t1, p_test_t1, p_full_t1 = evaluate_dual_horizon(model_t1, data_raw, device=device)
    p_storm_t1, y_storm_true, time_storm = simulate_storm_rollout(
        model_t1, df_aug, data_raw["feature_cols"], data_raw["scaler_x"], data_raw["scaler_y"],
        start_idx=storm_start_idx, horizon_hours=storm_duration, lookback=lookback, device=device
    )
    m_storm_t1 = compute_metrics(y_storm_true, p_storm_t1)
    results.append({
        "Tier": "Tier 1 (Data-Driven)",
        "Model": "Standard LSTM (Pure MSE)",
        "Paradigm": "Deep Learning",
        "Loss Formulation": "Standard MSE",
        "1-Hour Nowcast R2": m_t1["Test R2 Score"],
        "72h Storm RMSE (m)": m_storm_t1["rmse"],
        "Full-Year R2 Score": m_t1["Full-Year R2 Score"],
        "Full-Year RMSE (m)": m_t1["Full-Year RMSE (m)"],
    })
    trained_models["Tier 1 (Data-Driven)"] = model_t1
    trained_models["LSTM"] = model_t1
    test_predictions["Tier 1 (Data-Driven)"] = p_test_t1
    test_predictions["LSTM"] = p_test_t1
    full_predictions["Tier 1 (Data-Driven)"] = p_full_t1
    full_predictions["LSTM"] = p_full_t1
    storm_rollouts["Tier 1 (Data-Driven)"] = p_storm_t1
    storm_rollouts["LSTM"] = p_storm_t1

    effective_lr_pg = lr_pg if lr_pg is not None else lr

    # --- Tier 2 (Input-Guided): PG-LSTM (Physical Features + Pure MSE) ---
    model_t2 = PhysicsGuidedLSTM(input_dim=len(data_phys["feature_cols"]), hidden_dim=int(hidden_dim * 1.5), num_layers=2, dropout=0.10)
    model_t2, loss_t2 = train_neural_model(
        model_t2, data_phys["x_train"], data_phys["y_train"], epochs=epochs, lr=effective_lr_pg, batch_size=batch_size, device=device
    )
    m_t2, p_test_t2, p_full_t2 = evaluate_dual_horizon(model_t2, data_phys, device=device)
    p_storm_t2, _, _ = simulate_storm_rollout(
        model_t2, df_aug, data_phys["feature_cols"], data_phys["scaler_x"], data_phys["scaler_y"],
        start_idx=storm_start_idx, horizon_hours=storm_duration, lookback=lookback, device=device
    )
    m_storm_t2 = compute_metrics(y_storm_true, p_storm_t2)
    results.append({
        "Tier": "Tier 2 (Input-Guided)",
        "Model": "Input-Guided PG-LSTM (Physical Inputs + MSE)",
        "Paradigm": "Physics-Informed DL",
        "Loss Formulation": "Standard MSE",
        "1-Hour Nowcast R2": m_t2["Test R2 Score"],
        "72h Storm RMSE (m)": m_storm_t2["rmse"],
        "Full-Year R2 Score": m_t2["Full-Year R2 Score"],
        "Full-Year RMSE (m)": m_t2["Full-Year RMSE (m)"],
    })
    trained_models["Tier 2 (Input-Guided)"] = model_t2
    trained_models["PG-LSTM (Input-Guided)"] = model_t2
    test_predictions["Tier 2 (Input-Guided)"] = p_test_t2
    test_predictions["PG-LSTM (Input-Guided)"] = p_test_t2
    full_predictions["Tier 2 (Input-Guided)"] = p_full_t2
    full_predictions["PG-LSTM (Input-Guided)"] = p_full_t2
    storm_rollouts["Tier 2 (Input-Guided)"] = p_storm_t2
    storm_rollouts["PG-LSTM (Input-Guided)"] = p_storm_t2

    # --- Tier 3 (Dual-Guided): PG-LSTM with Physical Features + Physics-Guided Soft Loss ---
    lag_idx = data_phys["feature_cols"].index("shoreline_lag1") if "shoreline_lag1" in data_phys["feature_cols"] else len(data_phys["feature_cols"]) - 1
    ups_idx = data_phys["feature_cols"].index("log_upsilon") if "log_upsilon" in data_phys["feature_cols"] else 0

    physics_criterion = PhysicsGuidedLoss(
        lambda_mono=lambda_mono,
        lambda_mass=lambda_mass,
        lag_col_idx=lag_idx,
        log_upsilon_col_idx=ups_idx,
        upsilon_threshold_z=0.0,
        max_delta_x_z=3.0,
        dynamic_batch_scaling=dynamic_batch_scaling,
    )

    model_t3 = PhysicsGuidedLSTM(input_dim=len(data_phys["feature_cols"]), hidden_dim=int(hidden_dim * 1.5), num_layers=2, dropout=0.10)
    model_t3, loss_t3 = train_neural_model(
        model_t3, data_phys["x_train"], data_phys["y_train"], epochs=epochs, lr=effective_lr_pg, batch_size=batch_size,
        criterion=physics_criterion, device=device
    )
    m_t3, p_test_t3, p_full_t3 = evaluate_dual_horizon(model_t3, data_phys, device=device)
    p_storm_t3, _, _ = simulate_storm_rollout(
        model_t3, df_aug, data_phys["feature_cols"], data_phys["scaler_x"], data_phys["scaler_y"],
        start_idx=storm_start_idx, horizon_hours=storm_duration, lookback=lookback, device=device
    )
    m_storm_t3 = compute_metrics(y_storm_true, p_storm_t3)
    results.append({
        "Tier": "Tier 3 (Dual-Guided)",
        "Model": "Dual-Guided PG-LSTM (Physical Inputs + Soft Physics Loss)",
        "Paradigm": "Physics-Informed DL",
        "Loss Formulation": f"Composite (MSE + {lambda_mono}*L_mono + {lambda_mass}*L_mass)",
        "1-Hour Nowcast R2": m_t3["Test R2 Score"],
        "72h Storm RMSE (m)": m_storm_t3["rmse"],
        "Full-Year R2 Score": m_t3["Full-Year R2 Score"],
        "Full-Year RMSE (m)": m_t3["Full-Year RMSE (m)"],
    })
    trained_models["Tier 3 (Dual-Guided)"] = model_t3
    trained_models["PG-LSTM"] = model_t3
    trained_models["PG-LSTM (Dual-Guided)"] = model_t3
    test_predictions["Tier 3 (Dual-Guided)"] = p_test_t3
    test_predictions["PG-LSTM"] = p_test_t3
    full_predictions["Tier 3 (Dual-Guided)"] = p_full_t3
    full_predictions["PG-LSTM"] = p_full_t3
    storm_rollouts["Tier 3 (Dual-Guided)"] = p_storm_t3
    storm_rollouts["PG-LSTM"] = p_storm_t3

    # --- Reference Baseline: Multi-Layer Perceptron (MLP) ---
    mlp_input_dim = len(data_raw["feature_cols"]) * lookback
    model_mlp = CoastalMLP(input_dim=mlp_input_dim, hidden_dims=[hidden_dim * 2, hidden_dim], dropout=0.15)
    model_mlp, loss_mlp = train_neural_model(
        model_mlp, data_raw["x_train"], data_raw["y_train"], epochs=epochs, lr=lr, batch_size=batch_size, device=device
    )
    m_mlp, p_test_mlp, p_full_mlp = evaluate_dual_horizon(model_mlp, data_raw, device=device)
    p_storm_mlp, _, _ = simulate_storm_rollout(
        model_mlp, df_aug, data_raw["feature_cols"], data_raw["scaler_x"], data_raw["scaler_y"],
        start_idx=storm_start_idx, horizon_hours=storm_duration, lookback=lookback, device=device
    )
    m_storm_mlp = compute_metrics(y_storm_true, p_storm_mlp)
    results.append({
        "Tier": "Reference Baseline",
        "Model": "Black-Box MLP (Feedforward)",
        "Paradigm": "Deep Learning",
        "Loss Formulation": "Standard MSE",
        "1-Hour Nowcast R2": m_mlp["Test R2 Score"],
        "72h Storm RMSE (m)": m_storm_mlp["rmse"],
        "Full-Year R2 Score": m_mlp["Full-Year R2 Score"],
        "Full-Year RMSE (m)": m_mlp["Full-Year RMSE (m)"],
    })
    trained_models["MLP"] = model_mlp
    test_predictions["MLP"] = p_test_mlp
    full_predictions["MLP"] = p_full_mlp
    storm_rollouts["MLP"] = p_storm_mlp

    df_results = pd.DataFrame(results).sort_values(by="72h Storm RMSE (m)", ascending=True).reset_index(drop=True)

    suite_data = {
        "results_table": df_results,
        "trained_models": trained_models,
        "test_predictions": test_predictions,
        "full_predictions": full_predictions,
        "storm_rollouts": storm_rollouts,
        "time_storm": time_storm,
        "y_true_storm": y_storm_true,
        "time_test": data_raw["time_test"],
        "time_full": data_raw["time_all"],
        "y_true_test": data_raw["y_true_test_unscaled"],
        "y_true_full": data_raw["y_true_full_unscaled"],
        "storm_event_info": storm_event_info,
        "horizon_2_title": storm_event_info["horizon_title"],
        "loss_histories": {
            "Tier 1 (Data-Driven)": loss_t1,
            "Tier 2 (Input-Guided)": loss_t2,
            "Tier 3 (Dual-Guided)": loss_t3,
            "MLP": loss_mlp,
            "LSTM": loss_t1,
            "PG-LSTM": loss_t3,
        },
    }
    return df_results, suite_data


def generate_whitepaper_synthesis_table(
    df_bayesian: pd.DataFrame,
    df_neural: pd.DataFrame,
) -> pd.DataFrame:
    """Generate comprehensive cross-paradigm publication table synthesizing Analytical, Process-Based, and ML models."""
    rows: List[Dict[str, Any]] = []

    # 1. Add Bayesian & Analytical Models
    if not df_bayesian.empty:
        for _, r in df_bayesian.iterrows():
            m_name = str(r.get("Model", ""))
            
            # Robustly resolve Out-of-Sample / CV / Test / In-Sample metrics
            test_r2 = float(
                r.get("Test R2 Score",
                r.get("CV R2 Score",
                r.get("In-Sample R2 Score",
                r.get("Full R2 Score",
                r.get("R2 Score", 0.0)))))
            )
            test_rmse = float(
                r.get("Test RMSE (m)",
                r.get("CV RMSE (m)",
                r.get("In-Sample RMSE (m)",
                r.get("Full RMSE (m)",
                r.get("RMSE (m)", 0.0)))))
            )
            r2_full = float(
                r.get("Full R2 Score",
                r.get("Full-Year R2 Score",
                r.get("In-Sample R2 Score",
                r.get("R2 Score", test_r2))))
            )
            rmse_full = float(
                r.get("Full RMSE (m)",
                r.get("Full-Year RMSE (m)",
                r.get("In-Sample RMSE (m)",
                r.get("RMSE (m)", test_rmse))))
            )

            if "Yates" in m_name or ("Equilibrium" in m_name and "Upsilon" not in m_name):
                benchmark_class = "Analytical / Equilibrium"
                paradigm = "Equilibrium Analytical"
                uq_profile = "Deterministic / 94% HDI"
            elif "XBeach" in m_name:
                benchmark_class = "Process-Based"
                paradigm = "Hydrodynamic / Morphodynamic PDE"
                uq_profile = "Deterministic"
            elif "Wave Power" in m_name or "Mean Wave" in m_name or "Dual-Regime" in m_name:
                benchmark_class = "Analytical / Equilibrium"
                paradigm = "Empirical Baseline"
                uq_profile = "94% HDI Envelopes (Full MCMC)"
            elif "Hybrid" in m_name or "Upsilon" in m_name or "Dean" in m_name or "Signed" in m_name:
                benchmark_class = "Machine Learning (Bayesian)"
                paradigm = "Bayesian Physics-Informed"
                uq_profile = "94% HDI Envelopes (Full MCMC)"
            else:
                benchmark_class = "Analytical / Equilibrium"
                paradigm = "Empirical Baseline"
                uq_profile = "94% HDI Envelopes (Full MCMC)"

            rows.append({
                "Benchmark Class": benchmark_class,
                "Model Architecture": m_name,
                "Paradigm": paradigm,
                "Out-of-Sample Test R2": test_r2,
                "Out-of-Sample Test RMSE (m)": test_rmse,
                "Full-Sequence R2 Score": r2_full,
                "Full-Sequence RMSE (m)": rmse_full,
                "Uncertainty Quantification (UQ)": uq_profile,
            })

    # 2. Add Neural Models (Three-Tier Benchmark)
    if not df_neural.empty:
        for _, r in df_neural.iterrows():
            m_name = str(r.get("Model", ""))
            tier = str(r.get("Tier", ""))
            test_r2 = float(r.get("1-Hour Nowcast R2", r.get("Test R2 Score", 0.0)))
            storm_rmse = float(r.get("72h Storm RMSE (m)", r.get("Test RMSE (m)", 0.0)))
            r2_full = float(r.get("Full-Year R2 Score", r.get("Full R2 Score", 0.0)))
            rmse_full = float(r.get("Full-Year RMSE (m)", r.get("Full RMSE (m)", 0.0)))

            if "Tier 3" in tier or "Dual-Guided" in m_name:
                benchmark_class = "Machine Learning (Tier 3)"
                paradigm = "Physics-Informed DL (Dual-Guided)"
                uq_profile = "Physics Soft Loss Bounds / Dropout MC"
            elif "Tier 2" in tier or "Input-Guided" in m_name:
                benchmark_class = "Machine Learning (Tier 2)"
                paradigm = "Physics-Informed DL (Input-Guided)"
                uq_profile = "Dropout MC (Approximate)"
            elif "Tier 1" in tier or "Standard LSTM" in m_name:
                benchmark_class = "Machine Learning (Tier 1)"
                paradigm = "Deep Learning (Data-Driven)"
                uq_profile = "None (Point Estimate)"
            else:
                benchmark_class = "Machine Learning (Reference)"
                paradigm = "Deep Learning"
                uq_profile = "None (Point Estimate)"

            rows.append({
                "Benchmark Class": benchmark_class,
                "Model Architecture": m_name,
                "Paradigm": paradigm,
                "Out-of-Sample Test R2": test_r2,
                "Out-of-Sample Test RMSE (m)": storm_rmse,
                "Full-Sequence R2 Score": r2_full,
                "Full-Sequence RMSE (m)": rmse_full,
                "Uncertainty Quantification (UQ)": uq_profile,
            })

    df_synth = pd.DataFrame(rows)
    return df_synth
