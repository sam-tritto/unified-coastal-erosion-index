"""
Deep Learning Baseline Architectures for Coastal Morphodynamics.

Provides:
1. Multi-Layer Perceptron (MLP)
2. Standard LSTM (Black-Box Baseline with 24-72h lookback)
3. Physics-Guided LSTM (PG-LSTM with Upsilon, Dean Number Omega/Omega_crit, and Sandbar Dissipation)
4. Full Benchmark Training & Evaluation Suite
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler


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
        # If input has sequence dimension [batch, seq_len, features], flatten it
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


class CoastalLSTM(nn.Module):
    """Standard Black-Box Recurrent LSTM taking sequential raw forcings (Hs, Tp, Surge, Wind)."""

    def __init__(
        self,
        input_dim: int = 4,
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
        # x: [batch, seq_len, input_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # Use final time step hidden state
        final_step = lstm_out[:, -1, :]
        return self.head(final_step).squeeze(-1)


class PhysicsGuidedLSTM(nn.Module):
    """Physics-Guided Neural Network (PG-LSTM) augmented with Upsilon and Morphodynamic Channels.

    Combines raw sequential forcings with pre-computed physical groups:
    - log10(Upsilon)
    - Normalized Dean Number Omega / Omega_crit
    - Sandbar Wave Energy Dissipation Ratio Delta_P_dissip
    - Constructive Swash Accretion Potential
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.10,
        use_physics_residual_head: bool = True,
    ):
        super().__init__()
        self.use_physics_residual_head = use_physics_residual_head
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
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        final_step = lstm_out[:, -1, :]
        return self.fc_residual(final_step).squeeze(-1)


# ==============================================================================
# Data Preparation & Windowing
# ==============================================================================

def create_sliding_window_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "shoreline_pos_obs_m",
    lookback: int = 48,
    train_split: float = 0.75,
) -> Dict[str, Any]:
    """Create temporal lookback sequence windows and train/test splits."""
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from DataFrame.")

    x_raw = df[feature_cols].values
    y_raw = df[target_col].values
    n = len(df)

    if n <= lookback + 10:
        raise ValueError(f"Dataset length ({n}) must be greater than lookback ({lookback}) + 10.")

    # Split index
    split_idx = int(n * train_split)
    if split_idx <= lookback:
        split_idx = lookback + 1

    # Standardize features fit on train split only
    scaler_x = RobustScaler()
    x_train_unscaled = x_raw[:split_idx]
    scaler_x.fit(x_train_unscaled)
    x_scaled = scaler_x.transform(x_raw)

    scaler_y = StandardScaler()
    y_train_unscaled = y_raw[:split_idx].reshape(-1, 1)
    scaler_y.fit(y_train_unscaled)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel()

    x_seq, y_seq, time_seq = [], [], []
    for i in range(lookback, n):
        x_seq.append(x_scaled[i - lookback:i, :])
        y_seq.append(y_scaled[i])
        time_seq.append(df.index[i])

    x_seq_arr = np.array(x_seq, dtype=np.float32)
    y_seq_arr = np.array(y_seq, dtype=np.float32)
    time_seq_arr = pd.DatetimeIndex(time_seq)

    # Split sequences
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
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "lookback": lookback,
        "y_true_test_unscaled": y_raw[split_idx:],
    }


# ==============================================================================
# Training & Evaluation Engine
# ==============================================================================

def train_neural_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Tuple[nn.Module, List[float]]:
    """Train a PyTorch neural network model with AdamW and MSE loss."""
    model.to(device)
    model.train()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    loss_history: List[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)

    model.eval()
    return model, loss_history


def evaluate_neural_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_true_unscaled: np.ndarray,
    scaler_y: StandardScaler,
    device: str = "cpu",
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate model test predictions and compute RMSE, R2, and BSS metrics."""
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_tensor = x_test.to(device)
        preds_scaled = model(x_tensor).cpu().numpy()

    # Inverse transform to meters
    preds_unscaled = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    # Align lengths if sequence slicing has slight boundary offset
    min_len = min(len(y_true_unscaled), len(preds_unscaled))
    y_t = y_true_unscaled[:min_len]
    y_p = preds_unscaled[:min_len]

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # Brier Skill Score relative to initial shoreline baseline
    mse_model = np.mean((y_t - y_p) ** 2)
    mse_baseline = np.mean((y_t - y_t[0]) ** 2)
    bss = 1.0 - (mse_model / (mse_baseline + 1e-8))

    metrics = {
        "RMSE (m)": float(rmse),
        "R2 Score": float(r2),
        "BSS": float(bss),
    }
    return metrics, preds_unscaled


# ==============================================================================
# Unified Deep Learning Benchmark Suite
# ==============================================================================

def run_deep_learning_benchmark(
    df: pd.DataFrame,
    lookback: int = 48,
    train_split: float = 0.75,
    epochs: int = 50,
    batch_size: int = 64,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute high-level benchmark comparison across MLP, Standard LSTM, and PG-LSTM."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = "cpu"

    # 1. Feature sets
    raw_features = ["hs_m", "tp_s", "surge_m", "wind_normal_ms"]
    
    # Ensure physical columns exist
    df_aug = df.copy()
    if "log_upsilon" not in df_aug.columns and "upsilon" in df_aug.columns:
        df_aug["log_upsilon"] = np.log10(np.maximum(df_aug["upsilon"].values, 1e-4))
    if "omega_ratio" not in df_aug.columns and "dean_number" in df_aug.columns:
        df_aug["omega_ratio"] = df_aug["dean_number"].values / 1.75
    if "delta_p_dissip" not in df_aug.columns:
        df_aug["delta_p_dissip"] = 0.0
    if "accretion_potential" not in df_aug.columns:
        df_aug["accretion_potential"] = (9.81 * (df_aug["tp_s"] ** 2) / (2.0 * np.pi)) / (df_aug["hs_m"] + 0.1)

    physics_features = raw_features + ["log_upsilon", "omega_ratio", "delta_p_dissip", "accretion_potential"]

    # 2. Build Datasets
    data_raw = create_sliding_window_dataset(
        df_aug, feature_cols=raw_features, target_col="shoreline_pos_obs_m", lookback=lookback, train_split=train_split
    )
    data_phys = create_sliding_window_dataset(
        df_aug, feature_cols=physics_features, target_col="shoreline_pos_obs_m", lookback=lookback, train_split=train_split
    )

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    # --- Architecture 1: Multi-Layer Perceptron (MLP) ---
    mlp_input_dim = len(raw_features) * lookback
    model_mlp = CoastalMLP(input_dim=mlp_input_dim, hidden_dims=[64, 32], dropout=0.15)
    model_mlp, loss_mlp = train_neural_model(
        model_mlp, data_raw["x_train"], data_raw["y_train"], epochs=epochs, batch_size=batch_size, device=device
    )
    metrics_mlp, preds_mlp = evaluate_neural_model(
        model_mlp, data_raw["x_test"], data_raw["y_true_test_unscaled"], data_raw["scaler_y"], device=device
    )
    results.append({
        "Model": "Black-Box MLP (Feedforward)",
        "Type": "Deep Learning",
        "Input Features": "Raw (Hs, Tp, S, Wa)",
        "Lookback (h)": lookback,
        "RMSE (m)": metrics_mlp["RMSE (m)"],
        "R2 Score": metrics_mlp["R2 Score"],
        "BSS": metrics_mlp["BSS"],
    })
    trained_models["MLP"] = model_mlp
    test_predictions["MLP"] = preds_mlp

    # --- Architecture 2: Standard LSTM (Black-Box Recurrent) ---
    model_lstm = CoastalLSTM(input_dim=len(raw_features), hidden_dim=32, num_layers=2, dropout=0.10)
    model_lstm, loss_lstm = train_neural_model(
        model_lstm, data_raw["x_train"], data_raw["y_train"], epochs=epochs, batch_size=batch_size, device=device
    )
    metrics_lstm, preds_lstm = evaluate_neural_model(
        model_lstm, data_raw["x_test"], data_raw["y_true_test_unscaled"], data_raw["scaler_y"], device=device
    )
    results.append({
        "Model": "Standard LSTM (Black-Box Recurrent)",
        "Type": "Deep Learning",
        "Input Features": "Raw (Hs, Tp, S, Wa)",
        "Lookback (h)": lookback,
        "RMSE (m)": metrics_lstm["RMSE (m)"],
        "R2 Score": metrics_lstm["R2 Score"],
        "BSS": metrics_lstm["BSS"],
    })
    trained_models["LSTM"] = model_lstm
    test_predictions["LSTM"] = preds_lstm

    # --- Architecture 3: Physics-Guided LSTM (PG-LSTM) ---
    model_pgnn = PhysicsGuidedLSTM(input_dim=len(physics_features), hidden_dim=48, num_layers=2, dropout=0.10)
    model_pgnn, loss_pgnn = train_neural_model(
        model_pgnn, data_phys["x_train"], data_phys["y_train"], epochs=epochs, batch_size=batch_size, device=device
    )
    metrics_pgnn, preds_pgnn = evaluate_neural_model(
        model_pgnn, data_phys["x_test"], data_phys["y_true_test_unscaled"], data_phys["scaler_y"], device=device
    )
    results.append({
        "Model": "Physics-Guided LSTM (PG-LSTM)",
        "Type": "Physics-Informed DL",
        "Input Features": "Raw + Upsilon + Omega + Dissipation",
        "Lookback (h)": lookback,
        "RMSE (m)": metrics_pgnn["RMSE (m)"],
        "R2 Score": metrics_pgnn["R2 Score"],
        "BSS": metrics_pgnn["BSS"],
    })
    trained_models["PG-LSTM"] = model_pgnn
    test_predictions["PG-LSTM"] = preds_pgnn

    df_results = pd.DataFrame(results).sort_values(by="RMSE (m)", ascending=True).reset_index(drop=True)

    suite_data = {
        "results_table": df_results,
        "trained_models": trained_models,
        "test_predictions": test_predictions,
        "time_test": data_raw["time_test"],
        "y_true_test": data_raw["y_true_test_unscaled"],
        "loss_histories": {
            "MLP": loss_mlp,
            "LSTM": loss_lstm,
            "PG-LSTM": loss_pgnn,
        },
    }
    return df_results, suite_data
