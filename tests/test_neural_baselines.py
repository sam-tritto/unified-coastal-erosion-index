"""Tests for Deep Learning Baseline Architectures (MLP, LSTM, PG-LSTM)."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data import generate_frf_duck_hurricane_ian_dataset
from src.neural_baselines import (
    CoastalMLP,
    CoastalLSTM,
    PhysicsGuidedLSTM,
    PhysicsGuidedLoss,
    create_sliding_window_dataset,
    train_neural_model,
    evaluate_dual_horizon,
    simulate_storm_rollout,
    run_deep_learning_benchmark,
    generate_whitepaper_synthesis_table,
)


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    df = generate_frf_duck_hurricane_ian_dataset(random_seed=42)
    return df


def test_coastal_mlp_forward():
    batch_size = 8
    input_dim = 16
    model = CoastalMLP(input_dim=input_dim, hidden_dims=[32, 16], dropout=0.0)
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_coastal_lstm_forward():
    batch_size = 4
    seq_len = 24
    input_dim = 5
    model = CoastalLSTM(input_dim=input_dim, hidden_dim=16, num_layers=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_physics_guided_lstm_forward():
    batch_size = 4
    seq_len = 24
    input_dim = 9
    model = PhysicsGuidedLSTM(input_dim=input_dim, hidden_dim=24, num_layers=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_physics_guided_loss():
    criterion = PhysicsGuidedLoss(lambda_mono=0.5, lambda_mass=0.25, lag_col_idx=4, log_upsilon_col_idx=5)
    batch_size = 6
    lookback = 12
    n_features = 8

    batch_x = torch.randn(batch_size, lookback, n_features)
    targets = torch.randn(batch_size)
    preds = torch.randn(batch_size)

    loss_total, breakdown = criterion(preds, targets, batch_x)
    assert isinstance(loss_total, torch.Tensor)
    assert loss_total.ndim == 0
    assert not torch.isnan(loss_total)
    assert "loss_total" in breakdown
    assert "loss_mono" in breakdown
    assert "loss_mass" in breakdown
    assert breakdown["loss_total"] >= 0.0


def test_sliding_window_dataset(sample_dataset: pd.DataFrame):
    raw_features = ["hs_m", "tp_s", "surge_m", "wind_normal_ms"]
    lookback = 12
    data = create_sliding_window_dataset(
        sample_dataset, feature_cols=raw_features, target_col="shoreline_pos_obs_m", lookback=lookback, train_split=0.70
    )
    assert "x_train" in data and "x_test" in data
    assert data["x_train"].dim() == 3
    assert data["x_train"].size(1) == lookback
    assert data["x_train"].size(2) == len(raw_features) + 1


def test_run_deep_learning_benchmark_and_synthesis(sample_dataset: pd.DataFrame):
    df_results, suite_data = run_deep_learning_benchmark(
        sample_dataset, lookback=12, train_split=0.70, epochs=2, batch_size=16, storm_duration=24, random_seed=42
    )
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == 4
    assert "1-Hour Nowcast R2" in df_results.columns
    assert "72h Storm RMSE (m)" in df_results.columns
    assert "Full-Year R2 Score" in df_results.columns
    assert "Tier" in df_results.columns
    assert "Tier 1 (Data-Driven)" in suite_data["trained_models"]
    assert "Tier 2 (Input-Guided)" in suite_data["trained_models"]
    assert "Tier 3 (Dual-Guided)" in suite_data["trained_models"]
    assert "MLP" in suite_data["storm_rollouts"]

    # Test synthesis table with 3 standardized baseline classes
    df_bayesian_dummy = pd.DataFrame([
        {"Model": "Yates (2009) Equilibrium", "R2 Score": -0.0266, "RMSE (m)": 8.0875},
        {"Model": "XBeach Process Model", "R2 Score": -0.0364, "RMSE (m)": 8.1258},
        {"Model": "Unified Index (log-Upsilon)", "R2 Score": 0.1379, "RMSE (m)": 7.4113},
    ])
    df_synth = generate_whitepaper_synthesis_table(df_bayesian_dummy, df_results)
    assert len(df_synth) == 7
    assert "Benchmark Class" in df_synth.columns
    assert "Model Architecture" in df_synth.columns
    assert "Paradigm" in df_synth.columns
    assert "Analytical / Equilibrium" in df_synth["Benchmark Class"].values
    assert "Process-Based" in df_synth["Benchmark Class"].values
