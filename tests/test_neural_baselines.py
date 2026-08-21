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
    create_sliding_window_dataset,
    train_neural_model,
    evaluate_neural_model,
    run_deep_learning_benchmark,
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
    input_dim = 4
    model = CoastalLSTM(input_dim=input_dim, hidden_dim=16, num_layers=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_physics_guided_lstm_forward():
    batch_size = 4
    seq_len = 24
    input_dim = 8
    model = PhysicsGuidedLSTM(input_dim=input_dim, hidden_dim=24, num_layers=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_sliding_window_dataset(sample_dataset: pd.DataFrame):
    raw_features = ["hs_m", "tp_s", "surge_m", "wind_normal_ms"]
    lookback = 12
    data = create_sliding_window_dataset(
        sample_dataset, feature_cols=raw_features, target_col="shoreline_pos_obs_m", lookback=lookback, train_split=0.70
    )
    assert "x_train" in data and "x_test" in data
    assert data["x_train"].dim() == 3
    assert data["x_train"].size(1) == lookback
    assert data["x_train"].size(2) == len(raw_features)


def test_run_deep_learning_benchmark(sample_dataset: pd.DataFrame):
    df_results, suite_data = run_deep_learning_benchmark(
        sample_dataset, lookback=12, train_split=0.70, epochs=3, batch_size=16, random_seed=42
    )
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == 3
    assert "RMSE (m)" in df_results.columns
    assert "R2 Score" in df_results.columns
    assert "BSS" in df_results.columns
    assert "MLP" in suite_data["trained_models"]
    assert "LSTM" in suite_data["trained_models"]
    assert "PG-LSTM" in suite_data["trained_models"]
