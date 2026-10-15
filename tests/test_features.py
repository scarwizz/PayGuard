"""
Unit tests for causal feature engineering and zero-leakage verification.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    BANNED_COLUMNS,
    TIER1_FEATURE_COLS,
    TIER2_FEATURE_COLS,
    FeaturePipeline,
    extract_features,
    verify_no_leakage_columns,
)


@pytest.fixture
def toy_20row_df():
    """Generates a synthetic 20-row toy dataframe with timestamps, types, and accounts."""
    np.random.seed(42)
    n = 20
    steps = np.repeat(np.arange(1, 5), 5)
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"] * 4
    amounts = [10.0, 50.0, 100.0, 200.0, 500.0] * 4

    df = pd.DataFrame({
        "step": steps.astype(np.int16),
        "type": pd.Categorical(types),
        "amount": np.array(amounts, dtype=np.float32),
        "nameOrig": [f"C{i % 6}" for i in range(n)],
        "nameDest": [f"M{i % 4}" for i in range(n)],
        "oldbalanceOrg": np.random.uniform(100, 5000, size=n).astype(np.float32),
        "newbalanceOrig": np.random.uniform(0, 5000, size=n).astype(np.float32),
        "oldbalanceDest": np.random.uniform(0, 5000, size=n).astype(np.float32),
        "newbalanceDest": np.random.uniform(0, 5000, size=n).astype(np.float32),
        "isFraud": np.array([0] * 18 + [1, 0], dtype=np.int8),
        "isFlaggedFraud": np.zeros(n, dtype=np.int8),
    })
    return df


def test_banned_columns_not_in_output(toy_20row_df):
    """Asserts that banned settlement balances and raw IDs are absent from output features."""
    pipeline = FeaturePipeline(tier=2)
    X, _ = pipeline.transform(toy_20row_df)

    for banned_col in BANNED_COLUMNS:
        assert banned_col not in X.columns, f"Banned column '{banned_col}' leaked into feature matrix."
    verify_no_leakage_columns(X)

    X_t1, _ = pipeline.transform(toy_20row_df, tier=1)
    for banned_col in BANNED_COLUMNS:
        assert banned_col not in X_t1.columns, f"Banned column '{banned_col}' leaked into Tier 1."
    verify_no_leakage_columns(X_t1)


def test_future_perturbation_invariance(toy_20row_df):
    """Verifies that mutating future rows does not alter historical feature values."""
    pipeline = FeaturePipeline(tier=2)
    X_original, _ = pipeline.transform(toy_20row_df)

    df_mutated = toy_20row_df.copy()
    df_mutated.loc[15:, "amount"] = df_mutated.loc[15:, "amount"] * 1_000_000.0
    df_mutated.loc[15:, "isFraud"] = 1

    X_mutated, _ = pipeline.transform(df_mutated)

    assert X_original.shape == X_mutated.shape
    assert list(X_original.columns) == list(X_mutated.columns)

    numeric_cols = X_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        orig_vals = X_original.loc[:14, col].to_numpy()
        mutated_vals = X_mutated.loc[:14, col].to_numpy()
        np.testing.assert_allclose(
            orig_vals,
            mutated_vals,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Future perturbation leakage detected in feature '{col}' across rows 0..14.",
        )


def test_feature_tier_ablation_columns(toy_20row_df):
    """Verifies feature schema for Tier 1 and Tier 2 outputs."""
    X_tier1, _ = extract_features(toy_20row_df, tier=1)
    X_tier2, _ = extract_features(toy_20row_df, tier=2)

    assert set(X_tier1.columns) == set(TIER1_FEATURE_COLS)
    assert set(TIER2_FEATURE_COLS).issubset(set(X_tier2.columns))
    assert len(X_tier2.columns) > len(X_tier1.columns)


def test_cold_start_account_handling(toy_20row_df):
    """Verifies that first-seen accounts receive cold-start indicators and negative priors."""
    X, _ = extract_features(toy_20row_df, tier=2)
    assert X.iloc[0]["account_is_cold_start"] == 1
    assert X.iloc[0]["account_txn_count_past"] == 0
    assert X.iloc[0]["account_amount_mean_past"] == -1.0


def test_one_hot_encoding_option(toy_20row_df):
    """Verifies one-hot encoding of categorical transaction types."""
    X, _ = extract_features(toy_20row_df, tier=2, one_hot_type=True)
    assert "type" not in X.columns
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        assert f"type_{t}" in X.columns
