"""
Causal, leak-free feature engineering pipeline for pre-authorization financial fraud detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Banned settlement balance columns and entity IDs to prevent target leakage
BANNED_COLUMNS: Set[str] = {
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
}

TIER1_FEATURE_COLS: List[str] = [
    "amount",
    "step",
    "type",
]

TIER2_FEATURE_COLS: List[str] = [
    "amount",
    "step",
    "type",
    "log_amount",
    "hour_of_day",
    "day_index",
    "is_transfer",
    "is_cash_out",
    "amount_by_type",
    "account_txn_count_past",
    "account_amount_mean_past",
    "account_is_cold_start",
]


def verify_no_leakage_columns(df_features: pd.DataFrame) -> None:
    """Validates that no banned settlement balance columns or entity IDs exist in features."""
    leaked_cols = BANNED_COLUMNS.intersection(set(df_features.columns))
    if leaked_cols:
        raise AssertionError(
            f"Leakage detected: Banned columns present in feature matrix: {sorted(list(leaked_cols))}"
        )


class FeaturePipeline:
    """
    Transforms transaction logs into causal feature representations.
    Tier 1: Raw safe features (amount, step, type).
    Tier 2: Engineered temporal velocity, relative volume ratios, and cold-start indicators.
    """

    def __init__(
        self,
        tier: int = 2,
        one_hot_type: bool = False,
        global_type_means: Optional[Dict[str, float]] = None,
    ):
        if tier not in (1, 2):
            raise ValueError(f"Invalid feature tier: {tier}. Must be 1 or 2.")
        self.tier = tier
        self.one_hot_type = one_hot_type
        self.global_type_means = global_type_means

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        """Computes global baseline priors from training data for cold-start fallbacks."""
        if "amount" in df.columns and "type" in df.columns:
            self.global_type_means = (
                df.groupby("type", observed=True)["amount"].mean().to_dict()
            )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        tier: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transforms transactions into a causal feature matrix X and optional target label y."""
        active_tier = tier if tier is not None else self.tier
        orig_index = df.index

        # Deterministic chronological sort with original index tie-breaker
        df_sorted = df.copy()
        df_sorted["_orig_idx"] = np.arange(len(df_sorted))
        df_sorted = df_sorted.sort_values(by=["step", "_orig_idx"], kind="stable")

        y_sorted = df_sorted["isFraud"] if "isFraud" in df_sorted.columns else None

        feats = pd.DataFrame(index=df_sorted.index)
        feats["amount"] = df_sorted["amount"].astype(np.float32)
        feats["step"] = df_sorted["step"].astype(np.int16)
        feats["type"] = df_sorted["type"].astype("category")

        if active_tier == 2:
            feats["log_amount"] = np.log1p(df_sorted["amount"].values).astype(np.float32)
            feats["hour_of_day"] = (df_sorted["step"].values % 24).astype(np.int8)
            feats["day_index"] = (df_sorted["step"].values // 24).astype(np.int16)
            feats["is_transfer"] = (df_sorted["type"].values == "TRANSFER").astype(np.int8)
            feats["is_cash_out"] = (df_sorted["type"].values == "CASH_OUT").astype(np.int8)

            # Expanding amount by type: subtract current amount so stats use strictly prior rows < i
            type_grouped = df_sorted.groupby("type", observed=True)
            type_prior_count = type_grouped.cumcount().values
            type_prior_sum = type_grouped["amount"].cumsum().values - df_sorted["amount"].values

            if self.global_type_means is not None:
                type_fallback = df_sorted["type"].map(self.global_type_means).fillna(1.0).values
            else:
                type_fallback = np.ones(len(df_sorted), dtype=np.float32)

            type_prior_mean = np.where(
                type_prior_count > 0,
                type_prior_sum / np.maximum(type_prior_count, 1),
                type_fallback,
            )
            feats["amount_by_type"] = np.where(
                type_prior_count > 0,
                df_sorted["amount"].values / np.maximum(type_prior_mean, 1e-5),
                1.0,
            ).astype(np.float32)

            # Account expanding velocity: subtract current amount to ensure no contemporaneous leakage
            if "nameOrig" in df_sorted.columns:
                orig_grouped = df_sorted.groupby("nameOrig")
                account_txn_count_past = orig_grouped.cumcount().values.astype(np.int32)
                orig_cumsum = orig_grouped["amount"].cumsum().values - df_sorted["amount"].values

                feats["account_txn_count_past"] = account_txn_count_past
                feats["account_amount_mean_past"] = np.where(
                    account_txn_count_past > 0,
                    orig_cumsum / np.maximum(account_txn_count_past, 1),
                    -1.0,
                ).astype(np.float32)
                feats["account_is_cold_start"] = (account_txn_count_past == 0).astype(np.int8)
            else:
                feats["account_txn_count_past"] = np.zeros(len(df_sorted), dtype=np.int32)
                feats["account_amount_mean_past"] = np.full(len(df_sorted), -1.0, dtype=np.float32)
                feats["account_is_cold_start"] = np.ones(len(df_sorted), dtype=np.int8)

        if self.one_hot_type:
            type_dummies = pd.get_dummies(feats["type"], prefix="type", drop_first=False).astype(np.int8)
            feats = pd.concat([feats.drop(columns=["type"]), type_dummies], axis=1)

        # Restore original index ordering and verify zero leakage
        feats = feats.loc[orig_index]
        verify_no_leakage_columns(feats)

        if y_sorted is not None:
            y = y_sorted.loc[orig_index]
            return feats, y
        return feats, None


def extract_features(
    df: pd.DataFrame,
    tier: int = 2,
    one_hot_type: bool = False,
    pipeline: Optional[FeaturePipeline] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Convenience helper to fit and transform features in a single call."""
    if pipeline is None:
        pipeline = FeaturePipeline(tier=tier, one_hot_type=one_hot_type)
        pipeline.fit(df)
    X, y = pipeline.transform(df, tier=tier)
    return X, y
