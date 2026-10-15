"""
Chronological partitioning utility enforcing causal temporal integrity across transaction steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.60,
    "calibration": 0.10,
    "validation": 0.10,
    "test": 0.20,
}


@dataclass(frozen=True)
class SplitWindow:
    """Represents a discrete chronological split window."""
    name: str
    step_start: int
    step_end: int

    @property
    def total_steps(self) -> int:
        return self.step_end - self.step_start + 1

    def contains(self, step: int) -> bool:
        return self.step_start <= step <= self.step_end


def propose_step_cutoffs(
    min_step: int = 1,
    max_step: int = 744,
    ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[int, int]]:
    """Proposes contiguous integer step boundaries based on specified timeline ratios."""
    if ratios is None:
        ratios = DEFAULT_SPLIT_RATIOS

    # Cast to python int to prevent numpy int8/int16 overflow when calculating range
    total_steps = int(max_step) - int(min_step) + 1
    if total_steps <= 0:
        raise ValueError(f"Invalid step range: min_step={min_step}, max_step={max_step}")

    total_ratio = sum(ratios.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    train_steps = int(round(total_steps * ratios["train"]))
    cal_steps = int(round(total_steps * ratios["calibration"]))
    val_steps = int(round(total_steps * ratios["validation"]))

    train_start = int(min_step)
    train_end = train_start + train_steps - 1

    cal_start = train_end + 1
    cal_end = cal_start + cal_steps - 1

    val_start = cal_end + 1
    val_end = val_start + val_steps - 1

    test_start = val_end + 1
    test_end = int(max_step)

    boundaries: Dict[str, Tuple[int, int]] = {
        "train": (train_start, train_end),
        "calibration": (cal_start, cal_end),
        "validation": (val_start, val_end),
        "test": (test_start, test_end),
    }

    logger.info("Proposed chronological step cutoffs: %s", boundaries)
    return boundaries


def verify_split_integrity(
    df: pd.DataFrame,
    boundaries: Dict[str, Tuple[int, int]],
    step_col: str = "step",
    target_col: str = "isFraud",
) -> None:
    """Asserts that partitions are strictly non-overlapping, complete, and contain positive labels."""
    split_names = list(boundaries.keys())
    for i in range(len(split_names) - 1):
        curr_name = split_names[i]
        next_name = split_names[i + 1]
        curr_end = boundaries[curr_name][1]
        next_start = boundaries[next_name][0]
        if curr_end + 1 != next_start:
            raise AssertionError(f"Cutoff discontinuity between '{curr_name}' ({curr_end}) and '{next_name}' ({next_start})")

    min_step = boundaries[split_names[0]][0]
    max_step = boundaries[split_names[-1]][1]
    df_steps = df[step_col].dropna()
    if (df_steps < min_step).any() or (df_steps > max_step).any():
        raise AssertionError("Dataset contains steps falling outside defined split boundaries.")

    for name, (start, end) in boundaries.items():
        subset = df[(df[step_col] >= start) & (df[step_col] <= end)]
        if len(subset) == 0:
            raise AssertionError(f"Split '{name}' contains 0 transactions.")
        if target_col in subset.columns and subset[target_col].sum() == 0:
            raise AssertionError(f"Split '{name}' contains 0 fraud instances.")

        split_unique_steps = set(subset[step_col].unique())
        outside_steps = set(df[(df[step_col] < start) | (df[step_col] > end)][step_col].unique())
        leaked_steps = split_unique_steps.intersection(outside_steps)
        if leaked_steps:
            raise AssertionError(f"Step overlap detected! Step(s) {leaked_steps} appear both inside and outside '{name}'")


def inspect_temporal_fraud_distribution(
    df: pd.DataFrame,
    step_col: str = "step",
    target_col: str = "isFraud",
) -> pd.DataFrame:
    """Computes hourly transaction volume, fraud count, and empirical fraud prevalence."""
    step_stats = df.groupby(step_col, observed=True).agg(
        total_txns=(target_col, "count"),
        fraud_txns=(target_col, "sum"),
    ).reset_index()

    step_stats["fraud_rate_pct"] = (step_stats["fraud_txns"] / step_stats["total_txns"]) * 100.0
    step_stats["cum_fraud"] = step_stats["fraud_txns"].cumsum()
    step_stats["cum_txns"] = step_stats["total_txns"].cumsum()
    return step_stats


def generate_split_summary(
    df: pd.DataFrame,
    boundaries: Dict[str, Tuple[int, int]],
    step_col: str = "step",
    target_col: str = "isFraud",
) -> pd.DataFrame:
    """Generates an operational summary table reporting volume, fraud count, and prevalence per split."""
    verify_split_integrity(df, boundaries, step_col, target_col)

    total_df_rows = len(df)
    total_df_frauds = int(df[target_col].sum()) if target_col in df.columns else 0
    total_timeline_steps = sum(end - start + 1 for start, end in boundaries.values())

    records = []
    for name, (start, end) in boundaries.items():
        mask = (df[step_col] >= start) & (df[step_col] <= end)
        subset = df[mask]

        num_steps = end - start + 1
        steps_pct = (num_steps / total_timeline_steps) * 100.0
        num_rows = len(subset)
        rows_pct = (num_rows / total_df_rows) * 100.0 if total_df_rows > 0 else 0.0

        if target_col in subset.columns:
            fraud_count = int(subset[target_col].sum())
            non_fraud_count = num_rows - fraud_count
            fraud_share_pct = (fraud_count / total_df_frauds) * 100.0 if total_df_frauds > 0 else 0.0
            prevalence_pct = (fraud_count / num_rows) * 100.0 if num_rows > 0 else 0.0
        else:
            fraud_count = non_fraud_count = 0
            fraud_share_pct = prevalence_pct = 0.0

        records.append({
            "Split": name.capitalize(),
            "Step_Start": start,
            "Step_End": end,
            "Num_Steps": num_steps,
            "Timeline_%": round(steps_pct, 2),
            "Row_Count": num_rows,
            "Row_%": round(rows_pct, 2),
            "Non_Fraud": non_fraud_count,
            "Fraud_Count": fraud_count,
            "Fraud_Share_%": round(fraud_share_pct, 2),
            "Fraud_Prevalence_%": round(prevalence_pct, 4),
        })

    return pd.DataFrame(records)


def tag_split_column(
    df: pd.DataFrame,
    boundaries: Dict[str, Tuple[int, int]],
    step_col: str = "step",
) -> pd.DataFrame:
    """Appends a categorical 'split' partition column to the DataFrame."""
    df = df.copy()
    df["split"] = "unassigned"

    for name, (start, end) in boundaries.items():
        mask = (df[step_col] >= start) & (df[step_col] <= end)
        df.loc[mask, "split"] = name

    df["split"] = df["split"].astype("category")
    return df


def apply_chronological_splits(
    df: pd.DataFrame,
    boundaries: Dict[str, Tuple[int, int]],
    step_col: str = "step",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions the DataFrame into four disjoint splits: (train, calibration, validation, test)."""
    train_mask = (df[step_col] >= boundaries["train"][0]) & (df[step_col] <= boundaries["train"][1])
    cal_mask = (df[step_col] >= boundaries["calibration"][0]) & (df[step_col] <= boundaries["calibration"][1])
    val_mask = (df[step_col] >= boundaries["validation"][0]) & (df[step_col] <= boundaries["validation"][1])
    test_mask = (df[step_col] >= boundaries["test"][0]) & (df[step_col] <= boundaries["test"][1])

    return (
        df[train_mask].copy(),
        df[cal_mask].copy(),
        df[val_mask].copy(),
        df[test_mask].copy(),
    )
