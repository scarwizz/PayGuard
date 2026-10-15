"""
Evaluation metrics for fraud detection under severe class imbalance and asymmetric operational costs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Union[np.ndarray, pd.Series, List[float]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Computes Expected Calibration Error (ECE) across probability bins."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have identical lengths.")

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    n_samples = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fraud_value_captured(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    amounts: Union[np.ndarray, pd.Series],
) -> float:
    """Calculates proportion of total fraudulent monetary amount caught by positive alerts."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amounts = np.asarray(amounts, dtype=float)

    total_fraud_dollars = np.sum(amounts[y_true == 1])
    if total_fraud_dollars <= 0.0:
        return 0.0

    caught_fraud_dollars = np.sum(amounts[(y_true == 1) & (y_pred == 1)])
    return float(caught_fraud_dollars / total_fraud_dollars)


def fraud_value_captured_at_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    amounts: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> float:
    """Calculates fraud dollar value captured at a specified probability threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return fraud_value_captured(y_true, y_pred, amounts)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    amounts: Optional[Union[np.ndarray, pd.Series]] = None,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Computes a unified dictionary of ranking, calibration, and monetary fraud metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    positive_count = int(np.sum(y_true == 1))
    prevalence = float(positive_count / n_samples) if n_samples > 0 else 0.0

    # Primary metric: Average Precision (PR-AUC)
    ap = float(average_precision_score(y_true, y_prob)) if positive_count > 0 else 0.0

    # Secondary discrimination metric: ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        roc_auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=10))

    p_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, p_clipped))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    val_captured = None
    if amounts is not None:
        val_captured = fraud_value_captured(y_true, y_pred, amounts)

    results: Dict[str, Any] = {
        "model": model_name,
        "model_name": model_name,
        "n_samples": n_samples,
        "n_frauds": positive_count,
        "prevalence": prevalence,
        "average_precision": ap,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "log_loss": loss,
        "ece": ece,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fraud_value_captured": val_captured,
    }
    return results


def format_evaluation_table(eval_dicts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Formats multiple model evaluation results into an engineering summary table."""
    rows = []
    for d in eval_dicts:
        rows.append({
            "Model": d.get("model", d.get("model_name", "Unknown")),
            "Prevalence (%)": f"{d.get('prevalence', 0.0) * 100:.4f}%",
            "Average Precision (AP)": f"{d.get('average_precision', 0.0):.4f}",
            "ROC-AUC": f"{d.get('roc_auc', 0.0):.4f}",
            "Brier Score": f"{d.get('brier_score', 0.0):.6f}",
            "Log Loss": f"{d.get('log_loss', 0.0):.4f}",
            "ECE (10-bin)": f"{d.get('ece', 0.0):.4f}",
            f"Precision @ {d.get('threshold', 0.5):.2f}": f"{d.get('precision', 0.0) * 100:.2f}%",
            f"Recall @ {d.get('threshold', 0.5):.2f}": f"{d.get('recall', 0.0) * 100:.2f}%",
            "Fraud Value Captured (%)": (
                f"{d['fraud_value_captured'] * 100:.2f}%"
                if d.get("fraud_value_captured") is not None
                else "N/A"
            ),
        })
    return pd.DataFrame(rows)
