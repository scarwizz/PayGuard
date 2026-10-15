"""
Cost-aware decision optimization and capacity-constrained review queue policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostConfig:
    """Cost configuration parameters (in monetary units or dollars)."""
    c_miss: float = 500.0        # Cost of a False Negative (missed fraud)
    c_decline: float = 25.0      # Cost of a False Positive (false decline friction)
    c_review: float = 5.0        # Operational cost per manual review

    @property
    def bayes_optimal_threshold(self) -> float:
        # Derived from expected loss equality: P * c_miss = (1 - P) * c_decline => P = c_decline / (c_miss + c_decline)
        return self.c_decline / (self.c_miss + self.c_decline)


@dataclass
class PolicyResult:
    """Stores operational outcomes of a decision policy."""
    policy_name: str
    total_cost: float
    cost_per_txn: float
    n_approved: int
    n_reviewed: int
    n_declined: int
    n_miss: int                  # False Negatives
    n_false_decline: int         # False Positives among auto-declined
    n_caught_fraud: int          # True Positives (auto-declined + caught in review)
    fraud_recall: float
    precision_declines: float
    fraud_value_captured_pct: Optional[float] = None
    threshold: Optional[float] = None
    t_decline: Optional[float] = None
    k_capacity: Optional[int] = None


def evaluate_two_action_policy(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float,
    costs: CostConfig,
    amounts: Optional[Union[np.ndarray, pd.Series]] = None,
    policy_name: str = "Two-Action Policy",
) -> PolicyResult:
    """Evaluates binary decision policy: Decline if p >= threshold, else Approve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    is_decline = y_prob >= threshold
    is_approve = ~is_decline

    n_declined = int(np.sum(is_decline))
    n_approved = int(np.sum(is_approve))
    n_reviewed = 0

    n_caught = int(np.sum((is_decline) & (y_true == 1)))
    n_false_decline = int(np.sum((is_decline) & (y_true == 0)))
    n_miss = int(np.sum((is_approve) & (y_true == 1)))

    total_frauds = int(np.sum(y_true == 1))
    recall = float(n_caught / total_frauds) if total_frauds > 0 else 0.0
    precision = float(n_caught / n_declined) if n_declined > 0 else 0.0

    total_cost = float(costs.c_miss * n_miss + costs.c_decline * n_false_decline)
    cost_per_txn = float(total_cost / len(y_true)) if len(y_true) > 0 else 0.0

    val_captured = None
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        tot_dollars = np.sum(amounts[y_true == 1])
        caught_dollars = np.sum(amounts[(is_decline) & (y_true == 1)])
        val_captured = float((caught_dollars / tot_dollars) * 100.0) if tot_dollars > 0 else 0.0

    return PolicyResult(
        policy_name=policy_name,
        total_cost=total_cost,
        cost_per_txn=cost_per_txn,
        n_approved=n_approved,
        n_reviewed=n_reviewed,
        n_declined=n_declined,
        n_miss=n_miss,
        n_false_decline=n_false_decline,
        n_caught_fraud=n_caught,
        fraud_recall=recall,
        precision_declines=precision,
        fraud_value_captured_pct=val_captured,
        threshold=threshold,
    )


def evaluate_three_action_policy(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    costs: CostConfig,
    k_capacity: int = 500,
    t_decline: float = 0.85,
    amounts: Optional[Union[np.ndarray, pd.Series]] = None,
    policy_name: Optional[str] = None,
) -> PolicyResult:
    """Evaluates capacity-constrained policy: Auto-Decline (>= t_decline), Review (Top-K), Auto-Approve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    if policy_name is None:
        policy_name = f"Three-Action Policy (K={k_capacity}, t_dec={t_decline:.2f})"

    # High-confidence fraud is immediately declined without consuming analyst review capacity
    auto_decline_mask = y_prob >= t_decline

    candidate_review_indices = np.where(~auto_decline_mask)[0]
    candidate_probs = y_prob[candidate_review_indices]

    if len(candidate_review_indices) > 0 and k_capacity > 0:
        actual_k = min(k_capacity, len(candidate_review_indices))
        # Top-K highest probabilities among non-auto-declined candidates
        top_k_rel_indices = np.argsort(candidate_probs)[::-1][:actual_k]
        review_indices = candidate_review_indices[top_k_rel_indices]
        review_mask = np.zeros(n, dtype=bool)
        review_mask[review_indices] = True
    else:
        review_mask = np.zeros(n, dtype=bool)

    approve_mask = ~(auto_decline_mask | review_mask)

    n_declined = int(np.sum(auto_decline_mask))
    n_reviewed = int(np.sum(review_mask))
    n_approved = int(np.sum(approve_mask))

    auto_declined_frauds = int(np.sum((auto_decline_mask) & (y_true == 1)))
    reviewed_frauds = int(np.sum((review_mask) & (y_true == 1)))
    n_caught = auto_declined_frauds + reviewed_frauds

    n_false_decline = int(np.sum((auto_decline_mask) & (y_true == 0)))
    n_miss = int(np.sum((approve_mask) & (y_true == 1)))

    total_frauds = int(np.sum(y_true == 1))
    recall = float(n_caught / total_frauds) if total_frauds > 0 else 0.0
    precision_declines = float(auto_declined_frauds / n_declined) if n_declined > 0 else 0.0

    total_cost = float(
        costs.c_miss * n_miss
        + costs.c_decline * n_false_decline
        + costs.c_review * n_reviewed
    )
    cost_per_txn = float(total_cost / n) if n > 0 else 0.0

    val_captured = None
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        tot_dollars = np.sum(amounts[y_true == 1])
        caught_dollars = np.sum(amounts[(auto_decline_mask | review_mask) & (y_true == 1)])
        val_captured = float((caught_dollars / tot_dollars) * 100.0) if tot_dollars > 0 else 0.0

    return PolicyResult(
        policy_name=policy_name,
        total_cost=total_cost,
        cost_per_txn=cost_per_txn,
        n_approved=n_approved,
        n_reviewed=n_reviewed,
        n_declined=n_declined,
        n_miss=n_miss,
        n_false_decline=n_false_decline,
        n_caught_fraud=n_caught,
        fraud_recall=recall,
        precision_declines=precision_declines,
        fraud_value_captured_pct=val_captured,
        t_decline=t_decline,
        k_capacity=k_capacity,
    )


def sweep_optimal_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    costs: CostConfig,
    n_points: int = 100,
    min_t: float = 0.001,
    max_t: float = 0.999,
    amounts: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """Sweeps candidate thresholds to identify the empirical cost-minimizing threshold."""
    thresholds = np.linspace(min_t, max_t, n_points)
    results = []

    for t in thresholds:
        res = evaluate_two_action_policy(y_true, y_prob, threshold=t, costs=costs, amounts=amounts)
        results.append({
            "threshold": t,
            "cost": res.total_cost,
            "total_cost": res.total_cost,
            "cost_per_txn": res.cost_per_txn,
            "caught_fraud": res.n_caught_fraud,
            "fraud_recall": res.fraud_recall,
            "missed_fraud": res.n_miss,
            "false_declines": res.n_false_decline,
            "precision": res.precision_declines,
            "fraud_value_captured_pct": res.fraud_value_captured_pct,
        })

    curve_df = pd.DataFrame(results)
    best_row = curve_df.loc[curve_df["total_cost"].idxmin()]
    best_threshold = float(best_row["threshold"])
    min_cost = float(best_row["total_cost"])

    return best_threshold, min_cost, curve_df


def run_cost_sensitivity_suite(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    scenarios: Optional[Dict[str, CostConfig]] = None,
    k_capacities: Optional[List[int]] = None,
    amounts: Optional[Union[np.ndarray, pd.Series]] = None,
) -> pd.DataFrame:
    """Evaluates policies across loss-asymmetry regimes and manual review capacities."""
    if scenarios is None:
        scenarios = {
            "Baseline Operational": CostConfig(c_miss=500.0, c_decline=25.0, c_review=5.0),
            "High Customer Friction": CostConfig(c_miss=500.0, c_decline=100.0, c_review=5.0),
            "High Fraud Severity": CostConfig(c_miss=2000.0, c_decline=25.0, c_review=10.0),
        }

    k_list = k_capacities if k_capacities is not None else [0, 50, 200, 500, 1000]
    summary_rows = []

    for name, cost_cfg in scenarios.items():
        for k in k_list:
            if k == 0:
                best_t, _, _ = sweep_optimal_threshold(y_true, y_prob, costs=cost_cfg, amounts=amounts)
                res = evaluate_two_action_policy(y_true, y_prob, threshold=best_t, costs=cost_cfg, amounts=amounts, policy_name=f"{name} (Two-Action t*)")
            else:
                res = evaluate_three_action_policy(y_true, y_prob, costs=cost_cfg, k_capacity=k, amounts=amounts, policy_name=f"{name} (Review K={k})")

            summary_rows.append({
                "Scenario": name,
                "Review Capacity (K)": k,
                "C_miss ($)": cost_cfg.c_miss,
                "C_decline ($)": cost_cfg.c_decline,
                "C_review ($)": cost_cfg.c_review,
                "Total Cost ($)": f"${res.total_cost:,.2f}",
                "Cost / Txn ($)": f"${res.cost_per_txn:.4f}",
                "Caught Fraud": res.n_caught_fraud,
                "Fraud Recall (%)": f"{res.fraud_recall * 100:.2f}%",
                "False Declines": res.n_false_decline,
                "Manual Reviews": res.n_reviewed,
                "Fraud Value Captured (%)": (
                    f"{res.fraud_value_captured_pct:.2f}%"
                    if res.fraud_value_captured_pct is not None
                    else "N/A"
                ),
            })

    return pd.DataFrame(summary_rows)
