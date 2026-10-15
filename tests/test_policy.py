"""
Unit tests for cost-aware decision optimization and policy architectures.
"""

import numpy as np
import pandas as pd
import pytest

from src.decision.policy import (
    CostConfig,
    PolicyResult,
    evaluate_three_action_policy,
    evaluate_two_action_policy,
    run_cost_sensitivity_suite,
    sweep_optimal_threshold,
)


@pytest.fixture
def toy_decision_data():
    """Generates synthetic predictions and amounts for 100 transactions."""
    np.random.seed(42)
    n = 100
    y_true = np.array([0] * 85 + [1] * 15, dtype=int)
    # Calibrated probabilities: fraud around 0.6-0.9, legit around 0.01-0.2
    y_prob = np.concatenate([
        np.random.beta(1, 20, size=85),
        np.random.beta(5, 2, size=15),
    ])
    amounts = np.random.uniform(50, 5000, size=n)
    return y_true, y_prob, amounts


def test_cost_config_bayes_optimal():
    costs = CostConfig(c_miss=500.0, c_decline=25.0, c_review=5.0)
    expected_bayes = 25.0 / (500.0 + 25.0)  # ~0.0476
    assert np.isclose(costs.bayes_optimal_threshold, expected_bayes)


def test_evaluate_two_action_policy(toy_decision_data):
    y_true, y_prob, amounts = toy_decision_data
    costs = CostConfig(c_miss=500.0, c_decline=25.0)

    res = evaluate_two_action_policy(y_true, y_prob, threshold=0.5, costs=costs, amounts=amounts)

    assert isinstance(res, PolicyResult)
    # Check exact cost formula: c_miss * FN + c_decline * FP
    manual_cost = costs.c_miss * res.n_miss + costs.c_decline * res.n_false_decline
    assert np.isclose(res.total_cost, manual_cost)
    assert res.n_approved + res.n_declined == len(y_true)
    assert res.fraud_recall == res.n_caught_fraud / 15.0
    assert 0.0 <= res.fraud_value_captured_pct <= 100.0


def test_sweep_optimal_threshold(toy_decision_data):
    y_true, y_prob, _ = toy_decision_data
    costs = CostConfig(c_miss=500.0, c_decline=25.0)

    best_t, min_cost, curve_df = sweep_optimal_threshold(y_true, y_prob, costs, n_points=50)

    # Cost at optimal threshold must be <= cost at default 0.5
    res_default = evaluate_two_action_policy(y_true, y_prob, threshold=0.5, costs=costs)
    res_optimal = evaluate_two_action_policy(y_true, y_prob, threshold=best_t, costs=costs)

    assert res_optimal.total_cost <= res_default.total_cost
    assert len(curve_df) == 50
    assert "cost" in curve_df.columns


def test_evaluate_three_action_policy(toy_decision_data):
    y_true, y_prob, amounts = toy_decision_data
    costs = CostConfig(c_miss=500.0, c_decline=25.0, c_review=5.0)
    k_cap = 10

    res = evaluate_three_action_policy(
        y_true, y_prob, costs=costs, k_capacity=k_cap, t_decline=0.85, amounts=amounts
    )

    # Check review capacity is never exceeded
    assert res.n_reviewed <= k_cap
    assert res.n_approved + res.n_reviewed + res.n_declined == len(y_true)

    # Check cost formula
    expected_cost = (
        costs.c_miss * res.n_miss
        + costs.c_decline * res.n_false_decline
        + costs.c_review * res.n_reviewed
    )
    assert np.isclose(res.total_cost, expected_cost)


def test_run_cost_sensitivity_suite(toy_decision_data):
    y_true, y_prob, amounts = toy_decision_data
    table = run_cost_sensitivity_suite(
        y_true, y_prob, amounts=amounts, k_capacities=[5, 10]
    )
    assert isinstance(table, pd.DataFrame)
    assert len(table) > 0
    assert "Total Cost ($)" in table.columns
    assert "Caught Fraud" in table.columns
