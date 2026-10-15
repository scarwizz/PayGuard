"""
Interactive cost-aware financial fraud decision simulator and explainability dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

from src.decision.policy import (
    CostConfig,
    evaluate_three_action_policy,
    evaluate_two_action_policy,
)
from src.features.pipeline import FeaturePipeline

st.set_page_config(
    page_title="PayGuard | Fraud Decision Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .action-badge-approve {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .action-badge-review {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .action-badge-decline {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Loads calibrated models, feature pipeline, and test predictions with sample fallback."""
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"
    sample_dir = project_root / "data" / "sample"

    model_path = models_dir / "lgbm_calibrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Calibrated model not found at {model_path}")
    cal_model = joblib.load(model_path)
    lgbm_base = cal_model.base_estimator

    # Load pre-fitted pipeline or initialize with training baseline priors
    pipe_path = models_dir / "feature_pipeline.joblib"
    if pipe_path.exists():
        pipe = joblib.load(pipe_path)
    else:
        pipe = FeaturePipeline(
            tier=2,
            global_type_means={
                "CASH_IN": 168720.0,
                "CASH_OUT": 176273.0,
                "DEBIT": 5483.0,
                "PAYMENT": 13057.0,
                "TRANSFER": 910644.0,
            },
        )

    # Seamless fallback: load full test set if present, else load 500-row demo sample
    full_preds_path = processed_dir / "test_predictions.parquet"
    sample_preds_path = sample_dir / "test_cases_sample.parquet"

    is_sample = False
    if full_preds_path.exists():
        test_preds = pd.read_parquet(full_preds_path)
    elif sample_preds_path.exists():
        test_preds = pd.read_parquet(sample_preds_path)
        is_sample = True
    else:
        raise FileNotFoundError("Neither test_predictions.parquet nor test_cases_sample.parquet was found.")

    explainer = shap.TreeExplainer(lgbm_base)

    return {
        "cal_model": cal_model,
        "lgbm_base": lgbm_base,
        "pipe": pipe,
        "test_preds": test_preds,
        "explainer": explainer,
        "is_sample": is_sample,
    }


artifacts = load_artifacts()
cal_model = artifacts["cal_model"]
lgbm_base = artifacts["lgbm_base"]
pipe = artifacts["pipe"]
test_preds = artifacts["test_preds"]
explainer = artifacts["explainer"]
is_sample = artifacts["is_sample"]

# Sidebar controls
st.sidebar.markdown("## 🛡️ Cost & Policy Parameters")

preset = st.sidebar.selectbox(
    "Cost Scenario Preset",
    ["Baseline Operational", "High Customer Friction", "High Fraud Severity", "Custom"],
    index=0,
)

if preset == "Baseline Operational":
    c_miss_val, c_dec_val, c_rev_val = 500, 25, 5
elif preset == "High Customer Friction":
    c_miss_val, c_dec_val, c_rev_val = 500, 100, 5
elif preset == "High Fraud Severity":
    c_miss_val, c_dec_val, c_rev_val = 2000, 25, 10
else:
    c_miss_val, c_dec_val, c_rev_val = 500, 25, 5

c_miss = st.sidebar.slider("Cost of Missed Fraud (C_miss, $)", 100, 3000, c_miss_val, step=25)
c_dec = st.sidebar.slider("Cost of False Decline (C_decline, $)", 5, 200, c_dec_val, step=5)
c_rev = st.sidebar.slider("Cost of Manual Review (C_review, $)", 1, 50, c_rev_val, step=1)
k_cap = st.sidebar.slider("Manual Review Capacity (K txns)", 0, 2000, 1000 if not is_sample else 50, step=50 if not is_sample else 5)

costs = CostConfig(c_miss=float(c_miss), c_decline=float(c_dec), c_review=float(c_rev))
t_bayes = costs.bayes_optimal_threshold
asym_ratio = costs.c_miss / max(costs.c_decline, 1e-6)

st.sidebar.markdown("---")
st.sidebar.metric(
    label="Bayes-Optimal Threshold (t_Bayes)",
    value=f"{t_bayes:.4f}",
    help="t_Bayes = C_decline / (C_miss + C_decline). Optimal decision threshold under calibrated probabilities.",
)
st.sidebar.markdown(f"**Loss Asymmetry Ratio**: {asym_ratio:.1f} : 1")

st.markdown("<div class='main-header'>🛡️ PayGuard: Financial Fraud Decision Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Operational Risk Optimization, Probability Calibration & Explainability (SHAP)</div>", unsafe_allow_html=True)

if is_sample:
    st.info("ℹ️ **Demonstration Mode**: Operating on lightweight 500-transaction test sample (`data/sample/test_cases_sample.parquet`) including verified SHAP edge cases.")

with st.expander("ℹ️ Methodological Governance & Simulation Transparency Notice", expanded=False):
    st.markdown("""
    > **Methodological Transparency & Governance Principles**:
    > 1. **Synthetic Data Disclosure**: Built on the **PaySim mobile-money benchmark** (6.36M transactions across 743 simulation hours), not live credit card network or ACH settlements.
    > 2. **Decision-Policy Cost Modeling**: Total cost, per-transaction loss, and dollar savings reflect **decision-policy simulation outputs** governed by user-configured unit losses ($C_{\\text{miss}}, C_{\\text{decline}}, C_{\\text{review}}$).
    > 3. **Strict Zero-Leakage Guarantee**: Settlement balances (`oldbalanceOrg`, `newbalanceOrig`, etc.) and account IDs are permanently excluded. All features use strictly causal, non-leaking aggregations.
    """)

tab_overview, tab_scorer, tab_governance = st.tabs([
    "📊 Out-of-Time Policy Simulator",
    "🔍 Real-Time Transaction Scorer & SHAP Explainer",
    "⚖️ Governance, Leakage Audit & Model Cards",
])

# TAB 1: Policy Simulator
with tab_overview:
    st.markdown(f"### 📈 Policy Evaluation ({len(test_preds):,} Transactions)")
    st.markdown(f"Evaluating {len(test_preds):,} transactions ({test_preds['isFraud'].sum():,} actual frauds, prevalence: {test_preds['isFraud'].mean()*100:.2f}%) under selected costs:")

    y_true = test_preds["isFraud"].values
    amounts = test_preds["amount"].values
    p_cal = test_preds["p_cal"].values
    p_raw = test_preds["p_raw"].values

    res_raw_50 = evaluate_two_action_policy(y_true, p_raw, 0.5, costs, amounts, "LightGBM Raw (t=0.5)")
    res_cal_bayes = evaluate_two_action_policy(y_true, p_cal, t_bayes, costs, amounts, f"Calibrated (Bayes t={t_bayes:.4f})")
    res_three_action = evaluate_three_action_policy(y_true, p_cal, costs, k_capacity=k_cap, t_decline=0.85, amounts=amounts, policy_name=f"Calibrated + Review Queue (K={k_cap})")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Baseline Cost (t=0.5)</div>
            <div class='metric-value'>${res_raw_50.total_cost:,.0f}</div>
            <small style='color:#64748B;'>${res_raw_50.cost_per_txn:.2f} / txn</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Three-Action Policy Cost</div>
            <div class='metric-value' style='color:#16A34A;'>${res_three_action.total_cost:,.0f}</div>
            <small style='color:#16A34A;'>${res_three_action.cost_per_txn:.2f} / txn</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        savings = res_raw_50.total_cost - res_three_action.total_cost
        savings_pct = (savings / res_raw_50.total_cost) * 100 if res_raw_50.total_cost > 0 else 0.0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Cost Reduction</div>
            <div class='metric-value' style='color:#2563EB;'>{savings_pct:.1f}%</div>
            <small style='color:#2563EB;'>+${savings:,.0f} saved</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Fraud Recall</div>
            <div class='metric-value'>{res_three_action.fraud_recall*100:.1f}%</div>
            <small style='color:#64748B;'>{res_three_action.n_caught_fraud:,} / {len(y_true[y_true==1]):,} caught</small>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>False Declines</div>
            <div class='metric-value' style='color:#DC2626;'>{res_three_action.n_false_decline:,}</div>
            <small style='color:#64748B;'>{res_three_action.n_reviewed:,} in review queue</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Policy Comparison Matrix")
    policy_rows = [
        {
            "Policy Architecture": p.policy_name,
            "Total Cost ($)": f"${p.total_cost:,.2f}",
            "Cost / Txn": f"${p.cost_per_txn:.4f}",
            "Caught Frauds": f"{p.n_caught_fraud:,}",
            "Fraud Recall (%)": f"{p.fraud_recall*100:.2f}%",
            "Missed Frauds (FN)": f"{p.n_miss:,}",
            "False Declines (FP)": f"{p.n_false_decline:,}",
            "Manual Reviews (K)": f"{p.n_reviewed:,}",
            "Fraud Value Captured (%)": f"{p.fraud_value_captured_pct:.2f}%" if p.fraud_value_captured_pct is not None else "N/A",
        }
        for p in [res_raw_50, res_cal_bayes, res_three_action]
    ]
    st.dataframe(pd.DataFrame(policy_rows), use_container_width=True, hide_index=True)

    col_plot1, col_plot2 = st.columns([1.2, 0.8])
    with col_plot1:
        st.markdown("#### Expected Cost Curve vs. Decision Threshold")
        thresholds = np.linspace(0.001, 0.50, 60)
        costs_raw = [evaluate_two_action_policy(y_true, p_raw, t, costs, amounts).total_cost for t in thresholds]
        costs_cal = [evaluate_two_action_policy(y_true, p_cal, t, costs, amounts).total_cost for t in thresholds]

        fig_curve, ax_curve = plt.subplots(figsize=(8, 4.5))
        ax_curve.plot(thresholds, costs_raw, label="LightGBM Raw Probabilities", color="#EF4444", lw=2, ls="--")
        ax_curve.plot(thresholds, costs_cal, label="LightGBM Calibrated Probabilities", color="#2563EB", lw=2.5)
        ax_curve.axvline(t_bayes, color="#10B981", ls=":", lw=2, label=f"Bayes-Optimal t={t_bayes:.4f}")
        ax_curve.scatter([t_bayes], [evaluate_two_action_policy(y_true, p_cal, t_bayes, costs, amounts).total_cost], color="#10B981", s=80, zorder=5)

        ax_curve.set_xlabel("Decision Threshold (t)", fontsize=10)
        ax_curve.set_ylabel("Total Simulated Loss ($)", fontsize=10)
        ax_curve.set_title("Cost Curve across Thresholds", fontsize=11, fontweight="bold")
        ax_curve.legend(loc="upper right", fontsize=9)
        ax_curve.grid(True, alpha=0.3)
        st.pyplot(fig_curve)
        plt.close(fig_curve)

    with col_plot2:
        st.markdown("#### Three-Action Confusion Matrix")
        tp = res_three_action.n_caught_fraud
        fn = res_three_action.n_miss
        fp = res_three_action.n_false_decline
        tn = max(0, len(y_true) - tp - fn - fp - res_three_action.n_reviewed)

        cm_data = np.array([
            [tn, res_three_action.n_reviewed, fp],
            [fn, 0, tp]
        ])

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm_data,
            annot=True,
            fmt=",.0f",
            cmap="Blues",
            xticklabels=["Auto-Approve", f"Review (K={k_cap})", "Auto-Decline"],
            yticklabels=["Legitimate (0)", "Fraud (1)"],
            cbar=False,
            ax=ax_cm,
        )
        ax_cm.set_title("Decision Policy Confusion Matrix", fontsize=11, fontweight="bold")
        ax_cm.set_ylabel("Ground Truth", fontsize=10)
        ax_cm.set_xlabel("Policy Action", fontsize=10)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

# TAB 2: Transaction Scorer & SHAP
with tab_scorer:
    st.markdown("### 🔍 Real-Time Transaction Scorer & SHAP Waterfall Explainer")

    col_input, col_result = st.columns([1, 1.2])
    with col_input:
        st.markdown("#### Input Transaction Parameters")
        input_mode = st.radio("Choose Input Mode", ["Select Real Test Set Archetype", "Custom Transaction Parameters"], horizontal=True)

        if input_mode == "Select Real Test Set Archetype":
            archetype = st.selectbox(
                "Select Operational Test Case",
                [
                    "Case 1: High-Value Caught Fraud ($2.21M CASH_OUT) [True Positive]",
                    "Case 2: Legitimate False Decline ($851.8k CASH_OUT) [False Alarm]",
                    "Case 3: Stealthy Low-Value Missed Fraud ($63.4k TRANSFER) [False Negative]",
                ]
            )

            # Match archetype cleanly regardless of full or sample dataset
            if "case_study" in test_preds.columns:
                match = test_preds[test_preds["case_study"] == archetype]
                selected_row = match.iloc[0] if len(match) > 0 else test_preds.iloc[0]
            else:
                if "Case 1" in archetype:
                    target_idx = 3289 if len(test_preds) > 3289 else 0
                elif "Case 2" in archetype:
                    target_idx = 2006 if len(test_preds) > 2006 else 1
                else:
                    target_idx = 123570 if len(test_preds) > 123570 else 2
                selected_row = test_preds.iloc[target_idx]

            txn_type = selected_row["type"]
            txn_amount = float(selected_row["amount"])
            txn_step = int(selected_row["step"])
            txn_hour = txn_step % 24
            txn_day = txn_step // 24
            is_cold = bool(selected_row["account_is_cold_start"])
            prior_count = int(selected_row["account_txn_count_past"])
            prior_mean = float(selected_row["account_amount_mean_past"])
            ground_truth = int(selected_row["isFraud"])
            st.info(f"**Ground Truth Label**: {'🔴 FRAUD (1)' if ground_truth == 1 else '🟢 LEGITIMATE (0)'} | Step: {txn_step}")
        else:
            txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
            txn_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=20000000.0, value=250000.0, step=10000.0)
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                txn_hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            with col_t2:
                txn_day = st.slider("Day Index (0-30)", 0, 30, 25)
            txn_step = txn_day * 24 + txn_hour
            is_cold = st.checkbox("Is New Originating Account (Cold Start)?", value=True)
            if not is_cold:
                prior_count = st.number_input("Prior Transactions by Originator", min_value=1, max_value=50, value=3)
                prior_mean = st.number_input("Prior Expanding Mean Amount ($)", min_value=1.0, max_value=5000000.0, value=50000.0)
            else:
                prior_count = 0
                prior_mean = -1.0
            ground_truth = None

        type_means = pipe.global_type_means or {
            "CASH_IN": 168720.0,
            "CASH_OUT": 176273.0,
            "DEBIT": 5483.0,
            "PAYMENT": 13057.0,
            "TRANSFER": 910644.0,
        }
        type_mean = type_means.get(txn_type, 150000.0)
        amount_by_type = txn_amount / max(type_mean, 1.0)

        feature_dict = {
            "amount": np.float32(txn_amount),
            "step": np.int16(txn_step),
            "type": txn_type,
            "log_amount": np.float32(np.log1p(txn_amount)),
            "hour_of_day": np.int8(txn_hour),
            "day_index": np.int16(txn_day),
            "is_transfer": np.int8(1 if txn_type == "TRANSFER" else 0),
            "is_cash_out": np.int8(1 if txn_type == "CASH_OUT" else 0),
            "amount_by_type": np.float32(amount_by_type),
            "account_txn_count_past": np.int32(prior_count),
            "account_amount_mean_past": np.float32(prior_mean),
            "account_is_cold_start": np.int8(1 if is_cold else 0),
        }
        input_df = pd.DataFrame([feature_dict])
        input_df["type"] = pd.Categorical(
            input_df["type"],
            categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        )

    with col_result:
        st.markdown("#### Model Scoring & Policy Action")
        p_single_raw = float(lgbm_base.predict_proba(input_df)[:, 1][0])
        p_single_cal = float(cal_model.predict_proba(input_df)[:, 1][0])

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Raw LightGBM Probability", f"{p_single_raw*100:.2f}%")
        with col_p2:
            st.metric("Calibrated Probability (Platt)", f"{p_single_cal*100:.2f}%")

        if p_single_cal >= 0.85:
            action = "AUTO-DECLINE"
            badge_class = "action-badge-decline"
            action_desc = "High-confidence fraud signal. Transaction immediately blocked."
        elif p_single_cal >= t_bayes:
            action = "MANUAL REVIEW"
            badge_class = "action-badge-review"
            action_desc = "Suspicious activity exceeding loss-optimal threshold. Routed to fraud ops queue."
        else:
            action = "AUTO-APPROVE"
            badge_class = "action-badge-approve"
            action_desc = "Low fraud probability. Approved with zero customer friction."

        st.markdown(f"""
        <div style='margin-top:0.8rem; margin-bottom:1rem;'>
            <span class='{badge_class}'>{action}</span>
            <div style='margin-top:0.5rem; color:#475569; font-size:0.95rem;'>{action_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        exp_cost_approve = p_single_cal * c_miss
        exp_cost_decline = (1.0 - p_single_cal) * c_dec
        exp_cost_review = c_rev

        st.markdown("##### Expected Loss by Action:")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"**Approve**: ${exp_cost_approve:.2f}")
        with col_c2:
            st.markdown(f"**Decline**: ${exp_cost_decline:.2f}")
        with col_c3:
            st.markdown(f"**Review**: ${exp_cost_review:.2f}")

    st.markdown("---")
    st.markdown("#### Local SHAP Waterfall Attribution")
    sv_single = explainer(input_df)
    fig_sh, ax_sh = plt.subplots(figsize=(9, 4.5))
    shap.plots.waterfall(sv_single[0], show=False, max_display=10)
    plt.title(f"SHAP Waterfall Attribution | Calibrated P(Fraud) = {p_single_cal*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig_sh)
    plt.close(fig_sh)

# TAB 3: Governance
with tab_governance:
    st.markdown("### ⚖️ Governance, Leakage Audit & Production Readiness")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Zero Data-Leakage Protocol")
        st.markdown("""
        - **Banned Settlement Balances**: `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest` are post-authorization fields. In PaySim, `newbalanceOrig == 0` in **98.05% of frauds**, creating artificial 0.999 AUC benchmarks that fail in live pre-auth gateways.
        - **Banned Account IDs**: Raw `nameOrig` and `nameDest` have 99.8% single-use entities. Memorizing them causes severe out-of-time degradation.
        - **Causal Expanding Windows**: Behavioral velocity features use strictly prior rows ($t < t_{\\text{current}}$).
        """)

    with col_g2:
        st.markdown("#### Tree SHAP Collinearity & Causality Considerations")
        st.markdown("""
        - **Collinearity Splitting**: `amount`, `log_amount`, and `amount_by_type` are monotonic transforms. Tree SHAP splits credit across collinear groups.
        - **Attribution vs Causality**: SHAP calculates game-theoretic feature sensitivity within the model surface, not real-world interventionist causality.
        - **Reference Distribution**: SHAP base values depend on the background reference distribution prevalence.
        """)
