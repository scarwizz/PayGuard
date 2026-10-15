# Technical Report: Cost-Sensitive Fraud Decisioning under Temporal Shift

**Authors:** PayGuard Applied Machine Learning Team  
**Evaluation Scope:** Out-of-Time Test Set (`steps 595 to 743`, 123,580 transactions, 1,654 frauds)  
**Study Type:** Offline methodology study on synthetic mobile-money data  

---

## 1. Problem Formulation and Operational Context

In financial fraud detection, two asymmetries invalidate textbook machine learning workflows:
1. **Extreme Class Imbalance**: In the PaySim dataset, positive instances make up 0.1286% of the training distribution (1 in ~777 transactions). In the out-of-time test period, positive prevalence shifts to 1.3384% (1 in ~75 transactions). Accuracy is meaningless: a naive system that approves 100% of transactions scores 98.66% accuracy while allowing 100% of fraud dollars to execute.
2. **Asymmetric Error Losses**: A false negative (missed fraud) incurs direct financial loss, chargeback processing fees, and network penalties. A false positive (false decline) causes customer friction and support costs. Under our baseline assumptions ($C_{\text{miss}} = \$500$, $C_{\text{decline}} = \$25$), a missed fraud is 20 times more costly than a false decline.

Standard industry practice often pairs a gradient-boosted tree with an arbitrary probability cutoff ($t = 0.5$). Under severe class imbalance and asymmetric losses, this fails: the model scores very few transactions above 0.5, allowing most fraud to slip through.

This report evaluates an end-to-end operational decision pipeline:
- An offline temporal partition protocol that prevents lookahead bias.
- A causal, point-in-time feature pipeline that permanently excludes post-authorization balance fields.
- 1D probability calibration via Platt scaling to map tree leaf outputs to true posterior probabilities.
- A capacity-constrained three-action policy (Auto-Approve, Manual Review Queue $K$, Auto-Decline) that minimizes total operational loss under analyst bandwidth limits.

---

## 2. Dataset Profile, Leakage Audit and Temporal Partitioning

### 2.1 Dataset Profile
The PaySim synthetic mobile-money dataset contains 6,362,620 transactions across 743 hourly steps (30.9 simulated days). Transactions are divided into five categories, but fraudulent activity occurs exclusively in two:

| Transaction Type | Total Transactions | Volume Share (%) | Fraud Count | Channel Fraud Share (%) | Channel Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CASH_OUT` | 2,237,500 | 35.17% | 4,116 | 50.11% | 0.1840% |
| `PAYMENT` | 2,151,495 | 33.81% | 0 | 0.00% | 0.0000% |
| `CASH_IN` | 1,399,284 | 21.99% | 0 | 0.00% | 0.0000% |
| `TRANSFER` | 532,909 | 8.38% | 4,097 | 49.89% | 0.7688% |
| `DEBIT` | 41,432 | 0.65% | 0 | 0.00% | 0.0000% |
| **Total** | **6,362,620** | **100.00%** | **8,213** | **100.00%** | **0.1291%** |

### 2.2 Forensic Data Leakage Audit
In PaySim, origin accounts are emptied upon fraud completion:
- `newbalanceOrig == 0.0` in 98.05% of all frauds (8,053 / 8,213). For legitimate transactions, this occurs in 43.10% of cases.
- Transaction `amount == oldbalanceOrg` in 97.82% of all frauds (8,034 / 8,213).
- A single decision tree split on `newbalanceOrig` and `oldbalanceOrg` achieves 0.9991 Average Precision and 0.9997 ROC-AUC.

In an online pre-authorization transaction gateway, post-settlement ledger balances do not yet exist. Ingesting balance fields represents target leakage. All balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) and `isFlaggedFraud` are permanently removed from feature pipelines.

### 2.3 Chronological Splitting Protocol
Transactions are sorted strictly by simulation hour (`step`). A step is never split across periods:

| Split | Step Range | Hours | Row Count | Row Share (%) | Fraud Count | Prevalence (%) | Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 1 to 446 | 446 | 3,818,485 | 60.01% | 4,909 | 0.1286% | Train base estimators |
| **Calibration** | 447 to 520 | 74 | 760,776 | 11.96% | 827 | 0.1087% | Fit 1D probability calibrator |
| **Validation** | 521 to 594 | 74 | 1,659,779 | 26.09% | 823 | 0.0496% | Early stopping and threshold search |
| **Test (Locked)** | 595 to 743 | 149 | 123,580 | 1.94% | 1,654 | 1.3384% | Final out-of-time evaluation |

In the PaySim simulation, legitimate transaction volume drops sharply after step 400, while fraudulent transactions continue at a steady pace. This causes positive class prevalence to rise from 0.0496% in validation to 1.3384% in test (a 27x relative increase). This temporal drift provides a stress test for threshold selection.

---

## 3. Point-in-Time Feature Pipeline

All feature engineering runs in `src/features/pipeline.py` using vectorized NumPy and Pandas routines that execute on 6.36 million rows in under 16 seconds on standard CPU.

Two feature tiers were constructed:
- **Tier 1 (Raw Safe Features)**: `amount`, `step`, and categorical `type`.
- **Tier 2 (Engineered Behavioral Features)**:
  - `log_amount`: $\ln(1 + \text{amount})$ to handle right-skewed transaction dollar distributions.
  - `hour_of_day`: $\text{step} \pmod{24}$ capturing diurnal variations in transaction flow.
  - `day_index`: $\lfloor \text{step} / 24 \rfloor$ tracking multi-day trends.
  - `is_transfer`, `is_cash_out`: One-hot indicators for channels where fraud occurs.
  - `amount_by_type`: Ratio of current transaction `amount` to the strictly prior expanding mean of `amount` for that transaction type. Calculated via shifted cumulative sums:
    $$\mu_{\text{type}}(t < t_{\text{curr}}) = \frac{\sum_{\tau < t} \text{amount}_{\tau}}{\sum_{\tau < t} 1}, \quad \text{amount\_by\_type} = \frac{\text{amount}}{\mu_{\text{type}}(t < t_{\text{curr}})}$$
  - `account_txn_count_past`: Expanding transaction count for `nameOrig` strictly prior to the current transaction.
  - `account_amount_mean_past`: Expanding average amount for `nameOrig` strictly prior to the current event. Unseen accounts default to `-1.0`.
  - `account_is_cold_start`: Binary flag indicating whether the origin account has zero historical records in the data.

Raw string identifiers (`nameOrig`, `nameDest`) are dropped after computing expanding aggregations. Invariance against future information was verified in `tests/test_features.py` by asserting that mutating future rows ($t+1$) does not change features calculated for previous rows ($t$).

---

## 4. Model Training, Calibration and Ablation

### 4.1 Model Architectures and Feature Ablation
Models were evaluated on the held-out test split (`steps 595 to 743`, 123,580 transactions, 1,654 frauds):

| Model | Features | Calibration | AP | ROC-AUC | Brier Score | Log Loss | ECE (10 Bins) | Precision @ 0.5 | Recall @ 0.5 | Value Captured (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Tier 2 | Unweighted | 0.2371 | 0.9093 | 0.013204 | 0.0631 | 0.0128 | 0.00% | 0.00% | 0.00% |
| LightGBM | Tier 1 (Raw) | None | 0.0134 | 0.5000 | 0.013384 | 0.0694 | 0.0134 | 0.00% | 0.00% | 0.00% |
| LightGBM | Tier 2 (Engineered) | None | 0.4068 | 0.7601 | 0.009763 | 0.0693 | 0.0089 | 84.38% | 32.35% | 57.75% |
| **LightGBM** | **Tier 2 (Engineered)** | **Platt (Sigmoid)** | **0.4068** | **0.7601** | **0.009820** | **0.0535** | **0.0027** | **90.54%** | **27.21%** | **56.38%** |

### 4.2 Key Findings on Models and Calibration
1. **Failure of Raw Features**: Tier 1 LightGBM yields an AP of 0.0134, exactly matching the positive class prevalence in the test period. Tree models cannot separate fraudulent from legitimate high-dollar transfers without point-in-time historical context.
2. **Impact of Point-in-Time Features**: Tier 2 engineered features lift AP from 0.0134 to 0.4068 (a 30.4x relative gain), proving that expanding velocity and volume-to-type ratios supply the primary predictive signal.
3. **Probability Calibration Effect**: Uncalibrated LightGBM produces skewed probability scores. Platt scaling (Sigmoid) fitted on the Calibration split drops Expected Calibration Error from 0.0089 to 0.0027 (a 69.7% reduction) and log loss from 0.0693 to 0.0535. Precision at 0.5 improves from 84.38% to 90.54%.

---

## 5. Cost-Sensitive Decision Policies and Capacity Queuing

### 5.1 Financial Loss Model
Policies were evaluated under the loss formulation:
$$C_{\text{total}} = C_{\text{miss}} \cdot N_{\text{miss}} + C_{\text{decline}} \cdot N_{\text{false decline}} + C_{\text{review}} \cdot N_{\text{review}}$$

Baseline parameter settings:
- $C_{\text{miss}} = \$500$ (chargeback loss, stolen principal write-off)
- $C_{\text{decline}} = \$25$ (customer support handling, user friction)
- $C_{\text{review}} = \$5$ (manual analyst verification cost)

Under calibrated posterior probabilities, the theoretical Bayes-optimal cutoff for an unconstrained two-action policy is:
$$t_{\text{Bayes}} = \frac{C_{\text{decline}}}{C_{\text{miss}} + C_{\text{decline}}} = \frac{25}{500 + 25} \approx 0.047619$$

### 5.2 Out-of-Time Decision Policy Results

| Policy Architecture | Total Cost ($) | Cost / Txn ($) | Caught Fraud | Recall (%) | Missed Fraud (FN) | False Declines (FP) | Reviews (K) | Fraud Value Captured (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LogReg (Fixed $t=0.5$) | $827,000 | $6.6920 | 0 | 0.00% | 1,654 | 0 | 0 | 0.00% |
| LightGBM Raw (Fixed $t=0.5$) | $561,975 | $4.5475 | 535 | 32.35% | 1,119 | 99 | 0 | 57.75% |
| LightGBM Calibrated (Fixed $t=0.5$) | $603,175 | $4.8809 | 450 | 27.21% | 1,204 | 47 | 0 | 56.38% |
| LightGBM Calibrated (Val $t^*=0.0090$) | $519,675 | $4.2052 | 635 | 38.39% | 1,019 | 407 | 0 | 60.99% |
| LightGBM Calibrated (Bayes $t=0.0476$) | $561,975 | $4.5475 | 535 | 32.35% | 1,119 | 99 | 0 | 57.75% |
| LightGBM Calibrated (Review $K=500$) | $544,775 | $4.4083 | 570 | 34.46% | 1,084 | 11 | 500 | 59.67% |
| **LightGBM Calibrated (Review $K=1000$)** | **$513,550** | **$4.1556** | **635** | **38.39%** | **1,019** | **11** | **755** | **60.99%** |

### 5.3 Operational Analysis
- **Default $t=0.5$ Threshold Failure**: Because fraud prevalence is ~1.34%, most probabilities fall below 0.5. Logistic Regression outputs zero alerts, missing all 1,654 frauds and incurring $827,000 in loss.
- **Threshold Shift ($t^* = 0.0090$)**: Sweeping the threshold on validation data cuts total loss to $519,675 by catching 635 frauds (60.99% of fraud dollars). However, it generates 407 false declines on legitimate transactions.
- **Capacity-Constrained Review Queue ($K = 1000$)**: Routing transactions with $t^* \le P < 0.85$ to human review maintains the exact same fraud recall (635 caught frauds, 60.99% dollar volume) while reducing false declines from 407 to 11 (a 97.3% decline reduction). Net operating loss reaches the lowest observed level: $513,550 ($4.16/txn).

---

## 6. Explainability, Error Breakdown and Failure Modes

### 6.1 Global Feature Attribution (Tree SHAP)
Using `shap.TreeExplainer` on 2,500 test transactions:
- `amount_by_type`: Highest mean absolute SHAP value. Transactions exceeding 5x the expanding mean for that type receive positive log-odds adjustments (+10 to +18 SHAP units).
- `is_cash_out` and `is_transfer`: Provide initial channel base adjustments; non-fraud channels receive negative offsets.
- `amount` and `log_amount`: Supply non-linear dollar scaling.
- `day_index` and `hour_of_day`: Adjust for nocturnal volume drops and simulation drift.

### 6.2 Error Breakdown by Segment

#### Channel Breakdown
- `CASH_OUT`: 394 caught / 827 actual frauds (**47.64% recall**), 72 false declines (**84.55% precision**).
- `TRANSFER`: 241 caught / 827 actual frauds (**29.14% recall**), 335 false declines (**41.84% precision**). Large legitimate peer-to-peer transfers create false alarms in the transfer category.

#### Dollar Amount Breakdown
- `CASH_OUT > $1,000,000`: **98.8% to 100.0% recall** with zero false declines.
- Transactions `< $100,000`: **17.5% to 25.8% recall**. Across both fraud vectors, 276 low-dollar frauds slipped under the detection threshold.

### 6.3 Forensic Case Studies

#### Case 1: True Positive (Caught High-Value Cash-Out)
- Profile: Step 595 (Hour 19, Day 24) | Type: `CASH_OUT` | Amount: $2,209,035.25 | Ground Truth: Fraud (1)
- Scores: Raw $P = 100.00\%$, Calibrated $P = 98.50\%$ | Action: **AUTO-DECLINE**
- Attribution: Base log-odds = -7.0873. `amount_by_type = 16.41` contributes +16.68 SHAP units. `is_cash_out = 1` contributes +15.26 SHAP units. Total log-odds exceeds +20.

#### Case 2: False Positive (False Decline Rescued by Review Queue)
- Profile: Step 595 (Hour 19, Day 24) | Type: `CASH_OUT` | Amount: $851,839.25 | Ground Truth: Legitimate (0)
- Scores: Raw $P = 100.00\%$, Calibrated $P = 83.03\%$ | Action: **MANUAL REVIEW**
- Attribution: Legitimate commercial cash-out. `amount_by_type = 6.42` contributes +12.81 SHAP units. Under a two-action policy, this transaction was declined. Under the three-action policy, it routed to review, allowing analyst clearance.

#### Case 3: False Negative (Stealthy Missed Transfer)
- Profile: Step 742 (Hour 22, Day 30) | Type: `TRANSFER` | Amount: $63,416.99 | Ground Truth: Fraud (1)
- Scores: Raw $P = 0.08\%$, Calibrated $P = 0.71\%$ | Action: **AUTO-APPROVE**
- Attribution: Low-value transfer. `amount_by_type = 0.0867` (only 8.7% of mean transfer volume) yields a negative SHAP offset (-0.0045). Log-odds remained near the prior (-7.0873), causing an auto-approval failure.

---

## 7. Limitations, Deployment Constraints and Regulatory Compliance

### 7.1 Production Vulnerabilities
1. **Low-Dollar Fraud (Smurfing)**: When fraudsters execute transfers under $100,000 from fresh accounts, volume ratios stay low and expanding history defaults to -1.0. Catching structured micro-transactions requires secondary graph-network features (shared destination clustering, device ID linking) not present in PaySim.
2. **Label Latency**: PaySim assigns fraud ground truth immediately. In production payment processing, chargebacks arrive 30 to 90 days post-authorization. Threshold sweeps and calibration parameters must be updated using label-maturity cohorts.
3. **Feature Serving Latency**: Vectorized in-memory Pandas transforms work for offline batch evaluation. A production pre-authorization gateway operating under a 50ms SLA requires stateful streaming feature storage (e.g., Redis or Feast) to maintain atomic expanding counts.

### 7.2 Methodological Constraints of Tree SHAP
- **Collinearity Credit Splitting**: `amount`, `log_amount`, and `amount_by_type` are mathematically correlated. Tree SHAP splits attribution across correlated inputs, which can mask the importance of individual features.
- **Attribution vs. Causality**: SHAP measures sensitivity inside the model surface, not real-world intervention effects. Altering a transaction timestamp does not change a fraudster's intent.

### 7.3 Regulatory Disclosures
- **Adverse Action Reasons (FCRA / ECOA)**: Local SHAP waterfall attributions provide rank-ordered reason codes for declined transactions without exposing proprietary model parameters.
- **Fairness Audit**: No demographic or protected class attributes were ingested. All features represent transactional velocity and amount ratios.
