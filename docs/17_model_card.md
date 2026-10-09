# Credit Risk Decision Engine — Model Card

## Model Purpose

Estimate probability of default for BNPL-style credit applications using behavioral transaction, open-banking, and application features.

The model is designed as the risk-scoring component of a broader decision engine. It does not directly approve or reject applicants; predicted probability of default is passed to an independently versioned policy layer.

## Selected Model

**Unweighted Logistic Regression using raw predicted probabilities.**

The final model was selected after comparing Logistic Regression, XGBoost, and LightGBM on the scaled development portfolio.

Logistic Regression was selected because it delivered the strongest overall validation discrimination while maintaining stable and well-aligned probability estimates. Its interpretability and straightforward governance characteristics were also considered appropriate for credit-risk decisioning.

## Training Data

The model is trained on a synthetic historical credit portfolio containing:

- 12,000 users;
- 23,922 bank accounts;
- 2,342,981 transactions;
- 42,074 credit applications;
- 35,402 originated-loan outcomes;
- 141,608 repayment records;
- 1,368 observed defaults.

The generated portfolio produced an observed default rate of approximately **3.86%**, within the targeted 3%–5% range.

Synthetic data is used because labeled real-world credit performance data is not available for this portfolio project.

## External Data Integration

Plaid Sandbox is integrated as an inference-time open-banking data source.

Plaid transaction data is normalized into the same internal feature-engineering pipeline used during model development.

Plaid Sandbox observations are not used as labeled model-training data.

Unsupported balance-derived fields remain missing and are handled by the model preprocessing pipeline rather than being fabricated.

## Target

The model target is `default_status`.

A default is defined using the synthetic portfolio's repayment outcome logic based on **90+ days past due**.

## Data Splitting

The modeling dataset contains **35,402 labeled applications** and is split chronologically to preserve temporal ordering:

- Training: 24,781 rows;
- Validation: 5,310 rows;
- Test: 5,311 rows.

The test set remained untouched during model selection, probability-calibration analysis, and policy optimization.

## Model Inputs

Representative feature families include:

- income behavior;
- account balance behavior;
- spending behavior;
- transaction velocity;
- BNPL payment burden;
- loan-to-income ratio;
- requested-amount-to-balance ratio;
- cash-buffer metrics;
- income stability;
- data-confidence measures;
- merchant category;
- device type;
- application channel;
- bank-data availability.

Point-in-time feature construction is used to prevent future information from leaking into historical application features.

## Model Selection

The following model families were evaluated:

- Logistic Regression;
- XGBoost;
- LightGBM.

Representative validation results for the final scaled portfolio included:

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.7995 | 0.2053 | 0.0344 |
| XGBoost, unweighted | 0.7870 | 0.1871 | 0.0348 |
| LightGBM, unweighted | 0.7821 | 0.1997 | 0.0345 |

Weighted tree models improved recall at conventional classification thresholds but produced severely inflated probability estimates and substantially worse Brier scores. They were therefore rejected for probability-of-default estimation.

The final model is **unweighted Logistic Regression**.

## Probability Calibration

Raw, sigmoid-calibrated, and isotonic-calibrated probabilities were evaluated.

The final raw Logistic Regression model already showed strong portfolio-level calibration.

Validation calibration comparison:

- Raw Logistic Regression:
  - ROC-AUC: 0.8001
  - PR-AUC: 0.2042
  - Brier score: 0.0345
  - Mean predicted PD: 4.01%
  - Actual default rate: 3.99%

- Sigmoid calibration produced no ranking improvement and did not materially improve probability quality.

- Isotonic calibration slightly improved Brier score but reduced PR-AUC and introduced unnecessary complexity.

For these reasons, **raw Logistic Regression probabilities were retained**.

## Final Test Performance

The final model was retrained on the full training set and evaluated once on the untouched chronological test set.

Final test results:

- Test rows: 5,311;
- Test defaults: 224;
- ROC-AUC: **0.7680**;
- PR-AUC: **0.1801**;
- Brier score: **0.0371**;
- Mean predicted PD: **4.13%**;
- Actual default rate: **4.22%**.

The close alignment between mean predicted PD and observed default rate indicates good portfolio-level probability behavior on the final test set.

## Policy Layer

The model does not directly approve or reject applications.

Predicted probability of default is passed into a separately versioned policy engine that combines model risk, data availability, risk appetite, and unit economics.

Final simulated policy structure:

- **Full Approval:** PD below 5%;
- **Review:** PD from 5% to below 8%;
- **Reject / Review-out:** PD of 8% or above.

Additional policy controls include:

- missing bank-data fallback;
- data-confidence handling;
- expected-profit viability checks;
- versioned exposure limits;
- policy reason codes.

Applications with negative expected profit are rejected even when model risk alone would otherwise permit approval.

## Behavioral Guardrails

Behavioral hard-reject rules were evaluated through policy ablation.

Examples included:

- high loan-to-income ratio;
- elevated BNPL repayment burden;
- frequent negative-balance activity.

The ablation analysis showed that these hard rejects reduced profitable approvals while providing limited incremental risk improvement because the same behavioral signals were already incorporated into the probability-of-default model.

For the final production-style policy simulation, behavioral hard rejects are disabled.

The features remain available as model inputs and monitoring signals.

## Policy Economics

The policy engine estimates:

- expected loss;
- merchant-fee revenue;
- funding cost;
- servicing cost;
- expected profit.

A negative expected-profit decision is not automatically approved.

The economic layer is intentionally separated from the predictive model so model risk estimation and business-policy decisions can be evaluated independently.

## Final Policy Results

Using the final test portfolio and current economics logic:

- Applications: **5,311**;
- Approval rate: **81.51%**;
- Requested GMV: **$1.573M**;
- Approved GMV: **$1.038M**;
- GMV approval rate: **66.01%**;
- Expected loss: **$13.9K**;
- Expected loss rate: **1.34%**;
- Expected profit: **$25.0K**;
- Expected profit margin: **2.41%**;
- Approved observed default rate: **2.15%**;
- Rejected observed default rate: **13.34%**.

The substantially higher observed default rate among rejected applications provides evidence that the model and policy layer are separating higher-risk applications from the approved portfolio.

## Explainability

The final model is Logistic Regression, so model behavior can be inspected through feature coefficients after preprocessing.

Policy-level explanations are handled separately through reason codes such as:

- policy requirements met;
- bank data unavailable;
- insufficient data confidence;
- high probability of default;
- negative expected profit.

Policy reason codes are operational explanations generated by the decision framework and should not be interpreted as production adverse-action notices.

## Monitoring

The system supports monitoring of:

- approval rate;
- predicted PD distribution;
- observed default rate;
- approved vs rejected default rates;
- Brier score;
- mean predicted PD vs actual default rate;
- expected loss;
- expected profit;
- expected loss rate;
- policy-tier distribution;
- reason-code distribution;
- feature distribution drift;
- segment-level approval and risk performance.

A Streamlit risk-operations dashboard is used to visualize final test performance, portfolio economics, policy outcomes, model monitoring, and segment analysis.

## Deployment

The final model is persisted as a versioned artifact and served through a FastAPI underwriting service.

Deployment components include:

- versioned Logistic Regression model artifact;
- model metadata;
- FastAPI `/v1/underwrite` endpoint;
- `/health` endpoint;
- Docker containerization;
- API request validation;
- independently versioned policy configuration;
- Streamlit monitoring dashboard;
- GitHub Actions CI for automated testing and Docker build validation.

## Governance

Model and policy versions are tracked separately.

Decision records can include:

- application identifier;
- decision timestamp;
- predicted probability of default;
- model version;
- policy version;
- policy tier;
- final decision;
- approved exposure;
- reason code;
- expected loss;
- expected profit.

The project also includes:

- point-in-time feature construction;
- leakage review;
- chronological dataset splitting;
- model comparison;
- probability calibration analysis;
- policy threshold optimization;
- guardrail ablation;
- untouched final test evaluation;
- automated tests;
- CI/CD validation.

## Known Limitations

The training portfolio is synthetic and does not represent actual lender performance data.

Synthetic relationships may be cleaner or structurally different from real borrower behavior.

Plaid integration uses Sandbox data rather than live consumer banking data.

The economic assumptions, policy thresholds, servicing costs, loss-given-default assumptions, and approval limits are simulated for portfolio demonstration purposes.

The project has not undergone independent model-risk validation, legal review, fair-lending analysis, compliance review, or production security review.

This system is a portfolio engineering and decision-science demonstration and is **not intended for real consumer credit decisions**.