# Credit Risk Decision Engine

End-to-end BNPL underwriting system combining behavioral risk modeling, open-banking features, policy rules, unit economics, API serving, and portfolio monitoring.

**Portfolio Scale**

- 12K users
- 23.9K bank accounts
- 2.34M transactions
- 42.1K applications
- 35.4K labeled outcomes

**Final Held-Out Test Performance**

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.768 |
| PR-AUC | 0.180 |
| Brier Score | 0.0371 |
| Mean Predicted PD | 4.13% |
| Observed Default Rate | 4.22% |

**Policy Outcome**

| Metric | Value |
|----------|----------|
| Approval Rate | 81.5% |
| Approved GMV | $1.04M |
| Expected Loss Rate | 1.34% |
| Expected Profit | ~$25K |
| Approved Default Rate | 2.15% |

---

# What is Built

This project simulates how a modern lender or BNPL platform can move from raw application and open-banking-style data to a governed credit decision.

The system separates:

- Synthetic portfolio generation
- Point-in-time feature engineering
- Probability of default (PD) modeling
- Policy and exposure controls
- Expected loss and profit calculations
- Plaid Sandbox inference
- FastAPI serving
- Docker deployment
- Streamlit monitoring
- Testing and CI/CD

> **Design Principle**
>
> The model estimates risk. The policy decides what to do with that risk.

---

# Architecture

```text
Synthetic Portfolio / Plaid Sandbox
              │
              ▼
      Data Normalization
              │
              ▼
 Point-in-Time Feature Engineering
              │
              ▼
 Probability of Default Model
      Logistic Regression
              │
              ▼
         Policy Engine
   Risk + Data Confidence + Limits
              │
              ▼
      Economics Guardrail
 Expected Loss / Expected Profit
              │
              ▼
 APPROVE / REVIEW / REJECT
              │
              ▼
 FastAPI + Docker + Streamlit
```

The same feature-building pipeline is used for both historical model training and Plaid-based inference to reduce training-serving skew.

---

# Portfolio Scale

| Component | Scale |
|------------|------------|
| Users | 12,000 |
| Bank Accounts | 23,922 |
| Transactions | 2,342,981 |
| Credit Applications | 42,074 |
| Labeled Outcomes | 35,402 |
| Repayment Records | 141,608 |
| Observed Defaults | 1,368 |
| Portfolio Default Rate | 3.86% |
| Held-Out Test Applications | 5,311 |

> The 12K users and 42K applications represent the full synthetic portfolio.
>
> Final model and policy metrics are reported on the 5,311-application chronological held-out test set, not on training data.

---

# Model Strategy

Logistic Regression, XGBoost, and LightGBM were benchmarked.

| Model | ROC-AUC | PR-AUC | Brier Score |
|---------|---------|---------|---------|
| Logistic Regression | 0.7995 | 0.2053 | 0.0344 |
| XGBoost | 0.7870 | 0.1871 | 0.0348 |
| LightGBM | 0.7821 | 0.1997 | 0.0345 |

## Why Logistic Regression Won

The final model is an unweighted Logistic Regression model using raw probabilities because it delivered:

- Strongest validation ROC-AUC
- Strongest validation PR-AUC among final candidates
- Competitive Brier score
- Strong portfolio-level calibration
- Easier interpretation and governance
- Lower deployment complexity

The project intentionally did not choose the most complex model simply because it was available.

---

# Probability Calibration

Raw, sigmoid, and isotonic calibration approaches were evaluated.

Raw Logistic Regression already demonstrated strong alignment:

| Metric | Value |
|----------|----------|
| Mean Predicted PD | 4.01% |
| Observed Default Rate | 3.99% |

Additional calibration did not provide sufficient improvement to justify additional complexity, so raw probabilities were retained.

---

# Policy and Economics

The model does not directly approve or reject applications.

The predicted PD is passed into a separately versioned policy engine that also evaluates:

- Bank-data availability
- Data confidence
- Exposure limits
- Risk thresholds
- Expected loss
- Merchant-fee revenue
- Funding cost
- Servicing cost
- Expected profit

## Policy Parameters

```yaml
low_risk_threshold: 0.05
medium_risk_threshold: 0.055

full_approval_max: 1000
limited_exposure_max: 500
bank_data_fallback_max: 50
```

## Economic Assumptions

```yaml
loss_given_default: 0.85
merchant_fee_rate: 0.06
funding_cost_rate: 0.01
servicing_cost: 3
```

### Economics Guardrail

A candidate approval is automatically rejected when:

```text
Expected Profit <= 0
```

This keeps risk estimation and business economics separate.

---

# What We Learned

## 1. Simpler Can Be Better

Logistic Regression outperformed tree-based challengers on the validation metrics most relevant to PD-based underwriting.

## 2. A 0.50 Classification Threshold Is Not Credit Policy

With portfolio default rates near 4%, underwriting is better framed as:

```text
PD Estimate
    ↓
Policy Threshold
    ↓
Exposure Decision
    ↓
Economics
    ↓
Final Approval Decision
```

rather than treating underwriting as a generic binary classification problem.

## 3. More Guardrails Did Not Automatically Improve Results

Behavioral hard rejects were evaluated through ablation testing.

These rules:

- Reduced approvals
- Reduced expected profit
- Added limited risk reduction

Many of the same behavioral patterns were already captured by the model.

The signals remain available for monitoring but are not used as automatic hard-reject criteria.

## 4. Risk Approval Can Still Be Economically Wrong

Some applications passed risk thresholds but generated negative expected profit.

This prompted the addition of an economics veto:

```text
Negative Expected Profit → Reject
```

## 5. Final Metrics Are Truly Out-of-Sample

Reported model and policy results come from a chronological held-out test set that was not used for:

- Model selection
- Calibration selection
- Policy optimization

---

# Final Held-Out Test Results

## Model Performance

| Metric | Result |
|----------|----------|
| Test Applications | 5,311 |
| Defaults | 224 |
| ROC-AUC | 0.7680 |
| PR-AUC | 0.1801 |
| Brier Score | 0.0371 |
| Mean Predicted PD | 4.13% |
| Actual Default Rate | 4.22% |

## Policy Performance

| Metric | Result |
|----------|----------|
| Approval Rate | 81.51% |
| Requested GMV | $1.573M |
| Approved GMV | $1.038M |
| GMV Approval Rate | 66.01% |
| Expected Loss | $13.9K |
| Expected Loss Rate | 1.34% |
| Expected Profit | $25.0K |
| Expected Profit Margin | 2.41% |
| Approved Default Rate | 2.15% |
| Rejected Default Rate | 13.34% |

The rejected population exhibited a substantially higher realized default rate than the approved population, indicating meaningful risk separation.

---

# Dashboard

The Streamlit dashboard separates full portfolio scale from held-out evaluation performance.

Features include:

- Portfolio scale and data coverage
- Approval/rejection performance
- Predicted PD distributions
- Policy-tier economics
- Model monitoring
- Calibration analysis
- Segment analysis
- Decision-level drill downs

## Portfolio Scale & Final Test KPIs

![alt text](<Screenshot 2026-10-08 192757.png>)

![alt text](<Screenshot 2026-10-08 182444.png>)

## Model Monitoring

![alt text](<Screenshot 2026-10-08 191844.png>)

## Segment Analysis

![alt text](<Screenshot 2026-10-08 191856.png>)

---

# Open-Banking Integration

Plaid Sandbox is used as an inference-time open-banking source.

```text
Plaid Sandbox
    ↓
Normalize Accounts + Transactions
    ↓
Shared Feature Pipeline
    ↓
Logistic Regression PD
    ↓
Policy + Economics
    ↓
Underwriting Decision
```

Plaid Sandbox data is not used as labeled model-training data.

Unsupported or missing fields remain missing rather than being fabricated.

---

# Production-Style Components

The project includes:

- FastAPI underwriting service
- Docker containerization
- Streamlit risk operations dashboard
- MLflow experiment tracking
- Pytest automated testing
- GitHub Actions CI/CD
- Versioned model artifacts
- Versioned policy configuration
- Model card documentation
- Governance documentation

## API Endpoints

```http
GET  /health

POST /v1/underwrite
```

---

# Repository Structure

```text
credit-risk-decision-engine/
├── .github/
│   └── workflows/
├── configs/
├── dashboard/
├── data/
├── docs/
├── notebooks/
├── src/
│   ├── api/
│   ├── data_generation/
│   ├── features/
│   ├── ingestion/
│   │   └── plaid/
│   ├── models/
│   ├── monitoring/
│   └── policy/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

Local secrets, caches, MLflow run directories, and generated artifacts are excluded from source control.

---

# Tech Stack

## Modeling

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- MLflow

## Data

- Parquet
- PyArrow
- YAML

## Integration

- Plaid Sandbox

## Serving

- FastAPI
- Pydantic
- Uvicorn

## Deployment

- Docker

## Monitoring

- Streamlit
- Altair

## Testing

- pytest
- GitHub Actions

---

# Running Locally

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Run API

```bash
uvicorn src.api.main:app --reload
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Build Docker Image

```bash
docker build -t credit-risk-engine .
```

## Run Docker Container

```bash
docker run -p 8000:8000 credit-risk-engine
```

---

# Detailed Documentation

Detailed implementation and governance documentation is available under `docs/`, including:

- Business requirements
- Model requirements
- Synthetic data methodology
- Data dictionary
- Feature definitions
- Leakage review
- Model diagnostics
- Champion/challenger comparison
- Probability calibration
- Policy engine design
- Policy optimization
- Plaid integration
- API deployment
- Model card
- Model governance

---

# Limitations

This project is a portfolio demonstration and not a production lending platform.

Key limitations include:

- Training and evaluation data are synthetic
- Plaid integration uses Sandbox environments
- Policy thresholds are simulated
- Economic assumptions are simulated
- No independent model-risk validation has been performed
- No fair-lending review has been completed
- No legal or regulatory review has been completed
- No production security review has been completed
- Model relationships should not be interpreted as causal

> This system should not be used for real consumer credit decisions.

---

# Key Takeaway

This project demonstrates significantly more than model training.

It connects:

```text
Data Generation
      ↓
Feature Engineering
      ↓
PD Modeling
      ↓
Calibration
      ↓
Policy
      ↓
Economics
      ↓
API
      ↓
Monitoring
      ↓
Governance
```

into a single reproducible credit decision workflow.

The primary design decision throughout the project was to optimize for **credible decisioning**, not simply predictive complexity.