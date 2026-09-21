# Credit Risk Decision Engine

## 1. Business Problem

Digital lenders and BNPL platforms must make credit decisions quickly while balancing two competing objectives:

1. Approve enough customers and transactions to support growth.
2. Keep defaults and credit losses within an acceptable risk appetite.

Traditional underwriting may provide limited visibility for customers with thin credit histories or rapidly changing financial behavior.

The Credit Risk Decision Engine will simulate an underwriting system that combines application information, repayment history, and behavioral cash-flow signals to estimate the probability that a borrower will default.

The predicted probability of default will then be passed to a separate credit policy engine responsible for determining the appropriate action.

---

## 2. Primary Business Question

How can a lender increase profitable customer approvals while keeping portfolio credit losses within an acceptable risk threshold?

---

## 3. Project Objective

Develop an end-to-end credit risk decisioning system that:

* estimates Probability of Default (PD) for each credit application;
* incorporates behavioral and cash-flow-based risk indicators;
* applies configurable underwriting policies;
* supports low-and-grow credit limits for uncertain or borderline applicants;
* provides fallback decisions when open-banking data is unavailable;
* measures the impact of underwriting policy on growth, risk, and profitability;
* provides explainable decision reasons;
* supports model monitoring and governance.

---

## 4. Unit of Decision

The primary unit of analysis will be an individual credit application or transaction request.

Each application will receive:

* a probability of default;
* a policy tier;
* an underwriting decision;
* an approved credit limit;
* one or more reason codes;
* model and policy version identifiers.

---

## 5. Target Definition

The primary machine-learning target is:

`default_status`

Where:

* `0` = borrower successfully performs according to the repayment obligation;
* `1` = borrower reaches the project's defined default threshold.

For the initial project implementation, default will be defined as a borrower reaching at least **90 days past due** on the associated credit obligation.

This definition may be revised in later versions if the simulated BNPL product uses a shorter repayment horizon.

---

## 6. Core Business KPIs

The underwriting system will be evaluated using:

### Growth Metrics

* Approval Rate
* Customer Acceptance Rate
* Gross Merchandise Volume (GMV)
* Approved Credit Volume
* Number of New Customers Approved
* Number of Customers Approved Through Low-and-Grow

### Risk Metrics

* Portfolio Default Rate
* Expected Credit Loss
* Charge-Off Rate
* Exposure at Default
* Default Capture Rate

### Economic Metrics

* Expected Revenue
* Expected Credit Loss
* Expected Profit
* Profit per Approved Application

### Operational Metrics

* Decision Latency
* Percentage of Applications Using Fallback Underwriting
* Open-Banking Connection Rate
* Manual Review Rate
* Decision Error Rate

---

## 7. Business Success Criteria

The model will not be considered successful solely because it achieves strong predictive metrics.

A successful underwriting system should demonstrate at least one of the following:

* higher approval rates at approximately equivalent portfolio risk;
* lower credit losses at approximately equivalent approval rates;
* higher expected profit than the baseline policy;
* improved onboarding of low-risk customers who would otherwise receive conservative decisions.

The final project will compare:

1. Baseline rule-based underwriting
2. Machine-learning underwriting
3. ML + behavioral telemetry
4. ML + behavioral telemetry + optimized policy engine

---

## 8. Initial Risk Policy

The initial policy will use probability-of-default thresholds as a starting point.

Example:

* Low Risk → Full Approval
* Moderate Risk → Low-and-Grow / Micro-Limit
* High Risk → Reject or Manual Review

Initial thresholds are illustrative and must not be treated as optimal.

Later phases will determine thresholds using business economics and portfolio risk constraints.

---

## 9. Key Risk and Data Limitations

The system must account for:

### Open-Banking Refusal

Customers may decline to connect their financial accounts.

The system must provide a conservative fallback underwriting path rather than automatically rejecting the applicant.

### Income Volatility

Temporary low balances should not automatically imply high credit risk.

Income stability and longer-term cash-flow behavior should be considered.

### Partial Financial Visibility

A connected bank account may not represent the customer's complete financial activity.

The system must maintain a financial-data-confidence indicator.

### External Credit Exposure

External BNPL loans or other obligations may not be fully observable.

Any inferred external exposure must be treated as an incomplete risk signal.

### Data Leakage

Only information available at or before the underwriting decision may be used for prediction.

---

## 10. Model and Policy Separation

The machine-learning model is responsible for estimating risk.

The policy engine is responsible for determining the business action.

The architecture must preserve this separation:

Credit Application
→ Feature Engineering
→ Risk Model
→ Probability of Default
→ Policy Engine
→ Credit Decision

---

## 11. Governance Principles

The system should support:

* model versioning;
* policy versioning;
* reproducible training;
* decision logging;
* explainable reason codes;
* probability calibration;
* model monitoring;
* drift detection;
* fairness analysis;
* documented assumptions;
* human review where appropriate.

---

## 12. Project Scope

The project is a portfolio-grade simulation of a real credit-risk decisioning system.

It is not intended to represent a production-certified lending platform or substitute for legal, regulatory, security, or independent model-validation requirements used by financial institutions.
