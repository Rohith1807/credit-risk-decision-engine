# Credit Risk Decision Engine — Feature Quality Review

## Objective

Evaluate engineered underwriting features before model development to identify:

- missing-data issues;
- low-variance features;
- redundant features;
- suspicious synthetic artifacts;
- leakage risks;
- relationships between behavioral features and default outcomes.

## Dataset

The analysis uses historically originated applications with observed repayment outcomes.

Rejected applications are excluded because their repayment outcomes are unobserved.

## Missing Data

Bank-derived features are intentionally missing when open-banking data is unavailable.

Missing bank telemetry is represented as unavailable information rather than as a zero-valued observation.

## Variance Review

Features with zero or near-zero variance are reviewed before modeling.

Short-window transaction velocity features may exhibit limited variation because the first synthetic data generator does not explicitly simulate clustered checkout behavior.

Such features will be excluded or improved if they do not provide meaningful information.

## Predictive Signal Review

Numerical features are compared against observed default rates using quantile buckets.

The objective is not to select final features using univariate performance alone, but to verify that:

- several economically plausible variables contain signal;
- no single synthetic shortcut completely determines default;
- behavioral relationships are directionally reasonable.

## Correlation Review

Highly correlated numerical features are identified.

Correlation alone is not grounds for automatic removal, particularly for tree-based models.

Features may be removed when they are effectively duplicates or provide no additional conceptual value.

## Data Availability

Bank-derived variables are available only when open-banking telemetry is available.

The `bank_data_available` and `data_confidence_score` features distinguish data availability from observed financial behavior.

## Leakage Review

The analysis confirms that prohibited fields such as `latent_pd`, `default_status`, repayment outcomes, and future transactions are excluded from the predictor set.

## Modeling Decision

Only features passing the quality and leakage review will be passed into the baseline modeling pipeline.

Short-window velocity features showed limited variance in the first synthetic portfolio because transactions were generated mostly as dispersed historical activity. The 24-hour velocity features were retained for modeling, while 10-minute and 1-hour variants were excluded from the initial model and reserved for future enhanced event simulation.