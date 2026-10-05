# Credit Risk Decision Engine — Baseline Model

## Objective

Establish an interpretable benchmark for probability-of-default modeling before introducing gradient-boosted models.

## Algorithm

Logistic Regression

The baseline uses class-weighted training to address the low prevalence of default outcomes.

## Dataset

Training data contains historically originated applications with observed repayment outcomes.

Applications are split chronologically into:

- training;
- validation;
- test.

The test dataset remains untouched during model-development and model-selection activities.

## Preprocessing

Numerical features:

- median missing-value imputation;
- standard scaling.

Categorical features:

- most-frequent missing-value imputation;
- one-hot encoding.

Binary features:

- most-frequent missing-value imputation.

Bank-derived variables are intentionally missing when open-banking data is unavailable.

## Class Imbalance

The baseline Logistic Regression model uses:

`class_weight = balanced`

This provides an initial cost-sensitive approach to rare default events.

## Primary Evaluation Metric

Precision-Recall AUC (PR-AUC)

PR-AUC is emphasized because default events are rare and ranking performance among the positive class is particularly important.

## Secondary Metrics

- ROC-AUC
- Precision
- Recall
- F1 Score
- Brier Score

## Decision Threshold

A probability threshold of 0.50 is used only for the initial diagnostic confusion matrix.

It is not treated as the final underwriting policy threshold.

Final credit decisions will be made by a separate policy engine using calibrated probability of default and business constraints.

## Experiment Tracking

MLflow tracks:

- model configuration;
- dataset sizes;
- validation metrics;
- fitted model artifact.

This baseline will be compared with XGBoost and LightGBM challenger models.