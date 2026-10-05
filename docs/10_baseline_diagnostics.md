Baseline selected:
Unweighted Logistic Regression

Validation ROC-AUC: 0.7309
Validation PR-AUC: 0.1757
Validation Brier: 0.0267

Observed default rate: 2.85%
Mean predicted PD: 3.30%

Weighted Logistic Regression was rejected as the primary PD
baseline because class weighting inflated average predicted PD
to 32.61% and substantially worsened probability calibration.

Threshold analysis demonstrated meaningful ranking below the
conventional 0.50 threshold. No underwriting threshold was
selected during model development.

Calibration was directionally reasonable but noisy because the
validation sample contained only 13 defaults.

Several Logistic Regression coefficients had counterintuitive
signs. These are treated as potential effects of correlated
predictors and will be compared against nonlinear challenger
models and feature-importance diagnostics before any features
are removed.