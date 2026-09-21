Target:
default_status

Prediction:
P(default_status = 1)

Unit:
credit application

Expected default prevalence:
approximately 3–5%

Training:
historical applications only

Validation:
time-aware holdout preferred

Primary evaluation:
PR-AUC + calibration + business outcomes

Secondary evaluation:
ROC-AUC, precision, recall, F1

Baseline:
Logistic Regression

Champion candidates:
XGBoost / LightGBM

Probability calibration:
required

Explainability:
required

Decision logic:
must remain outside model

Fallback underwriting:
required

Prediction must be between 0 and 1.

Same input + same model + same policy
must produce the same decision.

Model version must be logged.

Policy version must be logged.

Future information must not enter features.

Missing open-banking data must not crash decisioning.

High-risk decisions must never bypass exposure caps.