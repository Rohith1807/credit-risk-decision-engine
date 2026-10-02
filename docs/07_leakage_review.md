# Credit Risk Decision Engine — Feature Leakage Review

## Objective

The underwriting model must use only information that would have been available at the time of the credit application.

Any information generated after the underwriting decision is prohibited from use as a predictive feature.

## Point-in-Time Rule

For application time `T`, transaction-based features may use only:

`transaction_timestamp <= T`

Future transactions are excluded.

## Prohibited Outcome Features

The following fields are never permitted as model predictors:

- default_status
- latent_pd
- max_days_past_due
- repayment payment status
- repayment amount paid
- future delinquency information
- final credit outcome

`latent_pd` is synthetic ground truth used only to generate simulated outcomes.

It is not an observable production feature.

## Open-Banking Availability

Bank-derived behavioral features are available only when `bank_data_available = 1`.

When open-banking data is unavailable, the corresponding bank-derived features are represented as missing rather than zero.

This distinction prevents the model from interpreting unavailable information as observed financial behavior.

## Historical Approval Selection

Repayment outcomes exist only for historically approved applications.

Rejected applications therefore do not have observed default outcomes and are excluded from supervised model training.

This creates historical selection bias.

The first project version documents this limitation rather than implementing reject inference.

## Dataset Split

Applications are split chronologically rather than purely randomly:

- earliest observations → training
- subsequent observations → validation
- latest observations → test

This provides a more realistic approximation of future model deployment.

## Leakage Controls

Automated tests verify:

- future transactions are excluded;
- identifiers and target fields are excluded from predictor lists;
- hidden synthetic risk values are excluded;
- post-origination repayment outcomes are excluded;
- train, validation, and test applications do not overlap.