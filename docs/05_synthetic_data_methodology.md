Purpose of synthetic data
Random seed = 42
Number of users
Number of accounts
Generation assumptions
Distribution assumptions
Known limitations
Future transaction generation methodology
Default generation methodology
Income Generation
Employment-Specific Behavior
Recurring Expenses
Discretionary Spending
BNPL Payment Simulation
Account Balance Simulation
Known Limitations

Employment-specific transaction distributions are synthetic modeling assumptions designed to create realistic behavioral heterogeneity. They are not estimates of actual population-level financial behavior.

## Historical Approval Selection

Loan repayment outcomes are only generated for historically approved applications.

This intentionally reproduces a common credit-risk modeling challenge: observed repayment outcomes exist only for originated credit.

As a result, the eventual modeling dataset may exhibit selection bias from the historical approval policy.

The first project version will document this limitation rather than implement full reject-inference methodology.

## Synthetic Default Mechanism

Defaults are not randomly assigned.

A hidden synthetic risk function determines the probability of default using pre-application behavioral characteristics including:

- negative balance behavior;
- income variability;
- requested exposure relative to recent income;
- observed BNPL repayment activity;
- cash-buffer strength.

The resulting latent probability is used only to generate synthetic repayment outcomes.

The latent risk score and latent probability are prohibited from use as predictive model features.