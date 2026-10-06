Objective:
Maximize profitable approval volume while keeping credit
risk within a defined risk appetite.

Scenarios:
- Conservative
- Current
- Growth
- Aggressive Growth

Metrics:
- Approval rate
- GMV approval rate
- Approved GMV
- Expected loss
- Expected loss rate
- Expected profit
- Expected profit margin
- Observed approved default rate
- Rejected default rate

Policy optimization is performed independently of model
training.

The validation portfolio is used for policy development.
The test portfolio remains untouched.

Model:
Raw unweighted LightGBM

Policy:
PD thresholds only
+ bank-data fallback
+ data-confidence handling

Thresholds:
Full approval: PD < 4%
Low-and-grow:  4% ≤ PD < 5%
Reject/review: PD ≥ 5%

Remove as hard rejection rules:
- loan-to-income ratio
- BNPL burden
- negative-balance rate