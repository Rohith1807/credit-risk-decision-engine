# Feature Definitions

## Income Features

### income_count_90d
Number of observed income transactions during the 90 days before the application.

### income_total_90d
Total observed positive income deposits during the 90-day pre-application period.

### income_cv_90d
Coefficient of variation of income deposits.

Used as an indicator of income variability.

### days_since_last_income
Number of days between the most recent observed income deposit and the application timestamp.

---

## Balance Features

### current_balance
Most recent observed account balance available before underwriting.

### avg_balance_30d
Average observed balance over the 30 days before the application.

### negative_balance_rate_30d
Percentage of observed balance events below zero during the 30-day period.

---

## Transaction Features

### spend_total_30d
Total transaction outflows during the 30 days before underwriting.

### bnpl_payment_count_90d
Count of observed external BNPL-style payments during the prior 90 days.

### discretionary_spend_30d
Observed restaurant, entertainment, and shopping outflows during the prior 30 days.