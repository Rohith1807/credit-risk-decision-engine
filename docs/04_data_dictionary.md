# Data Dictionary

## users

| Column | Type | Description |
|---|---|---|
| user_id | string | Unique customer identifier |
| signup_date | date | Date customer entered platform |
| age_band | category | Customer age range |
| region | category | Broad geographic region |
| employment_type | category | Salaried, self-employed, gig, etc. |
| account_tenure_months | integer | Months since customer joined |

## applications

| Column | Type | Description |
|---|---|---|
| application_id | string | Unique application identifier |
| user_id | string | Applicant identifier |
| application_timestamp | datetime | Exact underwriting decision point |
| requested_amount | float | Requested transaction amount |
| merchant_category | category | Merchant type |
| device_type | category | Mobile, desktop, tablet |
| channel | category | Checkout, app, web |
| bank_data_available | boolean | Whether banking telemetry is available |