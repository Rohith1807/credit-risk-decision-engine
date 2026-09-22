USER_COLUMNS = [
    "user_id",
    "signup_date",
    "age_band",
    "region",
    "employment_type",
    "account_tenure_months",
]


ACCOUNT_COLUMNS = [
    "account_id",
    "user_id",
    "account_type",
    "account_open_date",
    "current_balance",
    "available_balance",
    "primary_account_flag",
]


TRANSACTION_COLUMNS = [
    "transaction_id",
    "account_id",
    "transaction_timestamp",
    "amount",
    "transaction_type",
    "merchant_name",
    "merchant_category",
    "balance_after_transaction",
]


APPLICATION_COLUMNS = [
    "application_id",
    "user_id",
    "application_timestamp",
    "requested_amount",
    "merchant_category",
    "device_type",
    "channel",
    "bank_data_available",
]


LOAN_COLUMNS = [
    "loan_id",
    "application_id",
    "user_id",
    "approved_amount",
    "origination_date",
    "term_days",
    "installment_count",
    "interest_rate",
]


REPAYMENT_COLUMNS = [
    "payment_id",
    "loan_id",
    "installment_number",
    "scheduled_payment_date",
    "actual_payment_date",
    "amount_due",
    "amount_paid",
    "days_past_due",
    "payment_status",
]


DECISION_COLUMNS = [
    "decision_id",
    "application_id",
    "decision_timestamp",
    "probability_default",
    "policy_tier",
    "decision",
    "approved_limit",
    "reason_code",
    "model_version",
    "policy_version",
]