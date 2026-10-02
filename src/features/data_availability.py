import pandas as pd


BANK_DERIVED_FEATURES = [
    "income_count_90d",
    "income_total_90d",
    "avg_income_deposit_90d",
    "median_income_deposit_90d",
    "income_std_90d",
    "income_cv_90d",
    "days_since_last_income",

    "current_balance",
    "avg_balance_30d",
    "median_balance_30d",
    "min_balance_30d",
    "balance_std_30d",
    "negative_balance_events_30d",
    "negative_balance_rate_30d",

    "spend_total_30d",
    "transaction_count_30d",
    "unique_merchants_30d",
    "bnpl_payment_count_90d",
    "bnpl_payment_amount_90d",
    "discretionary_spend_30d",

    "loan_to_income_ratio",
    "requested_amount_to_balance",
    "paycheck_proximity_days",
    "discretionary_spend_ratio",
    "cash_buffer_days",
    "cash_buffer_velocity",
    "spend_acceleration",
    "bnpl_payment_burden",
    "income_stability_score",

    "transaction_count_10m",
    "transaction_count_1h",
    "transaction_count_24h",
    "unique_merchants_1h",
    "unique_merchants_24h",
]

def mask_unavailable_bank_features(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove bank-derived information when open-banking data
    is unavailable for an application.
    """

    output = dataset.copy()

    unavailable_mask = (
        output[
            "bank_data_available"
        ]
        == 0
    )

    existing_columns = [
        column
        for column
        in BANK_DERIVED_FEATURES
        if column
        in output.columns
    ]

    output.loc[
        unavailable_mask,
        existing_columns,
    ] = pd.NA

    return output