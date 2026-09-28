import numpy as np
import pandas as pd


def calculate_income_features(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> dict:
    """
    Calculate income-related features from pre-application transaction history.
    """

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    income = history[
        history[
            "transaction_type"
        ]
        == "income"
    ].copy()

    if income.empty:
        return {
            "income_count_90d": 0,
            "income_total_90d": 0.0,
            "avg_income_deposit_90d": 0.0,
            "median_income_deposit_90d": 0.0,
            "income_std_90d": 0.0,
            "income_cv_90d": 0.0,
            "days_since_last_income": 999,
        }

    cutoff_90d = (
        application_timestamp
        - pd.Timedelta(days=90)
    )

    income_90d = income[
        income[
            "transaction_timestamp"
        ]
        >= cutoff_90d
    ]

    if income_90d.empty:
        return {
            "income_count_90d": 0,
            "income_total_90d": 0.0,
            "avg_income_deposit_90d": 0.0,
            "median_income_deposit_90d": 0.0,
            "income_std_90d": 0.0,
            "income_cv_90d": 0.0,
            "days_since_last_income": 999,
        }

    income_values = income_90d[
        "amount"
    ]

    mean_income = float(
        income_values.mean()
    )

    std_income = float(
        income_values.std(ddof=0)
    )

    income_cv = (
        std_income / mean_income
        if mean_income > 0
        else 0.0
    )

    last_income_timestamp = pd.Timestamp(
        income_90d[
            "transaction_timestamp"
        ].max()
    )

    days_since_last_income = int(
        (
            application_timestamp
            - last_income_timestamp
        ).days
    )

    return {
        "income_count_90d":
            int(len(income_90d)),

        "income_total_90d":
            round(
                float(
                    income_values.sum()
                ),
                2,
            ),

        "avg_income_deposit_90d":
            round(
                mean_income,
                2,
            ),

        "median_income_deposit_90d":
            round(
                float(
                    income_values.median()
                ),
                2,
            ),

        "income_std_90d":
            round(
                std_income,
                2,
            ),

        "income_cv_90d":
            round(
                income_cv,
                4,
            ),

        "days_since_last_income":
            days_since_last_income,
    }