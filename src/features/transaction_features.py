import pandas as pd


def calculate_transaction_features(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> dict:
    """
    Calculate pre-application spending and BNPL behavior.
    """

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    if history.empty:
        return {
            "spend_total_30d": 0.0,
            "transaction_count_30d": 0,
            "unique_merchants_30d": 0,
            "bnpl_payment_count_90d": 0,
            "bnpl_payment_amount_90d": 0.0,
            "discretionary_spend_30d": 0.0,
        }

    cutoff_30d = (
        application_timestamp
        - pd.Timedelta(days=30)
    )

    cutoff_90d = (
        application_timestamp
        - pd.Timedelta(days=90)
    )

    recent_30d = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_30d
    ]

    recent_90d = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_90d
    ]

    spending = recent_30d[
        recent_30d["amount"] < 0
    ]

    bnpl = recent_90d[
        recent_90d[
            "transaction_type"
        ]
        == "bnpl_payment"
    ]

    discretionary_categories = {
        "restaurants",
        "entertainment",
        "shopping",
    }

    discretionary = recent_30d[
        recent_30d[
            "merchant_category"
        ].isin(
            discretionary_categories
        )
        &
        (
            recent_30d["amount"] < 0
        )
    ]

    return {
        "spend_total_30d":
            round(
                float(
                    spending[
                        "amount"
                    ].abs().sum()
                ),
                2,
            ),

        "transaction_count_30d":
            int(
                len(
                    recent_30d
                )
            ),

        "unique_merchants_30d":
            int(
                recent_30d[
                    "merchant_name"
                ].nunique()
            ),

        "bnpl_payment_count_90d":
            int(
                len(bnpl)
            ),

        "bnpl_payment_amount_90d":
            round(
                float(
                    bnpl[
                        "amount"
                    ].abs().sum()
                ),
                2,
            ),

        "discretionary_spend_30d":
            round(
                float(
                    discretionary[
                        "amount"
                    ].abs().sum()
                ),
                2,
            ),
    }