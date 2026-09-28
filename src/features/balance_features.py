import pandas as pd


def calculate_balance_features(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> dict:
    """
    Calculate account balance behavior before underwriting.
    """

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    if history.empty:
        return {
            "current_balance": 0.0,
            "avg_balance_30d": 0.0,
            "median_balance_30d": 0.0,
            "min_balance_30d": 0.0,
            "balance_std_30d": 0.0,
            "negative_balance_events_30d": 0,
            "negative_balance_rate_30d": 0.0,
        }

    cutoff_30d = (
        application_timestamp
        - pd.Timedelta(days=30)
    )

    recent = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_30d
    ]

    if recent.empty:
        recent = history.tail(1)

    balances = recent[
        "balance_after_transaction"
    ]

    current_balance = float(
        history.iloc[-1][
            "balance_after_transaction"
        ]
    )

    negative_events = int(
        (balances < 0).sum()
    )

    negative_rate = (
        negative_events
        / len(balances)
        if len(balances) > 0
        else 0.0
    )

    return {
        "current_balance":
            round(
                current_balance,
                2,
            ),

        "avg_balance_30d":
            round(
                float(
                    balances.mean()
                ),
                2,
            ),

        "median_balance_30d":
            round(
                float(
                    balances.median()
                ),
                2,
            ),

        "min_balance_30d":
            round(
                float(
                    balances.min()
                ),
                2,
            ),

        "balance_std_30d":
            round(
                float(
                    balances.std(
                        ddof=0
                    )
                ),
                2,
            ),

        "negative_balance_events_30d":
            negative_events,

        "negative_balance_rate_30d":
            round(
                negative_rate,
                4,
            ),
    }