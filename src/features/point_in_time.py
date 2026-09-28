import pandas as pd


def get_transaction_history(
    transactions: pd.DataFrame,
    user_id: str,
    application_timestamp: pd.Timestamp,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Return only transactions available at or before the application timestamp.

    Optionally restrict history to a fixed lookback window.
    """

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    history = transactions[
        (
            transactions["user_id"]
            == user_id
        )
        &
        (
            transactions[
                "transaction_timestamp"
            ]
            <= application_timestamp
        )
    ].copy()

    if lookback_days is not None:
        start_timestamp = (
            application_timestamp
            - pd.Timedelta(
                days=lookback_days
            )
        )

        history = history[
            history[
                "transaction_timestamp"
            ]
            >= start_timestamp
        ]

    return (
        history
        .sort_values(
            "transaction_timestamp"
        )
        .reset_index(
            drop=True
        )
    )