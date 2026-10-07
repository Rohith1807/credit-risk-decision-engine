import pandas as pd


def calculate_balance_features(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> dict:
    """
    Calculate account balance behavior before underwriting.

    If balance information is unavailable, return missing values
    rather than fabricating zero balances.
    """

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    if history.empty:
        return {
            "current_balance": pd.NA,
            "avg_balance_30d": pd.NA,
            "median_balance_30d": pd.NA,
            "min_balance_30d": pd.NA,
            "balance_std_30d": pd.NA,
            "negative_balance_events_30d": pd.NA,
            "negative_balance_rate_30d": pd.NA,
        }

    if (
        "balance_after_transaction"
        not in history.columns
    ):
        return {
            "current_balance": pd.NA,
            "avg_balance_30d": pd.NA,
            "median_balance_30d": pd.NA,
            "min_balance_30d": pd.NA,
            "balance_std_30d": pd.NA,
            "negative_balance_events_30d": pd.NA,
            "negative_balance_rate_30d": pd.NA,
        }

    # --------------------------------
    # Current balance
    # --------------------------------

    all_balances = pd.to_numeric(
        history[
            "balance_after_transaction"
        ],
        errors="coerce",
    )

    valid_all_balances = (
        all_balances.dropna()
    )

    if valid_all_balances.empty:
        current_balance = pd.NA
    else:
        current_balance = round(
            float(
                valid_all_balances.iloc[-1]
            ),
            2,
        )

    # --------------------------------
    # Recent 30-day history
    # --------------------------------

    cutoff_30d = (
        application_timestamp
        - pd.Timedelta(days=30)
    )

    recent = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_30d
    ].copy()

    if recent.empty:
        recent = history.tail(
            1
        ).copy()

    recent_balances = pd.to_numeric(
        recent[
            "balance_after_transaction"
        ],
        errors="coerce",
    ).dropna()

    # --------------------------------
    # Missing balance information
    # --------------------------------

    if recent_balances.empty:
        return {
            "current_balance":
                current_balance,

            "avg_balance_30d":
                pd.NA,

            "median_balance_30d":
                pd.NA,

            "min_balance_30d":
                pd.NA,

            "balance_std_30d":
                pd.NA,

            "negative_balance_events_30d":
                pd.NA,

            "negative_balance_rate_30d":
                pd.NA,
        }

    # --------------------------------
    # Balance statistics
    # --------------------------------

    avg_balance_30d = round(
        float(
            recent_balances.mean()
        ),
        2,
    )

    median_balance_30d = round(
        float(
            recent_balances.median()
        ),
        2,
    )

    min_balance_30d = round(
        float(
            recent_balances.min()
        ),
        2,
    )

    balance_std_30d = round(
        float(
            recent_balances.std(
                ddof=0
            )
        ),
        2,
    )

    negative_balance_events_30d = int(
        (
            recent_balances < 0
        ).sum()
    )

    negative_balance_rate_30d = round(
        float(
            (
                recent_balances < 0
            ).mean()
        ),
        4,
    )

    return {
        "current_balance":
            current_balance,

        "avg_balance_30d":
            avg_balance_30d,

        "median_balance_30d":
            median_balance_30d,

        "min_balance_30d":
            min_balance_30d,

        "balance_std_30d":
            balance_std_30d,

        "negative_balance_events_30d":
            negative_balance_events_30d,

        "negative_balance_rate_30d":
            negative_balance_rate_30d,
    }