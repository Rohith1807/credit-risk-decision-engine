import numpy as np
import pandas as pd


def is_missing(
    value,
) -> bool:
    """
    Return True when a scalar value is missing.
    """

    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )
    except (TypeError, ValueError):
        return False


def safe_divide(
    numerator,
    denominator,
    default=0.0,
):
    """
    Safely divide two values.

    Missing inputs return the supplied default.
    Division by zero or negative denominator
    also returns the supplied default.
    """

    if (
        is_missing(numerator)
        or is_missing(denominator)
    ):
        return default

    denominator = float(
        denominator
    )

    numerator = float(
        numerator
    )

    if denominator <= 0:
        return default

    return float(
        numerator
        / denominator
    )


def calculate_loan_to_income_ratio(
    requested_amount: float,
    income_total_90d,
):
    """
    Requested credit amount relative to
    estimated monthly income.

    90-day income is converted to an
    approximate monthly amount.
    """

    if is_missing(
        income_total_90d
    ):
        return pd.NA

    estimated_monthly_income = (
        float(
            income_total_90d
        )
        / 3
    )

    ratio = safe_divide(
        requested_amount,
        estimated_monthly_income,
        default=1.0,
    )

    return round(
        min(
            ratio,
            10.0,
        ),
        4,
    )


def calculate_requested_amount_to_balance(
    requested_amount: float,
    current_balance,
):
    """
    Calculate requested amount relative
    to current balance.

    Returns missing when current balance
    is unavailable.
    """

    if is_missing(
        current_balance
    ):
        return pd.NA

    current_balance = float(
        current_balance
    )

    return round(
        float(
            requested_amount
        )
        / max(
            current_balance,
            1.0,
        ),
        4,
    )


def calculate_paycheck_proximity(
    days_since_last_income,
):
    """
    Number of days since the latest
    observed income deposit.
    """

    if is_missing(
        days_since_last_income
    ):
        return pd.NA

    days_since_last_income = float(
        days_since_last_income
    )

    if days_since_last_income < 0:
        return 999

    return int(
        min(
            days_since_last_income,
            999,
        )
    )


def calculate_discretionary_spend_ratio(
    discretionary_spend_30d,
    spend_total_30d,
):
    """
    Share of recent spending classified
    as discretionary.
    """

    if (
        is_missing(
            discretionary_spend_30d
        )
        or is_missing(
            spend_total_30d
        )
    ):
        return pd.NA

    ratio = safe_divide(
        discretionary_spend_30d,
        spend_total_30d,
        default=0.0,
    )

    return round(
        float(
            np.clip(
                ratio,
                0.0,
                1.0,
            )
        ),
        4,
    )


def calculate_cash_buffer_days(
    current_balance,
    spend_total_30d,
):
    """
    Estimate how many days of recent
    spending can be covered by the
    current observed account balance.

    Returns missing when current balance
    is unavailable.
    """

    if (
        is_missing(
            current_balance
        )
        or is_missing(
            spend_total_30d
        )
    ):
        return pd.NA

    current_balance = float(
        current_balance
    )

    spend_total_30d = float(
        spend_total_30d
    )

    average_daily_spend = (
        spend_total_30d
        / 30
    )

    if average_daily_spend <= 0:
        return 30.0

    buffer_days = (
        current_balance
        / average_daily_spend
    )

    return round(
        float(
            np.clip(
                buffer_days,
                0.0,
                90.0,
            )
        ),
        2,
    )


def calculate_cash_buffer_velocity(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
):
    """
    Estimate balance trajectory by
    comparing recent and prior average
    balances.

    Returns missing when balance history
    is unavailable.
    """

    if history.empty:
        return pd.NA

    if (
        "balance_after_transaction"
        not in history.columns
    ):
        return pd.NA

    application_timestamp = (
        pd.Timestamp(
            application_timestamp
        )
    )

    recent_start = (
        application_timestamp
        - pd.Timedelta(
            days=15
        )
    )

    prior_start = (
        application_timestamp
        - pd.Timedelta(
            days=30
        )
    )

    recent = history[
        history[
            "transaction_timestamp"
        ]
        >= recent_start
    ].copy()

    prior = history[
        (
            history[
                "transaction_timestamp"
            ]
            >= prior_start
        )
        &
        (
            history[
                "transaction_timestamp"
            ]
            < recent_start
        )
    ].copy()

    if (
        recent.empty
        or prior.empty
    ):
        return pd.NA

    recent_balances = pd.to_numeric(
        recent[
            "balance_after_transaction"
        ],
        errors="coerce",
    ).dropna()

    prior_balances = pd.to_numeric(
        prior[
            "balance_after_transaction"
        ],
        errors="coerce",
    ).dropna()

    if (
        recent_balances.empty
        or prior_balances.empty
    ):
        return pd.NA

    recent_balance = float(
        recent_balances.mean()
    )

    prior_balance = float(
        prior_balances.mean()
    )

    if abs(
        prior_balance
    ) < 1:
        return 0.0

    velocity = (
        recent_balance
        - prior_balance
    ) / abs(
        prior_balance
    )

    return round(
        float(
            np.clip(
                velocity,
                -5.0,
                5.0,
            )
        ),
        4,
    )


def calculate_spend_acceleration(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> float:
    """
    Compare spending during the most
    recent 7 days with the prior 7 days.
    """

    if history.empty:
        return 0.0

    application_timestamp = (
        pd.Timestamp(
            application_timestamp
        )
    )

    recent_start = (
        application_timestamp
        - pd.Timedelta(
            days=7
        )
    )

    prior_start = (
        application_timestamp
        - pd.Timedelta(
            days=14
        )
    )

    recent = history[
        history[
            "transaction_timestamp"
        ]
        >= recent_start
    ]

    prior = history[
        (
            history[
                "transaction_timestamp"
            ]
            >= prior_start
        )
        &
        (
            history[
                "transaction_timestamp"
            ]
            < recent_start
        )
    ]

    recent_spend = float(
        recent.loc[
            recent[
                "amount"
            ] < 0,
            "amount",
        ]
        .abs()
        .sum()
    )

    prior_spend = float(
        prior.loc[
            prior[
                "amount"
            ] < 0,
            "amount",
        ]
        .abs()
        .sum()
    )

    if prior_spend <= 0:
        return 0.0

    acceleration = (
        recent_spend
        - prior_spend
    ) / prior_spend

    return round(
        float(
            np.clip(
                acceleration,
                -5.0,
                5.0,
            )
        ),
        4,
    )


def calculate_transaction_velocity(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
) -> dict:
    """
    Calculate short-window transaction
    activity.
    """

    application_timestamp = (
        pd.Timestamp(
            application_timestamp
        )
    )

    if history.empty:
        return {
            "transaction_count_10m": 0,
            "transaction_count_1h": 0,
            "transaction_count_24h": 0,
            "unique_merchants_1h": 0,
            "unique_merchants_24h": 0,
        }

    cutoff_10m = (
        application_timestamp
        - pd.Timedelta(
            minutes=10
        )
    )

    cutoff_1h = (
        application_timestamp
        - pd.Timedelta(
            hours=1
        )
    )

    cutoff_24h = (
        application_timestamp
        - pd.Timedelta(
            hours=24
        )
    )

    tx_10m = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_10m
    ]

    tx_1h = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_1h
    ]

    tx_24h = history[
        history[
            "transaction_timestamp"
        ]
        >= cutoff_24h
    ]

    return {
        "transaction_count_10m":
            int(
                len(
                    tx_10m
                )
            ),

        "transaction_count_1h":
            int(
                len(
                    tx_1h
                )
            ),

        "transaction_count_24h":
            int(
                len(
                    tx_24h
                )
            ),

        "unique_merchants_1h":
            int(
                tx_1h[
                    "merchant_name"
                ].nunique()
            ),

        "unique_merchants_24h":
            int(
                tx_24h[
                    "merchant_name"
                ].nunique()
            ),
    }


def calculate_bnpl_burden(
    bnpl_payment_amount_90d,
    income_total_90d,
):
    """
    Calculate BNPL repayments relative
    to observed income.
    """

    if (
        is_missing(
            bnpl_payment_amount_90d
        )
        or is_missing(
            income_total_90d
        )
    ):
        return pd.NA

    burden = safe_divide(
        bnpl_payment_amount_90d,
        income_total_90d,
        default=0.0,
    )

    return round(
        float(
            np.clip(
                burden,
                0.0,
                5.0,
            )
        ),
        4,
    )


def calculate_income_stability_score(
    income_cv_90d,
    income_count_90d,
):
    """
    Convert income variability and
    frequency into a 0-1 score.

    Higher values represent more stable
    observed income.
    """

    if (
        is_missing(
            income_cv_90d
        )
        or is_missing(
            income_count_90d
        )
    ):
        return pd.NA

    income_cv_90d = float(
        income_cv_90d
    )

    income_count_90d = float(
        income_count_90d
    )

    variability_component = (
        1.0
        / (
            1.0
            + max(
                income_cv_90d,
                0.0,
            )
        )
    )

    frequency_component = min(
        income_count_90d
        / 6,
        1.0,
    )

    stability = (
        0.7
        * variability_component
        +
        0.3
        * frequency_component
    )

    return round(
        float(
            np.clip(
                stability,
                0.0,
                1.0,
            )
        ),
        4,
    )


def calculate_data_confidence_score(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
    bank_data_available: int,
) -> float:
    """
    Estimate confidence in available
    bank telemetry.

    Balance availability contributes to
    confidence but is not required.
    """

    if bank_data_available == 0:
        return 0.0

    if history.empty:
        return 0.0

    application_timestamp = (
        pd.Timestamp(
            application_timestamp
        )
    )

    earliest_timestamp = (
        pd.Timestamp(
            history[
                "transaction_timestamp"
            ].min()
        )
    )

    history_days = max(
        (
            application_timestamp
            - earliest_timestamp
        ).days,
        0,
    )

    history_component = min(
        history_days
        / 90,
        1.0,
    )

    transaction_component = min(
        len(
            history
        )
        / 100,
        1.0,
    )

    if (
        "transaction_type"
        in history.columns
    ):
        income_detected = int(
            (
                history[
                    "transaction_type"
                ]
                == "income"
            ).any()
        )
    else:
        income_detected = 0

    if (
        "balance_after_transaction"
        in history.columns
    ):
        balance_available = int(
            history[
                "balance_after_transaction"
            ]
            .notna()
            .any()
        )
    else:
        balance_available = 0

    confidence = (
        0.35
        * history_component
        +
        0.25
        * transaction_component
        +
        0.25
        * income_detected
        +
        0.15
        * balance_available
    )

    return round(
        float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        ),
        4,
    )


def calculate_advanced_features(
    history: pd.DataFrame,
    application_timestamp: pd.Timestamp,
    requested_amount: float,
    bank_data_available: int,
    base_features: dict,
) -> dict:
    """
    Build advanced behavioral features
    for one application.

    Missing source data remains missing
    rather than being fabricated.
    """

    velocity_features = (
        calculate_transaction_velocity(
            history,
            application_timestamp,
        )
    )

    advanced = {
        "loan_to_income_ratio":
            calculate_loan_to_income_ratio(
                requested_amount,
                base_features.get(
                    "income_total_90d"
                ),
            ),

        "requested_amount_to_balance":
            calculate_requested_amount_to_balance(
                requested_amount,
                base_features.get(
                    "current_balance"
                ),
            ),

        "paycheck_proximity_days":
            calculate_paycheck_proximity(
                base_features.get(
                    "days_since_last_income"
                )
            ),

        "discretionary_spend_ratio":
            calculate_discretionary_spend_ratio(
                base_features.get(
                    "discretionary_spend_30d"
                ),
                base_features.get(
                    "spend_total_30d"
                ),
            ),

        "cash_buffer_days":
            calculate_cash_buffer_days(
                base_features.get(
                    "current_balance"
                ),
                base_features.get(
                    "spend_total_30d"
                ),
            ),

        "cash_buffer_velocity":
            calculate_cash_buffer_velocity(
                history,
                application_timestamp,
            ),

        "spend_acceleration":
            calculate_spend_acceleration(
                history,
                application_timestamp,
            ),

        "bnpl_payment_burden":
            calculate_bnpl_burden(
                base_features.get(
                    "bnpl_payment_amount_90d"
                ),
                base_features.get(
                    "income_total_90d"
                ),
            ),

        "income_stability_score":
            calculate_income_stability_score(
                base_features.get(
                    "income_cv_90d"
                ),
                base_features.get(
                    "income_count_90d"
                ),
            ),

        "data_confidence_score":
            calculate_data_confidence_score(
                history,
                application_timestamp,
                bank_data_available,
            ),
    }

    advanced.update(
        velocity_features
    )

    return advanced