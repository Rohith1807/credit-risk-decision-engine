import numpy as np
import pandas as pd


def safe_coefficient_of_variation(
    values: pd.Series,
) -> float:

    if len(values) < 2:
        return 0.0

    mean = values.mean()

    if mean <= 0:
        return 0.0

    return float(
        values.std() / mean
    )

def calculate_behavioral_risk_metrics(
    user_id: str,
    application_timestamp: pd.Timestamp,
    transactions: pd.DataFrame,
) -> dict:
    """
    Calculate synthetic risk indicators using only transaction
    history available at or before the application timestamp.
    """

    history = transactions[
        (transactions["user_id"] == user_id)
        &
        (
            transactions[
                "transaction_timestamp"
            ]
            <= application_timestamp
        )
    ].copy()

    if history.empty:
        return {
            "negative_balance_rate": 0.0,
            "income_variability": 1.0,
            "recent_income": 0.0,
            "recent_spend": 0.0,
            "bnpl_payment_count": 0,
            "cash_buffer": 0.0,
        }

    application_timestamp = pd.Timestamp(
        application_timestamp
    )

    last_30_days = history[
        history[
            "transaction_timestamp"
        ]
        >= application_timestamp
        - pd.Timedelta(days=30)
    ]

    last_90_days = history[
        history[
            "transaction_timestamp"
        ]
        >= application_timestamp
        - pd.Timedelta(days=90)
    ]

    income = last_90_days[
        last_90_days[
            "transaction_type"
        ]
        == "income"
    ]["amount"]

    expenses_30d = last_30_days[
        last_30_days["amount"] < 0
    ]["amount"].abs()

    negative_balance_rate = float(
        (
            history[
                "balance_after_transaction"
            ]
            < 0
        ).mean()
    )

    income_variability = (
        safe_coefficient_of_variation(
            income
        )
    )

    recent_income = float(
        income.sum()
    )

    recent_spend = float(
        expenses_30d.sum()
    )

    bnpl_payment_count = int(
        (
            last_90_days[
                "transaction_type"
            ]
            == "bnpl_payment"
        ).sum()
    )

    cash_buffer = float(
        history[
            "balance_after_transaction"
        ].tail(10).median()
    )

    return {
        "negative_balance_rate":
            negative_balance_rate,

        "income_variability":
            income_variability,

        "recent_income":
            recent_income,

        "recent_spend":
            recent_spend,

        "bnpl_payment_count":
            bnpl_payment_count,

        "cash_buffer":
            cash_buffer,
    }

def sigmoid(
    value: float,
) -> float:
    return float(
        1
        / (
            1
            + np.exp(-value)
        )
    )

def calculate_latent_default_probability(
    requested_amount: float,
    employment_type: str,
    behavioral_metrics: dict,
) -> float:
    """
    Generate the hidden synthetic probability of default.

    This function creates the ground-truth mechanism used only
    for synthetic outcome generation.

    It will NOT be available to the trained ML model.
    """

    recent_income = max(
        behavioral_metrics[
            "recent_income"
        ],
        1.0,
    )

    requested_to_income = (
        requested_amount
        / recent_income
    )

    risk_score = -5.2

    # Negative balance behavior
    risk_score += (
        3.0
        * behavioral_metrics[
            "negative_balance_rate"
        ]
    )

    # Unstable income
    risk_score += (
        0.8
        * min(
            behavioral_metrics[
                "income_variability"
            ],
            2.0,
        )
    )

    # High requested exposure relative to income
    risk_score += (
        2.2
        * min(
            requested_to_income,
            1.5,
        )
    )

    # Existing observed BNPL repayments
    risk_score += (
        0.10
        * min(
            behavioral_metrics[
                "bnpl_payment_count"
            ],
            10,
        )
    )

    # Very weak cash buffer
    cash_buffer = (
        behavioral_metrics[
            "cash_buffer"
        ]
    )

    if cash_buffer < 100:
        risk_score += 0.70

    elif cash_buffer < 300:
        risk_score += 0.35

    # Employment-specific uncertainty.
    # This does NOT mean these groups are inherently riskier.
    # It only simulates different cash-flow volatility patterns.
    if employment_type in {
        "gig_worker",
        "self_employed",
    }:
        risk_score += 0.12

    probability = sigmoid(
        risk_score
    )

    return float(
        np.clip(
            probability,
            0.002,
            0.35,
        )
    )