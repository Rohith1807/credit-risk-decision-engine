import pandas as pd

from src.features.point_in_time import (
    get_transaction_history,
)


def test_future_transactions_are_excluded():
    transactions = pd.DataFrame(
        {
            "user_id": [
                "U1",
                "U1",
                "U1",
            ],
            "transaction_timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-20",
                ]
            ),
            "amount": [
                100,
                -50,
                -75,
            ],
        }
    )

    application_timestamp = pd.Timestamp(
        "2026-01-10"
    )

    history = get_transaction_history(
        transactions=transactions,
        user_id="U1",
        application_timestamp=application_timestamp,
    )

    assert len(history) == 2

    assert (
        history[
            "transaction_timestamp"
        ]
        <= application_timestamp
    ).all()

def test_lookback_window_is_respected():
    transactions = pd.DataFrame(
        {
            "user_id": [
                "U1",
                "U1",
                "U1",
            ],
            "transaction_timestamp": pd.to_datetime(
                [
                    "2025-10-01",
                    "2025-12-20",
                    "2026-01-05",
                ]
            ),
            "amount": [
                100,
                200,
                300,
            ],
        }
    )

    history = get_transaction_history(
        transactions=transactions,
        user_id="U1",
        application_timestamp=pd.Timestamp(
            "2026-01-10"
        ),
        lookback_days=30,
    )

    assert len(history) == 2