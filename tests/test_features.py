import pandas as pd

from src.features.build_features import (
    build_application_features,
)


def test_feature_builder_excludes_future_transactions():
    application = {
        "application_id": "APP1",
        "user_id": "U1",
        "application_timestamp":
            pd.Timestamp(
                "2026-01-10"
            ),
        "requested_amount": 200,
        "merchant_category":
            "electronics",
        "device_type": "mobile",
        "channel": "checkout",
        "bank_data_available": 1,
    }

    transactions = pd.DataFrame(
        {
            "user_id": [
                "U1",
                "U1",
            ],
            "account_id": [
                "A1",
                "A1",
            ],
            "transaction_timestamp":
                pd.to_datetime(
                    [
                        "2026-01-05",
                        "2026-01-20",
                    ]
                ),
            "amount": [
                1000,
                999999,
            ],
            "transaction_type": [
                "income",
                "income",
            ],
            "merchant_name": [
                "Employer Payroll",
                "Future Payroll",
            ],
            "merchant_category": [
                "income",
                "income",
            ],
            "balance_after_transaction": [
                1000,
                999999,
            ],
        }
    )

    features = (
        build_application_features(
            application,
            transactions,
        )
    )

    assert (
        features[
            "income_total_90d"
        ]
        == 1000
    )

def test_feature_values_non_negative_where_expected():
    application = {
        "application_id": "APP1",
        "user_id": "U1",
        "application_timestamp":
            pd.Timestamp(
                "2026-01-10"
            ),
        "requested_amount": 200,
        "merchant_category":
            "electronics",
        "device_type": "mobile",
        "channel": "checkout",
        "bank_data_available": 1,
    }

    transactions = pd.DataFrame(
        {
            "user_id": ["U1"],
            "account_id": ["A1"],
            "transaction_timestamp":
                pd.to_datetime(
                    ["2026-01-05"]
                ),
            "amount": [1000],
            "transaction_type":
                ["income"],
            "merchant_name":
                ["Employer Payroll"],
            "merchant_category":
                ["income"],
            "balance_after_transaction":
                [1000],
        }
    )

    features = (
        build_application_features(
            application,
            transactions,
        )
    )

    assert (
        features[
            "income_total_90d"
        ]
        >= 0
    )

    assert (
        features[
            "transaction_count_30d"
        ]
        >= 0
    )