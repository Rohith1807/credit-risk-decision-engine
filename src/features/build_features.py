import pandas as pd

from src.features.point_in_time import (
    get_transaction_history,
)

from src.features.income_features import (
    calculate_income_features,
)

from src.features.balance_features import (
    calculate_balance_features,
)

from src.features.transaction_features import (
    calculate_transaction_features,
)


def build_application_features(
    application: dict,
    transactions: pd.DataFrame,
) -> dict:
    """
    Build a leakage-safe feature vector for one application.
    """

    user_id = application[
        "user_id"
    ]

    application_timestamp = pd.Timestamp(
        application[
            "application_timestamp"
        ]
    )

    history = get_transaction_history(
        transactions=transactions,
        user_id=user_id,
        application_timestamp=
            application_timestamp,
        lookback_days=180,
    )

    features = {
        "application_id":
            application[
                "application_id"
            ],

        "user_id":
            user_id,

        "application_timestamp":
            application_timestamp,

        "requested_amount":
            float(
                application[
                    "requested_amount"
                ]
            ),

        "merchant_category":
            application[
                "merchant_category"
            ],

        "device_type":
            application[
                "device_type"
            ],

        "channel":
            application[
                "channel"
            ],

        "bank_data_available":
            int(
                application[
                    "bank_data_available"
                ]
            ),
    }

    features.update(
        calculate_income_features(
            history,
            application_timestamp,
        )
    )

    features.update(
        calculate_balance_features(
            history,
            application_timestamp,
        )
    )

    features.update(
        calculate_transaction_features(
            history,
            application_timestamp,
        )
    )

    return features

def build_feature_dataset(
    applications: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature rows for every credit application.
    """

    records = []

    for application in applications.to_dict(
        orient="records"
    ):

        feature_row = (
            build_application_features(
                application=application,
                transactions=transactions,
            )
        )

        records.append(
            feature_row
        )

    return pd.DataFrame(
        records
    )

if __name__ == "__main__":
    applications = pd.read_parquet(
        "data/raw/applications.parquet"
    )

    transactions = pd.read_parquet(
        "data/raw/transactions.parquet"
    )

    features = build_feature_dataset(
        applications=applications,
        transactions=transactions,
    )

    features.to_parquet(
        "data/processed/application_features.parquet",
        index=False,
    )

    print(
        f"Feature rows generated: "
        f"{len(features):,}"
    )

    print(
        f"Feature columns generated: "
        f"{len(features.columns):,}"
    )

    print(
        features.head()
    )