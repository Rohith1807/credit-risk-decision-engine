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
from src.features.advanced_features import (
    calculate_advanced_features,
)


def build_application_features(
    application: dict,
    transactions: pd.DataFrame,
) -> dict:
    """
    Build a leakage-safe feature vector
    for one application.

    The transactions dataframe may contain
    either the full portfolio or only the
    current user's transactions.
    """

    user_id = application["user_id"]

    application_timestamp = pd.Timestamp(
        application["application_timestamp"]
    )

    history = get_transaction_history(
        transactions=transactions,
        user_id=user_id,
        application_timestamp=application_timestamp,
        lookback_days=180,
    )

    features = {
        "application_id":
            application["application_id"],

        "user_id":
            user_id,

        "application_timestamp":
            application_timestamp,

        "requested_amount":
            float(
                application["requested_amount"]
            ),

        "merchant_category":
            application["merchant_category"],

        "device_type":
            application["device_type"],

        "channel":
            application["channel"],

        "bank_data_available":
            int(
                application["bank_data_available"]
            ),
    }

    income_features = (
        calculate_income_features(
            history,
            application_timestamp,
        )
    )

    balance_features = (
        calculate_balance_features(
            history,
            application_timestamp,
        )
    )

    transaction_features = (
        calculate_transaction_features(
            history,
            application_timestamp,
        )
    )

    features.update(
        income_features
    )

    features.update(
        balance_features
    )

    features.update(
        transaction_features
    )

    advanced_base_features = {
        **income_features,
        **balance_features,
        **transaction_features,
    }

    advanced_features = (
        calculate_advanced_features(
            history=history,
            application_timestamp=
                application_timestamp,
            requested_amount=
                float(
                    application[
                        "requested_amount"
                    ]
                ),
            bank_data_available=
                int(
                    application[
                        "bank_data_available"
                    ]
                ),
            base_features=
                advanced_base_features,
        )
    )

    features.update(
        advanced_features
    )

    return features


def build_feature_dataset(
    applications: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature rows for every credit application.

    Optimization:
    Applications are processed user-by-user,
    so the full transaction table is not
    scanned separately for every application.
    """

    applications = applications.copy()
    transactions = transactions.copy()

    applications[
        "application_timestamp"
    ] = pd.to_datetime(
        applications[
            "application_timestamp"
        ]
    )

    transactions[
        "transaction_timestamp"
    ] = pd.to_datetime(
        transactions[
            "transaction_timestamp"
        ]
    )

    # Preserve original application ordering.
    applications["_original_order"] = range(
        len(applications)
    )

    # Sorting once improves repeated
    # point-in-time slicing.
    transactions = transactions.sort_values(
        [
            "user_id",
            "transaction_timestamp",
        ]
    )

    applications = applications.sort_values(
        [
            "user_id",
            "application_timestamp",
        ]
    )

    transaction_groups = (
        transactions.groupby(
            "user_id",
            sort=False,
        )
    )

    records = []

    processed = 0
    total = len(applications)

    empty_transactions = (
        transactions.iloc[0:0].copy()
    )

    for user_id, user_applications in (
        applications.groupby(
            "user_id",
            sort=False,
        )
    ):

        try:
            user_transactions = (
                transaction_groups.get_group(
                    user_id
                )
            )
        except KeyError:
            user_transactions = (
                empty_transactions
            )

        for application in (
            user_applications.to_dict(
                orient="records"
            )
        ):

            original_order = application.pop(
                "_original_order"
            )

            feature_row = (
                build_application_features(
                    application=application,
                    transactions=
                        user_transactions,
                )
            )

            feature_row[
                "_original_order"
            ] = original_order

            records.append(
                feature_row
            )

            processed += 1

            if (
                processed % 1000 == 0
                or processed == total
            ):
                print(
                    f"Processed "
                    f"{processed:,} / "
                    f"{total:,} applications"
                )

    features = pd.DataFrame.from_records(
        records
    )

    # Restore the exact original
    # application ordering.
    features = (
        features
        .sort_values(
            "_original_order"
        )
        .drop(
            columns=[
                "_original_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return features


if __name__ == "__main__":

    print(
        "Loading applications..."
    )

    applications = pd.read_parquet(
        "data/raw/applications.parquet"
    )

    print(
        f"Applications loaded: "
        f"{len(applications):,}"
    )

    print(
        "Loading transactions..."
    )

    transactions = pd.read_parquet(
        "data/raw/transactions.parquet"
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    print(
        "Building application features..."
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
        "\nFeature generation completed."
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