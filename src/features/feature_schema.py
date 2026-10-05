IDENTIFIER_COLUMNS = [
    "application_id",
    "user_id",
]

TIMESTAMP_COLUMNS = [
    "application_timestamp",
]

TARGET_COLUMN = (
    "default_status"
)

CATEGORICAL_FEATURES = [
    "merchant_category",
    "device_type",
    "channel",
]

BINARY_FEATURES = [
    "bank_data_available",
]

EXCLUDED_V1_FEATURES = [
    "transaction_count_10m",
    "transaction_count_1h",
    "unique_merchants_1h",
]

def get_numerical_features(
    dataset_columns: list[str],
) -> list[str]:
    excluded = set(
        IDENTIFIER_COLUMNS
        + TIMESTAMP_COLUMNS
        + CATEGORICAL_FEATURES
        + BINARY_FEATURES
        + [TARGET_COLUMN]
    )

    return [
        column
        for column in dataset_columns
        if column not in excluded
    ]