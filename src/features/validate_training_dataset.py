import pandas as pd

from src.features.build_training_dataset import (
    build_training_dataset,
    get_model_columns,
)


def validate_training_dataset():
    dataset = build_training_dataset()

    model_columns = get_model_columns(
        dataset
    )

    print("\n=== DATASET SHAPE ===")
    print(dataset.shape)

    print("\n=== TARGET DISTRIBUTION ===")
    print(
        dataset[
            "default_status"
        ].value_counts(
            normalize=True
        )
    )

    print("\n=== MISSING VALUES ===")

    missing = (
        dataset[
            model_columns
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[
            missing > 0
        ]
    )

    print("\n=== DATA TYPES ===")

    print(
        dataset[
            model_columns
        ].dtypes
    )

    print("\n=== DUPLICATES ===")

    print(
        "Duplicate application IDs:",
        dataset[
            "application_id"
        ].duplicated().sum(),
    )


if __name__ == "__main__":
    validate_training_dataset()