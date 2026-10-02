import pandas as pd
import numpy as np

from src.features.build_training_dataset import (
    build_training_dataset,
    get_model_columns,
)


def analyze_feature_quality():
    dataset = build_training_dataset()

    model_columns = get_model_columns(dataset)

    print("\n=== DATASET ===")
    print(f"Rows: {len(dataset):,}")
    print(f"Candidate features: {len(model_columns):,}")
    print(
        f"Default rate: "
        f"{dataset['default_status'].mean():.2%}"
    )

    print("\n=== MISSINGNESS ===")

    missing = (
        dataset[model_columns]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    print(
        missing[
            missing > 0
        ].to_string()
    )

    print("\n=== UNIQUE VALUE COUNTS ===")

    unique_counts = (
        dataset[model_columns]
        .nunique(dropna=False)
        .sort_values()
    )

    print(unique_counts.to_string())

    print("\n=== POSSIBLE LOW-VARIANCE FEATURES ===")

    low_variance = unique_counts[
        unique_counts <= 2
    ]

    print(low_variance.to_string())

    print("\n=== NUMERICAL SUMMARY ===")

    numeric_columns = (
        dataset[model_columns]
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    print(
        dataset[
            numeric_columns
        ]
        .describe()
        .T
        .to_string()
    )


if __name__ == "__main__":
    analyze_feature_quality()