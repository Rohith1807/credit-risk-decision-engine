import pandas as pd
import numpy as np

from src.features.build_training_dataset import (
    build_training_dataset,
    get_model_columns,
)


def analyze_numeric_signal():
    dataset = build_training_dataset()

    model_columns = get_model_columns(
        dataset
    )

    numeric_features = (
        dataset[model_columns]
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    results = []

    for feature in numeric_features:

        non_missing = dataset[
            [
                feature,
                "default_status",
            ]
        ].dropna()

        if (
            len(non_missing) < 20
            or non_missing[
                feature
            ].nunique() < 4
        ):
            continue

        try:
            non_missing["bucket"] = pd.qcut(
                non_missing[feature],
                q=5,
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            non_missing.groupby(
                "bucket",
                observed=True,
            )["default_status"]
            .mean()
        )

        results.append(
            {
                "feature": feature,
                "min_bucket_default":
                    grouped.min(),
                "max_bucket_default":
                    grouped.max(),
                "spread":
                    grouped.max()
                    - grouped.min(),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        print(
            "No numerical feature signal "
            "could be evaluated."
        )
        return

    result_df = (
        result_df.sort_values(
            "spread",
            ascending=False,
        )
    )

    print(
        result_df.head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    analyze_numeric_signal()