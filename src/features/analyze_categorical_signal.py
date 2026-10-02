from src.features.build_training_dataset import (
    build_training_dataset,
)
from src.features.feature_schema import (
    CATEGORICAL_FEATURES,
)


def analyze_categorical_signal():
    dataset = build_training_dataset()

    for feature in CATEGORICAL_FEATURES:

        print(
            f"\n=== {feature.upper()} ==="
        )

        summary = (
            dataset.groupby(
                feature,
                dropna=False,
            )["default_status"]
            .agg(
                [
                    "count",
                    "mean",
                ]
            )
            .sort_values(
                "mean",
                ascending=False,
            )
        )

        print(summary)


if __name__ == "__main__":
    analyze_categorical_signal()