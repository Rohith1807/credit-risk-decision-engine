import numpy as np
import pandas as pd

from src.features.build_training_dataset import (
    build_training_dataset,
    get_model_columns,
)


def analyze_correlations():
    dataset = build_training_dataset()

    model_columns = get_model_columns(
        dataset
    )

    numeric = (
        dataset[model_columns]
        .select_dtypes(
            include=np.number
        )
    )

    correlation = (
        numeric.corr()
        .abs()
    )

    pairs = []

    columns = correlation.columns

    for i in range(
        len(columns)
    ):
        for j in range(
            i + 1,
            len(columns)
        ):

            value = correlation.iloc[
                i,
                j,
            ]

            if value >= 0.90:
                pairs.append(
                    {
                        "feature_1":
                            columns[i],
                        "feature_2":
                            columns[j],
                        "correlation":
                            round(
                                float(value),
                                4,
                            ),
                    }
                )

    result = pd.DataFrame(
        pairs
    )

    if result.empty:
        print(
            "No feature pairs with "
            "|correlation| >= 0.90"
        )
    else:
        print(
            result.sort_values(
                "correlation",
                ascending=False,
            ).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    analyze_correlations()