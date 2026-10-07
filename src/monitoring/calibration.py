import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
)


def calculate_calibration_metrics(
    dataframe: pd.DataFrame,
) -> dict:

    required = {
        "default_status",
        "probability_default",
    }

    if not required.issubset(
        dataframe.columns
    ):
        return {}

    return {
        "actual_default_rate":
            float(
                dataframe[
                    "default_status"
                ].mean()
            ),

        "mean_predicted_pd":
            float(
                dataframe[
                    "probability_default"
                ].mean()
            ),

        "brier_score":
            float(
                brier_score_loss(
                    dataframe[
                        "default_status"
                    ],
                    dataframe[
                        "probability_default"
                    ],
                )
            ),
    }