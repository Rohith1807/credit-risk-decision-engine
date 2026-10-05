import pandas as pd


def calibration_table(
    y_true,
    probabilities,
    bins=10,
) -> pd.DataFrame:

    dataframe = pd.DataFrame(
        {
            "actual": y_true,
            "predicted_pd":
                probabilities,
        }
    )

    dataframe[
        "pd_bucket"
    ] = pd.qcut(
        dataframe[
            "predicted_pd"
        ],
        q=bins,
        duplicates="drop",
    )

    result = (
        dataframe.groupby(
            "pd_bucket",
            observed=True,
        )
        .agg(
            count=(
                "actual",
                "size",
            ),
            mean_predicted_pd=(
                "predicted_pd",
                "mean",
            ),
            actual_default_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    return result