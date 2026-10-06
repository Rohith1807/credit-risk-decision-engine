import pandas as pd


def split_fit_calibration(
    dataframe: pd.DataFrame,
    calibration_fraction: float = 0.20,
):
    """
    Split training data chronologically.

    Earlier observations -> model fitting
    Later observations   -> probability calibration
    """

    ordered = dataframe.sort_values(
        "application_timestamp"
    ).reset_index(drop=True)

    split_index = int(
        len(ordered)
        * (1 - calibration_fraction)
    )

    fit_data = ordered.iloc[
        :split_index
    ].copy()

    calibration_data = ordered.iloc[
        split_index:
    ].copy()

    return fit_data, calibration_data