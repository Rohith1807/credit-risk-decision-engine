import pandas as pd


def time_based_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split applications chronologically into train,
    validation, and test datasets.
    """

    dataset = dataset.sort_values(
        "application_timestamp"
    ).reset_index(
        drop=True
    )

    total_rows = len(dataset)

    train_end = int(
        total_rows
        * train_fraction
    )

    validation_end = int(
        total_rows
        * (
            train_fraction
            + validation_fraction
        )
    )

    train = dataset.iloc[
        :train_end
    ].copy()

    validation = dataset.iloc[
        train_end:
        validation_end
    ].copy()

    test = dataset.iloc[
        validation_end:
    ].copy()

    return (
        train,
        validation,
        test,
    )