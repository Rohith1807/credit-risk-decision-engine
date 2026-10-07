import pandas as pd


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """
    Simple PSI implementation for numeric features.
    """

    expected = pd.to_numeric(
        expected,
        errors="coerce",
    ).dropna()

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    ).dropna()

    if (
        expected.empty
        or actual.empty
    ):
        return 0.0

    breaks = expected.quantile(
        [
            i / bins
            for i in range(
                bins + 1
            )
        ]
    ).drop_duplicates()

    if len(breaks) < 3:
        return 0.0

    expected_bins = pd.cut(
        expected,
        bins=breaks,
        include_lowest=True,
    )

    actual_bins = pd.cut(
        actual,
        bins=breaks,
        include_lowest=True,
    )

    expected_dist = (
        expected_bins
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    actual_dist = (
        actual_bins
        .value_counts(
            normalize=True
        )
        .reindex(
            expected_dist.index,
            fill_value=0.0,
        )
    )

    epsilon = 1e-6

    expected_dist = (
        expected_dist + epsilon
    )

    actual_dist = (
        actual_dist + epsilon
    )

    psi = (
        (
            actual_dist
            - expected_dist
        )
        *
        (
            actual_dist
            / expected_dist
        ).apply(
            lambda value:
                __import__(
                    "math"
                ).log(value)
        )
    ).sum()

    return float(psi)