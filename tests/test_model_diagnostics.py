import numpy as np

from src.models.threshold_analysis import (
    evaluate_thresholds,
)
from src.models.calibration_analysis import (
    calibration_table,
)


def test_threshold_analysis():

    y_true = np.array(
        [0, 0, 0, 1, 1]
    )

    probabilities = np.array(
        [
            0.10,
            0.20,
            0.40,
            0.60,
            0.90,
        ]
    )

    result = evaluate_thresholds(
        y_true,
        probabilities,
    )

    assert len(result) > 0

    assert (
        result[
            "threshold"
        ]
        .between(
            0,
            1,
        )
        .all()
    )


def test_calibration_table():

    y_true = np.array(
        [
            0,
            0,
            1,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.30,
            0.40,
            0.70,
            0.90,
        ]
    )

    result = calibration_table(
        y_true,
        probabilities,
        bins=3,
    )

    assert (
        result[
            "count"
        ].sum()
        == len(y_true)
    )