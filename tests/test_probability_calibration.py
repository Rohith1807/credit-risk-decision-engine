import numpy as np
import pandas as pd

from src.models.calibration_split import (
    split_fit_calibration,
)
from src.models.probability_calibration import (
    IsotonicCalibrator,
    SigmoidCalibrator,
)


def test_temporal_calibration_split():

    dataframe = pd.DataFrame(
        {
            "application_timestamp":
                pd.date_range(
                    "2026-01-01",
                    periods=10,
                ),
            "default_status":
                [0] * 10,
        }
    )

    fit, calibration = (
        split_fit_calibration(
            dataframe,
            calibration_fraction=0.2,
        )
    )

    assert len(fit) == 8
    assert len(calibration) == 2

    assert (
        fit[
            "application_timestamp"
        ].max()
        <=
        calibration[
            "application_timestamp"
        ].min()
    )


def test_sigmoid_calibration():

    probabilities = np.array(
        [
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.40,
        ]
    )

    y = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
        ]
    )

    calibrator = (
        SigmoidCalibrator()
        .fit(
            probabilities,
            y,
        )
    )

    calibrated = (
        calibrator.predict(
            probabilities
        )
    )

    assert (
        (
            calibrated >= 0
        )
        &
        (
            calibrated <= 1
        )
    ).all()


def test_isotonic_calibration():

    probabilities = np.array(
        [
            0.01,
            0.03,
            0.10,
            0.20,
            0.40,
            0.60,
        ]
    )

    y = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    calibrator = (
        IsotonicCalibrator()
        .fit(
            probabilities,
            y,
        )
    )

    calibrated = (
        calibrator.predict(
            probabilities
        )
    )

    assert (
        (
            calibrated >= 0
        )
        &
        (
            calibrated <= 1
        )
    ).all()