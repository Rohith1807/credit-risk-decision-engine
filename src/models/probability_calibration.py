import numpy as np

from sklearn.isotonic import (
    IsotonicRegression,
)
from sklearn.linear_model import (
    LogisticRegression,
)


EPSILON = 1e-6


def probability_to_logit(
    probabilities,
):
    probabilities = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON,
    )

    return np.log(
        probabilities
        / (1 - probabilities)
    )


class SigmoidCalibrator:
    """
    Platt-style probability calibration.
    """

    def __init__(self):

        self.model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
        )

    def fit(
        self,
        probabilities,
        y_true,
    ):

        logits = (
            probability_to_logit(
                probabilities
            )
            .reshape(-1, 1)
        )

        self.model.fit(
            logits,
            y_true,
        )

        return self

    def predict(
        self,
        probabilities,
    ):

        logits = (
            probability_to_logit(
                probabilities
            )
            .reshape(-1, 1)
        )

        return self.model.predict_proba(
            logits
        )[:, 1]


class IsotonicCalibrator:

    def __init__(self):

        self.model = IsotonicRegression(
            out_of_bounds="clip"
        )

    def fit(
        self,
        probabilities,
        y_true,
    ):

        self.model.fit(
            probabilities,
            y_true,
        )

        return self

    def predict(
        self,
        probabilities,
    ):

        return self.model.predict(
            probabilities
        )