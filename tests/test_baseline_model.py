import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline

from src.models.preprocessing import (
    build_preprocessor,
)


def test_preprocessor_builds():
    data = pd.DataFrame(
        {
            "requested_amount": [
                100.0,
                200.0,
            ],
            "merchant_category": [
                "electronics",
                "apparel",
            ],
            "device_type": [
                "mobile",
                "desktop",
            ],
            "channel": [
                "checkout",
                "web",
            ],
            "bank_data_available": [
                1,
                0,
            ],
        }
    )

    preprocessor = (
        build_preprocessor(
            data
        )
    )

    assert preprocessor is not None


def test_logistic_pipeline_fits():
    data = pd.DataFrame(
        {
            "requested_amount": [
                100.0,
                200.0,
                700.0,
                900.0,
            ],
            "merchant_category": [
                "electronics",
                "apparel",
                "travel",
                "electronics",
            ],
            "device_type": [
                "mobile",
                "desktop",
                "mobile",
                "desktop",
            ],
            "channel": [
                "checkout",
                "web",
                "mobile_app",
                "checkout",
            ],
            "bank_data_available": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    target = [
        0,
        0,
        1,
        1,
    ]

    preprocessor = (
        build_preprocessor(
            data
        )
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    pipeline.fit(
        data,
        target,
    )

    probabilities = (
        pipeline.predict_proba(
            data
        )[:, 1]
    )

    assert len(
        probabilities
    ) == len(data)

    assert (
        (
            probabilities >= 0
        )
        &
        (
            probabilities <= 1
        )
    ).all()