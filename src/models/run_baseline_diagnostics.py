from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.feature_schema import EXCLUDED_V1_FEATURES
from src.models.preprocessing import build_preprocessor
from src.models.threshold_analysis import evaluate_thresholds
from src.models.calibration_analysis import calibration_table
from src.models.inspect_coefficients import extract_logistic_coefficients


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TARGET = "default_status"

NON_MODEL_COLUMNS = [
    "application_id",
    "user_id",
    "application_timestamp",
    TARGET,
]


def prepare_xy(
    dataframe,
):
    drop_columns = [
        column
        for column in (
            NON_MODEL_COLUMNS
            + EXCLUDED_V1_FEATURES
        )
        if column in dataframe.columns
    ]

    X = dataframe.drop(
        columns=drop_columns
    )

    y = dataframe[
        TARGET
    ].astype(int)

    return X, y


def main():

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_DIR
        / "validation.parquet"
    )

    X_train, y_train = prepare_xy(
        train
    )

    X_validation, y_validation = (
        prepare_xy(
            validation
        )
    )

    preprocessor = build_preprocessor(
        X_train
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
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    print(
        "\n=== THRESHOLD ANALYSIS ==="
    )

    print(
        evaluate_thresholds(
            y_validation,
            probabilities,
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== CALIBRATION TABLE ==="
    )

    print(
        calibration_table(
            y_validation,
            probabilities,
            bins=5,
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== TOP COEFFICIENTS ==="
    )

    coefficients = (
        extract_logistic_coefficients(
            pipeline
        )
    )

    print(
        coefficients.head(
            20
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()