from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.feature_schema import EXCLUDED_V1_FEATURES
from src.models.metrics import calculate_classification_metrics
from src.models.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "default_status"

NON_MODEL_COLUMNS = [
    "application_id",
    "user_id",
    "application_timestamp",
    TARGET,
]


def prepare_xy(dataframe: pd.DataFrame):
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

    y = dataframe[TARGET].astype(int)

    return X, y


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
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
                model,
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

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = calculate_classification_metrics(
        y_true=y_validation,
        probabilities=probabilities,
        predictions=predictions,
    )

    print(
        f"\n=== {name} ==="
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    print(
        "mean_predicted_pd:",
        round(
            probabilities.mean(),
            4,
        ),
    )

    print(
        "actual_default_rate:",
        round(
            y_validation.mean(),
            4,
        ),
    )

    return {
        "name": name,
        "pipeline": pipeline,
        "probabilities": probabilities,
        "metrics": metrics,
    }


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

    evaluate_model(
        name="Weighted Logistic Regression",
        model=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    evaluate_model(
        name="Unweighted Logistic Regression",
        model=LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42,
        ),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )


if __name__ == "__main__":
    main()