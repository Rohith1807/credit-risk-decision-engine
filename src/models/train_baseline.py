from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.feature_schema import (
    EXCLUDED_V1_FEATURES,
)
from src.models.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)
from src.models.preprocessing import (
    build_preprocessor,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

TARGET = "default_status"


NON_MODEL_COLUMNS = [
    "application_id",
    "user_id",
    "application_timestamp",
    TARGET,
]


def prepare_xy(
    dataframe: pd.DataFrame,
):
    """
    Separate predictors and target.
    """

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


def train_baseline():
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

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
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

    mlflow.set_experiment(
        "credit-risk-baseline"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model_type",
            "logistic_regression",
        )

        mlflow.log_param(
            "class_weight",
            "balanced",
        )

        mlflow.log_param(
            "max_iter",
            2000,
        )

        mlflow.log_param(
            "training_rows",
            len(train),
        )

        mlflow.log_param(
            "validation_rows",
            len(validation),
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

        metrics = (
            calculate_classification_metrics(
                y_true=y_validation,
                probabilities=
                    probabilities,
                predictions=
                    predictions,
            )
        )

        confusion = (
            calculate_confusion_matrix(
                y_true=y_validation,
                predictions=
                    predictions,
            )
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_metrics(
            confusion
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        ARTIFACT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            ARTIFACT_DIR
            / "baseline_logistic.joblib"
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        print(
            "\n=== BASELINE LOGISTIC REGRESSION ==="
        )

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"Validation default rate: "
            f"{y_validation.mean():.2%}"
        )

        print(
            "\n=== METRICS ==="
        )

        for key, value in metrics.items():
            print(
                f"{key}: "
                f"{value:.4f}"
            )

        print(
            "\n=== CONFUSION MATRIX ==="
        )

        for key, value in confusion.items():
            print(
                f"{key}: {value}"
            )

        print(
            f"\nModel saved to: "
            f"{model_path}"
        )


if __name__ == "__main__":
    train_baseline()