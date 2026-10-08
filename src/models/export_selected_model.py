import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.preprocessing import (
    build_preprocessor,
)
from src.models.train_baseline import (
    prepare_xy,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "credit_risk_model_v1.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "credit_risk_model_v1_metadata.json"
)


def build_final_model(
    X_train: pd.DataFrame,
) -> Pipeline:

    preprocessor = build_preprocessor(
        X_train
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
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


def main():

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    X_train, y_train = prepare_xy(
        train
    )

    pipeline = build_final_model(
        X_train
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    metadata = {
        "model_name":
            "Logistic Regression",

        "model_version":
            "v1.0",

        "calibration":
            "raw",

        "class_weighting":
            "none",

        "training_rows":
            int(
                len(train)
            ),

        "training_default_rate":
            float(
                y_train.mean()
            ),

        "feature_columns":
            list(
                X_train.columns
            ),

        "validation_roc_auc":
            0.800055,

        "validation_pr_auc":
            0.204199,

        "final_test_roc_auc":
            0.7680,

        "final_test_pr_auc":
            0.1801,

        "final_test_brier_score":
            0.0371,

        "final_test_mean_pd":
            0.0413,

        "final_test_actual_default_rate":
            0.0422,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "Final model saved:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Metadata saved:"
    )

    print(
        METADATA_PATH
    )


if __name__ == "__main__":
    main()