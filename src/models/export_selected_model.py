import json
from pathlib import Path

import joblib
import pandas as pd

from src.models.score_validation import (
    build_lightgbm,
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
    / "selected_lightgbm.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "selected_lightgbm_metadata.json"
)


def main():

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    X_train, y_train = prepare_xy(
        train
    )

    pipeline = build_lightgbm(
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
            "LightGBM",

        "model_version":
            "v1.0",

        "calibration":
            "raw",

        "training_rows":
            len(train),

        "training_default_rate":
            float(
                y_train.mean()
            ),

        "feature_columns":
            list(
                X_train.columns
            ),
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
        "Model saved:",
        MODEL_PATH,
    )

    print(
        "Metadata saved:",
        METADATA_PATH,
    )


if __name__ == "__main__":
    main()