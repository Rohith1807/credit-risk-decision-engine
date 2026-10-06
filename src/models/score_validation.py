from pathlib import Path

import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.features.feature_schema import (
    EXCLUDED_V1_FEATURES,
)
from src.models.tree_preprocessing import (
    build_tree_preprocessor,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "validation_scored.parquet"
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


def build_lightgbm(
    X_train: pd.DataFrame,
):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=7,
                    min_child_samples=40,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    scale_pos_weight=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


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

    X_validation, _ = prepare_xy(
        validation
    )

    pipeline = build_lightgbm(
        X_train
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

    scored = validation.copy()

    scored[
        "probability_default"
    ] = probabilities

    scored.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Scored rows:",
        len(scored),
    )

    print(
        "Mean predicted PD:",
        round(
            scored[
                "probability_default"
            ].mean(),
            4,
        ),
    )

    print(
        "Observed default rate:",
        round(
            scored[
                TARGET
            ].mean(),
            4,
        ),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()