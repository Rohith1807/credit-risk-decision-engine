from pathlib import Path

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


def build_logistic(
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

    pipeline = build_logistic(
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

    output_path = (
        PROCESSED_DIR
        / "validation_scored.parquet"
    )

    scored.to_parquet(
        output_path,
        index=False,
    )

    print(
        "=== FINAL VALIDATION SCORING ==="
    )

    print(
        f"Rows: {len(scored):,}"
    )

    print(
        f"Actual default rate: "
        f"{scored['default_status'].mean():.2%}"
    )

    print(
        f"Mean predicted PD: "
        f"{scored['probability_default'].mean():.2%}"
    )

    print(
        f"Min PD: "
        f"{scored['probability_default'].min():.4%}"
    )

    print(
        f"Max PD: "
        f"{scored['probability_default'].max():.4%}"
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()