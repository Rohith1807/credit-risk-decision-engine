from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.feature_schema import (
    EXCLUDED_V1_FEATURES,
)
from src.models.imbalance import (
    calculate_scale_pos_weight,
)
from src.models.metrics import (
    calculate_classification_metrics,
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


def evaluate_variant(
    name,
    scale_pos_weight,
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    preprocessor = (
        build_tree_preprocessor(
            X_train
        )
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
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

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\n=== TOP FEATURE IMPORTANCE ==="
    )

    print(
        importance.head(20)
        .to_string(index=False)
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
            y_validation,
            probabilities,
            predictions,
        )
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

    with mlflow.start_run(
        run_name=name
    ):

        mlflow.log_param(
            "model_type",
            "xgboost",
        )

        mlflow.log_param(
            "scale_pos_weight",
            scale_pos_weight,
        )

        mlflow.log_param(
            "n_estimators",
            300,
        )

        mlflow.log_param(
            "max_depth",
            3,
        )

        mlflow.log_param(
            "learning_rate",
            0.05,
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_metric(
            "mean_predicted_pd",
            probabilities.mean(),
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    return metrics


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

    imbalance_ratio = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    print(
        "Training default rate:",
        round(
            y_train.mean(),
            4,
        ),
    )

    print(
        "Calculated scale_pos_weight:",
        round(
            imbalance_ratio,
            2,
        ),
    )

    mlflow.set_experiment(
        "credit-risk-challengers"
    )

    evaluate_variant(
        name="XGBoost Unweighted",
        scale_pos_weight=1.0,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    evaluate_variant(
        name="XGBoost Weighted",
        scale_pos_weight=imbalance_ratio,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )


if __name__ == "__main__":
    main()