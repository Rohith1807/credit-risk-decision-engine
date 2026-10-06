from pathlib import Path

import mlflow
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline

from src.features.feature_schema import (
    EXCLUDED_V1_FEATURES,
)
from src.models.calibration_analysis import (
    calibration_table,
)
from src.models.calibration_split import (
    split_fit_calibration,
)
from src.models.metrics import (
    calculate_classification_metrics,
)
from src.models.preprocessing import (
    build_preprocessor,
)
from src.models.probability_calibration import (
    IsotonicCalibrator,
    SigmoidCalibrator,
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


def build_logistic(
    X,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X),
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


def build_lightgbm(
    X,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(
                    X
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


def evaluate_probabilities(
    model_name,
    calibration_method,
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = (
        calculate_classification_metrics(
            y_true,
            probabilities,
            predictions,
        )
    )

    result = {
        "model":
            model_name,
        "calibration":
            calibration_method,
        **metrics,
        "mean_predicted_pd":
            float(
                probabilities.mean()
            ),
        "actual_default_rate":
            float(
                y_true.mean()
            ),
    }

    return result


def evaluate_model(
    model_name,
    pipeline,
    X_fit,
    y_fit,
    X_calibration,
    y_calibration,
    X_validation,
    y_validation,
):

    pipeline.fit(
        X_fit,
        y_fit,
    )

    calibration_probabilities = (
        pipeline.predict_proba(
            X_calibration
        )[:, 1]
    )

    validation_raw = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    sigmoid = (
        SigmoidCalibrator()
        .fit(
            calibration_probabilities,
            y_calibration,
        )
    )

    isotonic = (
        IsotonicCalibrator()
        .fit(
            calibration_probabilities,
            y_calibration,
        )
    )

    validation_sigmoid = (
        sigmoid.predict(
            validation_raw
        )
    )

    validation_isotonic = (
        isotonic.predict(
            validation_raw
        )
    )

    results = []

    for method, probabilities in [
        (
            "raw",
            validation_raw,
        ),
        (
            "sigmoid",
            validation_sigmoid,
        ),
        (
            "isotonic",
            validation_isotonic,
        ),
    ]:

        result = (
            evaluate_probabilities(
                model_name,
                method,
                y_validation,
                probabilities,
            )
        )

        results.append(
            result
        )

        with mlflow.start_run(
            run_name=(
                f"{model_name}-{method}"
            )
        ):

            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_param(
                "calibration",
                method,
            )

            mlflow.log_metrics(
                {
                    key: value
                    for key, value
                    in result.items()
                    if isinstance(
                        value,
                        float,
                    )
                }
            )

    print(
        f"\n=== {model_name.upper()} "
        "CALIBRATION TABLE ==="
    )

    print(
        "\nRAW"
    )

    print(
        calibration_table(
            y_validation,
            validation_raw,
            bins=5,
        ).to_string(
            index=False
        )
    )

    print(
        "\nSIGMOID"
    )

    print(
        calibration_table(
            y_validation,
            validation_sigmoid,
            bins=5,
        ).to_string(
            index=False
        )
    )

    print(
        "\nISOTONIC"
    )

    print(
        calibration_table(
            y_validation,
            validation_isotonic,
            bins=5,
        ).to_string(
            index=False
        )
    )

    return results


def main():

    training = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_DIR
        / "validation.parquet"
    )

    (
        fit_data,
        calibration_data,
    ) = split_fit_calibration(
        training,
        calibration_fraction=0.20,
    )

    print(
        "Fit rows:",
        len(fit_data),
    )

    print(
        "Fit default rate:",
        round(
            fit_data[
                TARGET
            ].mean(),
            4,
        ),
    )

    print(
        "Calibration rows:",
        len(calibration_data),
    )

    print(
        "Calibration defaults:",
        int(
            calibration_data[
                TARGET
            ].sum()
        ),
    )

    print(
        "Calibration default rate:",
        round(
            calibration_data[
                TARGET
            ].mean(),
            4,
        ),
    )

    X_fit, y_fit = prepare_xy(
        fit_data
    )

    X_calibration, y_calibration = (
        prepare_xy(
            calibration_data
        )
    )

    X_validation, y_validation = (
        prepare_xy(
            validation
        )
    )

    mlflow.set_experiment(
        "credit-risk-calibration"
    )

    logistic_results = evaluate_model(
        model_name="logistic_regression",
        pipeline=build_logistic(
            X_fit
        ),
        X_fit=X_fit,
        y_fit=y_fit,
        X_calibration=X_calibration,
        y_calibration=y_calibration,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    lightgbm_results = evaluate_model(
        model_name="lightgbm",
        pipeline=build_lightgbm(
            X_fit
        ),
        X_fit=X_fit,
        y_fit=y_fit,
        X_calibration=X_calibration,
        y_calibration=y_calibration,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    results = pd.DataFrame(
        logistic_results
        + lightgbm_results
    )

    print(
        "\n=== CALIBRATION COMPARISON ==="
    )

    columns = [
        "model",
        "calibration",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "mean_predicted_pd",
        "actual_default_rate",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()