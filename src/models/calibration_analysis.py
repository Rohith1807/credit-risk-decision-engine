from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.preprocessing import (
    build_preprocessor,
)
from src.models.train_baseline import (
    prepare_xy as prepare_logistic_xy,
)
from src.models.score_validation import (
    build_lightgbm,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def calibration_table(
    y_true,
    probabilities,
    bins=10,
) -> pd.DataFrame:

    dataframe = pd.DataFrame(
        {
            "actual": y_true,
            "predicted_pd": probabilities,
        }
    )

    dataframe["pd_bucket"] = pd.qcut(
        dataframe["predicted_pd"],
        q=bins,
        duplicates="drop",
    )

    result = (
        dataframe.groupby(
            "pd_bucket",
            observed=True,
        )
        .agg(
            count=("actual", "size"),
            mean_predicted_pd=(
                "predicted_pd",
                "mean",
            ),
            actual_default_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    return result


def build_logistic_model(
    X_train: pd.DataFrame,
) -> Pipeline:
    """
    Build the unweighted Logistic Regression
    candidate using the project's standard
    preprocessing pipeline.
    """

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


def evaluate_probabilities(
    model_name: str,
    calibration_method: str,
    y_true,
    probabilities,
) -> dict:

    return {
        "model":
            model_name,

        "calibration":
            calibration_method,

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "mean_predicted_pd":
            float(
                probabilities.mean()
            ),

        "actual_default_rate":
            float(
                y_true.mean()
            ),
    }


def fit_prefit_calibrator(
    fitted_model,
    X_calibration,
    y_calibration,
    method: str,
):
    """
    Calibrate an already fitted model.

    Uses FrozenEstimator when available
    and falls back to cv='prefit' for
    compatibility with older sklearn.
    """

    try:
        from sklearn.frozen import (
            FrozenEstimator,
        )

        calibrator = (
            CalibratedClassifierCV(
                estimator=FrozenEstimator(
                    fitted_model
                ),
                method=method,
            )
        )

    except ImportError:

        calibrator = (
            CalibratedClassifierCV(
                estimator=fitted_model,
                method=method,
                cv="prefit",
            )
        )

    calibrator.fit(
        X_calibration,
        y_calibration,
    )

    return calibrator


def evaluate_model_family(
    model_name: str,
    base_model,
    X_fit,
    y_fit,
    X_calibration,
    y_calibration,
    X_validation,
    y_validation,
) -> list[dict]:

    results = []

    # -------------------------
    # RAW MODEL
    # -------------------------

    raw_model = clone(
        base_model
    )

    raw_model.fit(
        X_fit,
        y_fit,
    )

    raw_probabilities = (
        raw_model.predict_proba(
            X_validation
        )[:, 1]
    )

    results.append(
        evaluate_probabilities(
            model_name=model_name,
            calibration_method="raw",
            y_true=y_validation,
            probabilities=
                raw_probabilities,
        )
    )

    # -------------------------
    # SIGMOID CALIBRATION
    # -------------------------

    sigmoid_calibrator = (
        fit_prefit_calibrator(
            fitted_model=raw_model,
            X_calibration=
                X_calibration,
            y_calibration=
                y_calibration,
            method="sigmoid",
        )
    )

    sigmoid_probabilities = (
        sigmoid_calibrator.predict_proba(
            X_validation
        )[:, 1]
    )

    results.append(
        evaluate_probabilities(
            model_name=model_name,
            calibration_method=
                "sigmoid",
            y_true=y_validation,
            probabilities=
                sigmoid_probabilities,
        )
    )

    # -------------------------
    # ISOTONIC CALIBRATION
    # -------------------------

    isotonic_calibrator = (
        fit_prefit_calibrator(
            fitted_model=raw_model,
            X_calibration=
                X_calibration,
            y_calibration=
                y_calibration,
            method="isotonic",
        )
    )

    isotonic_probabilities = (
        isotonic_calibrator.predict_proba(
            X_validation
        )[:, 1]
    )

    results.append(
        evaluate_probabilities(
            model_name=model_name,
            calibration_method=
                "isotonic",
            y_true=y_validation,
            probabilities=
                isotonic_probabilities,
        )
    )

    return results


def main():

    print(
        "Loading scaled datasets..."
    )

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_DIR
        / "validation.parquet"
    )

    X_train, y_train = (
        prepare_logistic_xy(
            train
        )
    )

    X_validation, y_validation = (
        prepare_logistic_xy(
            validation
        )
    )

    # ---------------------------------
    # TEMPORAL CALIBRATION SPLIT
    # ---------------------------------

    split_index = int(
        len(X_train)
        * 0.80
    )

    X_fit = (
        X_train.iloc[
            :split_index
        ].copy()
    )

    y_fit = (
        y_train.iloc[
            :split_index
        ].copy()
    )

    X_calibration = (
        X_train.iloc[
            split_index:
        ].copy()
    )

    y_calibration = (
        y_train.iloc[
            split_index:
        ].copy()
    )

    print()
    print(
        "=== CALIBRATION SPLIT ==="
    )

    print(
        f"Model fit rows: "
        f"{len(X_fit):,}"
    )

    print(
        f"Calibration rows: "
        f"{len(X_calibration):,}"
    )

    print(
        f"Calibration defaults: "
        f"{int(y_calibration.sum()):,}"
    )

    print(
        f"Calibration default rate: "
        f"{y_calibration.mean():.2%}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"Validation defaults: "
        f"{int(y_validation.sum()):,}"
    )

    print(
        f"Validation default rate: "
        f"{y_validation.mean():.2%}"
    )

    # ---------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------

    logistic_model = (
        build_logistic_model(
            X_fit
        )
    )

    results = (
        evaluate_model_family(
            model_name="logistic",
            base_model=
                logistic_model,
            X_fit=X_fit,
            y_fit=y_fit,
            X_calibration=
                X_calibration,
            y_calibration=
                y_calibration,
            X_validation=
                X_validation,
            y_validation=
                y_validation,
        )
    )

    # ---------------------------------
    # LIGHTGBM
    # ---------------------------------

    lightgbm_model = (
        build_lightgbm(
            X_fit
        )
    )

    results.extend(
        evaluate_model_family(
            model_name="lightgbm",
            base_model=
                lightgbm_model,
            X_fit=X_fit,
            y_fit=y_fit,
            X_calibration=
                X_calibration,
            y_calibration=
                y_calibration,
            X_validation=
                X_validation,
            y_validation=
                y_validation,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    print()
    print(
        "=== CALIBRATION COMPARISON ==="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    output_path = (
        PROCESSED_DIR
        / "calibration_comparison.parquet"
    )

    results_df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved calibration comparison:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()