import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_thresholds(
    y_true,
    probabilities,
) -> pd.DataFrame:

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    ]

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
            )
            .ravel()
        )

        rows.append(
            {
                "threshold": threshold,
                "precision":
                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                "recall":
                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                "f1":
                    f1_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    ),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "flag_rate":
                    predictions.mean(),
            }
        )

    return pd.DataFrame(rows)