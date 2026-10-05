from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    probabilities,
    predictions,
) -> dict:
    """
    Calculate core credit-risk model metrics.
    """

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),
    }


def calculate_confusion_matrix(
    y_true,
    predictions,
) -> dict:

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
        )
        .ravel()
    )

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }