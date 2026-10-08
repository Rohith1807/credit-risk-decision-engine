from pathlib import Path

import yaml

import pandas as pd

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
    prepare_xy,
)
from src.policy.decision_engine import (
    make_credit_decision,
)
from src.policy.portfolio_metrics import (
    calculate_policy_metrics,
)

from src.policy.decision_engine import (
    make_credit_decision,
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
    / "final_test_results.parquet"
)

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "policy.yaml"
)


def build_logistic_model(
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

    print(
        "Loading train and untouched test datasets..."
    )

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    test = pd.read_parquet(
        PROCESSED_DIR
        / "test.parquet"
    )

    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    X_train, y_train = prepare_xy(
        train
    )

    X_test, y_test = prepare_xy(
        test
    )

    print()
    print(
        "=== FINAL MODEL TRAINING ==="
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Training default rate: "
        f"{y_train.mean():.2%}"
    )

    model = build_logistic_model(
        X_train
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    mean_pd = float(
        probabilities.mean()
    )

    actual_default_rate = float(
        y_test.mean()
    )

    print()
    print(
        "=== FINAL TEST MODEL PERFORMANCE ==="
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    print(
        f"Test defaults: "
        f"{int(y_test.sum()):,}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"Brier score: {brier:.4f}"
    )

    print(
        f"Mean predicted PD: "
        f"{mean_pd:.2%}"
    )

    print(
        f"Actual default rate: "
        f"{actual_default_rate:.2%}"
    )

    scored = test.copy()

    scored[
        "probability_default"
    ] = probabilities

    decisions = []

    for row in scored.to_dict(
        orient="records"
    ):

        decision = (
            make_credit_decision(
                probability_default=
                    float(
                        row[
                            "probability_default"
                        ]
                    ),

                requested_amount=
                    float(
                        row[
                            "requested_amount"
                        ]
                    ),

                bank_data_available=
                    int(
                        row[
                            "bank_data_available"
                        ]
                    ),

                data_confidence_score=
                    row.get(
                        "data_confidence_score"
                    ),

                loan_to_income_ratio=
                    row.get(
                        "loan_to_income_ratio"
                    ),

                bnpl_payment_burden=
                    row.get(
                        "bnpl_payment_burden"
                    ),

                negative_balance_rate_30d=
                    row.get(
                        "negative_balance_rate_30d"
                    ),

                config=config,

                apply_behavioral_guardrails=
                    False,
            )
        )

        decisions.append(
            {
                "application_id":
                    row[
                        "application_id"
                    ],

                "probability_default":
                    decision.probability_default,

                "decision":
                    decision.decision,

                "policy_tier":
                    decision.policy_tier,

                "approved_limit":
                    decision.approved_limit,

                "reason_code":
                    decision.reason_code,

                "expected_loss":
                    decision.expected_loss,

                "expected_profit":
                    decision.expected_profit,

                "default_status":
                    row[
                        "default_status"
                    ],

                "requested_amount":
                    row[
                        "requested_amount"
                    ],
            }
        )

    decisions_df = pd.DataFrame(
        decisions
    )

    metrics = (
        calculate_policy_metrics(
            decisions_df
        )
    )

    print()
    print(
        "=== FINAL TEST POLICY PERFORMANCE ==="
    )

    print(
        f"Applications: "
        f"{metrics['applications']:,}"
    )

    print(
        f"Approval rate: "
        f"{metrics['approval_rate']:.2%}"
    )

    print(
        f"Requested GMV: "
        f"${metrics['requested_gmv']:,.2f}"
    )

    print(
        f"Approved GMV: "
        f"${metrics['approved_gmv']:,.2f}"
    )

    print(
        f"GMV approval rate: "
        f"{metrics['gmv_approval_rate']:.2%}"
    )

    print(
        f"Expected loss: "
        f"${metrics['expected_loss']:,.2f}"
    )

    print(
        f"Expected loss rate: "
        f"{metrics['expected_loss_rate']:.2%}"
    )

    print(
        f"Expected profit: "
        f"${metrics['expected_profit']:,.2f}"
    )

    print(
        f"Expected profit margin: "
        f"{metrics['expected_profit_margin']:.2%}"
    )

    approved = decisions_df[
        decisions_df[
            "decision"
        ]
        == "APPROVE"
    ]

    rejected = decisions_df[
        decisions_df[
            "decision"
        ]
        == "REJECT"
    ]

    if not approved.empty:

        approved_default_rate = (
            approved[
                "default_status"
            ].mean()
        )

        print(
            f"Approved observed default rate: "
            f"{approved_default_rate:.2%}"
        )

    if not rejected.empty:

        rejected_default_rate = (
            rejected[
                "default_status"
            ].mean()
        )

        print(
            f"Rejected observed default rate: "
            f"{rejected_default_rate:.2%}"
        )

    print()
    print(
        "=== FINAL POLICY TIERS ==="
    )

    tier_summary = (
        decisions_df.groupby(
            "policy_tier"
        )
        .agg(
            applications=(
                "application_id",
                "count",
            ),
            approved_gmv=(
                "approved_limit",
                "sum",
            ),
            mean_pd=(
                "probability_default",
                "mean",
            ),
            observed_default_rate=(
                "default_status",
                "mean",
            ),
        )
    )

    print(
        tier_summary
    )

    decisions_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved final test results: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()