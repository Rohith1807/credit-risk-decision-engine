from pathlib import Path

import numpy as np

import pandas as pd
import yaml

from src.features.build_features import (
    build_application_features,
)
from src.features.feature_schema import (
    EXCLUDED_V1_FEATURES,
)
from src.ingestion.plaid.build_live_features import (
    prepare_internal_transactions,
)
from src.models.score_validation import (
    build_lightgbm,
    prepare_xy,
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

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "policy.yaml"
)


NON_MODEL_COLUMNS = [
    "application_id",
    "user_id",
    "application_timestamp",
    "default_status",
]


def create_demo_application() -> dict:
    """
    Create a demo underwriting application
    using Plaid transaction data.
    """

    return {
        "application_id":
            "plaid_demo_application",

        "user_id":
            "plaid_demo_user",

        "application_timestamp":
            pd.Timestamp.now(),

        "requested_amount":
            250.0,

        "merchant_category":
            "general_retail",

        "device_type":
            "mobile",

        "channel":
            "checkout",

        "bank_data_available":
            1,
    }


def prepare_scoring_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare one or more live feature rows
    for the trained sklearn/LightGBM pipeline.
    """

    drop_columns = [
        column
        for column in (
            NON_MODEL_COLUMNS
            + EXCLUDED_V1_FEATURES
        )
        if column in dataframe.columns
    ]

    scoring = dataframe.drop(
        columns=drop_columns
    ).copy()

    # sklearn works reliably with np.nan
    # rather than pandas scalar pd.NA
    scoring = scoring.replace(
        {
            pd.NA: np.nan,
        }
    )

    categorical_columns = {
        "merchant_category",
        "device_type",
        "channel",
    }

    for column in scoring.columns:

        if column not in categorical_columns:

            scoring[column] = (
                pd.to_numeric(
                    scoring[column],
                    errors="coerce",
                )
            )

    return scoring


def train_selected_model():
    """
    Train the selected unweighted LightGBM
    model using the existing training dataset.

    Later this will be replaced by loading
    a registered/versioned model artifact.
    """

    train = pd.read_parquet(
        PROCESSED_DIR
        / "train.parquet"
    )

    X_train, y_train = (
        prepare_xy(
            train
        )
    )

    pipeline = build_lightgbm(
        X_train
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def load_policy() -> dict:
    """
    Load the current underwriting policy.
    """

    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(
            file
        )


def safe_feature_value(
    dataframe: pd.DataFrame,
    column: str,
):
    """
    Safely retrieve a feature value.

    Missing values are converted to None
    for the policy engine.
    """

    if column not in dataframe.columns:
        return None

    value = dataframe.iloc[0][
        column
    ]

    if pd.isna(value):
        return None

    return float(value)


def main():

    # --------------------------------
    # 1. Load normalized Plaid data
    # --------------------------------

    transactions = (
        prepare_internal_transactions(
            user_id=
                "plaid_demo_user"
        )
    )

    print(
        "\n=== PLAID TRANSACTIONS ==="
    )

    print(
        "Transaction rows:",
        len(transactions),
    )

    # --------------------------------
    # 2. Create demo application
    # --------------------------------

    application = (
        create_demo_application()
    )

    print(
        "\n=== DEMO APPLICATION ==="
    )

    for key, value in (
        application.items()
    ):
        print(
            f"{key}: {value}"
        )

    # --------------------------------
    # 3. Reuse existing feature engine
    # --------------------------------

    feature_row = (
        build_application_features(
            application=application,
            transactions=transactions,
        )
    )

    live_features = (
        pd.DataFrame(
            [feature_row]
        )
    )

    print(
        "\n=== LIVE FEATURES ==="
    )

    print(
        live_features.T
    )

    # --------------------------------
    # 4. Prepare model inputs
    # --------------------------------

    X_live = (
        prepare_scoring_features(
            live_features
        )
    )

    # --------------------------------
    # 5. Train selected model
    # --------------------------------

    pipeline = (
        train_selected_model()
    )

    # --------------------------------
    # 6. Predict probability of default
    # --------------------------------

    probability_default = float(
        pipeline.predict_proba(
            X_live
        )[0, 1]
    )

    print(
        "\n=== MODEL SCORE ==="
    )

    print(
        "Predicted PD:",
        f"{probability_default:.2%}",
    )

    # --------------------------------
    # 7. Load policy
    # --------------------------------

    policy_config = (
        load_policy()
    )

    # --------------------------------
    # 8. Run underwriting policy
    # --------------------------------

    decision = (
        make_credit_decision(
            probability_default=
                probability_default,

            requested_amount=
                float(
                    live_features.iloc[
                        0
                    ][
                        "requested_amount"
                    ]
                ),

            bank_data_available=
                int(
                    live_features.iloc[
                        0
                    ][
                        "bank_data_available"
                    ]
                ),

            data_confidence_score=
                safe_feature_value(
                    live_features,
                    "data_confidence_score",
                ),

            loan_to_income_ratio=
                safe_feature_value(
                    live_features,
                    "loan_to_income_ratio",
                ),

            bnpl_payment_burden=
                safe_feature_value(
                    live_features,
                    "bnpl_payment_burden",
                ),

            negative_balance_rate_30d=
                safe_feature_value(
                    live_features,
                    "negative_balance_rate_30d",
                ),

            config=
                policy_config,

            apply_behavioral_guardrails=
                False,
        )
    )

    # --------------------------------
    # 9. Display decision
    # --------------------------------

    print(
        "\n=== LIVE UNDERWRITING DECISION ==="
    )

    print(
        "PD:",
        f"{decision.probability_default:.2%}",
    )

    print(
        "Decision:",
        decision.decision,
    )

    print(
        "Policy tier:",
        decision.policy_tier,
    )

    print(
        "Approved limit:",
        f"${decision.approved_limit:,.2f}",
    )

    print(
        "Reason code:",
        decision.reason_code,
    )

    print(
        "Expected loss:",
        f"${decision.expected_loss:,.2f}",
    )

    print(
        "Expected profit:",
        f"${decision.expected_profit:,.2f}",
    )


if __name__ == "__main__":
    main()