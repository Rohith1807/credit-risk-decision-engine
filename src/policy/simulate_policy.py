from pathlib import Path

import pandas as pd
import yaml

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

INPUT_PATH = (
    PROCESSED_DIR
    / "validation_scored.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "validation_policy_decisions.parquet"
)


def load_policy():
    with open(
        PROJECT_ROOT
        / "configs"
        / "policy.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def safe_value(
    value,
):
    if pd.isna(value):
        return None

    return float(value)


def apply_policy(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    config = load_policy()

    decisions = []

    for row in dataframe.itertuples():
        result = make_credit_decision(
            probability_default=float(
                row.probability_default
            ),
            requested_amount=float(
                row.requested_amount
            ),
            bank_data_available=int(
                row.bank_data_available
            ),
            data_confidence_score=safe_value(
                row.data_confidence_score
            ),
            loan_to_income_ratio=safe_value(
                row.loan_to_income_ratio
            ),
            bnpl_payment_burden=safe_value(
                row.bnpl_payment_burden
            ),
            negative_balance_rate_30d=safe_value(
                row.negative_balance_rate_30d
            ),
            config=config,
        )

        decisions.append(
            {
                "application_id":
                    row.application_id,

                "default_status":
                    row.default_status,

                "requested_amount":
                    row.requested_amount,

                "decision":
                    result.decision,

                "policy_tier":
                    result.policy_tier,

                "approved_limit":
                    result.approved_limit,

                "reason_code":
                    result.reason_code,

                "probability_default":
                    result.probability_default,

                "expected_loss":
                    result.expected_loss,

                "expected_profit":
                    result.expected_profit,
            }
        )

    return pd.DataFrame(
        decisions
    )


def main():
    scored = pd.read_parquet(
        INPUT_PATH
    )

    decisions = apply_policy(
        scored
    )

    decisions.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Policy rows:",
        len(decisions),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()