from pathlib import Path

import pandas as pd
import yaml

from src.policy.decision_engine import (
    make_credit_decision,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


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


def apply_policy(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    config = load_policy()

    decisions = []

    for row in dataframe.itertuples():

        result = make_credit_decision(
            probability_default=(
                row.probability_default
            ),
            requested_amount=(
                row.requested_amount
            ),
            bank_data_available=(
                row.bank_data_available
            ),
            data_confidence_score=(
                row.data_confidence_score
            ),
            loan_to_income_ratio=(
                row.loan_to_income_ratio
            ),
            bnpl_payment_burden=(
                row.bnpl_payment_burden
            ),
            negative_balance_rate_30d=(
                row.negative_balance_rate_30d
            ),
            config=config,
        )

        decisions.append(
            {
                "application_id":
                    row.application_id,
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