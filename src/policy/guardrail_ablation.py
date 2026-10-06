from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.policy.portfolio_metrics import (
    calculate_policy_metrics,
)
from src.policy.simulate_policy import (
    apply_policy,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_scored.parquet"
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


def main():

    scored = pd.read_parquet(
        INPUT_PATH
    )

    config = deepcopy(
        load_policy()
    )

    # Recommended thresholds from grid search
    config[
        "policy"
    ][
        "thresholds"
    ][
        "low_risk_max_pd"
    ] = 0.04

    config[
        "policy"
    ][
        "thresholds"
    ][
        "medium_risk_max_pd"
    ] = 0.05

    rows = []

    experiments = [
        (
            "pd_plus_guardrails",
            True,
        ),
        (
            "pd_only",
            False,
        ),
    ]

    for (
        name,
        guardrails_enabled,
    ) in experiments:

        decisions = apply_policy(
            scored,
            config=config,
            apply_behavioral_guardrails=(
                guardrails_enabled
            ),
        )

        metrics = (
            calculate_policy_metrics(
                decisions
            )
        )

        rejected = decisions[
            decisions[
                "decision"
            ] == "REJECT"
        ]

        rows.append(
            {
                "policy":
                    name,

                "approval_rate":
                    metrics[
                        "approval_rate"
                    ],

                "approved_gmv":
                    metrics[
                        "approved_gmv"
                    ],

                "gmv_approval_rate":
                    metrics[
                        "gmv_approval_rate"
                    ],

                "expected_loss":
                    metrics[
                        "expected_loss"
                    ],

                "expected_loss_rate":
                    metrics[
                        "expected_loss_rate"
                    ],

                "expected_profit":
                    metrics[
                        "expected_profit"
                    ],

                "expected_profit_margin":
                    metrics[
                        "expected_profit_margin"
                    ],

                "approved_default_rate":
                    metrics[
                        "observed_default_rate_approved"
                    ],

                "rejected_default_rate":
                    (
                        rejected[
                            "default_status"
                        ].mean()
                        if len(rejected)
                        else 0
                    ),

                "reject_count":
                    len(
                        rejected
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    print(
        "\n=== GUARDRAIL ABLATION ==="
    )

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()