from pathlib import Path

import pandas as pd
import yaml

from src.policy.portfolio_metrics import (
    calculate_policy_metrics,
)
from src.policy.scenario_analysis import (
    build_policy_scenarios,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_scenario_comparison.parquet"
)


def load_base_policy():
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

    base_config = load_base_policy()

    scenarios = (
        build_policy_scenarios(
            base_config
        )
    )

    rows = []

    for (
        scenario_name,
        config,
    ) in scenarios.items():

        decisions = apply_policy(
            scored,
            config=config,
        )

        metrics = (
            calculate_policy_metrics(
                decisions
            )
        )

        approved = decisions[
            decisions[
                "decision"
            ] == "APPROVE"
        ]

        rejected = decisions[
            decisions[
                "decision"
            ] == "REJECT"
        ]

        rows.append(
            {
                "scenario":
                    scenario_name,

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

                "average_approved_limit":
                    metrics[
                        "average_approved_limit"
                    ],

                "expected_loss":
                    metrics[
                        "expected_loss"
                    ],

                "expected_profit":
                    metrics[
                        "expected_profit"
                    ],

                "expected_loss_rate":
                    metrics[
                        "expected_loss_rate"
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

                "full_approval_count":
                    (
                        decisions[
                            "policy_tier"
                        ]
                        == "FULL_APPROVAL"
                    ).sum(),

                "low_and_grow_count":
                    (
                        decisions[
                            "policy_tier"
                        ]
                        == "LOW_AND_GROW"
                    ).sum(),

                "fallback_count":
                    (
                        decisions[
                            "policy_tier"
                        ]
                        == "FALLBACK"
                    ).sum(),

                "reject_count":
                    (
                        decisions[
                            "policy_tier"
                        ]
                        == "REJECT"
                    ).sum(),

                "approved_count":
                    len(
                        approved
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    current_row = (
        results[
            results[
                "scenario"
            ] == "current"
        ]
        .iloc[0]
    )

    results[
        "incremental_gmv_vs_current"
    ] = (
        results[
            "approved_gmv"
        ]
        - current_row[
            "approved_gmv"
        ]
    )

    results[
        "incremental_profit_vs_current"
    ] = (
        results[
            "expected_profit"
        ]
        - current_row[
            "expected_profit"
        ]
    )

    results[
        "incremental_loss_vs_current"
    ] = (
        results[
            "expected_loss"
        ]
        - current_row[
            "expected_loss"
        ]
    )

    results.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n=== POLICY SCENARIO COMPARISON ==="
    )

    display_columns = [
        "scenario",
        "approval_rate",
        "approved_gmv",
        "gmv_approval_rate",
        "expected_loss",
        "expected_profit",
        "approved_default_rate",
        "rejected_default_rate",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()