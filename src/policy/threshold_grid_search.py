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

    base_config = load_policy()

    low_thresholds = [
        0.02,
        0.03,
        0.04,
        0.05,
    ]

    medium_thresholds = [
        0.05,
        0.08,
        0.10,
        0.12,
    ]

    rows = []

    for low_pd in low_thresholds:
        for medium_pd in medium_thresholds:

            if (
                medium_pd
                <= low_pd
            ):
                continue

            config = deepcopy(
                base_config
            )

            config[
                "policy"
            ][
                "thresholds"
            ][
                "low_risk_max_pd"
            ] = low_pd

            config[
                "policy"
            ][
                "thresholds"
            ][
                "medium_risk_max_pd"
            ] = medium_pd

            decisions = apply_policy(
                scored,
                config=config,
            )

            metrics = (
                calculate_policy_metrics(
                    decisions
                )
            )

            rows.append(
                {
                    "low_pd_threshold":
                        low_pd,

                    "medium_pd_threshold":
                        medium_pd,

                    **metrics,
                }
            )

    results = pd.DataFrame(
        rows
    )

    results = (
        results.sort_values(
            "expected_profit",
            ascending=False,
        )
    )

    eligible = results[
        results[
            "observed_default_rate_approved"
        ] <= 0.03
    ]

    if not eligible.empty:

        recommended = (
            eligible.sort_values(
                "expected_profit",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            "\n=== RECOMMENDED POLICY ==="
        )

        print(
            recommended[
                [
                    "low_pd_threshold",
                    "medium_pd_threshold",
                    "approval_rate",
                    "approved_gmv",
                    "expected_loss",
                    "expected_profit",
                    "observed_default_rate_approved",
                ]
            ]
        )

    print(
        "\n=== TOP POLICY CONFIGURATIONS ==="
    )

    columns = [
        "low_pd_threshold",
        "medium_pd_threshold",
        "approval_rate",
        "approved_gmv",
        "expected_loss",
        "expected_profit",
        "expected_profit_margin",
        "observed_default_rate_approved",
    ]

    print(
        results[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()