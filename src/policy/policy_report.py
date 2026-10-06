from pathlib import Path

import pandas as pd

from src.policy.portfolio_metrics import (
    calculate_policy_metrics,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_policy_decisions.parquet"
)


def main():
    decisions = pd.read_parquet(
        INPUT_PATH
    )

    metrics = calculate_policy_metrics(
        decisions
    )

    print(
        "\n=== POLICY PORTFOLIO REPORT ==="
    )

    print(
        f"Applications: "
        f"{metrics['applications']:,}"
    )

    print(
        f"Approved: "
        f"{metrics['approved_count']:,}"
    )

    print(
        f"Rejected: "
        f"{metrics['rejected_count']:,}"
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
        f"Average approved limit: "
        f"${metrics['average_approved_limit']:,.2f}"
    )

    print(
        f"Expected loss: "
        f"${metrics['expected_loss']:,.2f}"
    )

    print(
        f"Expected profit: "
        f"${metrics['expected_profit']:,.2f}"
    )

    print(
        "Observed default rate "
        "among approved: "
        f"{metrics['observed_default_rate_approved']:.2%}"
    )

    print(
        "\n=== POLICY TIERS ==="
    )

    tier_summary = (
        decisions.groupby(
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
        .sort_values(
            "applications",
            ascending=False,
        )
    )

    print(
        tier_summary.to_string()
    )

    print(
        "\n=== REASON CODES ==="
    )

    print(
        decisions[
            "reason_code"
        ]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()