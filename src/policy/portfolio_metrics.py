import pandas as pd


def calculate_policy_metrics(
    decisions: pd.DataFrame,
) -> dict:

    total = len(
        decisions
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

    approved_count = len(
        approved
    )

    requested_gmv = (
        decisions[
            "requested_amount"
        ].sum()
    )

    approved_gmv = (
        approved[
            "approved_limit"
        ].sum()
    )

    metrics = {
        "applications":
            total,

        "approved_count":
            approved_count,

        "rejected_count":
            len(rejected),

        "approval_rate":
            (
                approved_count
                / total
                if total
                else 0
            ),

        "requested_gmv":
            requested_gmv,

        "approved_gmv":
            approved_gmv,

        "gmv_approval_rate":
            (
                approved_gmv
                / requested_gmv
                if requested_gmv
                else 0
            ),

        "average_approved_limit":
            (
                approved[
                    "approved_limit"
                ].mean()
                if approved_count
                else 0
            ),

        "expected_loss":
            approved[
                "expected_loss"
            ].sum(),

        "expected_profit":
            approved[
                "expected_profit"
            ].sum(),

        "observed_default_rate_approved":
            (
                approved[
                    "default_status"
                ].mean()
                if approved_count
                else 0
            ),
    }

    return metrics