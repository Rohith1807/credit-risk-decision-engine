import pandas as pd


def calculate_policy_metrics(
    decisions: pd.DataFrame,
) -> dict:

    total = len(decisions)

    approved = decisions[
        decisions["decision"] == "APPROVE"
    ]

    rejected = decisions[
        decisions["decision"] == "REJECT"
    ]

    approved_count = len(approved)

    requested_gmv = (
        decisions["requested_amount"].sum()
    )

    approved_gmv = (
        approved["approved_limit"].sum()
    )

    expected_loss_total = (
        approved["expected_loss"].sum()
    )

    expected_profit_total = (
        approved["expected_profit"].sum()
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
                approved_count / total
                if total
                else 0
            ),

        "requested_gmv":
            requested_gmv,

        "approved_gmv":
            approved_gmv,

        "gmv_approval_rate":
            (
                approved_gmv / requested_gmv
                if requested_gmv
                else 0
            ),

        "average_approved_limit":
            (
                approved["approved_limit"].mean()
                if approved_count
                else 0
            ),

        "expected_loss":
            expected_loss_total,

        "expected_profit":
            expected_profit_total,

        "expected_loss_rate":
            (
                expected_loss_total
                / approved_gmv
                if approved_gmv
                else 0
            ),

        "expected_profit_margin":
            (
                expected_profit_total
                / approved_gmv
                if approved_gmv
                else 0
            ),

        "observed_default_rate_approved":
            (
                approved["default_status"].mean()
                if approved_count
                else 0
            ),

        "profit_to_expected_loss_ratio":
            (
                expected_profit_total
                / expected_loss_total
                if expected_loss_total
                else 0
            ),
    }

    return metrics