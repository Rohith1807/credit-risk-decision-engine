import pandas as pd


def calculate_portfolio_monitoring(
    decisions: pd.DataFrame,
) -> dict:

    total = len(decisions)

    approved = decisions[
        decisions["decision"] == "APPROVE"
    ]

    rejected = decisions[
        decisions["decision"] == "REJECT"
    ]

    return {
        "applications":
            total,

        "approval_rate":
            (
                len(approved) / total
                if total
                else 0.0
            ),

        "mean_pd":
            float(
                decisions[
                    "probability_default"
                ].mean()
            ),

        "approved_mean_pd":
            (
                float(
                    approved[
                        "probability_default"
                    ].mean()
                )
                if len(approved)
                else 0.0
            ),

        "approved_default_rate":
            (
                float(
                    approved[
                        "default_status"
                    ].mean()
                )
                if (
                    len(approved)
                    and "default_status"
                    in approved.columns
                )
                else None
            ),

        "rejected_default_rate":
            (
                float(
                    rejected[
                        "default_status"
                    ].mean()
                )
                if (
                    len(rejected)
                    and "default_status"
                    in rejected.columns
                )
                else None
            ),

        "expected_loss":
            float(
                approved[
                    "expected_loss"
                ].sum()
            ),

        "expected_profit":
            float(
                approved[
                    "expected_profit"
                ].sum()
            ),
    }