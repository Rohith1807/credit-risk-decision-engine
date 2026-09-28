import pandas as pd


def test_repayment_integrity():

    repayments = pd.read_parquet(
        "data/raw/repayments.parquet"
    )

    assert repayments[
        "payment_id"
    ].is_unique

    assert (
        repayments[
            "amount_due"
        ]
        > 0
    ).all()

    assert (
        repayments[
            "amount_paid"
        ]
        >= 0
    ).all()

    assert (
        repayments[
            "days_past_due"
        ]
        >= 0
    ).all()


def test_default_definition():

    outcomes = pd.read_parquet(
        "data/raw/loan_outcomes.parquet"
    )

    expected_default = (
        outcomes[
            "max_days_past_due"
        ]
        >= 90
    ).astype(int)

    assert (
        outcomes[
            "default_status"
        ]
        == expected_default
    ).all()