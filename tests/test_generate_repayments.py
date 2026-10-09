import pandas as pd


def test_repayment_integrity(require_data_file):

    repayments = pd.read_parquet(
        require_data_file("data/raw/repayments.parquet")
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


def test_default_definition(require_data_file):

    outcomes = pd.read_parquet(
        require_data_file("data/raw/loan_outcomes.parquet")
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
