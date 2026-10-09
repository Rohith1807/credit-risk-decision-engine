from pathlib import Path

import pandas as pd
import pytest


RAW_DIR = Path("data/raw")

REQUIRED_FILES = [
    RAW_DIR / "users.parquet",
    RAW_DIR / "bank_accounts.parquet",
    RAW_DIR / "transactions.parquet",
    RAW_DIR / "applications.parquet",
    RAW_DIR / "loans.parquet",
    RAW_DIR / "repayments.parquet",
    RAW_DIR / "loan_outcomes.parquet",
]


if not all(path.exists() for path in REQUIRED_FILES):
    pytest.skip(
        "Raw generated portfolio files are not available in this environment.",
        allow_module_level=True,
    )


users = pd.read_parquet(RAW_DIR / "users.parquet")

accounts = pd.read_parquet(
    "data/raw/bank_accounts.parquet"
)

transactions = pd.read_parquet(
    "data/raw/transactions.parquet"
)

applications = pd.read_parquet(
    "data/raw/applications.parquet"
)

loans = pd.read_parquet(
    "data/raw/loans.parquet"
)

repayments = pd.read_parquet(
    "data/raw/repayments.parquet"
)

outcomes = pd.read_parquet(
    "data/raw/loan_outcomes.parquet"
)


def test_accounts_reference_valid_users():
    assert set(
        accounts["user_id"]
    ).issubset(
        set(users["user_id"])
    )


def test_transactions_reference_valid_accounts():
    assert set(
        transactions["account_id"]
    ).issubset(
        set(accounts["account_id"])
    )


def test_applications_reference_valid_users():
    assert set(
        applications["user_id"]
    ).issubset(
        set(users["user_id"])
    )


def test_loans_reference_valid_applications():
    assert set(
        loans["application_id"]
    ).issubset(
        set(applications["application_id"])
    )


def test_repayments_reference_valid_loans():
    assert set(
        repayments["loan_id"]
    ).issubset(
        set(loans["loan_id"])
    )


def test_outcomes_reference_valid_loans():
    assert set(
        outcomes["loan_id"]
    ).issubset(
        set(loans["loan_id"])
    )

def test_loans_not_before_application():
    merged = loans.merge(
        applications[
            [
                "application_id",
                "application_timestamp",
            ]
        ],
        on="application_id",
        how="left",
    )

    assert (
        pd.to_datetime(
            merged["origination_date"]
        )
        >=
        pd.to_datetime(
            merged["application_timestamp"]
        )
    ).all()


def test_repayments_not_before_origination():
    merged = repayments.merge(
        loans[
            [
                "loan_id",
                "origination_date",
            ]
        ],
        on="loan_id",
        how="left",
    )

    assert (
        pd.to_datetime(
            merged["scheduled_payment_date"]
        )
        >=
        pd.to_datetime(
            merged["origination_date"]
        )
    ).all()

def test_default_rate_within_target():
    default_rate = outcomes[
        "default_status"
    ].mean()

    assert 0.03 <= default_rate <= 0.05

def test_pd_buckets_rank_risk():
    temp = outcomes.copy()

    temp["pd_bucket"] = pd.qcut(
        temp["latent_pd"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    rates = (
        temp.groupby(
            "pd_bucket"
        )["default_status"]
        .mean()
    )

    assert rates.iloc[-1] > rates.iloc[0]
    assert rates.iloc[-1] > rates.mean()

analysis = outcomes.merge(
    users[
        [
            "user_id",
            "employment_type",
        ]
    ],
    on="user_id",
)

print(
    analysis.groupby(
        "employment_type"
    )["default_status"]
    .agg(["count", "mean"])
)

analysis = outcomes.merge(
    applications[
        [
            "application_id",
            "bank_data_available",
        ]
    ],
    on="application_id",
)

print(
    analysis.groupby(
        "bank_data_available"
    )["default_status"]
    .agg(["count", "mean"])
)

analysis = outcomes.merge(
    applications[
        [
            "application_id",
            "requested_amount",
        ]
    ],
    on="application_id",
)

analysis["amount_bucket"] = pd.qcut(
    analysis["requested_amount"],
    q=4,
    duplicates="drop",
)

print(
    analysis.groupby(
        "amount_bucket",
        observed=True,
    )["default_status"]
    .agg(["count", "mean"])
)

print(
    transactions[
        "transaction_type"
    ].value_counts()
)

print(
    transactions[
        "amount"
    ].describe()
)

print(
    transactions[
        "balance_after_transaction"
    ].describe()
)

negative_balance_rate = (
    transactions[
        "balance_after_transaction"
    ]
    < 0
).mean()

print(
    f"Negative balance transaction rate: "
    f"{negative_balance_rate:.2%}"
)