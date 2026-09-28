from src.data_generation.generate_accounts import (
    generate_accounts,
)
from src.data_generation.generate_transactions import (
    generate_transactions,
)
from src.data_generation.generate_users import (
    generate_users,
)


def build_test_data():
    users = (
        generate_users()
        .head(100)
        .copy()
    )

    accounts = generate_accounts(
        users
    )

    transactions = (
        generate_transactions(
            users,
            accounts,
        )
    )

    return (
        users,
        accounts,
        transactions,
    )


def test_transaction_ids_unique():
    _, _, transactions = (
        build_test_data()
    )

    assert transactions[
        "transaction_id"
    ].is_unique


def test_transactions_reference_valid_accounts():
    _, accounts, transactions = (
        build_test_data()
    )

    assert set(
        transactions["account_id"]
    ).issubset(
        set(accounts["account_id"])
    )


def test_income_is_positive():
    _, _, transactions = (
        build_test_data()
    )

    income = transactions[
        transactions[
            "transaction_type"
        ]
        == "income"
    ]

    assert (
        income["amount"] > 0
    ).all()


def test_purchases_are_negative():
    _, _, transactions = (
        build_test_data()
    )

    purchases = transactions[
        transactions[
            "transaction_type"
        ]
        == "purchase"
    ]

    assert (
        purchases["amount"] < 0
    ).all()


def test_transaction_timestamps_present():
    _, _, transactions = (
        build_test_data()
    )

    assert transactions[
        "transaction_timestamp"
    ].notna().all()


def test_balance_present():
    _, _, transactions = (
        build_test_data()
    )

    assert transactions[
        "balance_after_transaction"
    ].notna().all()