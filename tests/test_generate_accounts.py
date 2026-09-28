from src.data_generation.generate_accounts import generate_accounts
from src.data_generation.generate_users import generate_users


def test_account_ids_unique():
    users = generate_users()
    accounts = generate_accounts(users)

    assert accounts["account_id"].is_unique


def test_accounts_have_valid_users():
    users = generate_users()
    accounts = generate_accounts(users)

    assert set(accounts["user_id"]).issubset(
        set(users["user_id"])
    )


def test_every_user_has_primary_account():
    users = generate_users()
    accounts = generate_accounts(users)

    primary_counts = (
        accounts[
            accounts["primary_account_flag"] == 1
        ]
        .groupby("user_id")
        .size()
    )

    assert (primary_counts == 1).all()


def test_available_balance_not_above_current():
    users = generate_users()
    accounts = generate_accounts(users)

    assert (
        accounts["available_balance"]
        <= accounts["current_balance"]
    ).all()


def test_balances_non_negative():
    users = generate_users()
    accounts = generate_accounts(users)

    assert (accounts["current_balance"] >= 0).all()
    assert (accounts["available_balance"] >= 0).all()