from datetime import timedelta

import numpy as np
import pandas as pd

from src.data_generation.config import load_data_generation_config


TRANSACTION_REFERENCE_DATE = pd.Timestamp("2026-01-01")


EXPENSE_CATEGORIES = {
    "groceries": (20, 180),
    "restaurants": (10, 120),
    "transportation": (8, 100),
    "entertainment": (10, 150),
    "shopping": (15, 300),
    "healthcare": (10, 250),
    "utilities": (40, 220),
}


MERCHANTS = {
    "groceries": [
        "Fresh Market",
        "City Grocery",
        "Market Foods",
        "Neighborhood Market",
    ],
    "restaurants": [
        "Local Cafe",
        "Downtown Grill",
        "Quick Bites",
        "Corner Kitchen",
    ],
    "transportation": [
        "Metro Transit",
        "Fuel Station",
        "Ride Service",
        "Parking",
    ],
    "entertainment": [
        "Streaming Service",
        "Cinema",
        "Gaming Store",
    ],
    "shopping": [
        "General Retail",
        "Online Marketplace",
        "Clothing Store",
        "Electronics Store",
    ],
    "healthcare": [
        "Pharmacy",
        "Medical Clinic",
        "Dental Office",
    ],
    "utilities": [
        "Electric Utility",
        "Internet Provider",
        "Mobile Provider",
    ],
}


def get_income_profile(
    employment_type: str,
) -> dict:
    """
    Return synthetic income characteristics by employment type.

    These are simulation assumptions rather than empirical estimates.
    """

    profiles = {
        "salaried": {
            "monthly_income_mean": 5200,
            "monthly_income_std": 900,
            "payments_per_month": 2,
            "income_variability": 0.08,
        },
        "hourly": {
            "monthly_income_mean": 3600,
            "monthly_income_std": 850,
            "payments_per_month": 2,
            "income_variability": 0.18,
        },
        "self_employed": {
            "monthly_income_mean": 5800,
            "monthly_income_std": 2200,
            "payments_per_month": 3,
            "income_variability": 0.40,
        },
        "gig_worker": {
            "monthly_income_mean": 3200,
            "monthly_income_std": 1400,
            "payments_per_month": 8,
            "income_variability": 0.45,
        },
        "student": {
            "monthly_income_mean": 1800,
            "monthly_income_std": 800,
            "payments_per_month": 2,
            "income_variability": 0.35,
        },
    }

    return profiles[employment_type]


def random_timestamp(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Timestamp:

    total_seconds = int(
        (end - start).total_seconds()
    )

    random_seconds = int(
        rng.integers(
            0,
            max(total_seconds, 1),
        )
    )

    return start + pd.Timedelta(
        seconds=random_seconds
    )


def generate_income_transactions(
    rng: np.random.Generator,
    user_id: str,
    account_id: str,
    employment_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:

    profile = get_income_profile(
        employment_type
    )

    months = max(
        1,
        int(
            np.ceil(
                (end_date - start_date).days
                / 30
            )
        ),
    )

    transaction_records = []

    expected_payments = (
        profile["payments_per_month"]
        * months
    )

    monthly_income = max(
        300,
        rng.normal(
            profile["monthly_income_mean"],
            profile["monthly_income_std"],
        ),
    )

    base_payment = (
        monthly_income
        / profile["payments_per_month"]
    )

    if employment_type in {
        "salaried",
        "hourly",
    }:

        interval_days = (
            14
            if profile["payments_per_month"] == 2
            else 30
        )

        current_date = (
            start_date
            + pd.Timedelta(
                days=int(
                    rng.integers(0, 10)
                )
            )
        )

        while current_date <= end_date:

            amount = base_payment * (
                1
                + rng.normal(
                    0,
                    profile[
                        "income_variability"
                    ],
                )
            )

            transaction_records.append(
                {
                    "user_id": user_id,
                    "account_id": account_id,
                    "transaction_timestamp":
                        current_date,
                    "amount": round(
                        max(amount, 100),
                        2,
                    ),
                    "transaction_type":
                        "income",
                    "merchant_name":
                        "Employer Payroll",
                    "merchant_category":
                        "income",
                }
            )

            current_date += pd.Timedelta(
                days=interval_days
            )

    else:

        for _ in range(
            expected_payments
        ):

            timestamp = random_timestamp(
                rng,
                start_date,
                end_date,
            )

            amount = base_payment * (
                1
                + rng.normal(
                    0,
                    profile[
                        "income_variability"
                    ],
                )
            )

            transaction_records.append(
                {
                    "user_id": user_id,
                    "account_id": account_id,
                    "transaction_timestamp":
                        timestamp,
                    "amount": round(
                        max(amount, 50),
                        2,
                    ),
                    "transaction_type":
                        "income",
                    "merchant_name":
                        (
                            "Gig Platform"
                            if employment_type
                            == "gig_worker"
                            else
                            "Business Income"
                        ),
                    "merchant_category":
                        "income",
                }
            )

    return transaction_records

def generate_recurring_expenses(
    rng: np.random.Generator,
    user_id: str,
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:

    records = []

    current_month = pd.Timestamp(
        start_date.year,
        start_date.month,
        1,
    )

    while current_month <= end_date:

        # Housing
        housing_date = (
            current_month
            + pd.Timedelta(
                days=int(
                    rng.integers(0, 5)
                )
            )
        )

        if housing_date >= start_date:

            housing_amount = rng.uniform(
                700,
                2200,
            )

            records.append(
                {
                    "user_id": user_id,
                    "account_id": account_id,
                    "transaction_timestamp":
                        housing_date,
                    "amount":
                        round(
                            -housing_amount,
                            2,
                        ),
                    "transaction_type":
                        "rent",
                    "merchant_name":
                        "Housing Payment",
                    "merchant_category":
                        "housing",
                }
            )

        # Utilities
        utility_date = (
            current_month
            + pd.Timedelta(
                days=int(
                    rng.integers(
                        10,
                        24,
                    )
                )
            )
        )

        if (
            utility_date >= start_date
            and utility_date <= end_date
        ):

            amount = rng.uniform(
                80,
                350,
            )

            records.append(
                {
                    "user_id": user_id,
                    "account_id": account_id,
                    "transaction_timestamp":
                        utility_date,
                    "amount":
                        round(-amount, 2),
                    "transaction_type":
                        "utility",
                    "merchant_name":
                        "Monthly Utilities",
                    "merchant_category":
                        "utilities",
                }
            )

        current_month = (
            current_month
            + pd.offsets.MonthBegin(1)
        )

    return records

def generate_spending_transactions(
    rng: np.random.Generator,
    user_id: str,
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_monthly_events: int,
    max_monthly_events: int,
) -> list[dict]:

    records = []

    history_days = (end_date - start_date).days
    months = max(1, history_days / 30)

    monthly_events = int(
        rng.integers(
            min_monthly_events,
            max_monthly_events + 1,
        )
    )

    event_count = int(monthly_events * months)

    categories = list(EXPENSE_CATEGORIES.keys())

    category_probabilities = [
        0.28,
        0.19,
        0.16,
        0.10,
        0.15,
        0.05,
        0.07,
    ]

    for _ in range(event_count):

        category = rng.choice(
            categories,
            p=category_probabilities,
        )

        minimum, maximum = EXPENSE_CATEGORIES[category]

        amount = rng.uniform(
            minimum,
            maximum,
        )

        merchant = rng.choice(
            MERCHANTS[category]
        )

        timestamp = random_timestamp(
            rng,
            start_date,
            end_date,
        )

        records.append(
            {
                "user_id": user_id,
                "account_id": account_id,
                "transaction_timestamp": timestamp,
                "amount": round(-amount, 2),
                "transaction_type": "purchase",
                "merchant_name": merchant,
                "merchant_category": category,
            }
        )

    return records

def generate_bnpl_payments(
    rng: np.random.Generator,
    user_id: str,
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:

    records = []

    uses_bnpl = (
        rng.random() < 0.32
    )

    if not uses_bnpl:
        return records

    providers = [
        "BNPL Provider A",
        "BNPL Provider B",
        "BNPL Provider C",
    ]

    number_payments = int(
        rng.integers(
            1,
            10,
        )
    )

    for _ in range(
        number_payments
    ):

        timestamp = random_timestamp(
            rng,
            start_date,
            end_date,
        )

        amount = rng.uniform(
            20,
            180,
        )

        records.append(
            {
                "user_id": user_id,
                "account_id": account_id,
                "transaction_timestamp":
                    timestamp,
                "amount":
                    round(-amount, 2),
                "transaction_type":
                    "bnpl_payment",
                "merchant_name":
                    rng.choice(
                        providers
                    ),
                "merchant_category":
                    "bnpl",
            }
        )

    return records

def generate_transfers(
    rng: np.random.Generator,
    user_id: str,
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:

    records = []

    number_transfers = int(
        rng.integers(
            0,
            8,
        )
    )

    for _ in range(
        number_transfers
    ):

        timestamp = random_timestamp(
            rng,
            start_date,
            end_date,
        )

        direction = rng.choice(
            [-1, 1]
        )

        amount = rng.uniform(
            25,
            600,
        )

        records.append(
            {
                "user_id": user_id,
                "account_id": account_id,
                "transaction_timestamp":
                    timestamp,
                "amount":
                    round(
                        direction * amount,
                        2,
                    ),
                "transaction_type":
                    "transfer",
                "merchant_name":
                    "Account Transfer",
                "merchant_category":
                    "transfer",
            }
        )

    return records

def add_running_balances(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:

    starting_balances = (
        accounts[
            [
                "account_id",
                "current_balance",
            ]
        ]
        .set_index("account_id")[
            "current_balance"
        ]
        .to_dict()
    )

    transactions = (
        transactions.copy()
    )

    transactions[
        "balance_change"
    ] = (
        transactions.groupby(
            "account_id"
        )["amount"]
        .cumsum()
    )

    transactions[
        "starting_balance"
    ] = (
        transactions[
            "account_id"
        ]
        .map(
            starting_balances
        )
    )

    transactions[
        "balance_after_transaction"
    ] = (
        transactions[
            "starting_balance"
        ]
        + transactions[
            "balance_change"
        ]
    )

    transactions[
        "balance_after_transaction"
    ] = (
        transactions[
            "balance_after_transaction"
        ]
        .round(2)
    )

    return transactions.drop(
        columns=[
            "balance_change",
            "starting_balance",
        ]
    )

def generate_transactions(
    users: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:

    config = (
        load_data_generation_config()
    )

    seed = config["portfolio"]["seed"]
    rng = np.random.default_rng(
        seed + 2
    )

    transaction_config = (
        config["portfolio"][
            "transactions"
        ]
    )

    history_days = (
        transaction_config[
            "history_days"
        ]
    )

    start_date = (
        TRANSACTION_REFERENCE_DATE
        - pd.Timedelta(
            days=history_days
        )
    )

    end_date = (
        TRANSACTION_REFERENCE_DATE
    )

    user_employment = (
        users[
            [
                "user_id",
                "employment_type",
            ]
        ]
        .set_index("user_id")[
            "employment_type"
        ]
        .to_dict()
    )

    primary_accounts = (
        accounts[
            accounts[
                "primary_account_flag"
            ]
            == 1
        ]
    )

    all_records = []

    for account in (
        primary_accounts.itertuples(
            index=False
        )
    ):

        user_id = account.user_id
        account_id = account.account_id

        employment_type = (
            user_employment[user_id]
        )

        all_records.extend(
            generate_income_transactions(
                rng,
                user_id,
                account_id,
                employment_type,
                start_date,
                end_date,
            )
        )

        all_records.extend(
            generate_recurring_expenses(
                rng,
                user_id,
                account_id,
                start_date,
                end_date,
            )
        )

        all_records.extend(
            generate_spending_transactions(
                rng,
                user_id,
                account_id,
                start_date,
                end_date,
                transaction_config[
                    "monthly_spend_events"
                ]["min"],
                transaction_config[
                    "monthly_spend_events"
                ]["max"],
            )
        )

        all_records.extend(
            generate_bnpl_payments(
                rng,
                user_id,
                account_id,
                start_date,
                end_date,
            )
        )

        all_records.extend(
            generate_transfers(
                rng,
                user_id,
                account_id,
                start_date,
                end_date,
            )
        )

    transactions = pd.DataFrame(
        all_records
    )

    transactions = (
        transactions
        .sort_values(
            [
                "account_id",
                "transaction_timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    transactions.insert(
        0,
        "transaction_id",
        [
            f"T{i:010d}"
            for i in range(
                1,
                len(transactions) + 1,
            )
        ],
    )

    transactions = add_running_balances(
    transactions,
    accounts,
    )

    return transactions

