from pathlib import Path

from src.data_generation.generate_accounts import generate_accounts
from src.data_generation.generate_users import generate_users
from src.data_generation.generate_transactions import (
    generate_transactions,
)
from src.data_generation.generate_applications import (
    generate_applications,
)

from src.data_generation.generate_loans import (
    generate_loans,
)

from src.data_generation.generate_repayments import (
    generate_repayments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def generate_portfolio():
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating users...")
    users = generate_users()

    users.to_parquet(
        RAW_DATA_DIR / "users.parquet",
        index=False,
    )

    print(f"Users generated: {len(users):,}")

    print("Generating bank accounts...")
    accounts = generate_accounts(users)

    accounts.to_parquet(
        RAW_DATA_DIR / "bank_accounts.parquet",
        index=False,
    )

    print(f"Bank accounts generated: {len(accounts):,}")

    print("Generating transactions...")

    transactions = generate_transactions(
        users,
        accounts,
    )

    transactions.to_parquet(
        RAW_DATA_DIR
        / "transactions.parquet",
        index=False,
    )

    print(
        f"Transactions generated: "
        f"{len(transactions):,}"
    )

    print("Generating applications...")

    applications = generate_applications(
        users
    )

    applications.to_parquet(
        RAW_DATA_DIR
        / "applications.parquet",
        index=False,
    )

    print(
        f"Applications generated: "
        f"{len(applications):,}"
    )

    print("Generating loans...")

    loans = generate_loans(
        applications
    )

    loans.to_parquet(
        RAW_DATA_DIR
        / "loans.parquet",
        index=False,
    )

    print(
        f"Loans generated: "
        f"{len(loans):,}"
    )

    print(
    "Generating repayments "
    "and credit outcomes..."
)

    repayments, outcomes = (
        generate_repayments(
            loans=loans,
            applications=applications,
            users=users,
            transactions=transactions,
        )
    )

    repayments.to_parquet(
        RAW_DATA_DIR
        / "repayments.parquet",
        index=False,
    )

    outcomes.to_parquet(
        RAW_DATA_DIR
        / "loan_outcomes.parquet",
        index=False,
    )

    print(
        f"Repayments generated: "
        f"{len(repayments):,}"
    )

    print(
        f"Defaults generated: "
        f"{outcomes['default_status'].sum():,}"
    )

    print(
        "Default rate: "
        f"{outcomes['default_status'].mean():.2%}"
    )

    print("Portfolio generation completed.")


if __name__ == "__main__":
    generate_portfolio()