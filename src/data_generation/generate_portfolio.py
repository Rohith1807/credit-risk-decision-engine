from pathlib import Path

from src.data_generation.generate_accounts import generate_accounts
from src.data_generation.generate_users import generate_users
from src.data_generation.generate_transactions import (
    generate_transactions,
)
from src.data_generation.generate_applications import (
    generate_applications,
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

    print("Portfolio generation completed.")


if __name__ == "__main__":
    generate_portfolio()