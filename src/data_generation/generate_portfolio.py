from pathlib import Path

from src.data_generation.generate_accounts import generate_accounts
from src.data_generation.generate_users import generate_users


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

    print("Portfolio generation completed.")


if __name__ == "__main__":
    generate_portfolio()