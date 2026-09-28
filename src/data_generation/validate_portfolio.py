import pandas as pd


def validate_portfolio():
    users = pd.read_parquet("data/raw/users.parquet")
    accounts = pd.read_parquet("data/raw/bank_accounts.parquet")
    transactions = pd.read_parquet("data/raw/transactions.parquet")
    applications = pd.read_parquet("data/raw/applications.parquet")
    loans = pd.read_parquet("data/raw/loans.parquet")
    repayments = pd.read_parquet("data/raw/repayments.parquet")
    outcomes = pd.read_parquet("data/raw/loan_outcomes.parquet")

    print("\n=== PORTFOLIO SIZE ===")
    print(f"Users: {len(users):,}")
    print(f"Accounts: {len(accounts):,}")
    print(f"Transactions: {len(transactions):,}")
    print(f"Applications: {len(applications):,}")
    print(f"Loans: {len(loans):,}")
    print(f"Repayments: {len(repayments):,}")

    print("\n=== BUSINESS RATES ===")
    approval_rate = len(loans) / len(applications)
    default_rate = outcomes["default_status"].mean()

    print(f"Historical approval rate: {approval_rate:.2%}")
    print(f"Observed default rate: {default_rate:.2%}")

    print("\n=== LATENT PD ===")
    print(outcomes["latent_pd"].describe())

    print("\n=== DEFAULT RATE BY PD QUINTILE ===")

    outcomes["pd_bucket"] = pd.qcut(
        outcomes["latent_pd"],
        q=5,
        duplicates="drop",
    )

    print(
        outcomes.groupby(
            "pd_bucket",
            observed=True,
        )["default_status"]
        .agg(["count", "mean"])
    )

    print("\n=== BANK DATA AVAILABILITY ===")
    print(
        applications[
            "bank_data_available"
        ].value_counts(normalize=True)
    )

    print("\n=== APPLICATION AMOUNTS ===")
    print(
        applications[
            "requested_amount"
        ].describe()
    )

    print("\n=== EMPLOYMENT MIX ===")
    print(
        users[
            "employment_type"
        ].value_counts(normalize=True)
    )


if __name__ == "__main__":
    validate_portfolio()