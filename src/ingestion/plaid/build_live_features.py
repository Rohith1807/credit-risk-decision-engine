from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

PLAID_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "plaid_transactions.parquet"
)


def load_plaid_transactions():
    return pd.read_parquet(
        PLAID_PATH
    )


def prepare_internal_transactions(
    user_id: str = "plaid_demo_user",
):
    transactions = (
        load_plaid_transactions()
        .copy()
    )

    transactions[
        "user_id"
    ] = user_id

    if (
        "transaction_type"
        not in transactions.columns
    ):
        transactions[
            "transaction_type"
        ] = transactions[
            "amount"
        ].apply(
            lambda value:
                "income"
                if value > 0
                else "expense"
        )

    if (
        "balance_after_transaction"
        not in transactions.columns
    ):
        transactions[
            "balance_after_transaction"
        ] = pd.NA

    return transactions


if __name__ == "__main__":

    dataframe = (
        prepare_internal_transactions()
    )

    print(
        dataframe.head()
    )

    print(
        "\nRows:",
        len(dataframe),
    )

    print(
        "\nColumns:"
    )

    print(
        dataframe.columns.tolist()
    )