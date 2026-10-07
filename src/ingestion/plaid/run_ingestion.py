from pathlib import Path

from src.ingestion.plaid.normalize import (
    normalize_transactions,
)
from src.ingestion.plaid.transactions import (
    fetch_transactions,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "plaid_transactions.parquet"
)


def main():

    result = fetch_transactions()

    dataframe = (
        normalize_transactions(
            result[
                "added"
            ]
        )
    )

    dataframe.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Transactions:",
        len(dataframe),
    )

    print(
        dataframe.head()
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()