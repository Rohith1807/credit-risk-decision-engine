from pathlib import Path

from plaid.model.transactions_sync_request import (
    TransactionsSyncRequest,
)

from src.ingestion.plaid.client import (
    get_plaid_client,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


def load_access_token():

    path = (
        PROJECT_ROOT
        / "data"
        / "secrets"
        / "plaid_access_token.txt"
    )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def fetch_transactions():

    client = get_plaid_client()

    access_token = (
        load_access_token()
    )

    cursor = None

    added = []
    modified = []
    removed = []

    while True:

        if cursor is None:
            request = TransactionsSyncRequest(
                access_token=access_token,
            )
        else:
            request = TransactionsSyncRequest(
                access_token=access_token,
                cursor=cursor,
            )

        response = (
            client
            .transactions_sync(
                request
            )
        )

        added.extend(
            response[
                "added"
            ]
        )

        modified.extend(
            response[
                "modified"
            ]
        )

        removed.extend(
            response[
                "removed"
            ]
        )

        cursor = response[
            "next_cursor"
        ]

        if not response[
            "has_more"
        ]:
            break

    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "cursor": cursor,
    }


if __name__ == "__main__":

    result = fetch_transactions()

    print(
        "Added:",
        len(
            result[
                "added"
            ]
        ),
    )

    print(
        "Modified:",
        len(
            result[
                "modified"
            ]
        ),
    )

    print(
        "Removed:",
        len(
            result[
                "removed"
            ]
        ),
    )