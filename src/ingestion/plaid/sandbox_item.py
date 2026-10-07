import os

from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.products import Products
from plaid.model.sandbox_public_token_create_request import (
    SandboxPublicTokenCreateRequest,
)

from src.ingestion.plaid.client import (
    get_plaid_client,
)

from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

def create_sandbox_item():

    client = get_plaid_client()

    institution_id = os.getenv(
        "PLAID_INSTITUTION_ID",
        "ins_109508",
    )

    request = (
        SandboxPublicTokenCreateRequest(
            institution_id=
                institution_id,
            initial_products=[
                Products(
                    "transactions"
                )
            ],
        )
    )

    response = (
        client
        .sandbox_public_token_create(
            request
        )
    )

    public_token = (
        response[
            "public_token"
        ]
    )

    exchange_request = (
        ItemPublicTokenExchangeRequest(
            public_token=
                public_token
        )
    )

    exchange_response = (
        client
        .item_public_token_exchange(
            exchange_request
        )
    )

    return {
        "access_token":
            exchange_response[
                "access_token"
            ],

        "item_id":
            exchange_response[
                "item_id"
            ],
    }

def save_access_token(
    access_token: str,
):

    path = (
        PROJECT_ROOT
        / "data"
        / "secrets"
        / "plaid_access_token.txt"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        access_token,
        encoding="utf-8",
    )


if __name__ == "__main__":

    result = create_sandbox_item()

    save_access_token(
        result["access_token"]
    )

    print(
        "Sandbox Item created."
    )

    print(
        "Item ID:",
        result["item_id"],
    )

    print(
        "Access token obtained:",
        bool(
            result["access_token"]
        ),
    )