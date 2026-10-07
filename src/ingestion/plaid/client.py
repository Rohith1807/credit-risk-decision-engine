import os

import plaid
from dotenv import load_dotenv
from plaid.api import plaid_api


load_dotenv()


def get_plaid_client():

    client_id = os.getenv(
        "PLAID_CLIENT_ID"
    )

    secret = os.getenv(
        "PLAID_SECRET"
    )

    environment = os.getenv(
        "PLAID_ENV",
        "sandbox",
    )

    if not client_id:
        raise ValueError(
            "PLAID_CLIENT_ID is missing."
        )

    if not secret:
        raise ValueError(
            "PLAID_SECRET is missing."
        )

    environments = {
        "sandbox":
            plaid.Environment.Sandbox,

        "production":
            plaid.Environment.Production,
    }

    if environment not in environments:
        raise ValueError(
            f"Unsupported PLAID_ENV: "
            f"{environment}"
        )

    configuration = (
        plaid.Configuration(
            host=environments[
                environment
            ],
            api_key={
                "clientId":
                    client_id,
                "secret":
                    secret,
            },
        )
    )

    api_client = plaid.ApiClient(
        configuration
    )

    return plaid_api.PlaidApi(
        api_client
    )