from src.ingestion.plaid.normalize import (
    normalize_transactions,
)


def test_normalize_plaid_transaction():

    transactions = [
        {
            "transaction_id":
                "txn_001",

            "account_id":
                "acct_001",

            "date":
                "2026-01-15",

            "amount":
                25.50,

            "merchant_name":
                "Example Store",

            "name":
                "Example Store",

            "pending":
                False,

            "personal_finance_category":
                {
                    "primary":
                        "GENERAL_MERCHANDISE"
                },
        }
    ]

    result = (
        normalize_transactions(
            transactions
        )
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "amount"
        ]
        == -25.50
    )

    assert (
        result.iloc[0][
            "merchant_name"
        ]
        == "Example Store"
    )