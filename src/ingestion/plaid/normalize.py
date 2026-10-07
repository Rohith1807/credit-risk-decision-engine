import pandas as pd


def normalize_transactions(
    transactions,
) -> pd.DataFrame:

    rows = []

    for transaction in transactions:

        merchant_name = (
            transaction.get(
                "merchant_name"
            )
            or transaction.get(
                "name"
            )
        )

        personal_finance_category = (
            transaction.get(
                "personal_finance_category"
            )
        )

        if personal_finance_category:
            category = (
                personal_finance_category
                .get(
                    "primary"
                )
            )
        else:
            category = None

        # Plaid typically represents
        # spending as positive amounts.
        # Our internal schema uses
        # spending as negative cash flow.
        plaid_amount = float(
            transaction[
                "amount"
            ]
        )

        internal_amount = (
            -plaid_amount
        )

        rows.append(
            {
                "transaction_id":
                    transaction[
                        "transaction_id"
                    ],

                "account_id":
                    transaction[
                        "account_id"
                    ],

                "transaction_timestamp":
                    pd.to_datetime(
                        transaction[
                            "date"
                        ]
                    ),

                "amount":
                    internal_amount,

                "merchant_name":
                    merchant_name,

                "merchant_category":
                    category,

                "pending":
                    transaction.get(
                        "pending",
                        False,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )