import numpy as np
import pandas as pd

from src.data_generation.config import load_data_generation_config


def generate_accounts(
    users: pd.DataFrame,
) -> pd.DataFrame:

    config = load_data_generation_config()
    seed = config["portfolio"]["seed"]

    rng = np.random.default_rng(seed + 1)

    min_accounts = config["portfolio"]["accounts_per_user"]["min"]
    max_accounts = config["portfolio"]["accounts_per_user"]["max"]

    records = []
    account_counter = 1

    for row in users.itertuples(index=False):

        number_accounts = int(
            rng.integers(
                min_accounts,
                max_accounts + 1,
            )
        )

        for account_index in range(number_accounts):

            account_type = (
                "checking"
                if account_index == 0
                else rng.choice(["checking", "savings"])
            )

            current_balance = max(
                0,
                rng.lognormal(
                    mean=7.2,
                    sigma=1.0,
                ),
            )

            available_balance = max(
                0,
                current_balance
                - rng.uniform(0, min(current_balance, 250)),
            )

            account_open_date = (
                pd.Timestamp(row.signup_date)
                - pd.to_timedelta(
                    rng.integers(30, 1500),
                    unit="D",
                )
            )

            records.append(
                {
                    "account_id": f"A{account_counter:08d}",
                    "user_id": row.user_id,
                    "account_type": account_type,
                    "account_open_date": account_open_date,
                    "current_balance": round(current_balance, 2),
                    "available_balance": round(available_balance, 2),
                    "primary_account_flag": int(account_index == 0),
                }
            )

            account_counter += 1

    return pd.DataFrame(records)