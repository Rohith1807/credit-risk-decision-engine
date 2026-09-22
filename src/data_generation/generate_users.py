from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_generation.config import load_data_generation_config


AGE_BANDS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]

REGIONS = [
    "Midwest",
    "Northeast",
    "South",
    "West",
]

EMPLOYMENT_TYPES = [
    "salaried",
    "hourly",
    "self_employed",
    "gig_worker",
    "student",
]


def generate_users() -> pd.DataFrame:
    config = load_data_generation_config()

    seed = config["portfolio"]["seed"]
    user_count = config["portfolio"]["users"]["count"]

    rng = np.random.default_rng(seed)

    user_ids = [
        f"U{i:07d}"
        for i in range(1, user_count + 1)
    ]

    reference_date = datetime(2026, 1, 1)

    tenure_months = rng.integers(
        low=1,
        high=49,
        size=user_count,
    )

    signup_dates = [
        reference_date - timedelta(days=int(months * 30))
        for months in tenure_months
    ]

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": signup_dates,
            "age_band": rng.choice(
                AGE_BANDS,
                size=user_count,
                p=[0.16, 0.31, 0.23, 0.15, 0.10, 0.05],
            ),
            "region": rng.choice(
                REGIONS,
                size=user_count,
                p=[0.23, 0.18, 0.37, 0.22],
            ),
            "employment_type": rng.choice(
                EMPLOYMENT_TYPES,
                size=user_count,
                p=[0.45, 0.22, 0.12, 0.14, 0.07],
            ),
            "account_tenure_months": tenure_months,
        }
    )

    return users


if __name__ == "__main__":
    df = generate_users()

    output_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "users.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Generated {len(df):,} users")
    print(f"Saved to: {output_path}")