import numpy as np
import pandas as pd

from src.data_generation.config import load_data_generation_config


APPLICATION_REFERENCE_DATE = pd.Timestamp("2026-01-01")


MERCHANT_AMOUNT_RANGES = {
    "electronics": (100, 1500),
    "apparel": (25, 500),
    "home": (50, 1200),
    "travel": (100, 1500),
    "healthcare": (50, 1000),
    "education": (50, 1200),
    "entertainment": (25, 500),
    "general_retail": (25, 800),
}


MERCHANT_PROBABILITIES = {
    "electronics": 0.16,
    "apparel": 0.18,
    "home": 0.12,
    "travel": 0.08,
    "healthcare": 0.08,
    "education": 0.06,
    "entertainment": 0.12,
    "general_retail": 0.20,
}


DEVICE_TYPES = [
    "mobile",
    "desktop",
    "tablet",
]


DEVICE_PROBABILITIES = [
    0.72,
    0.23,
    0.05,
]


CHANNELS = [
    "checkout",
    "mobile_app",
    "web",
]


CHANNEL_PROBABILITIES = [
    0.50,
    0.32,
    0.18,
]

def generate_application_timestamp(
    rng: np.random.Generator,
    user_signup_date: pd.Timestamp,
    lookback_days: int,
) -> pd.Timestamp:
    """
    Generate an application timestamp within the underwriting window.

    Ensures the application does not occur before the user signed up.
    """

    earliest_allowed = (
        APPLICATION_REFERENCE_DATE
        - pd.Timedelta(days=lookback_days)
    )

    start_date = max(
        pd.Timestamp(user_signup_date),
        earliest_allowed,
    )

    end_date = APPLICATION_REFERENCE_DATE

    if start_date >= end_date:
        return end_date

    total_seconds = int(
        (end_date - start_date).total_seconds()
    )

    random_seconds = int(
        rng.integers(
            0,
            max(total_seconds, 1),
        )
    )

    return (
        start_date
        + pd.Timedelta(
            seconds=random_seconds
        )
    )

def generate_requested_amount(
    rng: np.random.Generator,
    merchant_category: str,
) -> float:
    """
    Generate a transaction amount appropriate for the merchant category.
    """

    minimum, maximum = (
        MERCHANT_AMOUNT_RANGES[
            merchant_category
        ]
    )

    amount = rng.lognormal(
        mean=np.log(
            max(
                minimum,
                (minimum + maximum) / 4,
            )
        ),
        sigma=0.65,
    )

    amount = np.clip(
        amount,
        minimum,
        maximum,
    )

    return round(
        float(amount),
        2,
    )

def generate_applications(
    users: pd.DataFrame,
) -> pd.DataFrame:

    config = (
        load_data_generation_config()
    )

    seed = config["portfolio"]["seed"]

    rng = np.random.default_rng(
        seed + 3
    )

    app_config = (
        config["portfolio"][
            "application_generation"
        ]
    )

    min_apps = (
        app_config[
            "applications_per_user"
        ]["min"]
    )

    max_apps = (
        app_config[
            "applications_per_user"
        ]["max"]
    )

    lookback_days = (
        app_config[
            "lookback_window_days"
        ]
    )

    bank_data_rate = (
        app_config[
            "bank_data_available_rate"
        ]
    )

    merchant_categories = list(
        MERCHANT_PROBABILITIES.keys()
    )

    merchant_probs = list(
        MERCHANT_PROBABILITIES.values()
    )

    records = []

    application_counter = 1

    for user in users.itertuples(
        index=False
    ):

        number_applications = int(
            rng.integers(
                min_apps,
                max_apps + 1,
            )
        )

        for _ in range(
            number_applications
        ):

            merchant_category = (
                rng.choice(
                    merchant_categories,
                    p=merchant_probs,
                )
            )

            requested_amount = (
                generate_requested_amount(
                    rng,
                    merchant_category,
                )
            )

            application_timestamp = (
                generate_application_timestamp(
                    rng,
                    user.signup_date,
                    lookback_days,
                )
            )

            device_type = rng.choice(
                DEVICE_TYPES,
                p=DEVICE_PROBABILITIES,
            )

            channel = rng.choice(
                CHANNELS,
                p=CHANNEL_PROBABILITIES,
            )

            bank_data_available = int(
                rng.random()
                < bank_data_rate
            )

            records.append(
                {
                    "application_id":
                        f"APP{application_counter:08d}",
                    "user_id":
                        user.user_id,
                    "application_timestamp":
                        application_timestamp,
                    "requested_amount":
                        requested_amount,
                    "merchant_category":
                        merchant_category,
                    "device_type":
                        device_type,
                    "channel":
                        channel,
                    "bank_data_available":
                        bank_data_available,
                }
            )

            application_counter += 1

    applications = pd.DataFrame(
        records
    )

    applications = (
        applications
        .sort_values(
            "application_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return applications