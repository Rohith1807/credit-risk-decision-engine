import numpy as np
import pandas as pd

from src.data_generation.config import load_data_generation_config


def calculate_historical_approval_probability(
    application: pd.Series,
) -> float:
    """
    Simulate a simple historical underwriting policy.

    This is intentionally simpler than the ML model we will build later.
    """

    probability = 0.86

    requested_amount = application["requested_amount"]

    # Larger exposures receive slightly more conservative treatment.
    if requested_amount > 1000:
        probability -= 0.14
    elif requested_amount > 600:
        probability -= 0.07

    # Missing bank telemetry makes the historical policy more conservative.
    if application["bank_data_available"] == 0:
        probability -= 0.08

    # Small requests are easier to approve.
    if requested_amount <= 150:
        probability += 0.04

    return float(
        np.clip(
            probability,
            0.10,
            0.98,
        )
    )

def generate_loans(
    applications: pd.DataFrame,
) -> pd.DataFrame:

    config = load_data_generation_config()

    seed = config["portfolio"]["seed"]
    rng = np.random.default_rng(seed + 4)

    loan_config = config["portfolio"]["loan_generation"]

    term_days = loan_config["term_days"]
    installment_count = loan_config["installment_count"]
    interest_rate = loan_config["interest_rate"]

    records = []
    loan_counter = 1

    for application in applications.to_dict(
        orient="records"
    ):

        approval_probability = (
            calculate_historical_approval_probability(
                application
            )
        )

        approved = (
            rng.random()
            < approval_probability
        )

        if not approved:
            continue

        approved_amount = float(
            application[
                "requested_amount"
            ]
        )

        origination_date = pd.Timestamp(
            application[
                "application_timestamp"
            ]
        )

        records.append(
            {
                "loan_id":
                    f"L{loan_counter:08d}",
                "application_id":
                    application[
                        "application_id"
                    ],
                "user_id":
                    application["user_id"],
                "approved_amount":
                    round(
                        approved_amount,
                        2,
                    ),
                "origination_date":
                    origination_date,
                "term_days":
                    term_days,
                "installment_count":
                    installment_count,
                "interest_rate":
                    interest_rate,
            }
        )

        loan_counter += 1

    loans = pd.DataFrame(records)

    return loans