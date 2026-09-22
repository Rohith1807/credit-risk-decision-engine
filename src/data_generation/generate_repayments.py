import numpy as np
import pandas as pd

from src.data_generation.config import (
    load_data_generation_config,
)
from src.data_generation.risk_signal import (
    calculate_behavioral_risk_metrics,
    calculate_latent_default_probability,
)

def generate_payment_schedule(
    loan: dict,
) -> list[dict]:

    installment_count = int(
        loan["installment_count"]
    )

    approved_amount = float(
        loan["approved_amount"]
    )

    installment_amount = (
        approved_amount
        / installment_count
    )

    origination_date = pd.Timestamp(
        loan["origination_date"]
    )

    schedule = []

    for installment_number in range(
        1,
        installment_count + 1,
    ):

        scheduled_date = (
            origination_date
            + pd.Timedelta(
                days=14
                * installment_number
            )
        )

        schedule.append(
            {
                "installment_number":
                    installment_number,
                "scheduled_payment_date":
                    scheduled_date,
                "amount_due":
                    round(
                        installment_amount,
                        2,
                    ),
            }
        )

    return schedule

def simulate_installment_outcome(
    rng: np.random.Generator,
    defaulted: bool,
    installment_number: int,
) -> tuple[int, str]:
    """
    Return:
        days_past_due,
        payment_status
    """

    if defaulted:

        # Defaults generally emerge later,
        # rather than immediately on installment 1.
        if installment_number >= 3:

            days_past_due = int(
                rng.integers(
                    90,
                    151,
                )
            )

            return (
                days_past_due,
                "defaulted",
            )

    random_value = rng.random()

    if random_value < 0.88:

        return (
            0,
            "paid",
        )

    elif random_value < 0.97:

        return (
            int(
                rng.integers(
                    1,
                    15,
                )
            ),
            "late",
        )

    else:

        return (
            int(
                rng.integers(
                    15,
                    45,
                )
            ),
            "late",
        )


def generate_repayments(
    loans: pd.DataFrame,
    applications: pd.DataFrame,
    users: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    config = (
        load_data_generation_config()
    )

    seed = config["portfolio"]["seed"]

    rng = np.random.default_rng(
        seed + 5
    )

    app_lookup = (
        applications
        .set_index(
            "application_id"
        )
        .to_dict(
            orient="index"
        )
    )

    user_lookup = (
        users
        .set_index(
            "user_id"
        )
        .to_dict(
            orient="index"
        )
    )

    repayment_records = []
    outcome_records = []

    payment_counter = 1

    for loan in loans.to_dict(
        orient="records"
    ):

        application = app_lookup[
            loan[
                "application_id"
            ]
        ]

        user = user_lookup[
            loan["user_id"]
        ]

        behavioral_metrics = (
            calculate_behavioral_risk_metrics(
                user_id=
                    loan["user_id"],

                application_timestamp=
                    pd.Timestamp(
                        application[
                            "application_timestamp"
                        ]
                    ),

                transactions=
                    transactions,
            )
        )

        latent_pd = (
            calculate_latent_default_probability(
                requested_amount=
                    float(
                        application[
                            "requested_amount"
                        ]
                    ),

                employment_type=
                    user[
                        "employment_type"
                    ],

                behavioral_metrics=
                    behavioral_metrics,
            )
        )

        defaulted = bool(
            rng.random()
            < latent_pd
        )

        schedule = (
            generate_payment_schedule(
                loan
            )
        )

        maximum_dpd = 0

        for installment in schedule:

            (
                days_past_due,
                payment_status,
            ) = (
                simulate_installment_outcome(
                    rng,
                    defaulted,
                    installment[
                        "installment_number"
                    ],
                )
            )

            scheduled_date = (
                installment[
                    "scheduled_payment_date"
                ]
            )

            actual_payment_date = (
                scheduled_date
                + pd.Timedelta(
                    days=days_past_due
                )
            )

            if (
                payment_status
                == "defaulted"
            ):

                amount_paid = 0.0

            else:

                amount_paid = (
                    installment[
                        "amount_due"
                    ]
                )

            maximum_dpd = max(
                maximum_dpd,
                days_past_due,
            )

            repayment_records.append(
                {
                    "payment_id":
                        f"P{payment_counter:010d}",

                    "loan_id":
                        loan["loan_id"],

                    "installment_number":
                        installment[
                            "installment_number"
                        ],

                    "scheduled_payment_date":
                        scheduled_date,

                    "actual_payment_date":
                        actual_payment_date,

                    "amount_due":
                        installment[
                            "amount_due"
                        ],

                    "amount_paid":
                        round(
                            amount_paid,
                            2,
                        ),

                    "days_past_due":
                        days_past_due,

                    "payment_status":
                        payment_status,
                }
            )

            payment_counter += 1

        outcome_records.append(
            {
                "loan_id":
                    loan["loan_id"],

                "application_id":
                    loan[
                        "application_id"
                    ],

                "user_id":
                    loan["user_id"],

                "latent_pd":
                    round(
                        latent_pd,
                        6,
                    ),

                "max_days_past_due":
                    maximum_dpd,

                "default_status":
                    int(
                        maximum_dpd >= 90
                    ),
            }
        )

    repayments = pd.DataFrame(
        repayment_records
    )

    outcomes = pd.DataFrame(
        outcome_records
    )

    return (
        repayments,
        outcomes,
    )