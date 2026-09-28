from src.data_generation.generate_applications import (
    generate_applications,
)
from src.data_generation.generate_users import (
    generate_users,
)


def build_test_applications():
    users = (
        generate_users()
        .head(200)
        .copy()
    )

    applications = (
        generate_applications(
            users
        )
    )

    return (
        users,
        applications,
    )


def test_application_ids_unique():
    _, applications = (
        build_test_applications()
    )

    assert applications[
        "application_id"
    ].is_unique


def test_applications_have_valid_users():
    users, applications = (
        build_test_applications()
    )

    assert set(
        applications["user_id"]
    ).issubset(
        set(users["user_id"])
    )


def test_requested_amount_positive():
    _, applications = (
        build_test_applications()
    )

    assert (
        applications[
            "requested_amount"
        ]
        > 0
    ).all()


def test_bank_data_flag_valid():
    _, applications = (
        build_test_applications()
    )

    assert set(
        applications[
            "bank_data_available"
        ]
    ).issubset(
        {0, 1}
    )


def test_application_timestamp_present():
    _, applications = (
        build_test_applications()
    )

    assert applications[
        "application_timestamp"
    ].notna().all()


def test_application_not_before_signup():
    users, applications = (
        build_test_applications()
    )

    merged = (
        applications.merge(
            users[
                [
                    "user_id",
                    "signup_date",
                ]
            ],
            on="user_id",
            how="left",
        )
    )

    assert (
        merged[
            "application_timestamp"
        ]
        >= merged[
            "signup_date"
        ]
    ).all()