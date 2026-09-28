from src.data_generation.generate_applications import (
    generate_applications,
)
from src.data_generation.generate_loans import (
    generate_loans,
)
from src.data_generation.generate_users import (
    generate_users,
)


def build_test_loans():

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

    loans = generate_loans(
        applications
    )

    return (
        applications,
        loans,
    )


def test_loan_ids_unique():

    _, loans = build_test_loans()

    assert loans[
        "loan_id"
    ].is_unique


def test_loans_reference_valid_applications():

    applications, loans = (
        build_test_loans()
    )

    assert set(
        loans[
            "application_id"
        ]
    ).issubset(
        set(
            applications[
                "application_id"
            ]
        )
    )


def test_approved_amount_positive():

    _, loans = build_test_loans()

    assert (
        loans[
            "approved_amount"
        ]
        > 0
    ).all()


def test_installment_count_valid():

    _, loans = build_test_loans()

    assert (
        loans[
            "installment_count"
        ]
        == 4
    ).all()