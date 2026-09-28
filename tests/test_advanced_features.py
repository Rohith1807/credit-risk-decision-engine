from src.features.advanced_features import (
    calculate_cash_buffer_days,
    calculate_discretionary_spend_ratio,
    calculate_income_stability_score,
    calculate_loan_to_income_ratio,
)


def test_loan_to_income_ratio_positive():

    ratio = (
        calculate_loan_to_income_ratio(
            requested_amount=500,
            income_total_90d=9000,
        )
    )

    assert ratio > 0


def test_discretionary_ratio_bounded():

    ratio = (
        calculate_discretionary_spend_ratio(
            discretionary_spend_30d=300,
            spend_total_30d=1000,
        )
    )

    assert 0 <= ratio <= 1


def test_cash_buffer_non_negative():

    buffer_days = (
        calculate_cash_buffer_days(
            current_balance=1000,
            spend_total_30d=1500,
        )
    )

    assert buffer_days >= 0


def test_income_stability_bounded():

    score = (
        calculate_income_stability_score(
            income_cv_90d=0.25,
            income_count_90d=6,
        )
    )

    assert 0 <= score <= 1