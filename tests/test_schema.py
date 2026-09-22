from src.data_generation.schema import (
    USER_COLUMNS,
    ACCOUNT_COLUMNS,
    TRANSACTION_COLUMNS,
    APPLICATION_COLUMNS,
    LOAN_COLUMNS,
    REPAYMENT_COLUMNS,
    DECISION_COLUMNS,
)


def test_user_schema_contains_primary_key():
    assert "user_id" in USER_COLUMNS


def test_account_schema_contains_foreign_key():
    assert "user_id" in ACCOUNT_COLUMNS


def test_transactions_have_timestamp():
    assert "transaction_timestamp" in TRANSACTION_COLUMNS


def test_applications_have_decision_timestamp():
    assert "application_timestamp" in APPLICATION_COLUMNS


def test_loans_reference_application():
    assert "application_id" in LOAN_COLUMNS


def test_repayments_reference_loan():
    assert "loan_id" in REPAYMENT_COLUMNS


def test_decisions_reference_application():
    assert "application_id" in DECISION_COLUMNS