from src.policy.economics import (
    expected_loss,
    expected_profit,
)


def test_expected_loss():

    result = expected_loss(
        probability_default=0.05,
        exposure=1000,
        loss_given_default=0.85,
    )

    assert result == 42.5


def test_expected_profit():

    result = expected_profit(
        probability_default=0.05,
        exposure=1000,
        loss_given_default=0.85,
        merchant_fee_rate=0.06,
        funding_cost_rate=0.01,
        servicing_cost=3.0,
    )

    assert round(
        result,
        2,
    ) == 4.50


from src.policy.decision_engine import (
    make_credit_decision,
)


TEST_CONFIG = {
    "policy": {
        "thresholds": {
            "low_risk_max_pd": 0.03,
            "medium_risk_max_pd": 0.08,
        },
        "limits": {
            "full_approval_max": 1000,
            "low_and_grow_max": 100,
            "fallback_max": 50,
        },
        "economics": {
            "loss_given_default": 0.85,
            "merchant_fee_rate": 0.06,
            "funding_cost_rate": 0.01,
            "servicing_cost": 3.0,
        },
        "confidence": {
            "minimum_full_approval_score":
                0.60,
        },
        "guardrails": {
            "max_loan_to_income_ratio":
                0.35,
            "max_bnpl_payment_burden":
                0.20,
            "max_negative_balance_rate_30d":
                0.50,
        },
    }
}


def test_low_risk_full_approval():

    result = make_credit_decision(
        probability_default=0.02,
        requested_amount=500,
        bank_data_available=1,
        data_confidence_score=0.90,
        loan_to_income_ratio=0.10,
        bnpl_payment_burden=0.05,
        negative_balance_rate_30d=0.05,
        config=TEST_CONFIG,
    )

    assert result.decision == "APPROVE"

    assert (
        result.policy_tier
        == "FULL_APPROVAL"
    )

    assert result.approved_limit == 500


def test_medium_risk_low_and_grow():

    result = make_credit_decision(
        probability_default=0.05,
        requested_amount=500,
        bank_data_available=1,
        data_confidence_score=0.90,
        loan_to_income_ratio=0.10,
        bnpl_payment_burden=0.05,
        negative_balance_rate_30d=0.05,
        config=TEST_CONFIG,
    )

    assert (
        result.policy_tier
        == "LOW_AND_GROW"
    )

    assert result.approved_limit == 100


def test_high_risk_rejected():

    result = make_credit_decision(
        probability_default=0.15,
        requested_amount=500,
        bank_data_available=1,
        data_confidence_score=0.90,
        loan_to_income_ratio=0.10,
        bnpl_payment_burden=0.05,
        negative_balance_rate_30d=0.05,
        config=TEST_CONFIG,
    )

    assert result.decision == "REJECT"

    assert result.approved_limit == 0


def test_missing_bank_data_fallback():

    result = make_credit_decision(
        probability_default=0.04,
        requested_amount=500,
        bank_data_available=0,
        data_confidence_score=None,
        loan_to_income_ratio=None,
        bnpl_payment_burden=None,
        negative_balance_rate_30d=None,
        config=TEST_CONFIG,
    )

    assert result.decision == "APPROVE"

    assert result.policy_tier == "FALLBACK"

    assert result.approved_limit == 50

import pandas as pd

from src.policy.portfolio_metrics import (
    calculate_policy_metrics,
)


def test_policy_metrics():
    dataframe = pd.DataFrame(
        {
            "decision": [
                "APPROVE",
                "APPROVE",
                "REJECT",
            ],
            "requested_amount": [
                100,
                200,
                300,
            ],
            "approved_limit": [
                100,
                100,
                0,
            ],
            "expected_loss": [
                5,
                10,
                0,
            ],
            "expected_profit": [
                4,
                3,
                0,
            ],
            "default_status": [
                0,
                1,
                1,
            ],
        }
    )

    metrics = (
        calculate_policy_metrics(
            dataframe
        )
    )

    assert (
        metrics[
            "applications"
        ]
        == 3
    )

    assert (
        metrics[
            "approved_count"
        ]
        == 2
    )

    assert round(
        metrics[
            "approval_rate"
        ],
        4,
    ) == round(
        2 / 3,
        4,
    )

    assert (
        metrics[
            "approved_gmv"
        ]
        == 200
    )

def test_behavioral_guardrails_can_be_disabled():

    with_guardrails = make_credit_decision(
        probability_default=0.02,
        requested_amount=500,
        bank_data_available=1,
        data_confidence_score=0.90,
        loan_to_income_ratio=0.50,
        bnpl_payment_burden=0.05,
        negative_balance_rate_30d=0.05,
        config=TEST_CONFIG,
        apply_behavioral_guardrails=True,
    )

    without_guardrails = make_credit_decision(
        probability_default=0.02,
        requested_amount=500,
        bank_data_available=1,
        data_confidence_score=0.90,
        loan_to_income_ratio=0.50,
        bnpl_payment_burden=0.05,
        negative_balance_rate_30d=0.05,
        config=TEST_CONFIG,
        apply_behavioral_guardrails=False,
    )

    assert (
        with_guardrails.decision
        == "REJECT"
    )

    assert (
        without_guardrails.decision
        == "APPROVE"
    )