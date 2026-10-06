from src.policy.scenario_analysis import (
    build_policy_scenarios,
)


BASE_POLICY_CONFIG = {
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
            "servicing_cost": 3.00,
            "funding_cost_rate": 0.01,
        },
        "confidence": {
            "minimum_full_approval_score": 0.60,
        },
        "guardrails": {
            "max_loan_to_income_ratio": 0.35,
            "max_bnpl_payment_burden": 0.20,
            "max_negative_balance_rate_30d": 0.50,
        },
    }
}


def test_policy_scenarios():

    scenarios = (
        build_policy_scenarios(
            BASE_POLICY_CONFIG
        )
    )

    assert "conservative" in scenarios
    assert "current" in scenarios
    assert "growth" in scenarios
    assert "aggressive_growth" in scenarios

    assert (
        scenarios[
            "conservative"
        ][
            "policy"
        ][
            "thresholds"
        ][
            "low_risk_max_pd"
        ]
        <
        scenarios[
            "growth"
        ][
            "policy"
        ][
            "thresholds"
        ][
            "low_risk_max_pd"
        ]
    )