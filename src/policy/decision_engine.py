from dataclasses import dataclass

from src.policy.economics import (
    expected_loss,
    expected_profit,
)
from src.policy.reason_codes import (
    APPROVED,
    BANK_DATA_UNAVAILABLE,
    HIGH_BNPL_BURDEN,
    HIGH_LTI,
    HIGH_PD,
    LOW_AND_GROW,
    LOW_DATA_CONFIDENCE,
    NEGATIVE_BALANCE_ACTIVITY,
    NEGATIVE_EXPECTED_PROFIT
)


@dataclass
class PolicyDecision:
    decision: str
    policy_tier: str
    approved_limit: float
    reason_code: str
    probability_default: float
    expected_loss: float
    expected_profit: float


def make_credit_decision(
    probability_default: float,
    requested_amount: float,
    bank_data_available: int,
    data_confidence_score: float | None,
    loan_to_income_ratio: float | None,
    bnpl_payment_burden: float | None,
    negative_balance_rate_30d: float | None,
    config: dict,
    apply_behavioral_guardrails: bool = True,
) -> PolicyDecision:

    policy = config["policy"]

    thresholds = policy["thresholds"]
    limits = policy["limits"]
    economics = policy["economics"]
    guardrails = policy["guardrails"]
    confidence = policy["confidence"]

    # --------------------------------
    # 1. Missing bank-data fallback
    # --------------------------------

    if bank_data_available == 0:

        approved_limit = min(
            requested_amount,
            limits["fallback_max"],
        )

        loss = expected_loss(
            probability_default,
            approved_limit,
            economics["loss_given_default"],
        )

        profit = expected_profit(
            probability_default,
            approved_limit,
            economics["loss_given_default"],
            economics["merchant_fee_rate"],
            economics["funding_cost_rate"],
            economics["servicing_cost"],
        )

        return PolicyDecision(
            decision="APPROVE",
            policy_tier="FALLBACK",
            approved_limit=approved_limit,
            reason_code=BANK_DATA_UNAVAILABLE,
            probability_default=probability_default,
            expected_loss=loss,
            expected_profit=profit,
        )

    # --------------------------------
    # 2. Data-confidence guardrail
    # --------------------------------

    if (
        data_confidence_score is not None
        and data_confidence_score
        < confidence[
            "minimum_full_approval_score"
        ]
    ):

        approved_limit = min(
            requested_amount,
            limits["low_and_grow_max"],
        )

        tier = "LOW_AND_GROW"
        reason = LOW_DATA_CONFIDENCE

    # --------------------------------
    # 3. Behavioral guardrails
    # --------------------------------

    elif (
        apply_behavioral_guardrails
        and loan_to_income_ratio is not None
        and loan_to_income_ratio
        > guardrails[
            "max_loan_to_income_ratio"
        ]
    ):

        approved_limit = 0.0
        tier = "REJECT"
        reason = HIGH_LTI

    elif (
        apply_behavioral_guardrails
         and bnpl_payment_burden is not None
        and bnpl_payment_burden
        > guardrails[
            "max_bnpl_payment_burden"
        ]
    ):

        approved_limit = 0.0
        tier = "REJECT"
        reason = HIGH_BNPL_BURDEN

    elif (
        apply_behavioral_guardrails
        and negative_balance_rate_30d
        is not None
        and negative_balance_rate_30d
        > guardrails[
            "max_negative_balance_rate_30d"
        ]
    ):

        approved_limit = 0.0
        tier = "REJECT"
        reason = NEGATIVE_BALANCE_ACTIVITY

    # --------------------------------
    # 4. PD policy
    # --------------------------------

    elif (
        probability_default
        < thresholds[
            "low_risk_max_pd"
        ]
    ):

        approved_limit = min(
            requested_amount,
            limits[
                "full_approval_max"
            ],
        )

        tier = "FULL_APPROVAL"
        reason = APPROVED

    elif (
        probability_default
        < thresholds[
            "medium_risk_max_pd"
        ]
    ):

        approved_limit = min(
            requested_amount,
            limits[
                "low_and_grow_max"
            ],
        )

        tier = "LOW_AND_GROW"
        reason = LOW_AND_GROW

    else:

        approved_limit = 0.0
        tier = "REJECT"
        reason = HIGH_PD

    # --------------------------------
    # 5. Final decision
    # --------------------------------

    decision = (
        "APPROVE"
        if approved_limit > 0
        else "REJECT"
    )

    loss = expected_loss(
        probability_default,
        approved_limit,
        economics[
            "loss_given_default"
        ],
    )

    profit = expected_profit(
        probability_default,
        approved_limit,
        economics[
            "loss_given_default"
        ],
        economics[
            "merchant_fee_rate"
        ],
        economics[
            "funding_cost_rate"
        ],
        economics[
            "servicing_cost"
        ],
    )

    if approved_limit > 0 and profit <= 0:

        approved_limit = 0.0
        decision = "REJECT"
        tier = "REJECT"
        reason = NEGATIVE_EXPECTED_PROFIT

        loss = 0.0
        profit = 0.0

    return PolicyDecision(
        decision=decision,
        policy_tier=tier,
        approved_limit=approved_limit,
        reason_code=reason,
        probability_default=probability_default,
        expected_loss=loss,
        expected_profit=profit,
    )