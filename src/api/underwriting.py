import numpy as np
import pandas as pd

from src.api.dependencies import (
    get_model,
    get_model_metadata,
    get_policy,
)
from src.api.schemas import (
    UnderwritingRequest,
    UnderwritingResponse,
)
from src.policy.decision_engine import (
    make_credit_decision,
)


def request_to_dataframe(
    request: UnderwritingRequest,
) -> pd.DataFrame:

    payload = request.model_dump()

    dataframe = pd.DataFrame(
        [payload]
    )

    dataframe = dataframe.replace(
        {
            None: np.nan,
        }
    )

    return dataframe


def score_application(
    request: UnderwritingRequest,
) -> UnderwritingResponse:

    model = get_model()

    metadata = (
        get_model_metadata()
    )

    policy = get_policy()

    X = request_to_dataframe(
        request
    )

    # Ensure the model receives the same
    # columns and ordering used at training.
    expected_columns = metadata[
        "feature_columns"
    ]

    for column in expected_columns:

        if column not in X.columns:
            X[column] = np.nan

    X = X[
        expected_columns
    ]

    probability_default = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    decision = make_credit_decision(
        probability_default=
            probability_default,

        requested_amount=
            request.requested_amount,

        bank_data_available=
            request.bank_data_available,

        data_confidence_score=
            request.data_confidence_score,

        loan_to_income_ratio=
            request.loan_to_income_ratio,

        bnpl_payment_burden=
            request.bnpl_payment_burden,

        negative_balance_rate_30d=
            request.negative_balance_rate_30d,

        config=
            policy,

        apply_behavioral_guardrails=
            False,
    )

    policy_version = (
        policy[
            "policy"
        ].get(
            "version",
            "unknown",
        )
    )

    return UnderwritingResponse(
        probability_default=
            probability_default,

        decision=
            decision.decision,

        policy_tier=
            decision.policy_tier,

        approved_limit=
            decision.approved_limit,

        reason_code=
            decision.reason_code,

        expected_loss=
            decision.expected_loss,

        expected_profit=
            decision.expected_profit,

        model_version=
            metadata[
                "model_version"
            ],

        policy_version=
            policy_version,
    )