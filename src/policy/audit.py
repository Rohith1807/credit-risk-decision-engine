from datetime import (
    datetime,
    timezone,
)

import uuid


def build_decision_audit_record(
    application_id: str,
    decision,
    model_version: str,
    policy_version: str,
) -> dict:

    return {
        "decision_id":
            str(
                uuid.uuid4()
            ),

        "application_id":
            application_id,

        "decision_timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "probability_default":
            decision.probability_default,

        "policy_tier":
            decision.policy_tier,

        "decision":
            decision.decision,

        "approved_limit":
            decision.approved_limit,

        "reason_code":
            decision.reason_code,

        "expected_loss":
            decision.expected_loss,

        "expected_profit":
            decision.expected_profit,

        "model_version":
            model_version,

        "policy_version":
            policy_version,
    }