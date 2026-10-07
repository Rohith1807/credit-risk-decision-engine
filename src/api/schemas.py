from pydantic import (
    BaseModel,
    Field,
)


class UnderwritingRequest(
    BaseModel
):
    requested_amount: float = Field(
        gt=0,
        le=10000,
    )

    merchant_category: str = (
        "general_retail"
    )

    device_type: str = "mobile"

    channel: str = "checkout"

    bank_data_available: int = Field(
        ge=0,
        le=1,
    )

    income_count_90d: float | None = None

    income_total_90d: float | None = None

    avg_income_deposit_90d: float | None = None

    median_income_deposit_90d: float | None = None

    income_std_90d: float | None = None

    income_cv_90d: float | None = None

    days_since_last_income: float | None = None

    current_balance: float | None = None

    avg_balance_30d: float | None = None

    median_balance_30d: float | None = None

    min_balance_30d: float | None = None

    balance_std_30d: float | None = None

    negative_balance_events_30d: float | None = None

    negative_balance_rate_30d: float | None = None

    spend_total_30d: float | None = None

    transaction_count_30d: float | None = None

    unique_merchants_30d: float | None = None

    bnpl_payment_count_90d: float | None = None

    bnpl_payment_amount_90d: float | None = None

    discretionary_spend_30d: float | None = None

    loan_to_income_ratio: float | None = None

    requested_amount_to_balance: float | None = None

    paycheck_proximity_days: float | None = None

    discretionary_spend_ratio: float | None = None

    cash_buffer_days: float | None = None

    cash_buffer_velocity: float | None = None

    spend_acceleration: float | None = None

    bnpl_payment_burden: float | None = None

    income_stability_score: float | None = None

    data_confidence_score: float | None = None

    transaction_count_24h: float | None = None

    unique_merchants_24h: float | None = None


class UnderwritingResponse(
    BaseModel
):
    probability_default: float

    decision: str

    policy_tier: str

    approved_limit: float

    reason_code: str

    expected_loss: float

    expected_profit: float

    model_version: str

    policy_version: str