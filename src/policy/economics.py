def expected_loss(
    probability_default: float,
    exposure: float,
    loss_given_default: float,
) -> float:
    """
    Expected credit loss.

    EL = PD * LGD * EAD
    """

    return (
        probability_default
        * loss_given_default
        * exposure
    )


def expected_revenue(
    exposure: float,
    merchant_fee_rate: float,
) -> float:
    """
    Simplified BNPL merchant-fee revenue.
    """

    return (
        exposure
        * merchant_fee_rate
    )


def expected_funding_cost(
    exposure: float,
    funding_cost_rate: float,
) -> float:

    return (
        exposure
        * funding_cost_rate
    )


def expected_profit(
    probability_default: float,
    exposure: float,
    loss_given_default: float,
    merchant_fee_rate: float,
    funding_cost_rate: float,
    servicing_cost: float,
) -> float:

    revenue = expected_revenue(
        exposure,
        merchant_fee_rate,
    )

    credit_loss = expected_loss(
        probability_default,
        exposure,
        loss_given_default,
    )

    funding_cost = expected_funding_cost(
        exposure,
        funding_cost_rate,
    )

    return (
        revenue
        - credit_loss
        - funding_cost
        - servicing_cost
    )