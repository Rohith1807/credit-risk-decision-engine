from copy import deepcopy


def build_policy_scenarios(
    base_config: dict,
) -> dict[str, dict]:
    """
    Create policy scenarios around the current policy.

    Thresholds are portfolio simulation assumptions,
    not external industry standards.
    """

    conservative = deepcopy(
        base_config
    )

    conservative[
        "policy"
    ][
        "thresholds"
    ][
        "low_risk_max_pd"
    ] = 0.02

    conservative[
        "policy"
    ][
        "thresholds"
    ][
        "medium_risk_max_pd"
    ] = 0.05

    conservative[
        "policy"
    ][
        "limits"
    ][
        "low_and_grow_max"
    ] = 75


    current = deepcopy(
        base_config
    )


    growth = deepcopy(
        base_config
    )

    growth[
        "policy"
    ][
        "thresholds"
    ][
        "low_risk_max_pd"
    ] = 0.04

    growth[
        "policy"
    ][
        "thresholds"
    ][
        "medium_risk_max_pd"
    ] = 0.10

    growth[
        "policy"
    ][
        "limits"
    ][
        "low_and_grow_max"
    ] = 150


    aggressive_growth = deepcopy(
        base_config
    )

    aggressive_growth[
        "policy"
    ][
        "thresholds"
    ][
        "low_risk_max_pd"
    ] = 0.05

    aggressive_growth[
        "policy"
    ][
        "thresholds"
    ][
        "medium_risk_max_pd"
    ] = 0.12

    aggressive_growth[
        "policy"
    ][
        "limits"
    ][
        "low_and_grow_max"
    ] = 200


    return {
        "conservative":
            conservative,

        "current":
            current,

        "growth":
            growth,

        "aggressive_growth":
            aggressive_growth,
    }