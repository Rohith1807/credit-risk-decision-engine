def calculate_scale_pos_weight(
    y,
) -> float:

    negatives = (
        y == 0
    ).sum()

    positives = (
        y == 1
    ).sum()

    if positives == 0:
        raise ValueError(
            "Training dataset contains no positive examples."
        )

    return (
        negatives
        / positives
    )