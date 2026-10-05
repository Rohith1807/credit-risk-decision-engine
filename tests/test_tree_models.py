import pandas as pd

from src.models.imbalance import (
    calculate_scale_pos_weight,
)
from src.models.tree_preprocessing import (
    build_tree_preprocessor,
)


def test_scale_pos_weight():

    y = pd.Series(
        [0, 0, 0, 0, 1]
    )

    result = (
        calculate_scale_pos_weight(
            y
        )
    )

    assert result == 4.0


def test_tree_preprocessor_builds():

    dataframe = pd.DataFrame(
        {
            "requested_amount":
                [100.0, 200.0],
            "merchant_category":
                ["electronics", "apparel"],
            "device_type":
                ["mobile", "desktop"],
            "channel":
                ["checkout", "web"],
            "bank_data_available":
                [1, 0],
        }
    )

    preprocessor = (
        build_tree_preprocessor(
            dataframe
        )
    )

    assert preprocessor is not None