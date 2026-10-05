import pandas as pd


def extract_logistic_coefficients(
    pipeline,
) -> pd.DataFrame:

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        model.coef_[0]
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient":
                coefficients,
        }
    )

    result[
        "absolute_coefficient"
    ] = (
        result[
            "coefficient"
        ]
        .abs()
    )

    return (
        result.sort_values(
            "absolute_coefficient",
            ascending=False,
        )
    )