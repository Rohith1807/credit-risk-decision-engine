import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.feature_schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_V1_FEATURES,
)


def get_tree_feature_groups(
    dataframe: pd.DataFrame,
):

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in dataframe.columns
    ]

    binary = [
        column
        for column in BINARY_FEATURES
        if column in dataframe.columns
    ]

    excluded = set(
        categorical
        + binary
        + EXCLUDED_V1_FEATURES
    )

    numerical = [
        column
        for column in dataframe.select_dtypes(
            include="number"
        ).columns
        if column not in excluded
    ]

    return (
        numerical,
        categorical,
        binary,
    )


def build_tree_preprocessor(
    dataframe: pd.DataFrame,
):

    (
        numerical_features,
        categorical_features,
        binary_features,
    ) = get_tree_feature_groups(
        dataframe
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "binary",
                binary_pipeline,
                binary_features,
            ),
        ],
        remainder="drop",
    )