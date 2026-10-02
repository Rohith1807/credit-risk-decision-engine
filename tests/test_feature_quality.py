import pandas as pd

from src.features.build_training_dataset import (
    build_training_dataset,
    get_model_columns,
)


def test_training_dataset_has_predictors():
    dataset = build_training_dataset()

    features = get_model_columns(
        dataset
    )

    assert len(features) > 0


def test_target_has_both_classes():
    dataset = build_training_dataset()

    assert dataset[
        "default_status"
    ].nunique() == 2


def test_requested_amount_has_variation():
    dataset = build_training_dataset()

    assert dataset[
        "requested_amount"
    ].nunique() > 10


def test_income_features_have_variation():
    dataset = build_training_dataset()

    observed = dataset[
        "income_total_90d"
    ].dropna()

    assert observed.nunique() > 10