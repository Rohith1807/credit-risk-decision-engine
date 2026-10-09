import pandas as pd

from src.features.build_training_dataset import (
    get_model_columns,
)


def test_training_dataset_has_predictors(training_dataset):
    features = get_model_columns(
        training_dataset
    )

    assert len(features) > 0


def test_target_has_both_classes(training_dataset):
    assert training_dataset[
        "default_status"
    ].nunique() == 2


def test_requested_amount_has_variation(training_dataset):
    assert training_dataset[
        "requested_amount"
    ].nunique() > 10


def test_income_features_have_variation(training_dataset):
    observed = training_dataset[
        "income_total_90d"
    ].dropna()

    assert observed.nunique() > 10
