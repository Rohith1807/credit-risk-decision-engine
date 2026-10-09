from src.features.build_training_dataset import (
    FORBIDDEN_MODEL_COLUMNS,
    get_model_columns,
)

from src.features.data_availability import (
    BANK_DERIVED_FEATURES,
)


def test_training_dataset_has_target(training_dataset):
    assert "default_status" in training_dataset.columns


def test_target_is_binary(training_dataset):
    assert set(
        training_dataset["default_status"].unique()
    ).issubset(
        {0, 1}
    )


def test_application_ids_unique(training_dataset):
    assert training_dataset[
        "application_id"
    ].is_unique


def test_model_columns_exclude_target(training_dataset):
    columns = get_model_columns(
        training_dataset
    )

    assert "default_status" not in columns


def test_forbidden_columns_not_in_model_features(training_dataset):
    columns = set(
        get_model_columns(
            training_dataset
        )
    )

    assert not (
        columns
        & FORBIDDEN_MODEL_COLUMNS
    )


def test_training_dataset_not_empty(training_dataset):
    assert len(training_dataset) > 0

def test_bank_features_missing_when_bank_data_unavailable(training_dataset):
    dataset = training_dataset

    unavailable = dataset[
        dataset[
            "bank_data_available"
        ]
        == 0
    ]

    if unavailable.empty:
        return

    existing_features = [
        column
        for column
        in BANK_DERIVED_FEATURES
        if column
        in unavailable.columns
    ]

    assert (
        unavailable[
            existing_features
        ]
        .isna()
        .all()
        .all()
    )
