from src.features.build_training_dataset import (
    build_training_dataset,
)
from src.features.split_dataset import (
    time_based_split,
)


def test_time_split_has_no_overlap():
    dataset = build_training_dataset()

    train, validation, test = (
        time_based_split(
            dataset
        )
    )

    train_ids = set(
        train["application_id"]
    )

    validation_ids = set(
        validation["application_id"]
    )

    test_ids = set(
        test["application_id"]
    )

    assert train_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert validation_ids.isdisjoint(
        test_ids
    )


def test_time_split_order():
    dataset = build_training_dataset()

    train, validation, test = (
        time_based_split(
            dataset
        )
    )

    assert (
        train[
            "application_timestamp"
        ].max()
        <=
        validation[
            "application_timestamp"
        ].min()
    )

    assert (
        validation[
            "application_timestamp"
        ].max()
        <=
        test[
            "application_timestamp"
        ].min()
    )