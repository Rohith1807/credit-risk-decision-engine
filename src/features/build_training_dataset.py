from pathlib import Path

import pandas as pd
from pyarrow import dataset

from src.features.data_availability import (
    mask_unavailable_bank_features,
)

from src.features.split_dataset import (
    time_based_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


FORBIDDEN_MODEL_COLUMNS = {
    # identifiers
    "application_id",
    "user_id",

    # timestamps
    "application_timestamp",

    # synthetic hidden ground truth
    "latent_pd",

    # target / outcome information
    "default_status",
    "max_days_past_due",
    "loan_id",

    # anything explicitly generated after underwriting
    "decision",
    "approved_limit",
    "probability_default",
}


def build_training_dataset() -> pd.DataFrame:
    """
    Assemble the supervised learning dataset.

    Only originated loans have observed repayment outcomes.
    """

    features = pd.read_parquet(
        PROCESSED_DIR
        / "application_features.parquet"
    )

    outcomes = pd.read_parquet(
        RAW_DIR
        / "loan_outcomes.parquet"
    )

    # Only keep information needed for the supervised target.
    outcome_targets = outcomes[
        [
            "application_id",
            "default_status",
        ]
    ].copy()

    dataset = features.merge(
        outcome_targets,
        on="application_id",
        how="inner",
        validate="one_to_one",
    )

    dataset = mask_unavailable_bank_features(
    dataset
    )

    return dataset


def get_model_columns(
    dataset: pd.DataFrame,
) -> list[str]:
    """
    Return candidate predictor columns after explicit leakage exclusions.
    """

    return [
        column
        for column in dataset.columns
        if column
        not in FORBIDDEN_MODEL_COLUMNS
    ]

POST_OUTCOME_COLUMNS = {
    "latent_pd",
    "default_status",
    "max_days_past_due",
    "days_past_due",
    "payment_status",
    "amount_paid",
}

def validate_no_forbidden_features(
    model_columns: list[str],
) -> None:
    """
    Fail fast if any known leakage field enters the predictor list.
    """

    leaked = (
        set(model_columns)
        & FORBIDDEN_MODEL_COLUMNS
    )

    if leaked:
        raise ValueError(
            "Forbidden model columns detected: "
            f"{sorted(leaked)}"
        )

if __name__ == "__main__":
    dataset = build_training_dataset()

    output_path = (
        PROCESSED_DIR
        / "training_dataset.parquet"
    )

    dataset.to_parquet(
        output_path,
        index=False,
    )

    model_columns = get_model_columns(
        dataset
    )

    validate_no_forbidden_features(
    model_columns
    )

    train, validation, test = (
    time_based_split(
        dataset
    )
)

    train.to_parquet(
        PROCESSED_DIR
        / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        PROCESSED_DIR
        / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        PROCESSED_DIR
        / "test.parquet",
        index=False,
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Training rows: {len(dataset):,}"
    )

    print(
        f"Total columns: {len(dataset.columns):,}"
    )

    print(
        f"Candidate model features: "
        f"{len(model_columns):,}"
    )

    print(
        f"Default rate: "
        f"{dataset['default_status'].mean():.2%}"
    )

    print("\nCandidate features:")

    for column in model_columns:
        print(f" - {column}")

