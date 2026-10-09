from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_file(relative_path: str) -> Path:
    """
    Return the requested project-relative file path if it exists.

    Skip the current test when the required local/generated artifact
    is unavailable. This is useful for CI environments where large
    generated Parquet datasets are intentionally not committed.
    """

    file_path = PROJECT_ROOT / relative_path

    if not file_path.exists():
        pytest.skip(
            f"Required local data artifact not available: {relative_path}"
        )

    return file_path


@pytest.fixture
def require_data_file():
    """
    Pytest fixture wrapper around require_file().

    Usage:
        def test_something(require_data_file):
            path = require_data_file(
                "data/processed/application_features.parquet"
            )
    """

    return require_file


@pytest.fixture(scope="module")
def training_dataset():
    """
    Module-scoped fixture that guards the two data artifacts required by
    build_training_dataset() and returns the assembled DataFrame.

    Tests that call build_training_dataset() should accept this fixture
    instead of calling the function directly so they skip cleanly in CI
    when the underlying Parquet files are absent.
    """
    require_file("data/processed/application_features.parquet")
    require_file("data/raw/loan_outcomes.parquet")

    from src.features.build_training_dataset import build_training_dataset

    return build_training_dataset()