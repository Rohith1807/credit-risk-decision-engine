import json
from functools import lru_cache
from pathlib import Path

import joblib
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "credit_risk_model_v1.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "credit_risk_model_v1_metadata.json"
)

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "policy.yaml"
)


@lru_cache
def get_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Selected model artifact "
            "does not exist."
        )

    return joblib.load(
        MODEL_PATH
    )


@lru_cache
def get_model_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


@lru_cache
def get_policy():

    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(
            file
        )