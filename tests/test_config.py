from src.config import load_config


def test_config_loads():
    config = load_config()

    assert config is not None
    assert "project" in config
    assert "target" in config
    assert "policy" in config


def test_target_configuration():
    config = load_config()

    target = config["target"]

    assert target["column"] == "default_status"
    assert target["positive_class"] == 1


def test_default_rate_range():
    config = load_config()

    minimum = config["target"]["expected_default_rate_min"]
    maximum = config["target"]["expected_default_rate_max"]

    assert 0 < minimum < maximum < 1


def test_policy_threshold_order():
    config = load_config()

    thresholds = config["policy"]["initial_thresholds"]

    low = thresholds["low_risk_max_pd"]
    moderate = thresholds["moderate_risk_max_pd"]

    assert 0 < low < moderate < 1


def test_exposure_limits_are_positive():
    config = load_config()

    limits = config["policy"]["initial_limits"]

    assert limits["full_approval"] > 0
    assert limits["low_and_grow"] > 0
    assert limits["fallback"] > 0