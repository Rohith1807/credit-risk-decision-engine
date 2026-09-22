from src.data_generation.generate_users import generate_users


def test_user_count():
    users = generate_users()

    assert len(users) == 1000


def test_user_ids_unique():
    users = generate_users()

    assert users["user_id"].is_unique


def test_user_ids_not_null():
    users = generate_users()

    assert users["user_id"].notna().all()


def test_account_tenure_positive():
    users = generate_users()

    assert (users["account_tenure_months"] > 0).all()


def test_valid_employment_types():
    users = generate_users()

    allowed = {
        "salaried",
        "hourly",
        "self_employed",
        "gig_worker",
        "student",
    }

    assert set(users["employment_type"]).issubset(allowed)