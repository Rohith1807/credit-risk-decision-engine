from src.data_generation.generate_users import (
    generate_users,
)


def test_generation_reproducible():
    first = generate_users()
    second = generate_users()

    assert first.equals(second)