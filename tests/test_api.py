from fastapi.testclient import (
    TestClient,
)

from src.api.main import app


client = TestClient(
    app
)


def test_health():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "status"
        ]
        == "healthy"
    )


def test_invalid_amount():

    response = client.post(
        "/v1/underwrite",
        json={
            "requested_amount": -100,
            "merchant_category":
                "general_retail",
            "device_type":
                "mobile",
            "channel":
                "checkout",
            "bank_data_available": 1,
        },
    )

    assert (
        response.status_code
        == 422
    )