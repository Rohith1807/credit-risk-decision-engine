from fastapi import (
    FastAPI,
    HTTPException,
)

from src.api.schemas import (
    UnderwritingRequest,
    UnderwritingResponse,
)
from src.api.underwriting import (
    score_application,
)


app = FastAPI(
    title=(
        "Credit Risk Decision Engine"
    ),
    description=(
        "Probability-of-default scoring "
        "and credit-policy decision API."
    ),
    version="1.0.0",
)


@app.get(
    "/health"
)
def health_check():

    return {
        "status": "healthy"
    }


@app.post(
    "/v1/underwrite",
    response_model=
        UnderwritingResponse,
)
def underwrite(
    request:
        UnderwritingRequest,
):

    try:

        return score_application(
            request
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Underwriting failed: "
                f"{exc}"
            ),
        ) from exc