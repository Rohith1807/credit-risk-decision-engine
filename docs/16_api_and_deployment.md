Service:
Credit Risk Decision Engine API

Endpoints:

GET /health
POST /v1/underwrite

Underwriting flow:

Request
→ schema validation
→ model scoring
→ probability of default
→ policy engine
→ approval decision

Model:
Raw unweighted LightGBM

Model version:
v1.0

Policy:
PD-based policy with bank-data fallback
and confidence handling.

Deployment:
FastAPI + Uvicorn + Docker

Important:
The portfolio service is a simulation and
engineering demonstration, not a production
credit-decision system.