Purpose:
Provide an external open-banking ingestion path for
inference-time behavioral features.

Development environment:
Plaid Sandbox.

Training:
Synthetic historical portfolio with repayment/default labels.

Inference:
Plaid transaction/account telemetry can be normalized and
processed using the same behavioral feature definitions.

Sandbox integration:
- create Sandbox Item
- exchange public token for access token
- retrieve transactions using Transactions Sync
- normalize external data into internal schema

Security:
Credentials and access tokens are never committed to source
control.

Production considerations:
- encrypted token storage
- webhook processing
- consent lifecycle
- Item login recovery
- cursor persistence
- transaction updates/removals
- data retention and privacy controls