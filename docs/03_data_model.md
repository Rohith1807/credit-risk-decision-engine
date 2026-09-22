# Credit Risk Decision Engine — Data Model

## Purpose

The project uses a relational data model representing a simplified BNPL / digital lending environment.

The model separates:

- customer information
- banking activity
- credit applications
- loan decisions
- repayment outcomes

This separation helps preserve realistic data lineage and prevents future repayment information from leaking into underwriting features.

---

## Core Entities

### users

Represents a customer profile.

Primary Key:
- user_id

One user can have:
- multiple bank accounts
- multiple applications
- multiple loans

---

### bank_accounts

Represents bank accounts associated with a user.

Primary Key:
- account_id

Foreign Key:
- user_id -> users.user_id

---

### transactions

Represents historical bank transactions.

Primary Key:
- transaction_id

Foreign Key:
- account_id -> bank_accounts.account_id

Transactions occurring before an application timestamp may be used for underwriting features.

Transactions occurring after an application timestamp must not be used for that decision.

---

### applications

Represents a credit request.

Primary Key:
- application_id

Foreign Key:
- user_id -> users.user_id

Each application has a timestamp representing the underwriting decision point.

---

### loans

Represents approved credit obligations.

Primary Key:
- loan_id

Foreign Keys:
- application_id -> applications.application_id
- user_id -> users.user_id

Only approved applications create loans.

---

### repayments

Represents scheduled and actual repayment behavior.

Primary Key:
- payment_id

Foreign Key:
- loan_id -> loans.loan_id

Repayment outcomes occur after underwriting and must never be used as prediction features for the original application.

They are used to construct the model target.

---

### credit_decisions

Stores underwriting output.

Primary Key:
- decision_id

Foreign Key:
- application_id -> applications.application_id

Stores:
- model score
- policy tier
- approved limit
- reason codes
- model version
- policy version

## Point-in-Time Rule

All underwriting features must be calculated using information available at or before the application timestamp.

No transaction, repayment event, account update, or outcome occurring after the application timestamp may be included in the application's feature vector.

## Data Availability Classes

### Available Before Underwriting

Examples:
- application amount
- merchant category
- historical bank transactions
- account balances
- historical repayment behavior from previous loans

These may be used as model features.

### Available After Underwriting

Examples:
- repayment performance for the new loan
- future balances
- future transactions
- eventual default status

These must not be used as features for the original credit decision.