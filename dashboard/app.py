from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DECISIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_policy_decisions.parquet"
)


st.set_page_config(
    page_title=(
        "Credit Risk Operations"
    ),
    layout="wide",
)


st.title(
    "Credit Risk Decision Engine"
)

st.caption(
    "Portfolio risk operations dashboard"
)


if not DECISIONS_PATH.exists():

    st.error(
        "Policy decisions dataset "
        "not found."
    )

    st.stop()


df = pd.read_parquet(
    DECISIONS_PATH
)


approved = df[
    df["decision"]
    == "APPROVE"
]

rejected = df[
    df["decision"]
    == "REJECT"
]


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Applications",
    f"{len(df):,}",
)

col2.metric(
    "Approval Rate",
    f"{len(approved) / len(df):.1%}",
)

col3.metric(
    "Expected Loss",
    f"${approved['expected_loss'].sum():,.0f}",
)

col4.metric(
    "Expected Profit",
    f"${approved['expected_profit'].sum():,.0f}",
)


st.subheader(
    "Policy Tier Distribution"
)

tier_counts = (
    df[
        "policy_tier"
    ]
    .value_counts()
)

st.bar_chart(
    tier_counts
)


st.subheader(
    "Probability of Default"
)

st.bar_chart(
    df[
        "probability_default"
    ]
    .value_counts(
        bins=10
    )
    .sort_index()
)


st.subheader(
    "Reason Codes"
)

reason_counts = (
    df[
        "reason_code"
    ]
    .value_counts()
)

st.bar_chart(
    reason_counts
)


if (
    "default_status"
    in df.columns
):

    st.subheader(
        "Observed Risk by Policy Tier"
    )

    default_by_tier = (
        df.groupby(
            "policy_tier"
        )[
            "default_status"
        ]
        .mean()
        .sort_values()
    )

    st.bar_chart(
        default_by_tier
    )


st.subheader(
    "Decision Explorer"
)

st.dataframe(
    df[
        [
            "application_id",
            "probability_default",
            "decision",
            "policy_tier",
            "approved_limit",
            "reason_code",
            "expected_loss",
            "expected_profit",
        ]
    ],
    use_container_width=True,
)

st.sidebar.header(
    "Risk Appetite Simulator"
)

low_threshold = (
    st.sidebar.slider(
        "Full approval PD threshold",
        min_value=0.01,
        max_value=0.10,
        value=0.04,
        step=0.01,
    )
)

medium_threshold = (
    st.sidebar.slider(
        "Reject PD threshold",
        min_value=0.02,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
)

st.sidebar.write(
    "Current simulated thresholds:"
)

st.sidebar.write(
    f"Full approval < "
    f"{low_threshold:.0%}"
)

st.sidebar.write(
    f"Reject ≥ "
    f"{medium_threshold:.0%}"
)