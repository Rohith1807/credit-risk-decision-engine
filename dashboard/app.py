from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

FINAL_RESULTS_PATH = (
    PROCESSED_DIR
    / "final_test_results.parquet"
)

TEST_DATA_PATH = (
    PROCESSED_DIR
    / "test.parquet"
)

TRAINING_DATA_PATH = (
    PROCESSED_DIR
    / "training_dataset.parquet"
)

CALIBRATION_PATH = (
    PROCESSED_DIR
    / "calibration_comparison.parquet"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "credit_risk_model_v1_metadata.json"
)

USERS_PATH = (
    RAW_DIR
    / "users.parquet"
)

ACCOUNTS_PATH = (
    RAW_DIR
    / "bank_accounts.parquet"
)

TRANSACTIONS_PATH = (
    RAW_DIR
    / "transactions.parquet"
)

APPLICATIONS_PATH = (
    RAW_DIR
    / "applications.parquet"
)

LOANS_PATH = (
    RAW_DIR
    / "loans.parquet"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Decision Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 2.15rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h2 {
        margin-top: 1.1rem !important;
    }

    h3 {
        margin-top: 0.8rem !important;
    }

    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: -0.8rem;
        margin-bottom: 1.2rem;
    }

    .section-note {
        color: #888;
        font-size: 0.88rem;
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def parquet_row_count(
    path: Path,
) -> int | None:
    """
    Read only Parquet metadata when possible so large
    files such as transactions.parquet are not loaded
    into memory just to obtain their row count.
    """

    if not path.exists():
        return None

    if pq is not None:
        parquet_file = pq.ParquetFile(
            path
        )

        return int(
            parquet_file.metadata.num_rows
        )

    # Fallback if pyarrow metadata access
    # is unavailable.
    dataframe = pd.read_parquet(
        path
    )

    return len(dataframe)


def format_large_number(
    value: int | None,
) -> str:

    if value is None:
        return "N/A"

    if value >= 1_000_000:
        return (
            f"{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:
        return (
            f"{value / 1_000:.1f}K"
        )

    return f"{value:,}"


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_final_results():
    return pd.read_parquet(
        FINAL_RESULTS_PATH
    )


@st.cache_data
def load_test_data():
    return pd.read_parquet(
        TEST_DATA_PATH
    )


@st.cache_data
def load_calibration():

    if CALIBRATION_PATH.exists():

        return pd.read_parquet(
            CALIBRATION_PATH
        )

    return pd.DataFrame()


@st.cache_data
def load_model_metadata():

    if not MODEL_METADATA_PATH.exists():
        return {}

    with open(
        MODEL_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


@st.cache_data
def load_portfolio_scale():

    return {
        "users":
            parquet_row_count(
                USERS_PATH
            ),

        "accounts":
            parquet_row_count(
                ACCOUNTS_PATH
            ),

        "transactions":
            parquet_row_count(
                TRANSACTIONS_PATH
            ),

        "applications":
            parquet_row_count(
                APPLICATIONS_PATH
            ),

        "labeled_outcomes":
            parquet_row_count(
                TRAINING_DATA_PATH
            ),

        "test_applications":
            parquet_row_count(
                FINAL_RESULTS_PATH
            ),
    }


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

if not FINAL_RESULTS_PATH.exists():

    st.error(
        "Final test results were not found. "
        "Run the final test evaluation first."
    )

    st.stop()


if not TEST_DATA_PATH.exists():

    st.error(
        "Test dataset was not found."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

decisions = load_final_results()
test_data = load_test_data()
calibration = load_calibration()
metadata = load_model_metadata()
portfolio_scale = load_portfolio_scale()


# ============================================================
# MERGE FINAL DECISIONS WITH TEST FEATURES
# ============================================================

segment_columns = [
    "application_id",
    "merchant_category",
    "device_type",
    "channel",
    "bank_data_available",
]

available_segment_columns = [
    column
    for column in segment_columns
    if column in test_data.columns
]

segment_data = (
    test_data[
        available_segment_columns
    ]
    .drop_duplicates(
        subset=["application_id"]
    )
)

df = decisions.merge(
    segment_data,
    on="application_id",
    how="left",
)


# ============================================================
# FINAL TEST PORTFOLIO METRICS
# ============================================================

approved = df[
    df["decision"] == "APPROVE"
].copy()

rejected = df[
    df["decision"] == "REJECT"
].copy()

applications = len(df)

approval_rate = (
    len(approved) / applications
    if applications
    else 0
)

requested_gmv = float(
    df[
        "requested_amount"
    ].sum()
)

approved_gmv = float(
    approved[
        "approved_limit"
    ].sum()
)

gmv_approval_rate = (
    approved_gmv / requested_gmv
    if requested_gmv
    else 0
)

expected_loss = float(
    approved[
        "expected_loss"
    ].sum()
)

expected_profit = float(
    approved[
        "expected_profit"
    ].sum()
)

expected_loss_rate = (
    expected_loss / approved_gmv
    if approved_gmv
    else 0
)

expected_profit_margin = (
    expected_profit / approved_gmv
    if approved_gmv
    else 0
)

approved_default_rate = float(
    approved[
        "default_status"
    ].mean()
)

rejected_default_rate = float(
    rejected[
        "default_status"
    ].mean()
)

mean_pd = float(
    df[
        "probability_default"
    ].mean()
)

actual_default_rate = float(
    df[
        "default_status"
    ].mean()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Risk Operations"
)

st.sidebar.caption(
    "Credit Risk Decision Engine v1.0"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Production-Style Policy"
)

st.sidebar.write(
    "**Full Approval**  \nPD < 5%"
)

st.sidebar.write(
    "**Review**  \n5% ≤ PD < 8%"
)

st.sidebar.write(
    "**Reject**  \nPD ≥ 8%"
)

st.sidebar.caption(
    "Bank-data fallback, data-confidence "
    "controls, and expected-profit viability "
    "checks are applied separately."
)

st.sidebar.divider()

st.sidebar.subheader(
    "Model"
)

st.sidebar.write(
    metadata.get(
        "model_name",
        "Logistic Regression",
    )
)

st.sidebar.write(
    f"Version: "
    f"{metadata.get('model_version', 'v1.0')}"
)

st.sidebar.write(
    "Calibration: Raw"
)

st.sidebar.divider()

decision_filter = (
    st.sidebar.multiselect(
        "Decision",
        options=sorted(
            df[
                "decision"
            ]
            .dropna()
            .unique()
        ),
        default=sorted(
            df[
                "decision"
            ]
            .dropna()
            .unique()
        ),
    )
)

tier_filter = (
    st.sidebar.multiselect(
        "Policy tier",
        options=sorted(
            df[
                "policy_tier"
            ]
            .dropna()
            .unique()
        ),
        default=sorted(
            df[
                "policy_tier"
            ]
            .dropna()
            .unique()
        ),
    )
)

filtered_df = df[
    df[
        "decision"
    ].isin(
        decision_filter
    )
    &
    df[
        "policy_tier"
    ].isin(
        tier_filter
    )
].copy()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Credit Risk Decision Engine"
)

st.markdown(
    """
    <div class="dashboard-subtitle">
    Model performance, underwriting decisions,
    portfolio economics, and risk monitoring
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Synthetic BNPL portfolio with final "
    "out-of-sample model and policy evaluation. "
    "Portfolio demonstration only — not a production "
    "consumer credit system."
)


# ============================================================
# PORTFOLIO SCALE
# ============================================================

st.subheader(
    "Portfolio Scale & Data Coverage"
)

st.caption(
    "Full synthetic portfolio scale is shown below. "
    "Model and policy performance metrics in this "
    "dashboard are calculated only on the held-out "
    "chronological test portfolio."
)

scale1, scale2, scale3 = (
    st.columns(3)
)

scale1.metric(
    "Synthetic Users",
    format_large_number(
        portfolio_scale[
            "users"
        ]
    ),
)

scale2.metric(
    "Bank Accounts",
    format_large_number(
        portfolio_scale[
            "accounts"
        ]
    ),
)

scale3.metric(
    "Transactions",
    format_large_number(
        portfolio_scale[
            "transactions"
        ]
    ),
)

scale4, scale5, scale6 = (
    st.columns(3)
)

scale4.metric(
    "Credit Applications",
    format_large_number(
        portfolio_scale[
            "applications"
        ]
    ),
)

scale5.metric(
    "Labeled Outcomes",
    format_large_number(
        portfolio_scale[
            "labeled_outcomes"
        ]
    ),
)

scale6.metric(
    "Held-Out Test Applications",
    format_large_number(
        portfolio_scale[
            "test_applications"
        ]
    ),
)

st.info(
    "The 12K-user portfolio represents the full "
    "simulated lending environment. The 5.3K held-out "
    "applications represent the untouched test sample "
    "used to report out-of-sample model and policy "
    "performance."
)


# ============================================================
# FINAL TEST KPI HEADER
# ============================================================

st.subheader(
    "Final Held-Out Test Performance"
)

st.caption(
    "The following KPIs are calculated on the "
    "chronological test set only and were not used "
    "for model training, calibration selection, "
    "or policy threshold optimization."
)


# ============================================================
# TOP KPI ROW
# ============================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

kpi1.metric(
    "Test Applications",
    f"{applications:,}",
)

kpi2.metric(
    "Approval Rate",
    f"{approval_rate:.1%}",
)

kpi3.metric(
    "Approved GMV",
    f"${approved_gmv / 1_000_000:.2f}M",
)

kpi4.metric(
    "Expected Profit",
    f"${expected_profit:,.0f}",
)

kpi5.metric(
    "Approved Default Rate",
    f"{approved_default_rate:.2%}",
)


# ============================================================
# SECOND KPI ROW
# ============================================================

kpi6, kpi7, kpi8, kpi9 = (
    st.columns(4)
)

kpi6.metric(
    "GMV Approval Rate",
    f"{gmv_approval_rate:.1%}",
)

kpi7.metric(
    "Expected Loss Rate",
    f"{expected_loss_rate:.2%}",
)

kpi8.metric(
    "Portfolio Mean PD",
    f"{mean_pd:.2%}",
)

kpi9.metric(
    "Observed Default Rate",
    f"{actual_default_rate:.2%}",
)


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    risk_tab,
    policy_tab,
    model_tab,
    segment_tab,
    explorer_tab,
) = st.tabs(
    [
        "Executive Overview",
        "Portfolio Risk",
        "Policy & Economics",
        "Model Monitoring",
        "Segment Analysis",
        "Decision Explorer",
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with overview_tab:

    st.subheader(
        "Portfolio Decision Summary"
    )

    left, right = st.columns(
        [1.15, 1]
    )

    with left:

        decision_summary = (
            df[
                "decision"
            ]
            .value_counts()
            .rename_axis(
                "decision"
            )
            .reset_index(
                name="applications"
            )
        )

        decision_chart = (
            alt.Chart(
                decision_summary
            )
            .mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
            )
            .encode(
                x=alt.X(
                    "decision:N",
                    title=None,
                ),
                y=alt.Y(
                    "applications:Q",
                    title="Applications",
                ),
                tooltip=[
                    "decision",
                    alt.Tooltip(
                        "applications",
                        format=",",
                    ),
                ],
            )
            .properties(
                height=300,
                title=(
                    "Approval vs Rejection"
                ),
            )
        )

        st.altair_chart(
            decision_chart,
            use_container_width=True,
        )

    with right:

        tier_summary = (
            df.groupby(
                "policy_tier",
                observed=True,
            )
            .agg(
                applications=(
                    "application_id",
                    "count",
                ),
                mean_pd=(
                    "probability_default",
                    "mean",
                ),
                observed_default_rate=(
                    "default_status",
                    "mean",
                ),
            )
            .reset_index()
        )

        tier_chart = (
            alt.Chart(
                tier_summary
            )
            .mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
            )
            .encode(
                x=alt.X(
                    "policy_tier:N",
                    title=None,
                    sort="-y",
                ),
                y=alt.Y(
                    "applications:Q",
                    title="Applications",
                ),
                tooltip=[
                    "policy_tier",
                    "applications",
                    alt.Tooltip(
                        "mean_pd:Q",
                        format=".2%",
                    ),
                    alt.Tooltip(
                        "observed_default_rate:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=300,
                title=(
                    "Policy Tier Distribution"
                ),
            )
        )

        st.altair_chart(
            tier_chart,
            use_container_width=True,
        )

    st.subheader(
        "Risk Separation"
    )

    risk1, risk2 = (
        st.columns(2)
    )

    risk1.metric(
        "Approved Observed Default Rate",
        f"{approved_default_rate:.2%}",
    )

    risk2.metric(
        "Rejected Observed Default Rate",
        f"{rejected_default_rate:.2%}",
    )

    st.caption(
        "Rejected applications exhibit materially "
        "higher realized default risk than approved "
        "applications on the held-out test portfolio."
    )


# ============================================================
# PORTFOLIO RISK
# ============================================================

with risk_tab:

    st.subheader(
        "Probability of Default Distribution"
    )

    pd_chart = (
        alt.Chart(
            filtered_df
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "probability_default:Q",
                bin=alt.Bin(
                    maxbins=30
                ),
                title=(
                    "Predicted Probability "
                    "of Default"
                ),
                axis=alt.Axis(
                    format="%"
                ),
            ),
            y=alt.Y(
                "count():Q",
                title="Applications",
            ),
            tooltip=[
                alt.Tooltip(
                    "count():Q",
                    title="Applications",
                )
            ],
        )
        .properties(
            height=340,
        )
    )

    st.altair_chart(
        pd_chart,
        use_container_width=True,
    )

    st.subheader(
        "Observed Risk by Policy Tier"
    )

    tier_risk = (
        df.groupby(
            "policy_tier",
            observed=True,
        )
        .agg(
            applications=(
                "application_id",
                "count",
            ),
            mean_pd=(
                "probability_default",
                "mean",
            ),
            observed_default_rate=(
                "default_status",
                "mean",
            ),
        )
        .reset_index()
    )

    tier_risk_long = (
        tier_risk.melt(
            id_vars=[
                "policy_tier",
                "applications",
            ],
            value_vars=[
                "mean_pd",
                "observed_default_rate",
            ],
            var_name="metric",
            value_name="rate",
        )
    )

    risk_chart = (
        alt.Chart(
            tier_risk_long
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "policy_tier:N",
                title=None,
            ),
            xOffset="metric:N",
            y=alt.Y(
                "rate:Q",
                title="Rate",
                axis=alt.Axis(
                    format="%"
                ),
            ),
            color=alt.Color(
                "metric:N",
                title=None,
            ),
            tooltip=[
                "policy_tier",
                "metric",
                alt.Tooltip(
                    "rate:Q",
                    format=".2%",
                ),
            ],
        )
        .properties(
            height=340,
        )
    )

    st.altair_chart(
        risk_chart,
        use_container_width=True,
    )


# ============================================================
# POLICY & ECONOMICS
# ============================================================

with policy_tab:

    st.subheader(
        "Portfolio Economics"
    )

    econ1, econ2, econ3, econ4 = (
        st.columns(4)
    )

    econ1.metric(
        "Requested GMV",
        (
            f"${requested_gmv / 1_000_000:.2f}M"
        ),
    )

    econ2.metric(
        "Approved GMV",
        (
            f"${approved_gmv / 1_000_000:.2f}M"
        ),
    )

    econ3.metric(
        "Expected Loss",
        f"${expected_loss:,.0f}",
    )

    econ4.metric(
        "Expected Profit",
        f"${expected_profit:,.0f}",
    )

    st.subheader(
        "Economics by Policy Tier"
    )

    economics_by_tier = (
        df.groupby(
            "policy_tier",
            observed=True,
        )
        .agg(
            applications=(
                "application_id",
                "count",
            ),
            approved_gmv=(
                "approved_limit",
                "sum",
            ),
            expected_loss=(
                "expected_loss",
                "sum",
            ),
            expected_profit=(
                "expected_profit",
                "sum",
            ),
            mean_pd=(
                "probability_default",
                "mean",
            ),
            observed_default_rate=(
                "default_status",
                "mean",
            ),
        )
        .reset_index()
    )

    st.dataframe(
        economics_by_tier.style.format(
            {
                "approved_gmv":
                    "${:,.2f}",

                "expected_loss":
                    "${:,.2f}",

                "expected_profit":
                    "${:,.2f}",

                "mean_pd":
                    "{:.2%}",

                "observed_default_rate":
                    "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Decision Reason Codes"
    )

    reason_summary = (
        df[
            "reason_code"
        ]
        .value_counts()
        .rename_axis(
            "reason_code"
        )
        .reset_index(
            name="applications"
        )
    )

    reason_chart = (
        alt.Chart(
            reason_summary
        )
        .mark_bar()
        .encode(
            y=alt.Y(
                "reason_code:N",
                sort="-x",
                title=None,
            ),
            x=alt.X(
                "applications:Q",
                title="Applications",
            ),
            tooltip=[
                "reason_code",
                "applications",
            ],
        )
        .properties(
            height=max(
                250,
                len(
                    reason_summary
                ) * 38,
            )
        )
    )

    st.altair_chart(
        reason_chart,
        use_container_width=True,
    )


# ============================================================
# MODEL MONITORING
# ============================================================

with model_tab:

    st.subheader(
        "Final Out-of-Sample Performance"
    )

    model1, model2, model3, model4 = (
        st.columns(4)
    )

    model1.metric(
        "ROC-AUC",
        (
            f"{metadata.get(
                'final_test_roc_auc',
                0.7680
            ):.3f}"
        ),
    )

    model2.metric(
        "PR-AUC",
        (
            f"{metadata.get(
                'final_test_pr_auc',
                0.1801
            ):.3f}"
        ),
    )

    model3.metric(
        "Brier Score",
        (
            f"{metadata.get(
                'final_test_brier_score',
                0.0371
            ):.4f}"
        ),
    )

    model4.metric(
        "Mean PD vs Actual",
        (
            f"{mean_pd:.2%} "
            f"/ {actual_default_rate:.2%}"
        ),
    )

    st.caption(
        "Final model metrics are reported on "
        "the previously untouched chronological "
        "test set."
    )

    st.subheader(
        "Model Selection"
    )

    st.markdown(
        """
        **Champion: Unweighted Logistic Regression**

        Logistic Regression, XGBoost, and LightGBM
        were evaluated on the scaled modeling
        portfolio.

        Logistic Regression was selected because it
        delivered the strongest overall validation
        discrimination while retaining well-behaved
        probability estimates, straightforward
        interpretation, and governance-friendly
        deployment characteristics.
        """
    )

    if not calibration.empty:

        st.subheader(
            "Calibration Comparison"
        )

        calibration_display = (
            calibration.copy()
        )

        calibration_display[
            "model_calibration"
        ] = (
            calibration_display[
                "model"
            ].str.title()
            + " — "
            + calibration_display[
                "calibration"
            ].str.title()
        )

        calibration_chart = (
            alt.Chart(
                calibration_display
            )
            .mark_bar()
            .encode(
                y=alt.Y(
                    "model_calibration:N",
                    sort="-x",
                    title=None,
                ),
                x=alt.X(
                    "brier_score:Q",
                    title="Brier Score",
                    scale=alt.Scale(
                        zero=False
                    ),
                ),
                tooltip=[
                    "model",
                    "calibration",
                    alt.Tooltip(
                        "roc_auc:Q",
                        format=".4f",
                    ),
                    alt.Tooltip(
                        "pr_auc:Q",
                        format=".4f",
                    ),
                    alt.Tooltip(
                        "brier_score:Q",
                        format=".4f",
                    ),
                    alt.Tooltip(
                        "mean_predicted_pd:Q",
                        format=".2%",
                    ),
                    alt.Tooltip(
                        "actual_default_rate:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=300,
            )
        )

        st.altair_chart(
            calibration_chart,
            use_container_width=True,
        )

    st.subheader(
        "Model Governance"
    )

    gov1, gov2, gov3 = (
        st.columns(3)
    )

    gov1.info(
        "Model Version\n\n"
        + metadata.get(
            "model_version",
            "v1.0",
        )
    )

    gov2.info(
        "Calibration\n\n"
        "Raw probabilities"
    )

    gov3.info(
        "Behavioral Hard Rejects\n\n"
        "Disabled after ablation"
    )


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

with segment_tab:

    st.subheader(
        "Portfolio Segment Analysis"
    )

    st.caption(
        "Segment metrics below are calculated on "
        "the held-out test portfolio."
    )

    segment_options = [
        column
        for column in [
            "merchant_category",
            "device_type",
            "channel",
            "bank_data_available",
        ]
        if column in df.columns
    ]

    if not segment_options:

        st.info(
            "No segment fields are available "
            "in the final test dataset."
        )

    else:

        selected_segment = (
            st.selectbox(
                "Analyze by",
                options=
                    segment_options,
                format_func=lambda value:
                    value
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),
            )
        )

        segment_summary = (
            df.groupby(
                selected_segment,
                dropna=False,
            )
            .agg(
                applications=(
                    "application_id",
                    "count",
                ),
                approval_rate=(
                    "decision",
                    lambda values:
                        (
                            values
                            == "APPROVE"
                        ).mean(),
                ),
                mean_pd=(
                    "probability_default",
                    "mean",
                ),
                observed_default_rate=(
                    "default_status",
                    "mean",
                ),
                approved_gmv=(
                    "approved_limit",
                    "sum",
                ),
                expected_profit=(
                    "expected_profit",
                    "sum",
                ),
            )
            .reset_index()
        )

        segment_chart_data = (
            segment_summary.melt(
                id_vars=[
                    selected_segment
                ],
                value_vars=[
                    "approval_rate",
                    "observed_default_rate",
                ],
                var_name="metric",
                value_name="rate",
            )
        )

        segment_chart = (
            alt.Chart(
                segment_chart_data
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{selected_segment}:N",
                    title=None,
                ),
                xOffset="metric:N",
                y=alt.Y(
                    "rate:Q",
                    title="Rate",
                    axis=alt.Axis(
                        format="%"
                    ),
                ),
                color=alt.Color(
                    "metric:N",
                    title=None,
                ),
                tooltip=[
                    selected_segment,
                    "metric",
                    alt.Tooltip(
                        "rate:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=350,
            )
        )

        st.altair_chart(
            segment_chart,
            use_container_width=True,
        )

        st.dataframe(
            segment_summary.style.format(
                {
                    "approval_rate":
                        "{:.2%}",

                    "mean_pd":
                        "{:.2%}",

                    "observed_default_rate":
                        "{:.2%}",

                    "approved_gmv":
                        "${:,.2f}",

                    "expected_profit":
                        "${:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DECISION EXPLORER
# ============================================================

with explorer_tab:

    st.subheader(
        "Application Decision Explorer"
    )

    st.caption(
        "Use the sidebar filters to inspect specific "
        "decision and policy populations from the "
        "held-out test portfolio."
    )

    display_columns = [
        "application_id",
        "probability_default",
        "decision",
        "policy_tier",
        "requested_amount",
        "approved_limit",
        "reason_code",
        "expected_loss",
        "expected_profit",
        "default_status",
    ]

    display_columns += [
        column
        for column in [
            "merchant_category",
            "device_type",
            "channel",
            "bank_data_available",
        ]
        if column in filtered_df.columns
    ]

    explorer = (
        filtered_df[
            display_columns
        ]
        .sort_values(
            "probability_default",
            ascending=False,
        )
    )

    st.dataframe(
        explorer,
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability_default":
                st.column_config.NumberColumn(
                    "Predicted PD",
                    format="%.2%%",
                ),

            "requested_amount":
                st.column_config.NumberColumn(
                    "Requested",
                    format="$%.2f",
                ),

            "approved_limit":
                st.column_config.NumberColumn(
                    "Approved",
                    format="$%.2f",
                ),

            "expected_loss":
                st.column_config.NumberColumn(
                    "Expected Loss",
                    format="$%.2f",
                ),

            "expected_profit":
                st.column_config.NumberColumn(
                    "Expected Profit",
                    format="$%.2f",
                ),
        },
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Decision Engine • "
    "12K-user synthetic portfolio • "
    "Final model: Logistic Regression • "
    "Held-out chronological test evaluation • "
    "FastAPI + Docker + Streamlit • "
    "Portfolio demonstration only"
)