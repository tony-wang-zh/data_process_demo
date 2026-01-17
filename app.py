# main.py
from __future__ import annotations

from pathlib import Path
import streamlit as st

from storage import connect, init_schema, get_latest_dataset_meta, fetch_latest_insight
from ingest import ingest_csv_into_db
from analytics import compute_analytics_bundle
from llm_client import generate_and_store_insight_structured


# -----------------------------
# Hardcoded paths (per OA spec)
# -----------------------------
DB_PATH = "data/trades.db"
CSV_PATH = "sample_transactions.csv"   
SOURCE_NAME = Path(CSV_PATH).name


def ensure_db_and_data() -> str:
    """
    Ensure DB exists + schema exists + at least one dataset ingested.
    Returns dataset_id to use.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = connect(DB_PATH)
    init_schema(conn)

    meta = get_latest_dataset_meta(conn)
    if meta is None:
        # First run: ingest
        ingest_result = ingest_csv_into_db(
            conn=conn,
            csv_path=CSV_PATH,
            source_name=SOURCE_NAME,
            replace_if_exists=False,
        )
        dataset_id = ingest_result.dataset_meta.dataset_id
    else:
        dataset_id = meta.dataset_id

    conn.close()
    return dataset_id


def main() -> None:
    st.set_page_config(page_title="Trade Surveillance Dashboard", layout="wide")
    st.title("Trade Surveillance Dashboard")

    # ---- startup init ----
    dataset_id = ensure_db_and_data()
    conn = connect(DB_PATH)

    # ---- load analytics ----
    bundle = compute_analytics_bundle(conn, dataset_id)

    # ---- header ----
    meta = get_latest_dataset_meta(conn)
    if meta is not None:
        st.caption(
            f"Dataset: `{meta.dataset_id}` | Source: `{meta.source_name}` | "
            f"Trades: {meta.accepted_rows} accepted / {meta.rejected_rows} rejected"
        )

    # -----------------------------
    # UI: core stats + anomalies
    # -----------------------------
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Per-ticker stats")
        st.dataframe(bundle.per_ticker, use_container_width=True, height=320)

        st.subheader("Per-trader stats")
        st.dataframe(bundle.per_trader, use_container_width=True, height=320)

    with right:
        st.subheader("Anomalies")
        st.dataframe(bundle.anomalies, use_container_width=True, height=420)

        st.subheader("Time series (daily)")
        # simple chart: total notional per day (across tickers)
        if not bundle.time_series.empty:
            ts = bundle.time_series.copy()
            ts_daily = ts.groupby("time_bucket", as_index=False).agg(
                notional=("notional", "sum"),
                volume=("volume", "sum"),
                trade_count=("trade_count", "sum"),
            )
            st.line_chart(ts_daily.set_index("time_bucket")[["notional"]], height=220)
        else:
            st.info("No time series data available.")

    st.divider()

    # -----------------------------
    # UI: LLM Insights
    # -----------------------------
    st.subheader("LLM insights")

    cached = fetch_latest_insight(conn, dataset_id)
    if cached is not None:
        with st.expander("Show cached insight", expanded=True):
            st.markdown(cached.content_markdown)

    col_a, col_b = st.columns([0.25, 0.75], gap="large")
    with col_a:
        run = st.button("Generate / Refresh insight", type="primary")

    with col_b:
        st.caption(
            "Uses aggregated stats to generate a dataset summary with openai"
        )

    if run:
        with st.spinner("Generating insights ... "):
            insight = generate_and_store_insight_structured(
                conn=conn,
                dataset_id=dataset_id,
                llm_summary=bundle.llm_summary,
                force=True,
            )
        st.success("Insight generated.")
        st.markdown(insight.content_markdown)

        with st.expander("Raw structured JSON"):
            st.json(insight.raw_json)

    conn.close()


if __name__ == "__main__":
    main()
