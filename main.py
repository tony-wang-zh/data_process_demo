import ingest
import storage
import analytics
import llm_client
from datetime import datetime
from pathlib import Path
from model import DatasetMeta, Transaction


def main():
    # read
    csv_path = "sample_transactions.csv"
    DB_PATH = "data/test_trades.db"
    Path("data").mkdir(exist_ok=True)

    conn = storage.connect(DB_PATH)
    storage.init_schema(conn)

    ingest.ingest_csv_into_db(
        conn,
        csv_path,
        csv_path,
    )

    dataset_meta = storage.get_latest_dataset_meta(conn)

    assert dataset_meta is not None

    # ---- check query back data ----
    print(" --- Overview --- ")
    print("\nLatest dataset meta:")
    print(dataset_meta)

    dataset_id = dataset_meta.dataset_id
    print("\nFirst 5 transactions:")
    print(storage.fetch_transactions(conn, dataset_id, limit=5))

    print("\nFirst 5 rejected rows:")
    print(storage.fetch_rejected_rows(conn, dataset_id, limit=5))

    # --- check analytics ---
    bundle = analytics.compute_analytics_bundle(conn, dataset_id)
    print(" --- analytics --- ")
    print(bundle.per_ticker.head(5))
    print(bundle.per_trader.head(5))
    print(bundle.time_series.head(5))
    print(bundle.anomalies.head(5))
    print(bundle.llm_summary)

    # --- insight  ---
    # ---- call OpenAI (structured) ----
    insight = llm_client.generate_and_store_insight_structured(
        conn=conn,
        dataset_id=dataset_id,
        llm_summary=bundle.llm_summary,
        force=True,          # force refresh for testing
    )

    # ---- display result ----
    print("\n===== LLM MARKDOWN OUTPUT =====\n")
    print(insight.content_markdown)

    print("\n===== RAW STRUCTURED JSON =====\n")
    print(insight.raw_json)

    conn.close()

    conn.close()

if __name__ == "__main__":
    main()