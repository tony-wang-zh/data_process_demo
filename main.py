import ingest
from datetime import datetime
from pathlib import Path

from storage import connect, init_schema
from storage import (
    upsert_dataset_meta,
    insert_transactions,
    insert_rejected_rows,
    fetch_transactions,
    fetch_rejected_rows,
    get_latest_dataset_meta,
)
from model import DatasetMeta, Transaction


def main():
    # read
    csv_path = "sample_transactions.csv"
    print("dataset_id:", ingest.compute_dataset_id(csv_path))
    df = ingest.read_csv_bytes(csv_path)
    # print(df.head())

    # process
    df_clean, df_rejected, report = ingest.validate_and_clean(df)

    # store 
    DB_PATH = "data/test_trades.db"
    Path("data").mkdir(exist_ok=True)

    conn = connect(DB_PATH)
    init_schema(conn)

    dataset_id = "test_dataset_001"

    meta = DatasetMeta(
        dataset_id=dataset_id,
        source_name="sample.csv",
        ingested_at=datetime.now(),
        total_rows=report.total_rows,
        accepted_rows=report.accepted_rows,
        rejected_rows=report.rejected_rows,
    )

    upsert_dataset_meta(conn, meta, report)

    inserted = insert_transactions(conn, dataset_id, df_clean)
    print(f"Inserted {inserted} clean transactions")

    # ---- store rejected rows ----
    rejected_inserted = insert_rejected_rows(conn, dataset_id, df_rejected)
    print(f"Inserted {rejected_inserted} rejected rows")

    # ---- query back data ----
    print("\nLatest dataset meta:")
    print(get_latest_dataset_meta(conn))

    print("\nFirst 5 transactions:")
    print(fetch_transactions(conn, dataset_id, limit=5))

    print("\nFirst 5 rejected rows:")
    print(fetch_rejected_rows(conn, dataset_id, limit=5))

    conn.close()

if __name__ == "__main__":
    main()