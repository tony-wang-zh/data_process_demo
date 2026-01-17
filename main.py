import ingest
import storage
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

    if not dataset_meta:
        print("\n no dataset found")
    else:
        # ---- query back data ----
        print("\nLatest dataset meta:")
        print(dataset_meta)

        dataset_id = dataset_meta.dataset_id
        print("\nFirst 5 transactions:")
        print(storage.fetch_transactions(conn, dataset_id, limit=5))

        print("\nFirst 5 rejected rows:")
        print(storage.fetch_rejected_rows(conn, dataset_id, limit=5))

    conn.close()

if __name__ == "__main__":
    main()