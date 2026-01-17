from __future__ import annotations
from typing import Optional, Iterable, List, Any, Dict
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path\

from model import Transaction, DatasetMeta, DataQualityReport, InsightResult

SCHEMA_VERSION = "1"
SCHEMA_SQL = """
        -- ---------- meta ----------
        CREATE TABLE IF NOT EXISTS app_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO app_meta(key, value)
        VALUES ('schema_version', '1');

        -- ---------- datasets ----------
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id     TEXT PRIMARY KEY,
            source_name    TEXT NOT NULL,
            ingested_at    TEXT NOT NULL, -- ISO 8601 string
            total_rows     INTEGER NOT NULL,
            accepted_rows  INTEGER NOT NULL,
            rejected_rows  INTEGER NOT NULL,
            dq_reasons_json TEXT NOT NULL, -- JSON dict
            dq_notes_json   TEXT NOT NULL  -- JSON list, not populated 
        );

        -- ---------- transactions ----------
        CREATE TABLE IF NOT EXISTS transactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,  -- ISO 8601 string (sortable)
            ticker    TEXT NOT NULL,
            action    TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
            quantity  REAL NOT NULL CHECK (quantity > 0),
            price     REAL NOT NULL CHECK (price > 0),
            trader_id TEXT NOT NULL,

            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_txn_dataset_time
            ON transactions(dataset_id, timestamp);

        CREATE INDEX IF NOT EXISTS idx_txn_dataset_ticker_time
            ON transactions(dataset_id, ticker, timestamp);

        CREATE INDEX IF NOT EXISTS idx_txn_dataset_trader_time
            ON transactions(dataset_id, trader_id, timestamp);

        -- ---------- rejected rows ----------
        CREATE TABLE IF NOT EXISTS rejected_rows (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            timestamp  TEXT,
            ticker     TEXT,
            action     TEXT,
            quantity   TEXT,
            price      TEXT,
            trader_id  TEXT,
            reasons_json TEXT NOT NULL,

            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_rejected_dataset
            ON rejected_rows(dataset_id);

        -- ---------- llm insights ----------
        CREATE TABLE IF NOT EXISTS insights (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id    TEXT NOT NULL,
            generated_at  TEXT NOT NULL, -- ISO 8601
            model         TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            content_markdown TEXT NOT NULL,
            raw_json      TEXT,

            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_insights_dataset_generated
            ON insights(dataset_id, generated_at);
        """

"""
    connect to db 
    this will create db file if doesn't already exist
"""
def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    conn.row_factory = sqlite3.Row

    # Sensible defaults
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    return conn

# ---- initialization ---- 
"""
    Create tables, if they do not exist, 
    safe to call multiple times
"""
def init_schema(conn: sqlite3.Connection) -> None:
    
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(SCHEMA_SQL)

    # Ensure schema version is what we expect (simple check)
    row = conn.execute(
        "SELECT value FROM app_meta WHERE key='schema_version'"
    ).fetchone()
    if row and str(row["value"]) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: db has {row['value']} but app expects {SCHEMA_VERSION}"
        )

    conn.commit()


# ---- ingestion metadata ----
"""
    Store info 
    dataset metadata and data quality report summary
"""
def upsert_dataset_meta(conn: sqlite3.Connection, meta: DatasetMeta, dqr: DataQualityReport) -> None:
    # Ensure we serialize datetimes consistently
    ingested_at = meta.ingested_at
    if isinstance(ingested_at, datetime):
        ingested_at_str = ingested_at.isoformat(sep=" ", timespec="seconds")
    else:
        # If your DatasetMeta ingested_at is already a string, keep it
        ingested_at_str = str(ingested_at)

    dq_reasons_json = json.dumps(dqr.reasons_count, sort_keys=True)
    dq_notes_json = ""

    conn.execute(
        """
        INSERT INTO datasets (
            dataset_id, source_name, ingested_at,
            total_rows, accepted_rows, rejected_rows,
            dq_reasons_json, dq_notes_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            source_name      = excluded.source_name,
            ingested_at      = excluded.ingested_at,
            total_rows       = excluded.total_rows,
            accepted_rows    = excluded.accepted_rows,
            rejected_rows    = excluded.rejected_rows,
            dq_reasons_json  = excluded.dq_reasons_json,
            dq_notes_json    = excluded.dq_notes_json
        """,
        (
            meta.dataset_id,
            meta.source_name,
            ingested_at_str,
            int(dqr.total_rows),
            int(dqr.accepted_rows),
            int(dqr.rejected_rows),
            dq_reasons_json,
            dq_notes_json,
        ),
    )
    conn.commit()

"""
    Return dataset meta by id
    can be used to check if a specific csv is ingested 
"""
def get_dataset_meta_by_id(conn: sqlite3.Connection, dataset_id: str) -> Optional[DatasetMeta]:
    row = conn.execute(
        """
        SELECT dataset_id, source_name, ingested_at, total_rows, accepted_rows, rejected_rows
        FROM datasets
        WHERE dataset_id = ?
        """,
        (dataset_id,),
    ).fetchone()

    if row is None:
        return None

    try:
        ingested_at_dt = datetime.fromisoformat(row["ingested_at"])
    except Exception:
        ingested_at_dt = datetime.strptime(row["ingested_at"], "%Y-%m-%d %H:%M:%S")

    return DatasetMeta(
        dataset_id=row["dataset_id"],
        source_name=row["source_name"],
        ingested_at=ingested_at_dt,
        total_rows=row["total_rows"],
        accepted_rows=row["accepted_rows"],
        rejected_rows=row["rejected_rows"],
    )


"""
    Return newest dataset meta, if any.
    can be used to check any dataset is populated 
"""
def get_latest_dataset_meta(conn: sqlite3.Connection) -> Optional[DatasetMeta]:
    row = conn.execute(
        """
        SELECT
            dataset_id,
            source_name,
            ingested_at,
            total_rows,
            accepted_rows,
            rejected_rows
        FROM datasets
        ORDER BY ingested_at DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        return None

    # Parse ISO timestamp back to datetime
    ingested_at = row["ingested_at"]
    try:
        ingested_at_dt = datetime.fromisoformat(ingested_at)
    except Exception:
        # Fallback: keep as raw string if parsing fails
        ingested_at_dt = datetime.strptime(ingested_at, "%Y-%m-%d %H:%M:%S")

    return DatasetMeta(
        dataset_id=row["dataset_id"],
        source_name=row["source_name"],
        ingested_at=ingested_at_dt,
        total_rows=row["total_rows"],
        accepted_rows=row["accepted_rows"],
        rejected_rows=row["rejected_rows"],
    )

def get_latest_dataset_id(conn: sqlite3.Connection) -> str | None:
    latest_dataset_meta = get_latest_dataset_meta(conn)
    if latest_dataset_meta is not None:
        return latest_dataset_meta.dataset_id
    else:
        return None

# ---- transactions ----
"""
    Bulk insert transactions for a dataset. 
    Returns inserted row count.
"""
def insert_transactions(conn: sqlite3.Connection, dataset_id: str, df_clean: pd.DataFrame) -> int:
    txns = [
        Transaction(
            timestamp=row["timestamp"],
            ticker=row["ticker"],
            action=row["action"],
            quantity=row["quantity"],
            price=row["price"],
            trader_id=row["trader_id"],
        )
        for _, row in df_clean.iterrows()
    ]

    rows = []
    for t in txns:
        ts = t.timestamp
        if isinstance(ts, datetime):
            ts_str = ts.isoformat(sep=" ", timespec="seconds")
        else:
            ts_str = str(ts)

        rows.append((
            dataset_id,
            ts_str,
            t.ticker,
            t.action,
            float(t.quantity),
            float(t.price),
            t.trader_id,
        ))

    if not rows:
        return 0

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO transactions (
            dataset_id, timestamp, ticker, action, quantity, price, trader_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return cur.rowcount

# helper for timestamp consistent with insertion format
def _dt_to_iso(ts: datetime) -> str:
    return ts.isoformat(sep=" ", timespec="seconds")

"""
    Main query method 
    Return filtered transactions 
    sorted by ascending time
"""
def fetch_transactions(
    conn: sqlite3.Connection,
    dataset_id: str,
    ticker: Optional[str] = None,
    trader_id: Optional[str] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    where: List[str] = ["dataset_id = ?"]
    params: List[Any] = [dataset_id]

    if ticker is not None and str(ticker).strip() != "":
        where.append("ticker = ?")
        params.append(str(ticker).strip().upper())

    if trader_id is not None and str(trader_id).strip() != "":
        where.append("trader_id = ?")
        params.append(str(trader_id).strip())

    if start_ts is not None:
        where.append("timestamp >= ?")
        params.append(_dt_to_iso(start_ts))

    if end_ts is not None:
        where.append("timestamp < ?")
        params.append(_dt_to_iso(end_ts))

    where_sql = " AND ".join(where)

    sql = f"""
        SELECT
            id, dataset_id, timestamp, ticker, action, quantity, price, trader_id
        FROM transactions
        WHERE {where_sql}
        ORDER BY timestamp ASC, id ASC
    """
    if limit is not None:
        sql += "LIMIT ?"
        params.append(int(limit))

    df = pd.read_sql_query(sql, conn, params=params)

    # Optional: parse timestamp back to datetime for analytics/UI
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# ---- utils ----
"""
    Return distinct tickers for a dataset.
"""
def list_tickers(conn: sqlite3.Connection, dataset_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM transactions
        WHERE dataset_id = ?
        ORDER BY ticker ASC
        """,
        (dataset_id,),
    ).fetchall()

    return [row["ticker"] for row in rows]

"""
    Return distinct trader_ids for a dataset.
"""
def list_traders(conn: sqlite3.Connection, dataset_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT trader_id
        FROM transactions
        WHERE dataset_id = ?
        ORDER BY trader_id ASC
        """,
        (dataset_id,),
    ).fetchall()

    return [row["trader_id"] for row in rows]
    
# ---- rejected rows utils ----
"""
    store rejected rows for analysis
    Expected columns in `rejected` df to be stored
      - rejection_reasons: list[str] OR a string representation
      - other columns representing the raw row values
    
"""
def insert_rejected_rows(conn: sqlite3.Connection, dataset_id: str, rejected: pd.DataFrame) -> int:
    if rejected is None or rejected.empty:
        return 0

    df = rejected.copy()

    # Ensure required columns exist (fail fast)
    expected = ["timestamp", "ticker", "action", "quantity", "price", "trader_id", "rejection_reasons"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Rejected dataframe missing columns: {missing}")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            dataset_id,
            str(r["timestamp"]),
            str(r["ticker"]),
            str(r["action"]),
            str(r["quantity"]),
            str(r["price"]),
            str(r["trader_id"]),
            json.dumps(r["rejection_reasons"]),
        ))

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO rejected_rows (
            dataset_id, timestamp, ticker, action, quantity, price, trader_id, reasons_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return cur.rowcount

"""
    return rejected rows for inspection in UI.
"""
def fetch_rejected_rows(conn: sqlite3.Connection, dataset_id: str, limit: int = 10) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            timestamp, ticker, action, quantity, price, trader_id, reasons_json
        FROM rejected_rows
        WHERE dataset_id = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        conn,
        params=(dataset_id, int(limit)),
    )

    if df.empty:
        # keep the expected column name even if empty
        df["rejection_reasons"] = []
        return df.drop(columns=["reasons_json"], errors="ignore")

    def parse_reasons(s: Optional[str]):
        if s is None:
            return []
        try:
            val = json.loads(s)
            return val if isinstance(val, list) else [str(val)]
        except Exception:
            return [str(s)]

    df["rejection_reasons"] = df["reasons_json"].apply(parse_reasons)
    df = df.drop(columns=["reasons_json"])

    return df
    
# ---- LLM insights utils ----
"""
    Save the LLM insight for a dataset
"""
def save_insight(conn: sqlite3.Connection, insight: InsightResult) -> None:
    """
    Persist an InsightResult to the insights table.
    """
    generated_at = insight.generated_at
    if isinstance(generated_at, datetime):
        generated_at_str = generated_at.isoformat(sep=" ", timespec="seconds")
    else:
        generated_at_str = str(generated_at)

    raw_json_str = json.dumps(insight.raw_json) if insight.raw_json is not None else None

    conn.execute(
        """
        INSERT INTO insights (
            dataset_id, generated_at, model, prompt_version, content_markdown, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            insight.dataset_id,
            generated_at_str,
            insight.model,
            insight.prompt_version,
            insight.content_markdown,
            raw_json_str,
        ),
    )
    conn.commit()

"""
    Return the most recent cached insight for 
    return none if none exists 
"""
def fetch_latest_insight(conn: sqlite3.Connection, dataset_id: str) -> Optional[InsightResult]:
    
    row = conn.execute(
        """
        SELECT
            dataset_id, generated_at, model, prompt_version, content_markdown, raw_json
        FROM insights
        WHERE dataset_id = ?
        ORDER BY generated_at DESC, id DESC
        LIMIT 1
        """,
        (dataset_id,),
    ).fetchone()

    if row is None:
        return None

    # Parse generated_at
    gen = row["generated_at"]
    try:
        generated_at_dt = datetime.fromisoformat(gen)
    except Exception:
        generated_at_dt = datetime.strptime(gen, "%Y-%m-%d %H:%M:%S")

    # Parse raw_json
    raw_json_val = None
    if row["raw_json"] is not None:
        try:
            raw_json_val = json.loads(row["raw_json"])
        except Exception:
            raw_json_val = None

    return InsightResult(
        dataset_id=row["dataset_id"],
        generated_at=generated_at_dt,
        model=row["model"],
        prompt_version=row["prompt_version"],
        content_markdown=row["content_markdown"],
        raw_json=raw_json_val,
    )

