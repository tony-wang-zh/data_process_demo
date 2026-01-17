from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import sqlite3
import hashlib
from pathlib import Path
import pandas as pd
from typing import Dict, List

from model import DataQualityReport, DatasetMeta

@dataclass(frozen=True)
class IngestResult:
    dataset_meta: DatasetMeta
    data_quality: DataQualityReport
    accepted_preview: pd.DataFrame   # small preview for UI
    rejected_preview: pd.DataFrame   # small preview for UI

# timestamp,ticker,action,quantity,price,trader_id
REQUIRED_COLUMNS = ["timestamp", "ticker", "action", "quantity", "price", "trader_id"]
VALID_ACTIONS = {"BUY", "SELL"}

"""
    just a simple method to generate an id for our dataset using hashing
"""
def compute_dataset_id(csv_path: str) -> str:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
    
"""
    Read CSV file into a raw DataFrame.
"""
def read_csv_bytes(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(
            path,
            dtype=str,            # read everything as string first
            keep_default_na=False # prevent silent NaN coercion
        )
    except Exception as e:
        raise ValueError(f"Error reading read CSV file: {csv_path}") from e

    return df

"""
    helper for rejecting invalid rows, reject empty values 
"""
def _is_missing_series(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")

"""
    validate and clean a dataframe 
    return:
        df_clean: normalized + typed accepted rows
        df_rejected: rejected rows with 'reasons' column
        report: counts + notes
"""
def validate_and_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    if df_raw is None:
        raise ValueError("input df_raw is None")

    # ---
    # sanity check: columns and normalize name
    # exit if missing require column 
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Fatal: CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    #  --- 
    # normalize string type values 
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["action"] = df["action"].astype(str).str.strip().str.upper()
    df["trader_id"] = df["trader_id"].astype(str).str.strip()
    
    # ---
    # per row check 
    # for each row, find rejection reasons 
    # TODO: this should be its own method 
    reasons: Dict[int, List[str]] = {}

    def add_reason(mask: pd.Series, reason: str) -> None:
        idxs = df.index[mask]
        for i in idxs:
            reasons.setdefault(int(i), []).append(reason)

    for col in REQUIRED_COLUMNS:
        add_reason(_is_missing_series(df[col]), f"missing_{col}")

    # action 
    action_missing = _is_missing_series(df["action"])
    add_reason((~action_missing) & (~df["action"].isin(VALID_ACTIONS)), "invalid_action")

    # timestamp
    ts_missing = _is_missing_series(df["timestamp"])
    ts_parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    add_reason((~ts_missing) & (ts_parsed.isna()), "invalid_timestamp")

    # quantiy and price (numeric values)
    qty_missing = _is_missing_series(df["quantity"])
    price_missing = _is_missing_series(df["price"])
    # type  
    qty_num = pd.to_numeric(df["quantity"], errors="coerce")
    price_num = pd.to_numeric(df["price"], errors="coerce")
    # check missing
    add_reason((~qty_missing) & (qty_num.isna()), "invalid_quantity")
    add_reason((~price_missing) & (price_num.isna()), "invalid_price")
    # expected to be positive 
    add_reason((~qty_missing) & (~qty_num.isna()) & (qty_num <= 0), "non_positive_quantity")
    add_reason((~price_missing) & (~price_num.isna()) & (price_num <= 0), "non_positive_price")

    # ---
    # create clean and rejected data frames from reason 
    if reasons:
        rejected_idx = sorted(reasons.keys())
        rejected_mask = df.index.isin(rejected_idx)
    else:
        rejected_mask = pd.Series(False, index=df.index)

    df_orig = df.copy()
    df_rejected = df_orig.loc[rejected_mask].copy()
    df_rejected["rejection_reasons"] = [
        reasons.get(int(i), []) for i in df_rejected.index
    ]
    df_clean = df.loc[~rejected_mask].copy()

    # ---
    # type casting for clean data 
    df_clean["timestamp"] = ts_parsed.loc[df_clean.index]

    # Quantity/price: numeric
    df_clean["quantity"] = qty_num.loc[df_clean.index].astype(float)
    df_clean["price"] = price_num.loc[df_clean.index].astype(float)

    # optional, but nice
    df_clean = df_clean.sort_values("timestamp", ascending=True)

    # optional but nice 
    df_clean = df_clean.reset_index(drop=True)
    df_rejected = df_rejected.reset_index(drop=True)

    # --- 
    # generate data processing report
    # TODO: this should be its own method 
    reasons_count: Dict[str, int] = {}
    for rs in reasons.values():
        for r in rs:
            reasons_count[r] = reasons_count.get(r, 0) + 1

    report = DataQualityReport(
        total_rows=len(df),
        accepted_rows=len(df_clean),
        rejected_rows=len(df_rejected),
        reasons_count=reasons_count,
    ) 

    return df_clean, df_rejected, report


"""
read csv, validate and clean rows, 
call into storage method to create and populate database
output: info about dataset (IngestResult)
"""
def ingest_csv_into_db(
    conn: sqlite3.Connection,
    csv_bytes: bytes,
    source_name: str,
    replace_if_exists: bool = False,
) -> IngestResult:
    """
    End-to-end:
      - dataset_id = hash(csv)
      - if exists and not replace: no-op (or return existing meta)
      - else: clean -> insert accepted -> store rejected -> store meta
    """
    return None
