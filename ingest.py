from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import sqlite3
import hashlib
from pathlib import Path
import pandas as pd

from model import DataQualityReport, DatasetMeta

@dataclass(frozen=True)
class IngestResult:
    dataset_meta: DatasetMeta
    data_quality: DataQualityReport
    accepted_preview: pd.DataFrame   # small preview for UI
    rejected_preview: pd.DataFrame   # small preview for UI

"""
    Return id for the dataset using hash 
"""
def compute_dataset_id(csv_path: str) -> str:
    return "0"
    
"""
    Read CSV file into a raw DataFrame.
"""
def read_csv_bytes(csv_path: str) -> pd.DataFrame:
    return pd.DataFrame()

"""
    validate and clean a dataframe 
    return:
        df_clean: normalized + typed accepted rows
        df_rejected: rejected rows with 'reasons' column
        report: counts + notes
"""
def validate_and_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    return None
    

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
