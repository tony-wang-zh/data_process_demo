from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import sqlite3
from datetime import datetime

@dataclass(frozen=True)
class AnalyticsBundle:
    per_ticker: pd.DataFrame
    per_trader: pd.DataFrame
    time_series: pd.DataFrame          # e.g., volume per ticker per day/hour
    anomalies: pd.DataFrame            # rows = flags (type, severity, evidence)
    llm_summary: Dict[str, Any]        # bounded JSON for prompting

def compute_basic_stats(conn: sqlite3.Connection, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_ticker_df, per_trader_df)."""

def compute_time_series(
    conn: sqlite3.Connection,
    dataset_id: str,
    freq: str = "D",
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Time-bucketed stats (volume/notional/trade_count)."""

def detect_anomalies(conn: sqlite3.Connection, dataset_id: str) -> pd.DataFrame:
    """
    Return anomaly flags (volume spikes, price jumps, concentration, rapid flips, etc.)
    Columns should include: type, severity, ticker, trader_id, window_start, window_end, description, evidence_json
    """

def build_llm_summary(
    per_ticker: pd.DataFrame,
    per_trader: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_items: int = 25,
) -> Dict[str, Any]:
    """Create bounded, structured summary for LLM prompt (no raw CSV)."""

def compute_analytics_bundle(conn: sqlite3.Connection, dataset_id: str) -> AnalyticsBundle:
    """Convenience: compute all + return bundle."""
