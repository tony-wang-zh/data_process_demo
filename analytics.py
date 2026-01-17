from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import sqlite3
from datetime import datetime

import storage

@dataclass(frozen=True)
class AnalyticsBundle:
    per_ticker: pd.DataFrame
    per_trader: pd.DataFrame
    time_series: pd.DataFrame          # e.g., volume per ticker per day/hour
    anomalies: pd.DataFrame            # rows = flags (type, severity, evidence)
    llm_summary: Dict[str, Any]        # bounded JSON for prompting

"""
    computers per ticker and per trader stats as dataframes 

    per_ticker columns:
      ticker, trade_count, total_volume, buy_volume, sell_volume,
      net_position, total_notional, vwap

    per_trader columns:
      trader_id, trade_count, total_volume, buy_volume, sell_volume,
      net_position, total_notional
"""
def compute_basic_stats(conn: sqlite3.Connection, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ---- load data ----
    df = storage.fetch_transactions(
        conn=conn,
        dataset_id=dataset_id,
    )

    if df.empty:
        per_ticker = pd.DataFrame(columns=[
            "ticker", "trade_count", "total_volume", "buy_volume", "sell_volume",
            "net_position", "total_notional", "vwap"
        ])
        per_trader = pd.DataFrame(columns=[
            "trader_id", "trade_count", "total_volume", "buy_volume", "sell_volume",
            "net_position", "total_notional"
        ])
        return per_ticker, per_trader

    # Ensure numeric types (DB should store REAL but be defensive)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["notional"] = df["quantity"] * df["price"]

    # Helper columns
    df["buy_qty"] = df["quantity"].where(df["action"] == "BUY", 0.0)
    df["sell_qty"] = df["quantity"].where(df["action"] == "SELL", 0.0)
    df["signed_qty"] = df["buy_qty"] - df["sell_qty"]

    # ---- per ticker ----
    g_t = df.groupby("ticker", dropna=False)
    per_ticker = g_t.agg(
        trade_count=("ticker", "size"),
        total_volume=("quantity", "sum"),
        buy_volume=("buy_qty", "sum"),
        sell_volume=("sell_qty", "sum"),
        net_position=("signed_qty", "sum"),
        total_notional=("notional", "sum"),
        _notional_sum=("notional", "sum"),
        _qty_sum=("quantity", "sum"),
    ).reset_index()

    # VWAP = sum(notional)/sum(quantity) (guard div-by-zero)
    per_ticker["vwap"] = per_ticker["_notional_sum"] / per_ticker["_qty_sum"].replace({0.0: pd.NA})
    per_ticker = per_ticker.drop(columns=["_notional_sum", "_qty_sum"])

    # Sort for nicer UI
    per_ticker = per_ticker.sort_values(["total_notional", "total_volume"], ascending=False).reset_index(drop=True)

    # ---- per trader ----
    g_u = df.groupby("trader_id", dropna=False)
    per_trader = g_u.agg(
        trade_count=("trader_id", "size"),
        total_volume=("quantity", "sum"),
        buy_volume=("buy_qty", "sum"),
        sell_volume=("sell_qty", "sum"),
        net_position=("signed_qty", "sum"),
        total_notional=("notional", "sum"),
    ).reset_index()

    per_trader = per_trader.sort_values(["total_notional", "total_volume"], ascending=False).reset_index(drop=True)

    return per_ticker, per_trader

"""
    Compute time-bucketed transaction statistics.

    freq: defines time series bucket size, such as:
    "T", "min",
    "5T", "15T", "30T",
    "H", "2H", "4H",
    "D",
    "W",
    "M",

    Returns a DataFrame with columns:
      time_bucket, ticker, trade_count, volume, notional
"""
def compute_time_series(
    conn: sqlite3.Connection,
    dataset_id: str,
    freq: str = "D",
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    # ---- load data ----
    df = storage.fetch_transactions(
        conn=conn,
        dataset_id=dataset_id,
        ticker=ticker,
    )[["timestamp", "ticker", "quantity", "price"]]

    if df.empty:
        return pd.DataFrame(
            columns=["time_bucket", "ticker", "trade_count", "volume", "notional"]
        )

    # ---- ensure proper types ----
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["notional"] = df["quantity"] * df["price"]

    # this shouldn't do anything (should be handled in ingestion time) but for sanity's sake
    df = df.dropna(subset=["timestamp"])

    # ---- time bucketing ----
    ts = (
        df.groupby([pd.Grouper(key="timestamp", freq=freq), "ticker"])
        .agg(trade_count=("ticker", "size"), volume=("quantity", "sum"), notional=("notional", "sum"))
        .reset_index()
        .rename(columns={"timestamp": "time_bucket"})
    )
    return ts


"""
    Detect simple, explainable anomalies suitable for a trade-surveillance dashboard.

    Returns a DataFrame with columns:
      type, severity, ticker, trader_id, window_start, window_end, description, evidence_json
"""
def detect_anomalies(conn: sqlite3.Connection, dataset_id: str) -> pd.DataFrame:
    
    df = storage.fetch_transactions(conn=conn, dataset_id=dataset_id)

    cols = ["type", "severity", "ticker", "trader_id", "window_start", "window_end", "description", "evidence_json"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Defensive typing
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["action"] = df["action"].astype(str).str.upper()
    df["trader_id"] = df["trader_id"].astype(str)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["notional"] = df["quantity"] * df["price"]

    anomalies = []

    def add_anomaly(
        typ: str,
        severity: int,
        ticker: Optional[str],
        trader_id: Optional[str],
        w_start: Optional[pd.Timestamp],
        w_end: Optional[pd.Timestamp],
        desc: str,
        evidence: Dict[str, Any],
    ):
        anomalies.append(
            {
                "type": typ,
                "severity": int(severity),
                "ticker": ticker,
                "trader_id": trader_id,
                "window_start": w_start,
                "window_end": w_end,
                "description": desc,
                "evidence_json": evidence,
            }
        )

    # ----------------------------
    # A) Volume spike by ticker (hourly buckets; z-score on volume)
    # ----------------------------
    # Bucket to hour to find bursty activity
    df["time_bucket"] = df["timestamp"].dt.floor("H") # floor() here causes static checker error but is fine 
    vol = (
        df.groupby(["ticker", "time_bucket"])
          .agg(volume=("quantity", "sum"), trade_count=("ticker", "size"), notional=("notional", "sum"))
          .reset_index()
    )

    # z-score within each ticker across time buckets
    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mu) / sigma

    vol["z"] = vol.groupby("ticker")["volume"].transform(_zscore)

    # Flag large spikes
    for _, r in vol[vol["z"] >= 3.0].iterrows():
        add_anomaly(
            typ="VOLUME_SPIKE",
            severity=80 if r["z"] >= 4.0 else 60,
            ticker=r["ticker"],
            trader_id=None,
            w_start=r["time_bucket"],
            w_end=r["time_bucket"] + pd.Timedelta(hours=1),
            desc=f"Hourly volume spike for {r['ticker']}: volume={r['volume']:.2f} (z={r['z']:.2f}).",
            evidence={
                "bucket": str(r["time_bucket"]),
                "volume": float(r["volume"]),
                "trade_count": int(r["trade_count"]),
                "notional": float(r["notional"]),
                "z": float(r["z"]),
                "freq": "H",
            },
        )

    # ----------------------------
    # B) Price jump by ticker (hourly avg price; z-score on returns)
    # ----------------------------
    # Use average trade price per hour, then percent change
    px = (
        df.groupby(["ticker", "time_bucket"])
          .agg(avg_price=("price", "mean"))
          .reset_index()
          .sort_values(["ticker", "time_bucket"])
    )
    px["pct_change"] = px.groupby("ticker")["avg_price"].pct_change()

    # z-score on pct_change within ticker
    px["ret_z"] = px.groupby("ticker")["pct_change"].transform(lambda s: _zscore(s.fillna(0.0)))

    # Flag big moves; also use absolute move threshold to avoid tiny-noise
    flagged_px = px[(px["ret_z"].abs() >= 3.0) & (px["pct_change"].abs() >= 0.05)]
    for _, r in flagged_px.iterrows():
        direction = "up" if r["pct_change"] > 0 else "down"
        add_anomaly(
            typ="PRICE_JUMP",
            severity=85 if abs(r["ret_z"]) >= 4.0 else 65,
            ticker=r["ticker"],
            trader_id=None,
            w_start=r["time_bucket"],
            w_end=r["time_bucket"] + pd.Timedelta(hours=1),
            desc=(
                f"Hourly average price jump for {r['ticker']} {direction}: "
                f"{r['pct_change']*100:.1f}% (z={r['ret_z']:.2f})."
            ),
            evidence={
                "bucket": str(r["time_bucket"]),
                "avg_price": float(r["avg_price"]),
                "pct_change": float(r["pct_change"]),
                "ret_z": float(r["ret_z"]),
                "freq": "H",
            },
        )

    # ----------------------------
    # C) Trader concentration by ticker (top trader share of volume)
    # ----------------------------
    tv = (
        df.groupby(["ticker", "trader_id"])
          .agg(volume=("quantity", "sum"), trade_count=("ticker", "size"), notional=("notional", "sum"))
          .reset_index()
    )
    total_by_ticker = tv.groupby("ticker")["volume"].sum().rename("ticker_total_volume").reset_index()
    tv = tv.merge(total_by_ticker, on="ticker", how="left")
    tv["share"] = tv["volume"] / tv["ticker_total_volume"].replace({0.0: pd.NA})

    # Flag if a single trader dominates a ticker
    top = tv.sort_values(["ticker", "share"], ascending=[True, False]).groupby("ticker").head(1)
    for _, r in top.iterrows():
        share = r["share"]
        if pd.isna(share):
            continue
        if share >= 0.60:
            sev = 75 if share < 0.80 else 90
            add_anomaly(
                typ="TRADER_CONCENTRATION",
                severity=sev,
                ticker=r["ticker"],
                trader_id=r["trader_id"],
                w_start=None,
                w_end=None,
                desc=(
                    f"High trader concentration in {r['ticker']}: "
                    f"trader {r['trader_id']} accounts for {share*100:.1f}% of volume."
                ),
                evidence={
                    "ticker_total_volume": float(r["ticker_total_volume"]),
                    "top_trader_volume": float(r["volume"]),
                    "top_trader_trade_count": int(r["trade_count"]),
                    "top_trader_notional": float(r["notional"]),
                    "share": float(share),
                    "threshold": 0.60,
                },
            )

    # ----------------------------
    # D) Rapid buy/sell flip by same trader & ticker (within 5 minutes)
    # ----------------------------
    df_sorted = df.sort_values(["trader_id", "ticker", "timestamp"]).reset_index(drop=True)
    df_sorted["prev_action"] = df_sorted.groupby(["trader_id", "ticker"])["action"].shift(1)
    df_sorted["prev_ts"] = df_sorted.groupby(["trader_id", "ticker"])["timestamp"].shift(1)

    dt = (df_sorted["timestamp"] - df_sorted["prev_ts"])
    flip_mask = (
        df_sorted["prev_action"].notna()
        & (df_sorted["action"] != df_sorted["prev_action"])
        & dt.notna()
        & (dt <= pd.Timedelta(minutes=5))
    )

    flips = df_sorted[flip_mask].copy()
    for _, r in flips.iterrows():
        add_anomaly(
            typ="RAPID_SIDE_FLIP",
            severity=70,
            ticker=r["ticker"],
            trader_id=r["trader_id"],
            w_start=r["prev_ts"],
            w_end=r["timestamp"],
            desc=(
                f"Rapid side flip for trader {r['trader_id']} on {r['ticker']}: "
                f"{r['prev_action']} -> {r['action']} within {dt.loc[r.name].total_seconds():.0f}s."
            ),
            evidence={
                "prev_action": r["prev_action"],
                "action": r["action"],
                "prev_timestamp": str(r["prev_ts"]),
                "timestamp": str(r["timestamp"]),
                "delta_seconds": float(dt.loc[r.name].total_seconds()),
                "quantity": float(r["quantity"]),
                "price": float(r["price"]),
            },
        )

    # Assemble output
    out = pd.DataFrame(anomalies, columns=cols)
    if out.empty:
        return out

    # Sort for UI: highest severity first
    out = out.sort_values(["severity", "type"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_llm_summary(
    per_ticker: pd.DataFrame,
    per_trader: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_items: int = 25,
) -> Dict[str, Any]:
    """
    Build a bounded, structured summary for LLM prompting.
    No raw transaction data is included.
    """
    summary: Dict[str, Any] = {}

    # ----------------------------
    # Dataset-level overview
    # ----------------------------
    summary["overview"] = {
        "num_tickers": int(per_ticker["ticker"].nunique()) if not per_ticker.empty else 0,
        "num_traders": int(per_trader["trader_id"].nunique()) if not per_trader.empty else 0,
        "total_trades": int(per_ticker["trade_count"].sum()) if not per_ticker.empty else 0,
        "total_volume": float(per_ticker["total_volume"].sum()) if not per_ticker.empty else 0.0,
        "total_notional": float(per_ticker["total_notional"].sum()) if not per_ticker.empty else 0.0,
    }

    # ----------------------------
    # Top tickers (by notional, then volume)
    # ----------------------------
    if not per_ticker.empty:
        top_tickers = (
            per_ticker
            .sort_values(["total_notional", "total_volume"], ascending=False)
            .head(max_items)
        )
        summary["top_tickers"] = [
            {
                "ticker": r["ticker"],
                "trade_count": int(r["trade_count"]),
                "total_volume": float(r["total_volume"]),
                "total_notional": float(r["total_notional"]),
                "vwap": float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
                "net_position": float(r["net_position"]),
            }
            for _, r in top_tickers.iterrows()
        ]
    else:
        summary["top_tickers"] = []

    # ----------------------------
    # Top traders (by notional, then volume)
    # ----------------------------
    if not per_trader.empty:
        top_traders = (
            per_trader
            .sort_values(["total_notional", "total_volume"], ascending=False)
            .head(max_items)
        )
        summary["top_traders"] = [
            {
                "trader_id": r["trader_id"],
                "trade_count": int(r["trade_count"]),
                "total_volume": float(r["total_volume"]),
                "total_notional": float(r["total_notional"]),
                "net_position": float(r["net_position"]),
            }
            for _, r in top_traders.iterrows()
        ]
    else:
        summary["top_traders"] = []

    # ----------------------------
    # Meta
    # ----------------------------
    summary["meta"] = {
        "max_items": max_items,
        "anomaly_count": int(len(anomalies)) if anomalies is not None else 0,
        "note": (
            "Summary contains ranked top items only. "
            "Metrics are precomputed; no raw transactions included."
        ),
    }

    # # ----------------------------
    # # Anomalies (already scored)
    # # ----------------------------
    # if anomalies is not None and not anomalies.empty:
    #     top_anomalies = (
    #         anomalies
    #         .sort_values(["severity"], ascending=False)
    #         .head(max_items)
    #     )
    #     summary["anomalies"] = [
    #         {
    #             "type": r["type"],
    #             "severity": int(r["severity"]),
    #             "ticker": r.get("ticker"),
    #             "trader_id": r.get("trader_id"),
    #             "window_start": str(r.get("window_start")) if r.get("window_start") is not None else None,
    #             "window_end": str(r.get("window_end")) if r.get("window_end") is not None else None,
    #             "description": r["description"],
    #         }
    #         for _, r in top_anomalies.iterrows()
    #     ]
    # else:
    #     summary["anomalies"] = []

    # ----------------------------
    # Anomalies (collapsed)
    # ----------------------------
    if anomalies is not None and not anomalies.empty:
        a = anomalies.copy()

        # Ensure these exist; your detect_anomalies already provides them
        for c in ["type", "severity", "ticker", "trader_id", "window_start", "window_end", "description"]:
            if c not in a.columns:
                raise ValueError(f"anomalies missing required column '{c}'")

        # Normalize keys to avoid None/NaN grouping weirdness
        a["ticker"] = a["ticker"].fillna("")
        a["trader_id"] = a["trader_id"].fillna("")

        # Parse window timestamps if they are strings (safe if already datetime-like)
        a["window_start"] = pd.to_datetime(a["window_start"], errors="coerce")
        a["window_end"] = pd.to_datetime(a["window_end"], errors="coerce")

        group_cols = ["type", "ticker", "trader_id"]

        grouped = (
            a.groupby(group_cols, dropna=False)
            .agg(
                count=("type", "size"),
                max_severity=("severity", "max"),
                first_seen=("window_start", "min"),
                last_seen=("window_end", "max"),
            )
            .reset_index()
        )

        # Rank groups by max severity then count
        grouped = grouped.sort_values(["max_severity", "count"], ascending=False)

        # Add a few example descriptions per group (kept small)
        example_k = 3
        examples_map = (
            a.sort_values(["severity"], ascending=False)
            .groupby(group_cols)["description"]
            .apply(lambda s: list(s.head(example_k)))
            .to_dict()
        )

        top_groups = grouped.head(max_items)

        summary["anomalies"] = [
            {
                "type": r["type"],
                "ticker": r["ticker"] or None,
                "trader_id": r["trader_id"] or None,
                "count": int(r["count"]),
                "max_severity": int(r["max_severity"]),
                "first_seen": r["first_seen"].isoformat(sep=" ", timespec="seconds") if pd.notna(r["first_seen"]) else None,
                "last_seen": r["last_seen"].isoformat(sep=" ", timespec="seconds") if pd.notna(r["last_seen"]) else None,
                "example_descriptions": examples_map.get((r["type"], r["ticker"], r["trader_id"]), []),
            }
            for _, r in top_groups.iterrows()
        ]

        summary["meta"]["anomaly_group_count"] = int(len(grouped))
        summary["meta"]["anomaly_example_k"] = example_k
    else:
        summary["anomalies"] = []
        summary["meta"]["anomaly_group_count"] = 0
        summary["meta"]["anomaly_example_k"] = 0

    return summary


def compute_analytics_bundle(conn: sqlite3.Connection, dataset_id: str) -> AnalyticsBundle:
    per_ticker, per_trader = compute_basic_stats(conn, dataset_id)

    # Daily time series for all tickers (good default for dashboards)
    time_series = compute_time_series(conn, dataset_id, freq="D", ticker=None)

    anomalies = detect_anomalies(conn, dataset_id)

    llm_summary = build_llm_summary(
        per_ticker=per_ticker,
        per_trader=per_trader,
        anomalies=anomalies,
        max_items=25,
    )

    return AnalyticsBundle(
        per_ticker=per_ticker,
        per_trader=per_trader,
        time_series=time_series,
        anomalies=anomalies,
        llm_summary=llm_summary,
    )
