from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any, Literal

Action = Literal["BUY", "SELL"]

@dataclass(frozen=True)
class Transaction:
    timestamp: datetime
    ticker: str
    action: Action
    quantity: float
    price: float
    trader_id: str

@dataclass(frozen=True)
class RejectedRow:
    row_index: int
    raw: Dict[str, Any]
    reasons: List[str]

@dataclass(frozen=True)
class DataQualityReport:
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    reasons_count: Dict[str, int]  # e.g., {"missing_price": 3, "bad_timestamp": 2}

@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str               # e.g. sha256 of CSV bytes
    source_name: str              # filename
    ingested_at: datetime
    total_rows: int
    accepted_rows: int
    rejected_rows: int

@dataclass(frozen=True)
class InsightResult:
    dataset_id: str
    generated_at: datetime
    model: str
    prompt_version: str
    content_markdown: str         # ready to render in UI
    raw_json: Optional[Dict[str, Any]] = None  # optional structured output
