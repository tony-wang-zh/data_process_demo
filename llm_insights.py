from __future__ import annotations
from typing import Optional, Dict, Any
import sqlite3

from .models import InsightResult

PROMPT_VERSION = "v1"

def build_prompt(llm_summary: Dict[str, Any]) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    Keep it stable + versioned for reproducibility.
    """

def generate_insights(
    llm_summary: Dict[str, Any],
    model: str,
    api_key: Optional[str] = None,
    timeout_s: int = 60,
) -> InsightResult:
    """Call LLM and return InsightResult (markdown + optional JSON)."""

def get_or_generate_insights(
    conn: sqlite3.Connection,
    dataset_id: str,
    llm_summary: Dict[str, Any],
    model: str,
    force_refresh: bool = False,
) -> InsightResult:
    """Use cached insight if present unless force_refresh."""
