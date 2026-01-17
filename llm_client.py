from __future__ import annotations


import storage
from typing import Optional, Dict, Any
import sqlite3

from model import InsightResult
from typing import List, Optional
from pydantic import BaseModel, Field
import json
from typing import Dict, Any, Tuple

PROMPT_VERSION = "v1"

class UnusualActivity(BaseModel):
    title: str = Field(..., description="Short label for the unusual pattern")
    description: str = Field(..., description="What happened and why it stands out")
    tickers: List[str] = Field(default_factory=list)
    trader_ids: List[str] = Field(default_factory=list)
    severity: int = Field(..., ge=0, le=100, description="0-100 severity estimate")
    evidence: List[str] = Field(default_factory=list, description="Key supporting facts from the summary")


class LlmInsights(BaseModel):
    executive_summary: str
    key_observations: List[str]
    unusual_or_risky_activity: List[UnusualActivity]
    context_and_caveats: List[str]
    suggested_followups: List[str]

"""
    create prompt to send to llm api

    system prompt is general, about role llm plays 
    user prompt is for this specific query
"""
def build_prompt(llm_summary: Dict[str, Any]) -> Tuple[str, str]:
    
    # ----------------------------
    # System prompt: role & rules
    # ----------------------------
    system_prompt = (
        "You are a financial risk and market surveillance analyst.\n\n"
        "Your task is to analyze summarized trading activity and identify:\n"
        "- notable patterns\n"
        "- potential risks or unusual behavior\n"
        "- concentrations or anomalies worth follow-up\n\n"
        "Rules:\n"
        "- Base your analysis ONLY on the provided summary.\n"
        "- Do NOT assume intent or legality.\n"
        "- Use cautious, professional language.\n"
        "- If something may have benign explanations, mention that.\n"
        "- Prefer concise, structured insights over speculation.\n"
    )

    # ----------------------------
    # User prompt: task + data
    # ----------------------------
    summary_json = json.dumps(llm_summary, indent=2)

    user_prompt = (
        "Below is a structured summary of trading activity derived from transaction data.\n"
        "The summary includes aggregate statistics and detected anomalies.\n\n"
        "Analyze this summary and provide insights in the following structure:\n\n"
        "1. **Key Observations**\n"
        "   - Major patterns or concentrations in activity\n\n"
        "2. **Unusual or Risky Activity**\n"
        "   - Behaviors that stand out relative to the rest of the dataset\n\n"
        "3. **Context & Caveats**\n"
        "   - Benign explanations or data limitations\n\n"
        "4. **Suggested Follow-ups**\n"
        "   - Specific next steps an analyst might take\n\n"
        "Do NOT repeat raw numbers unless necessary to support a point.\n"
        "Do NOT reference missing data.\n"
        "Respond using the provided JSON schema.\n\n"
        "=== TRADING SUMMARY ===\n"
        f"{summary_json}\n"
        "=== END SUMMARY ==="
    )

    return system_prompt, user_prompt


import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI

def _render_markdown(insights: LlmInsights) -> str:
    """Turn structured insights into a readable Markdown block for the dashboard."""
    lines = []
    lines.append(f"## Executive summary\n{insights.executive_summary}\n")

    lines.append("## Key observations")
    for x in insights.key_observations:
        lines.append(f"- {x}")
    lines.append("")

    lines.append("## Unusual or risky activity")
    if insights.unusual_or_risky_activity:
        for item in insights.unusual_or_risky_activity:
            lines.append(f"**{item.title}** (severity {item.severity}/100)")
            lines.append(f"- {item.description}")
            if item.tickers:
                lines.append(f"- Tickers: {', '.join(item.tickers)}")
            if item.trader_ids:
                lines.append(f"- Traders: {', '.join(item.trader_ids)}")
            if item.evidence:
                lines.append("- Evidence:")
                for e in item.evidence:
                    lines.append(f"  - {e}")
            lines.append("")
    else:
        lines.append("- None detected.")
        lines.append("")

    lines.append("## Context & caveats")
    for x in insights.context_and_caveats:
        lines.append(f"- {x}")
    lines.append("")

    lines.append("## Suggested follow-ups")
    for x in insights.suggested_followups:
        lines.append(f"- {x}")
    lines.append("")

    return "\n".join(lines).strip()


def generate_and_store_insight_structured(
    conn,
    dataset_id: str,
    llm_summary: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
    prompt_version: str = "v2_structured",
    force: bool = False,
    max_retries: int = 3,
) -> InsightResult:
    """
    Like generate_and_store_insight, but uses Structured Outputs (Pydantic) for reliable JSON.

    Note: Structured Outputs are supported on newer models (GPT-4o and later). :contentReference[oaicite:3]{index=3}
    """
    if not force:
        cached = storage.fetch_latest_insight(conn, dataset_id)
        if cached is not None:
            return cached

    system_prompt, user_prompt = build_prompt(llm_summary)

    client = OpenAI()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=LlmInsights,  # <-- Structured Outputs via Pydantic :contentReference[oaicite:4]{index=4}
            )

            assert resp.output_parsed is not None
            parsed: LlmInsights = resp.output_parsed

            raw_json = parsed.model_dump()

            content_md = _render_markdown(parsed)

            insight = InsightResult(
                dataset_id=dataset_id,
                generated_at=datetime.utcnow(),
                model=model,
                prompt_version=prompt_version,
                content_markdown=content_md,
                raw_json=raw_json,
            )
            storage.save_insight(conn, insight)
            return insight

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(0.8 * (2 ** attempt))
                continue
            break

    raise RuntimeError(f"OpenAI structured call failed after {max_retries} attempts: {last_err}") from last_err

def generate_and_store_insight(
    conn,
    dataset_id: str,
    llm_summary: Dict[str, Any],
    *,
    model: str = "gpt-4.1-mini",
    prompt_version: str = "v1",
    force: bool = False,
    max_retries: int = 3,
) -> InsightResult:
    """
    Generate LLM insights for a dataset (from llm_summary), store them in SQLite, and return result.

    Caching behavior:
      - If force=False and a cached insight exists for this dataset, return it.
      - If force=True, always call the API and store a new insight row.
    """
    if not force:
        cached = storage.fetch_latest_insight(conn, dataset_id)
        if cached is not None:
            return cached

    system_prompt, user_prompt = build_prompt(llm_summary)

    client = OpenAI()

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content_md = (resp.output_text or "").strip()

            insight = InsightResult(
                dataset_id=dataset_id,
                generated_at=datetime.utcnow(),
                model=model,
                prompt_version=prompt_version,
                content_markdown=content_md,
                raw_json=None,
            )
            storage.save_insight(conn, insight)
            return insight

        except Exception as e:
            last_err = e
            # Simple exponential backoff (good enough for OA / demo)
            if attempt < max_retries - 1:
                time.sleep(0.8 * (2 ** attempt))
                continue
            break

    # If we got here, all retries failed
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}") from last_err
