from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import RealEstateAgent


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_l = answer.lower()
    hit_count = sum(1 for kw in expected_keywords if kw.lower() in answer_l)
    return hit_count / max(len(expected_keywords), 1)


def _source_hit(citations: list[str], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return False
    source_blob = " ".join(citations).lower()
    return any(src.lower() in source_blob for src in expected_sources)


def run_benchmark(agent: RealEstateAgent, cases: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []

    for case in cases:
        question = case["question"]
        result = agent.ask(question)

        expected_keywords = case.get("expected_keywords", [])
        expected_sources = case.get("expected_sources", [])
        threshold = float(case.get("coverage_threshold", 0.5))

        coverage = _keyword_coverage(result.answer, expected_keywords)
        source_hit = _source_hit(result.citations, expected_sources)
        passed = (coverage >= threshold) and source_hit

        rows.append(
            {
                "question": question,
                "route": result.route,
                "confidence": round(result.confidence, 3),
                "latency_ms": round(result.latency_ms, 1),
                "keyword_coverage": round(coverage, 3),
                "coverage_threshold": round(threshold, 3),
                "source_hit": source_hit,
                "passed": passed,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "pass_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "avg_keyword_coverage": 0.0,
        }

    summary = {
        "pass_rate": float(df["passed"].mean()),
        "avg_confidence": float(df["confidence"].mean()),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "avg_keyword_coverage": float(df["keyword_coverage"].mean()),
    }
    return df, summary
