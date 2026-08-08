"""Persistent, user-approved correction memory for DataSense AI."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = Path("memory") / "datasense_feedback.db"
ALLOWED_OPERATIONS = {"sum", "mean", "count", "min", "max", "median", "std", "nunique"}
ALLOWED_TIME_GRAINS = {"day", "week", "month", "quarter", "year"}
ALLOWED_CHARTS = {"bar", "line", "area", "pie", "scatter", "histogram", "table", "3d_bar"}


def schema_fingerprint(df: pd.DataFrame) -> str:
    schema = "|".join(sorted(f"{col}:{df[col].dtype}" for col in df.columns))
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()[:20]


def _connect(db_path=DEFAULT_DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_hash TEXT NOT NULL,
            schema_columns TEXT NOT NULL,
            original_question TEXT NOT NULL,
            feedback_text TEXT NOT NULL,
            rule_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def is_feedback_message(text: str, previous_plan: dict | None) -> bool:
    if not previous_plan:
        return False
    normalized = text.lower().strip()
    markers = (
        "this is wrong", "that is wrong", "wrong column", "incorrect",
        "not correct", "should use", "use instead", "instead use",
        "my expectation", "you should fetch", "do not use", "don't use",
        "made a mistake", "you used the wrong",
        "i prefer", "i want to see", "always show", "always use",
        "whenever i ask", "for correlation use", "for correlation i want",
    )
    return any(marker in normalized for marker in markers)


def _important_terms(text: str) -> list[str]:
    stopwords = {
        "show", "give", "find", "what", "which", "with", "from", "this",
        "that", "only", "please", "data", "dataset", "the", "and", "for",
        "into", "using", "use", "want", "need", "make", "calculate", "me",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return list(dict.fromkeys(token for token in tokens if len(token) > 2 and token not in stopwords))[:10]


def _explicit_columns(text: str, df: pd.DataFrame) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    found = []
    for column in df.columns:
        col_normalized = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        if col_normalized and re.search(rf"\b{re.escape(col_normalized)}\b", normalized):
            found.append(column)
    return found


def _fallback_proposal(feedback_text, original_question, previous_plan, df):
    mentioned = _explicit_columns(feedback_text, df)
    measure = next((c for c in mentioned if pd.api.types.is_numeric_dtype(df[c])), None)
    date_column = next((
        c for c in mentioned
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or "date" in str(c).lower() or "time" in str(c).lower()
    ), None)
    group_by = [
        c for c in mentioned
        if c not in (measure, date_column) and not pd.api.types.is_numeric_dtype(df[c])
    ]
    text = feedback_text.lower()
    operation = None
    if any(term in text for term in ("average", "mean")):
        operation = "mean"
    elif "count" in text:
        operation = "count"
    elif any(term in text for term in ("total", "sum")):
        operation = "sum"

    time_granularity = None
    for grain, variants in {
        "month": ("month", "monthly"),
        "quarter": ("quarter", "quarterly"),
        "year": ("year", "yearly", "annual"),
        "week": ("week", "weekly"),
        "day": ("day", "daily"),
    }.items():
        if any(term in text for term in variants):
            time_granularity = grain
            break

    chart = None
    chart_patterns = (
        (("scatter plot", "scatter chart", "scatterplot"), "scatter"),
        (("3d bar", "3d chart"), "3d_bar"),
        (("bar chart", "bar graph"), "bar"),
        (("line chart", "line graph"), "line"),
        (("area chart",), "area"),
        (("pie chart",), "pie"),
        (("histogram",), "histogram"),
        (("table",), "table"),
    )
    for variants, chart_name in chart_patterns:
        if any(term in text for term in variants):
            chart = chart_name
            break

    explicit_preference = any(marker in text for marker in (
        "i prefer", "i want to see", "always show", "always use",
        "whenever i ask", "for correlation use", "for correlation i want",
    ))

    return {
        "instruction": feedback_text.strip(),
        "trigger_terms": _important_terms(original_question),
        "measure": measure,
        "group_by": group_by,
        "date_column": date_column,
        "operation": operation,
        "time_granularity": time_granularity,
        "chart": chart,
        "auto_save": explicit_preference and chart is not None,
    }


def propose_feedback_rule(feedback_text, original_question, previous_plan, df):
    """Convert natural-language feedback into a validated correction proposal."""
    columns = [f"{column} ({df[column].dtype})" for column in df.columns]
    prompt = f"""
Extract a reusable correction rule for a data-analysis planner.

Dataset columns:
{chr(10).join(columns)}

Original question:
{original_question}

Plan that produced the wrong result:
{json.dumps(previous_plan, default=str)}

User correction:
{feedback_text}

Return JSON only:
{{
  "instruction": "short correction in plain English",
  "trigger_terms": ["terms that identify similar questions"],
  "measure": "exact numeric dataset column or null",
  "group_by": ["exact categorical dataset columns"],
  "date_column": "exact dataset date column or null",
  "operation": "sum|mean|count|min|max|median|std|nunique|null",
  "time_granularity": "day|week|month|quarter|year|null",
  "chart": "bar|line|area|pie|scatter|histogram|table|3d_bar|null"
}}

Never invent a column. Use null when the correction does not specify a field.
"""

    try:
        # Import lazily: browsing and applying already-saved rules should not
        # need the Ollama client to be imported or running.
        from llm_agent import ask_llm

        raw = ask_llm(
            "You extract structured, reusable user corrections for DataSense AI.",
            prompt,
            json_mode=True,
        )
        proposal = json.loads(raw)
    except Exception:
        proposal = _fallback_proposal(feedback_text, original_question, previous_plan, df)

    fallback = _fallback_proposal(feedback_text, original_question, previous_plan, df)
    columns_set = set(df.columns)

    measure = proposal.get("measure")
    if measure not in columns_set or not pd.api.types.is_numeric_dtype(df[measure]):
        measure = fallback.get("measure")

    date_column = proposal.get("date_column")
    if date_column not in columns_set:
        date_column = fallback.get("date_column")

    groups = [
        column for column in proposal.get("group_by", [])
        if column in columns_set and column not in (measure, date_column)
    ] or fallback.get("group_by", [])

    operation = proposal.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        operation = fallback.get("operation")

    time_granularity = proposal.get("time_granularity")
    if time_granularity not in ALLOWED_TIME_GRAINS:
        time_granularity = fallback.get("time_granularity")

    chart = str(proposal.get("chart") or "").lower().replace(" ", "_") or None
    if chart not in ALLOWED_CHARTS:
        chart = fallback.get("chart")

    instruction = str(proposal.get("instruction") or feedback_text).strip()
    extracted_trigger_terms = [
        str(term).lower().strip() for term in proposal.get("trigger_terms", [])
        if str(term).strip()
    ]
    # Always retain terms from the original question. This keeps the rule
    # discoverable even when the LLM returns trigger words that are too broad.
    trigger_terms = list(dict.fromkeys(
        extracted_trigger_terms + fallback["trigger_terms"]
    ))[:10]

    correction_fields = [measure, date_column, operation, time_granularity, chart, *groups]
    valid = any(value for value in correction_fields)
    return {
        "valid": valid,
        "instruction": instruction,
        "trigger_terms": trigger_terms,
        "measure": measure,
        "group_by": groups,
        "date_column": date_column,
        "operation": operation,
        "time_granularity": time_granularity,
        "chart": chart,
        "auto_save": bool(fallback.get("auto_save") and chart),
        "original_question": original_question,
        "feedback_text": feedback_text,
    }


def save_feedback_rule(df, proposal, db_path=DEFAULT_DB_PATH) -> int:
    rule = {key: proposal.get(key) for key in (
        "instruction", "trigger_terms", "measure", "group_by", "date_column",
        "operation", "time_granularity",
        "chart",
    )}
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback_rules (
                schema_hash, schema_columns, original_question, feedback_text,
                rule_json, active, created_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (
                schema_fingerprint(df),
                json.dumps(list(map(str, df.columns))),
                proposal.get("original_question", ""),
                proposal.get("feedback_text", ""),
                json.dumps(rule),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_feedback_rules(df=None, db_path=DEFAULT_DB_PATH) -> list[dict]:
    with _connect(db_path) as connection:
        if df is None:
            rows = connection.execute(
                "SELECT * FROM feedback_rules WHERE active = 1 ORDER BY id DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM feedback_rules WHERE active = 1 AND schema_hash = ? ORDER BY id DESC",
                (schema_fingerprint(df),),
            ).fetchall()

    rules = []
    for row in rows:
        item = dict(row)
        item.update(json.loads(item.pop("rule_json")))
        rules.append(item)
    return rules


def delete_feedback_rule(rule_id: int, db_path=DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute("UPDATE feedback_rules SET active = 0 WHERE id = ?", (rule_id,))
        connection.commit()


def relevant_feedback_rules(df, question: str, db_path=DEFAULT_DB_PATH, limit=5) -> list[dict]:
    question_terms = set(_important_terms(question))
    scored = []
    for rule in list_feedback_rules(df, db_path):
        trigger_terms = set(rule.get("trigger_terms", []))
        overlap = len(question_terms & trigger_terms)
        original = rule.get("original_question", "").strip().lower()
        exact_bonus = 3 if original and original == question.strip().lower() else 0
        score = overlap + exact_bonus
        if score > 0:
            scored.append((score, rule))
    scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
    return [rule for _, rule in scored[:limit]]


def inject_feedback_context(question: str, rules: list[dict]) -> str:
    if not rules:
        return question
    instructions = "\n".join(f"- {rule['instruction']}" for rule in rules)
    return (
        f"{question}\n\n"
        "Approved user corrections for this dataset schema. Follow them exactly:\n"
        f"{instructions}"
    )


def apply_feedback_rules(plan: dict, rules: list[dict]) -> dict:
    corrected = dict(plan)
    applied_ids = []
    for rule in reversed(rules):
        if rule.get("measure"):
            corrected["measure"] = rule["measure"]
        if rule.get("group_by"):
            corrected["group_by"] = list(rule["group_by"])
        if rule.get("date_column"):
            corrected["date_column"] = rule["date_column"]
        if rule.get("operation"):
            corrected["operation"] = rule["operation"]
        if rule.get("time_granularity"):
            corrected["time_granularity"] = rule["time_granularity"]
        if rule.get("chart"):
            corrected["chart"] = rule["chart"]
        applied_ids.append(rule["id"])
    if applied_ids:
        corrected["learned_rule_ids"] = applied_ids
    return corrected


__all__ = [
    "apply_feedback_rules", "delete_feedback_rule", "inject_feedback_context",
    "is_feedback_message", "list_feedback_rules", "propose_feedback_rule",
    "relevant_feedback_rules", "save_feedback_rule", "schema_fingerprint",
]
