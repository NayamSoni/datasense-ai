"""Small, transparent analysis memory for conversational follow-ups."""

from __future__ import annotations

import re


FOLLOW_UP_MARKERS = (
    "only ", "now ", "same ", "instead", "also ", "filter ", "exclude ",
    "include ", "change it", "make it", "what about", "previous",
)


def is_follow_up(question: str, previous_plan: dict | None) -> bool:
    if not previous_plan:
        return False
    text = question.strip().lower()
    if any(marker in text for marker in FOLLOW_UP_MARKERS):
        return True
    if re.search(r"\bonly\b", text):
        return True

    # A bare year fragment is normally a refinement; a complete request such
    # as "Show sales for 2023" is a new analysis and must not inherit filters.
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", text))
    new_request_terms = (
        "show", "plot", "chart", "compare", "top", "bottom", "average",
        "total", "sales", "revenue", "profit", "correlation", "outlier", "pareto",
    )
    return has_year and len(text.split()) <= 4 and not any(
        term in text for term in new_request_terms
    )


def contextualize_question(question: str, previous_plan: dict | None) -> tuple[str, bool]:
    """Give the planner explicit prior-analysis context without hiding it in the LLM."""
    if not is_follow_up(question, previous_plan):
        return question, False

    plan = previous_plan or {}
    groups = ", ".join(map(str, plan.get("group_by", []))) or "none"
    filters = ", ".join(
        f"{item.get('column')}={item.get('value')}"
        for item in plan.get("filters", [])
    ) or "none"

    context = (
        "Continue the previous analysis. Preserve its measure, grouping, operation, "
        "analysis type, and chart unless the follow-up explicitly changes them.\n"
        f"Previous measure: {plan.get('measure')}\n"
        f"Previous grouping: {groups}\n"
        f"Previous operation: {plan.get('operation')}\n"
        f"Previous analysis type: {plan.get('analysis_type')}\n"
        f"Previous chart: {plan.get('chart')}\n"
        f"Previous filters: {filters}\n"
        f"Follow-up instruction: {question}"
    )
    return context, True


def merge_follow_up_plan(new_plan: dict, previous_plan: dict | None, question: str) -> dict:
    """Preserve prior choices that a short follow-up did not explicitly replace."""
    if not is_follow_up(question, previous_plan):
        return new_plan

    previous_plan = previous_plan or {}
    merged = dict(new_plan)
    for key in ("measure", "operation", "analysis_type", "chart"):
        if not merged.get(key):
            merged[key] = previous_plan.get(key)

    if not merged.get("group_by"):
        merged["group_by"] = list(previous_plan.get("group_by", []))

    old_filters = {
        item.get("column"): item for item in previous_plan.get("filters", [])
        if item.get("column")
    }
    for item in merged.get("filters", []):
        if item.get("column"):
            old_filters[item["column"]] = item
    merged["filters"] = list(old_filters.values())
    return merged


__all__ = ["contextualize_question", "is_follow_up", "merge_follow_up_plan"]
