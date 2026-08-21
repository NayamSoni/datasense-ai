import re

print("### INTENT_AGENT.PY LOADED — VERSION 2026-07-08-v4 (added missing ANALYSIS intent) ###")


def _matches_any(text, keywords):
    """
    Whole-word / whole-phrase matching instead of naive substring
    containment. Plain `keyword in text` false-positives constantly:
    "decline" contains "line", "outline" contains "line", "airline"
    contains "line", "laptop" contains "top". Regex word boundaries
    fix this while still matching multi-word phrases like "top " or
    "business metrics" exactly as before (a boundary exists at the
    space too, so trailing/leading spaces in keywords are harmless).

    An optional trailing "s" is allowed so plural forms ("profits",
    "revenues", "customers") still match their singular keyword
    ("profit", "revenue", "customer").
    """

    for keyword in keywords:

        pattern = r"\b" + re.escape(keyword.strip()) + r"s?\b"

        if re.search(pattern, text):
            return True

    return False


def detect_intent(question, knowledge_available=False):

    q = question.lower().strip()

    # -------------------------
    # Business Knowledge / RAG
    # -------------------------

    knowledge = [
        "knowledge base",
        "according to the document",
        "according to our document",
        "according to the policy",
        "according to our policy",
        "what is the definition",
        "define",
        "definition of",
        "business rule",
        "policy says",
        "documentation says",
        "how is this calculated",
        "how do we calculate",
        "formula for",
    ]

    means_question = bool(re.search(r"\bwhat does\b.+\bmean\b", q))

    if means_question or _matches_any(q, knowledge):
        return "KNOWLEDGE"

    # -------------------------
    # Explain Dataset
    # -------------------------

    explain = [
        "explain the dataset",
        "explain dataset",
        "explain this dataset",
        "describe the dataset",
        "describe dataset",
        "describe this dataset",
        "about this dataset",
        "what is this dataset",
        "what kind of dataset"
    ]

    if _matches_any(q, explain):
        return "EXPLAIN"

    # When a knowledge index exists, a short "What is X?" question normally
    # asks for business meaning. Keep explicit aggregations in the deterministic
    # calculation path (for example, "What is total revenue by city?").
    definition_style = bool(re.match(r"^what (?:is|are)\b", q))
    calculation_cues = [
        "total", "sum", "average", "avg", "mean", "median", "minimum",
        "maximum", "min", "max", "count", "top", "bottom", "by", "vs",
        "compare", "growth", "trend", "distribution", "correlation",
        "outlier", "pareto", "monthly", "weekly", "daily", "yearly",
    ]
    if (
        knowledge_available
        and definition_style
        and not _matches_any(q, calculation_cues)
    ):
        return "KNOWLEDGE"

    # -------------------------
    # Business Analyses
    # -------------------------
    # FIX: this bucket did not exist before, even though app.py has
    # a whole "elif intent == 'ANALYSIS':" branch waiting for it.
    # Since detect_intent() could never actually return "ANALYSIS",
    # every "Suggest business analyses" style question fell through
    # every list below and hit the default CALCULATE fallback at
    # the bottom of this function — which is exactly why clicking
    # "Suggest Analyses" tried to run a data calculation on the
    # literal question text and crashed with a pivot/time-column
    # error instead of returning a text answer.

    analysis = [
        "suggest analyses",
        "suggest analysis",
        "suggest business analyses",
        "suggest business analysis",
        "business analyses",
        "business analysis",
        "recommend analyses",
        "recommend analysis",
        "analysis ideas",
        "what analyses",
        "what analysis",
        "suggest some analyses"
    ]

    if _matches_any(q, analysis):
        return "ANALYSIS"

    # -------------------------
    # KPI
    # -------------------------

    kpi = [
        "kpi",
        "metric",
        "metrics",
        "measure",
        "business metrics"
    ]

    if _matches_any(q, kpi):
        return "KPI"

    # -------------------------
    # Statistical Summary
    # -------------------------

    summary = [
        "summary",
        "statistics",
        "statistical",
        "describe data",
        "profile"
    ]

    if _matches_any(q, summary):
        return "SUMMARY"

    # -------------------------
    # Visualization
    # -------------------------

    charts = [
        "chart",
        "graph",
        "plot",
        "visual",
        "dashboard",
        "pie",
        "bar",
        "line",
        "scatter",
        "histogram",
        "heatmap"
    ]

    if _matches_any(q, charts):
        return "VISUALIZE"

    # -------------------------
    # Calculation
    # -------------------------

    calculate_keywords = [

        "sum",
        "total",
        "average",
        "avg",
        "mean",
        "median",
        "minimum",
        "maximum",
        "min",
        "max",
        "count",

        "top",
        "bottom",

        "sales",
        "revenue",
        "profit",
        "loss",
        "quantity",
        "discount",
        "adr",
        "occupancy",

        "customer",
        "customers",
        "product",
        "products",

        "region",
        "state",
        "city",
        "country",

        "hotel",
        "property",

        "monthly",
        "weekly",
        "daily",
        "yearly",
        "quarterly",

        "month over month",
        "mom",
        "year over year",
        "yoy",
        "quarter over quarter",
        "qoq",
        "week over week",
        "wow",

        "trend",

        "compare",

        "growth",

        "distribution",

        "correlation",

        "outlier",

        "pareto",

        "80/20",

        "filter",

        "where",

        "by",

        "vs"
    ]

    if _matches_any(q, calculate_keywords):
        return "CALCULATE"

    # -------------------------
    # Truly conversational (no data intent at all)
    # -------------------------

    conversational = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
        "help"
    ]

    if _matches_any(q, conversational):
        return "CHAT"

    # -------------------------
    # Default fallback
    # -------------------------

    return "CALCULATE"
