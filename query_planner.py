import json
import re

import pandas as pd

print("### QUERY_PLANNER.PY LOADED — VERSION 2026-07-21-v6 (deterministic Average Order Value planning) ###")

from prompts import CALCULATION_PROMPT

from utils import (
    sanitize_plan,
    resolve_measure,
    resolve_dimension,
    resolve_business_column,
    detect_primary_measure,
    detect_dimension,
    detect_date_column,
    date_columns,
)

# ==========================================================
# Build Dataset Schema
# ==========================================================

def build_schema(df):

    schema = []

    for col in df.columns:

        schema.append({

            "column": col,

            "dtype": str(df[col].dtype)

        })

    return schema


# ==========================================================
# Prompt
# ==========================================================

def build_prompt(question, df):

    schema = build_schema(df)

    columns = []

    for item in schema:

        columns.append(

            f"{item['column']} ({item['dtype']})"

        )

    return f"""
You are a Senior Data Analyst.

Dataset Columns

{chr(10).join(columns)}

User Question

{question}

Return ONLY valid JSON.

Example:

{{
    "title":"Sales by Product",
    "analysis_type":"aggregation",
    "operation":"sum",
    "measure":"Sales",
    "group_by":["Product"],
    "filters":[],
    "sort":"desc",
    "limit":10,
    "time_granularity":null,
    "pivot":false,
    "chart":"bar"
}}

Rules

1. Never invent columns.

2. Always use dataset columns.

3. If user asks Top N
analysis_type = top_bottom

4. If user asks Bottom N
analysis_type = top_bottom

5. If user asks Month over Month
analysis_type = mom

6. If user asks Year over Year
analysis_type = yoy

7. If user asks Pareto
analysis_type = pareto

8. If user asks Contribution
analysis_type = pareto

9. If question contains

by product

by category

by region

by segment

and also month/year/quarter

set

pivot = true

Return ONLY JSON.
"""


# ==========================================================
# JSON Extraction
# ==========================================================

def extract_json(response):
    """
    Robustly pull JSON plan(s) out of a raw LLM response.

    Handles, in order of preference:
      1. A clean JSON array or object.
      2. The same wrapped in markdown fences.
      3. The failure mode where the model ignores instructions and
         returns multiple top-level {...} objects back to back
         without wrapping them in an array (e.g. "{...}\\n{...}").
         A naive greedy regex would splice these into one broken
         blob; this scans bracket-by-bracket instead, so each
         object is parsed independently and bad ones are dropped
         rather than corrupting the whole batch.
    """

    response = response.strip()

    # Strip markdown fences if present
    response = re.sub(r"^```(json)?", "", response.strip(), flags=re.IGNORECASE)
    response = re.sub(r"```$", "", response.strip())
    response = response.strip()

    # 1. Direct parse (covers both a clean array and a single object)
    try:
        parsed = json.loads(response)
        return parsed
    except Exception:
        pass

    # 2. Look for an explicit array anywhere in the text
    match = re.search(r"\[.*\]", response, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # 3. Bracket-balanced scan: extract every top-level {...} block
    #    independently, so multiple bare objects don't get merged.
    objects = []
    depth = 0
    start = None

    for i, ch in enumerate(response):

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1

        elif ch == "}":
            if depth > 0:
                depth -= 1

                if depth == 0 and start is not None:

                    chunk = response[start:i + 1]

                    try:
                        objects.append(json.loads(chunk))
                    except Exception:
                        pass

                    start = None

    if objects:
        return objects

    return None

# ==========================================================
# Rule Engine
# ==========================================================

def apply_rule_engine(question, plan):

    question_lower = question.lower()
    title = (plan.get("title") or "").strip().lower()

    if not title or title in ("analysis", "untitled", "result"):
        q = question.lower()
    else:
        q = f"{title} {question.lower()}"

    # --------------------------------------------------
    # Analysis Type
    # --------------------------------------------------

    if is_average_order_value_question(question_lower):

        plan["analysis_type"] = "average_order_value"
        plan["operation"] = "ratio"
        plan["chart"] = "kpi"

    elif any(x in q for x in [
        "top ",
        "bottom "
    ]):

        plan["analysis_type"] = "top_bottom"

    elif any(x in q for x in [
        "month over month",
        "mom"
    ]):

        plan["analysis_type"] = "mom"

    elif any(x in q for x in [
        "year over year",
        "yoy"
    ]):

        plan["analysis_type"] = "yoy"

    elif any(x in q for x in [
        "pareto",
        "80/20",
        "abc analysis",
        "contribution"
    ]):

        plan["analysis_type"] = "pareto"

    elif any(x in q for x in [
        "correlation",
        "correlate",
        "relationship between"
    ]):

        plan["analysis_type"] = "correlation"
        # A relationship is most readable as a scatter plot. The numerical
        # coefficient is still returned in the result table below the chart.
        # A saved user preference may override this later.
        plan["chart"] = "scatter"

    elif any(x in q for x in [
        "outlier",
        "anomaly",
        "anomalies",
        "unusual"
    ]):

        plan["analysis_type"] = "outlier"

    elif any(x in q for x in [
        "distribution",
        "histogram",
        "spread of"
    ]):

        plan["analysis_type"] = "distribution"

    elif any(x in q for x in [
        "trend",
        "monthly",
        "quarterly",
        "yearly",
        "weekly",
        "daily"
    ]):

        plan["analysis_type"] = "time_series"

    else:

        plan.setdefault("analysis_type", "aggregation")

    # --------------------------------------------------
    # Time Granularity
    # --------------------------------------------------
    # Only the user's own words may activate time analysis. A local LLM can
    # occasionally hallucinate month/pivot fields in its JSON plan even when
    # the request is simply "subcategory-wise units sold".

    if any(x in question_lower for x in [
        "monthly",
        "month",
        "mom"
    ]):

        plan["time_granularity"] = "month"

    elif any(x in question_lower for x in [
        "quarter",
        "quarterly",
        "qoq"
    ]):

        plan["time_granularity"] = "quarter"

    elif any(x in question_lower for x in [
        "year",
        "yearly",
        "annual",
        "yoy"
    ]):

        plan["time_granularity"] = "year"

    elif any(x in question_lower for x in [
        "week",
        "weekly"
    ]):

        plan["time_granularity"] = "week"

    elif any(x in question_lower for x in [
        "day",
        "daily"
    ]):

        plan["time_granularity"] = "day"

    else:

        plan["time_granularity"] = None
        plan["pivot"] = False

        if plan.get("analysis_type") in ("time_series", "mom", "yoy"):
            plan["analysis_type"] = "aggregation"

    # --------------------------------------------------
    # Top / Bottom
    # --------------------------------------------------

    m = re.search(r"top\s+(\d+)", q)

    if m:

        plan["sort"] = "desc"
        plan["limit"] = int(m.group(1))

    m = re.search(r"bottom\s+(\d+)", q)

    if m:

        plan["sort"] = "asc"
        plan["limit"] = int(m.group(1))

    # --------------------------------------------------
    # Pivot Detection
    # --------------------------------------------------

    if plan.get("time_granularity"):

        if any(x in question_lower for x in [

            "by product",

            "by customer",

            "by region",

            "by state",

            "by city",

            "by segment",

            "by category",

            "by sub category",

            "product wise",

            "customer wise",

            "segment wise",

            "category wise",

            "region wise"

        ]):

            plan["pivot"] = True

        else:

            plan["pivot"] = False

    else:

        plan["pivot"] = False

    return plan


# ==========================================================
# LLM Planner
# ==========================================================

def llm_plan(question, df):

    from llm_agent import ask_llm

    prompt = build_prompt(question, df)

    response = ask_llm(

        CALCULATION_PROMPT,

        prompt,

        json_mode=True

    ).strip()

    print("=" * 80)
    print("RAW PLANNER")
    print(response)
    print("=" * 80)

    plans = extract_json(response)

    if plans is None:

        return []

    if isinstance(plans, dict):

        plans = [plans]

    return plans

# ==========================================================
# Normalize Plan
# ==========================================================

def normalize_plan(plan):

    return {

        "title": plan.get("title", "Analysis"),

        "analysis_type": plan.get("analysis_type", "aggregation"),

        "operation": str(
            plan.get("operation", "sum")
        ).lower(),

        "measure": plan.get("measure"),

        "measure2": plan.get("measure2"),

        "order_column": plan.get("order_column"),

        "metric_label": plan.get("metric_label"),

        "group_by": plan.get("group_by", []),

        "filters": plan.get("filters", []),

        "sort": plan.get("sort"),

        "limit": plan.get("limit"),

        "time_granularity": plan.get("time_granularity"),

        "pivot": plan.get("pivot", False),

        "chart": plan.get("chart", "table")

    }


# ==========================================================
# Shared dimension keyword list
# ==========================================================

DIMENSION_KEYWORDS = [
    "customer",
    "product",
    "sub category",
    "sub-category",
    "subcategory",
    "category",
    "segment",
    "region",
    "state",
    "city",
    "country",
    "ship mode",
    "shipping",
    "brand",
    "channel"
]


def is_average_order_value_question(question):
    """Return True when the user is asking for revenue per distinct order."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(question).lower()).strip()

    if re.search(r"\baov\b", normalized):
        return True

    explicit_phrases = (
        "average order value",
        "average sales value per order",
        "average sale value per order",
        "average revenue value per order",
        "average sales per order",
        "average revenue per order",
        "sales per order",
        "revenue per order",
    )

    return any(phrase in normalized for phrase in explicit_phrases)


def _detect_order_identifier(df):
    """Prefer a true order identifier and never mistake Order Date for it."""
    normalized_columns = {
        re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip(): column
        for column in df.columns
    }

    exact_names = (
        "order id",
        "order number",
        "order no",
        "sales order id",
        "invoice id",
        "invoice number",
        "transaction id",
        "booking id",
    )

    for name in exact_names:
        if name in normalized_columns:
            return normalized_columns[name]

    for normalized_name, column in normalized_columns.items():
        has_order_word = "order" in normalized_name.split()
        has_id_word = any(
            token in normalized_name.split()
            for token in ("id", "number", "no")
        )
        if has_order_word and has_id_word and "date" not in normalized_name.split():
            return column

    fallback = resolve_business_column(df, "order")
    if fallback is not None and "date" not in str(fallback).lower():
        return fallback

    return None


def _keyword_variants(keyword):
    """
    FIX: plain substring matching missed simple plurals whose
    spelling changes when pluralized -- "country" is NOT a
    substring of "countries", "category" is NOT a substring of
    "categories", "city" is NOT a substring of "cities" (the
    y -> ies swap breaks a naive `keyword in text` check). Without
    this, a perfectly valid "top 10 countries" or "by category"
    style question could have its own correctly-named dimension
    incorrectly stripped out by the validation step below, simply
    because the plural form didn't literally contain the singular
    keyword as a substring.
    """

    variants = {keyword}

    if keyword.endswith("y") and len(keyword) > 1:
        variants.add(keyword[:-1] + "ies")

    variants.add(keyword + "s")

    return variants


def _dimension_mentioned_in_question(col, q_check):
    """
    Checks whether a resolved group_by column is actually
    something the user's question named — either via one of the
    known dimension keywords contained in the column name (e.g.
    "Product Category" contains both "product" and "category"; ANY
    of them appearing in the question is enough), or, for columns
    with no recognized keyword, the column name itself.

    FIX (root cause of "top 10 customers" resolving to Country):
    this function is now called for EVERY analysis_type, including
    top_bottom/pareto. Previously those two types skipped this
    check entirely on the assumption that "some entity is always
    needed for a ranking, so whatever the LLM provided must be
    fine" — but a small local model can hallucinate a plausible-
    looking but WRONG entity (e.g. defaulting to "Country" for a
    "top 10 customers?" question, likely pattern-matching on
    common "top 10 countries" business examples rather than the
    actual question). Validating unconditionally catches that; see
    resolve_plan() below for how an empty result after validation
    is then handled differently depending on analysis_type.
    """

    col_lower = col.lower()

    matched_keywords = [k for k in DIMENSION_KEYWORDS if k in col_lower]

    if matched_keywords:

        for k in matched_keywords:
            for variant in _keyword_variants(k):
                if variant in q_check:
                    return True

        return False

    for variant in _keyword_variants(col_lower):
        if variant in q_check:
            return True

    return False


def _numeric_columns_mentioned(df, question):
    """Return distinct numeric columns in the order named by the user."""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    matches = []

    for column in df.select_dtypes(include="number").columns:
        column_text = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        if not column_text:
            continue

        # Accept common plurals: Profit/profits and Discount/discounts.
        pattern = rf"\b{re.escape(column_text)}(?:s|es)?\b"
        match = re.search(pattern, normalized)
        if match:
            matches.append((match.start(), column))

    matches.sort(key=lambda item: item[0])
    return list(dict.fromkeys(column for _, column in matches))


# ==========================================================
# Semantic Resolution
# ==========================================================

def resolve_plan(df, plan, question=""):

    plan = normalize_plan(plan)

    # -------------------------------
    # Measure
    # -------------------------------

    if plan["measure"]:

        plan["measure"] = resolve_measure(

            df,

            plan["measure"]

        )

    else:

        plan["measure"] = detect_primary_measure(df)

    # Deterministic business-language overrides. These are applied from the
    # user's question, so "units sold" reliably maps to Quantity even if the
    # small local model selected Sales or left the measure blank.
    question_lower = question.lower()
    explicit_measures = (
        (("units sold", "unit sold", "units", "quantity", "qty", "volume"), "quantity"),
        (("profit", "margin", "earnings"), "profit"),
        (("sales", "revenue", "turnover"), "sales"),
        (("discount", "rebate"), "discount"),
        (("cost", "expense"), "cost"),
    )

    for phrases, semantic_measure in explicit_measures:
        if any(phrase in question_lower for phrase in phrases):
            resolved_measure = resolve_measure(df, semantic_measure)
            if resolved_measure in df.columns and pd.api.types.is_numeric_dtype(df[resolved_measure]):
                plan["measure"] = resolved_measure
            break

    # Average Order Value is a ratio, not mean(Sales) and not Sales / Quantity.
    # Resolve it deterministically so the local LLM cannot route it to a total,
    # ranking, contribution, or row-average calculation.
    if is_average_order_value_question(question_lower):
        sales_column = resolve_measure(df, "sales")
        order_column = _detect_order_identifier(df)

        plan.update({
            "title": "Average Order Value",
            "analysis_type": "average_order_value",
            "operation": "ratio",
            "measure": sales_column,
            "order_column": order_column,
            "metric_label": "Average Order Value",
            "chart": "kpi",
            "sort": None,
            "limit": None,
            "pivot": False,
        })

        if sales_column is None or order_column is None:
            missing = []
            if sales_column is None:
                missing.append("a numeric Sales/Revenue column")
            if order_column is None:
                missing.append("an Order ID column")
            plan["no_data_message"] = (
                "Average Order Value cannot be calculated because this dataset "
                f"does not contain {', '.join(missing)}."
            )

    # -------------------------------
    # Measure 2 (correlation only)
    # -------------------------------

    if plan.get("measure2"):

        plan["measure2"] = resolve_measure(

            df,

            plan["measure2"]

        )

    # Correlation must use two different Series. Small local models sometimes
    # return the same field for measure and measure2 even when the user clearly
    # named two columns. Resolve the pair deterministically from the question.
    if any(term in question_lower for term in (
        "correlation", "correlate", "relationship between"
    )):
        correlation_columns = _numeric_columns_mentioned(df, question)
        if len(correlation_columns) >= 2:
            plan["measure"] = correlation_columns[0]
            plan["measure2"] = correlation_columns[1]
        elif plan.get("measure2") == plan.get("measure"):
            plan["measure2"] = None

    # -------------------------------
    # Group By
    # -------------------------------

    date_cols = set(date_columns(df))

    groups = []

    for col in plan["group_by"]:

        resolved = resolve_dimension(

            df,

            col

        )

        if not resolved:
            continue

        if resolved in date_cols:
            continue

        groups.append(resolved)

    plan["group_by"] = groups

    # -------------------------------
    # Validate grouping against the actual question
    # -------------------------------
    # FIX: this now runs for EVERY analysis_type (see the
    # docstring on _dimension_mentioned_in_question above for why
    # skipping it for top_bottom/pareto was the actual root cause
    # of wrong-entity results like "top 10 customers" -> Country).
    # If everything gets stripped, the Smart Defaults section
    # further down still supplies a sensible entity for
    # top_bottom/pareto — it just won't be a hallucinated one.

    needs_fallback_dimension = plan["analysis_type"] in (
        "top_bottom",
        "pareto"
    )

    if plan["group_by"]:

        q_check = question.lower()

        validated_groups = [
            col for col in plan["group_by"]
            if _dimension_mentioned_in_question(col, q_check)
        ]

        if validated_groups != plan["group_by"]:

            print("=" * 80)
            print("GROUP_BY VALIDATION: stripped dimension(s) not mentioned in question")
            print("Question   :", question)
            print("Before     :", plan["group_by"])
            print("After      :", validated_groups)
            print("=" * 80)

            # Only relabel as "(Overall)" when we're NOT about to
            # hand this to the Smart Defaults fallback below — for
            # top_bottom/pareto, a real dimension is still coming,
            # so calling it "(Overall)" here would be misleading.
            if not validated_groups and not needs_fallback_dimension:
                plan["title"] = f"{question.strip().capitalize()} (Overall)"

        plan["group_by"] = validated_groups

    # -------------------------------
    # Filters
    # -------------------------------

    valid_filters = []

    for f in plan["filters"]:

        column = resolve_business_column(

            df,

            f.get("column", "")

        )

        if column:

            valid_filters.append({

                "column": column,

                "value": f.get("value")

            })

    plan["filters"] = valid_filters

    # -------------------------------
    # Smart Defaults
    # -------------------------------

    q = f"{question} {plan['title']}".lower()

    wants_overall = any(x in q for x in [
        "overall",
        "total company",
        "company wide",
        "company-wide",
        "entire business",
        "across all",
        "as a whole"
    ])

    if not plan["group_by"]:

        dim = None

        if "customer" in q:

            dim = resolve_dimension(df, "customer")

        elif "product" in q:

            dim = resolve_dimension(df, "product")

        elif "sub category" in q or "subcategory" in q or "sub-category" in q:

            dim = resolve_dimension(df, "subcategory")

        elif "category" in q:

            dim = resolve_dimension(df, "category")

        elif "segment" in q:

            dim = resolve_dimension(df, "segment")

        elif "region" in q:

            dim = resolve_dimension(df, "region")

        elif "state" in q:

            dim = resolve_dimension(df, "state")

        elif "city" in q or "cities" in q:

            dim = resolve_dimension(df, "city")

        elif "country" in q or "countries" in q:

            dim = resolve_dimension(df, "country")

        elif "brand" in q:

            dim = resolve_dimension(df, "brand")

        elif "channel" in q:

            dim = resolve_dimension(df, "channel")

        elif needs_fallback_dimension and not wants_overall:

            dim = detect_dimension(df)

        if dim:

            plan["group_by"] = [dim]

    # -------------------------------
    # Explicit Year Detection / Validation
    # -------------------------------
    # FIX (root cause of "Compare Sales 2023 vs 2024" silently
    # showing 2014-2017 data): the planner previously had no
    # concept of a specific YEAR VALUE — only "time_granularity",
    # which controls HOW to bucket time (month/quarter/year), not
    # WHICH years to include. A question naming years that simply
    # don't exist in the dataset was never checked against what
    # actually exists, so pivot/time-series just showed whatever
    # years happened to be present. Now: read any 4-digit year
    # mentioned in the question, compare against the dataset's
    # real date range, and either flag a clear "no data" message
    # (none of the requested years exist) or restrict the plan to
    # only the years that do exist, noting any that don't.

    requested_years = sorted(set(
        int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)
    ))

    if requested_years:

        date_col = detect_date_column(df)

        if date_col is not None:

            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

            available_years = set(
                parsed_dates.dt.year.dropna().astype(int).tolist()
            )

            present_years = [y for y in requested_years if y in available_years]
            missing_years = [y for y in requested_years if y not in available_years]

            if not present_years:

                years_str = ", ".join(str(y) for y in requested_years)

                if available_years:
                    avail_str = f"{min(available_years)}\u2013{max(available_years)}"
                else:
                    avail_str = "no valid dates"

                plan["no_data_message"] = (
                    f"No data found for {years_str}. "
                    f"This dataset only contains records from {avail_str}."
                )

            else:

                plan["year_filter"] = present_years

                if missing_years:

                    missing_str = ", ".join(str(y) for y in sorted(missing_years))
                    present_str = ", ".join(str(y) for y in sorted(present_years))

                    plan["title"] = (
                        f"{plan.get('title', 'Analysis')} "
                        f"(Note: no data for {missing_str} \u2014 showing {present_str} only)"
                    )

    return sanitize_plan(

        df,

        plan

    )

# ==========================================================
# Create Execution Plan
# ==========================================================

def create_execution_plan(user_question, df):

    raw_plans = llm_plan(

        user_question,

        df

    )

    if not raw_plans:

        return [

            {

                "title": user_question,

                "analysis_type": "aggregation",

                "operation": "sum",

                "measure": detect_primary_measure(df),

                "group_by": [],

                "filters": [],

                "sort": None,

                "limit": None,

                "time_granularity": None,

                "pivot": False,

                "chart": "table"

            }

        ]

    execution_plans = []

    for raw in raw_plans:

        try:

            plan = resolve_plan(

                df,

                raw,

                question=user_question

            )

            plan = apply_rule_engine(

                user_question,

                plan

            )

            print("=" * 80)
            print("FINAL RESOLVED PLAN (after validation)")
            print("Title      :", plan.get("title"))
            print("Measure    :", plan.get("measure"))
            print("Group By   :", plan.get("group_by"))
            print("Analysis   :", plan.get("analysis_type"))
            print("Year Filter:", plan.get("year_filter"))
            print("=" * 80)

            execution_plans.append(

                plan

            )

        except Exception as e:

            print("=" * 80)

            print("PLANNER ERROR")

            print(e)

            print("=" * 80)

    if len(execution_plans) == 0:

        execution_plans.append(

            {

                "title": user_question,

                "analysis_type": "aggregation",

                "operation": "sum",

                "measure": detect_primary_measure(df),

                "group_by": [],

                "filters": [],

                "sort": None,

                "limit": None,

                "time_granularity": None,

                "pivot": False,

                "chart": "table"

            }

        )

    return execution_plans


# ==========================================================
# Debug Helper
# ==========================================================

def preview_execution_plan(question, df):

    plans = create_execution_plan(

        question,

        df

    )

    print("=" * 80)

    print("FINAL EXECUTION PLAN")

    print(

        json.dumps(

            plans,

            indent=4

        )

    )

    print("=" * 80)

    return plans
