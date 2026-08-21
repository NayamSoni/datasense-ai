import re
import difflib
import warnings
from typing import List, Optional

import pandas as pd

print("### UTILS.PY LOADED — VERSION 2026-07-11-v3 (added debug prints to resolve_business_column/resolve_dimension to find why Customer still resolves to Country) ###")


# ============================================================
# Universal Business Synonyms
# ============================================================

BUSINESS_SYNONYMS = {

    "customer": [
        "customer",
        "customer name",
        "client",
        "client name",
        "buyer",
        "consumer",
        "account",
        "party",
        "company",
        "organisation",
        "organization"
    ],

    "product": [
        "product",
        "product name",
        "item",
        "item name",
        "sku",
        "material",
        "article",
        "model"
    ],

    "category": [
        "category",
        "segment",
        "department"
    ],

    "subcategory": [
        "subcategory",
        "sub category",
        "sub-category",
        "family"
    ],

    "sales": [
        "sales",
        "sale",
        "revenue",
        "amount",
        "order amount",
        "invoice amount",
        "net sales",
        "gross sales",
        "turnover"
    ],

    "profit": [
        "profit",
        "margin",
        "net profit",
        "gross profit",
        "earning",
        "earnings"
    ],

    "cost": [
        "cost",
        "expense",
        "price",
        "buying price",
        "purchase cost"
    ],

    "quantity": [
        "qty",
        "quantity",
        "units",
        "volume"
    ],

    "discount": [
        "discount",
        "discount percent",
        "discount %",
        "rebate"
    ],

    "state": [
        "state",
        "province"
    ],

    "city": [
        "city",
        "town"
    ],

    "country": [
        "country",
        "nation"
    ],

    "region": [
        "region",
        "territory",
        "zone",
        "market"
    ],

    "date": [
        "date",
        "order date",
        "invoice date",
        "created date",
        "booking date",
        "purchase date",
        "transaction date"
    ],
    "order": [
    "order",
    "order id",
    "sales order",
    "invoice",
    "invoice id"
],

"customer_segment": [
    "segment",
    "customer segment"
],

"ship_mode": [
    "ship mode",
    "shipping mode",
    "delivery mode"
],

"brand": [
    "brand"
],

"channel": [
    "channel",
    "sales channel"
],

"agent": [
    "agent",
    "advisor",
    "executive",
    "employee",
    "owner"
],

"lead": [
    "lead",
    "lead id"
],

"hospital": [
    "hospital",
    "hospital name",
    "medical center",
    "medical centre",
    "health facility"
],

"facility": [
    "facility",
    "facility name",
    "hospital",
    "hospital name"
],

"provider": [
    "provider",
    "provider name",
    "hospital",
    "hospital name"
]
}


# ============================================================
# Helpers
# ============================================================

def normalize(text: str) -> str:

    text = str(text).lower()

    text = text.replace("_", " ")

    text = text.replace("-", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def similarity(a: str, b: str) -> float:

    return difflib.SequenceMatcher(

        None,

        normalize(a),

        normalize(b)

    ).ratio()


def all_columns(df):

    return list(df.columns)


def numeric_columns(df):

    return list(

        df.select_dtypes(include="number").columns

    )


def categorical_columns(df):

    return list(

        df.select_dtypes(

            exclude="number"

        ).columns

    )

# ============================================================
# Date Detection
# ============================================================

def date_columns(df):

    cols = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols.append(col)
            continue

        # Numeric columns (IDs, measures, counts) are never genuine
        # date columns. pd.to_datetime does NOT raise on a plain
        # number -- it silently treats small integers/floats as
        # nanoseconds-since-epoch, which lands near 1970 and then
        # falsely "passes" the validity check below. Row IDs, Sales,
        # Quantity, Discount, Profit, etc. would otherwise all get
        # misdetected as date columns.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        try:

            sample = df[col].dropna().head(20)

            if len(sample) == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce")

            valid = parsed.dropna()

            if len(valid) < max(3, int(len(sample) * 0.6)):
                continue

            # Extra sanity check: real dates should fall in a
            # plausible calendar range. This guards against any
            # other silent-misparse edge case beyond pure numerics.
            if valid.dt.year.between(1900, 2100).mean() < 0.9:
                continue

            cols.append(col)

        except Exception:
            pass

    return cols


# ============================================================
# Best Semantic Match
# ============================================================

def semantic_match(target, columns):

    target = normalize(target)

    best_column = None
    best_score = 0

    for col in columns:

        score = similarity(target, col)

        if score > best_score:
            best_score = score
            best_column = col

    return best_column, best_score


# ============================================================
# Business Entity Resolver
# ============================================================

def resolve_business_column(df, entity):

    original_entity = entity

    entity = normalize(entity)

    columns = all_columns(df)

    # 1. Exact column name match

    for col in columns:

        if normalize(col) == entity:
            print(f"### resolve_business_column('{original_entity}') -> EXACT MATCH -> '{col}' ###")
            return col

    synonyms = BUSINESS_SYNONYMS.get(entity, [entity])

    substring_candidates = []

    for word in synonyms:

        word_norm = normalize(word)

        for col in columns:

            if word_norm in normalize(col):
                substring_candidates.append(col)

    if substring_candidates:

        name_cols = [
            c for c in substring_candidates
            if "name" in normalize(c).split()
        ]

        if name_cols:
            print(f"### resolve_business_column('{original_entity}') -> SUBSTRING (name preferred) -> '{name_cols[0]}' | all candidates: {substring_candidates} ###")
            return name_cols[0]

        print(f"### resolve_business_column('{original_entity}') -> SUBSTRING -> '{substring_candidates[0]}' | all candidates: {substring_candidates} ###")
        return substring_candidates[0]

    # 3. Fuzzy fallback (unbiased)

    best = None
    best_score = 0

    for word in synonyms:

        col, score = semantic_match(word, columns)

        if score > best_score:

            best = col
            best_score = score

    if best_score >= 0.60:
        print(f"### resolve_business_column('{original_entity}') -> FUZZY SYNONYM -> '{best}' (score={best_score:.3f}) ###")
        return best

    col, score = semantic_match(entity, columns)

    if score >= 0.65:
        print(f"### resolve_business_column('{original_entity}') -> FUZZY DIRECT -> '{col}' (score={score:.3f}) ###")
        return col

    print(f"### resolve_business_column('{original_entity}') -> NO MATCH (best_score={best_score:.3f}) ###")
    return None


# ============================================================
# Measure Resolver
# ============================================================

def resolve_measure(df, measure):

    col = resolve_business_column(df, measure)

    if col is not None:
        return col

    nums = numeric_columns(df)

    if len(nums) == 1:
        return nums[0]

    return None


# ============================================================
# Dimension Resolver
# ============================================================

def resolve_dimension(df, dimension):

    col = resolve_business_column(df, dimension)

    if col is not None:
        return col

    cats = categorical_columns(df)

    if len(cats) == 1:
        print(f"### resolve_dimension('{dimension}') -> FALLBACK single categorical column -> '{cats[0]}' ###")
        return cats[0]

    print(f"### resolve_dimension('{dimension}') -> None (categorical columns: {cats}) ###")
    return None

# ============================================================
# Automatic Measure Detection
# ============================================================

def detect_primary_measure(df):

    priority = [
        "sales",
        "revenue",
        "profit",
        "amount",
        "quantity",
        "cost"
    ]

    for item in priority:

        col = resolve_measure(df, item)

        if col is not None:
            return col

    nums = numeric_columns(df)

    if len(nums):
        return nums[0]

    return None


# ============================================================
# Automatic Date Detection
# ============================================================

def detect_date_column(df):

    dates = date_columns(df)

    if len(dates):
        return dates[0]

    return resolve_business_column(df, "date")


# ============================================================
# Detect Best Dimension
# ============================================================

def detect_dimension(df):

    priority = [
        "customer",
        "product",
        "category",
        "subcategory",
        "region",
        "state",
        "city",
        "country",
        "hospital",
        "facility",
        "provider"
    ]

    for item in priority:

        col = resolve_dimension(df, item)

        if col is not None:
            return col

    cats = categorical_columns(df)

    if len(cats):
        return cats[0]

    return None


# ============================================================
# Validate Execution Plan
# ============================================================

def validate_plan(df, plan):

    plan = plan.copy()

    # Measure

    if plan.get("measure"):

        plan["measure"] = resolve_measure(
            df,
            plan["measure"]
        )

    else:

        plan["measure"] = detect_primary_measure(df)

    # Group By

    groups = []

    for col in plan.get("group_by", []):

        resolved = resolve_dimension(df, col)

        if resolved:

            groups.append(resolved)

    plan["group_by"] = groups

    # Filters

    validated_filters = []

    for f in plan.get("filters", []):

        column = resolve_business_column(
            df,
            f.get("column", "")
        )

        if column:

            validated_filters.append({
                "column": column,
                "value": f.get("value")
            })

    plan["filters"] = validated_filters

    return plan

# ============================================================
# Time Granularity Resolver
# ============================================================

def resolve_time_column(df):

    return detect_date_column(df)


# ============================================================
# Dataset Profile
# ============================================================

def dataset_profile(df):

    return {

        "rows": len(df),

        "columns": len(df.columns),

        "numeric_columns": numeric_columns(df),

        "categorical_columns": categorical_columns(df),

        "date_columns": date_columns(df),

        "primary_measure": detect_primary_measure(df),

        "primary_dimension": detect_dimension(df)

    }


# ============================================================
# KPI Suggestions
# ============================================================

def suggest_kpis(df):

    measure = detect_primary_measure(df)

    dimension = detect_dimension(df)

    date = detect_date_column(df)

    suggestions = []

    if measure:

        suggestions.extend([
            f"Total {measure}",
            f"Average {measure}",
            f"Top 10 by {measure}",
            f"Bottom 10 by {measure}"
        ])

    if measure and dimension:

        suggestions.extend([
            f"{measure} by {dimension}",
            f"Top 10 {dimension}",
            f"Bottom 10 {dimension}"
        ])

    if measure and date:

        suggestions.extend([
            f"Monthly {measure}",
            f"Quarterly {measure}",
            f"Yearly {measure}",
            f"Month over Month {measure}",
            f"Year over Year {measure}"
        ])

    return suggestions


# ============================================================
# Execution Plan Sanitizer
# ============================================================

def sanitize_plan(df, plan):

    plan = validate_plan(df, plan)

    if not plan.get("measure"):
        plan["measure"] = detect_primary_measure(df)

    if not plan.get("group_by"):

        # PREVIOUSLY: this unconditionally called detect_dimension(df)
        # and added it to group_by any time group_by was empty — no
        # matter what analysis_type was, no matter whether the
        # question asked for a breakdown at all. This ran as the
        # LAST step of resolve_plan() in query_planner.py (via
        # `return sanitize_plan(df, plan)`), which meant it silently
        # undid every upstream fix: query_planner's own "smart
        # defaults" logic and its group_by validation step could
        # correctly empty group_by, and this function would put a
        # dimension (typically "Customer ID", first in
        # detect_dimension's priority list) right back in immediately
        # after — with zero visibility, since none of that showed up
        # in query_planner.py's own debug prints. This is the actual
        # root cause of the recurring "asked for an overall
        # trend/total, got a per-customer breakdown" bug.
        #
        # FIX: only apply this fallback for analysis types that
        # genuinely need SOME entity to rank/split against even when
        # none was named (top_bottom, pareto) — matching the same
        # rule query_planner.py already applies upstream. Also skip
        # it if the plan's title indicates the user explicitly asked
        # for an overall/aggregate figure.

        analysis_type = plan.get("analysis_type", "aggregation")

        title_text = str(plan.get("title", "")).lower()

        wants_overall = any(x in title_text for x in [
            "overall",
            "total company",
            "company wide",
            "company-wide",
            "entire business",
            "across all",
            "as a whole"
        ])

        needs_fallback_dimension = analysis_type in (
            "top_bottom",
            "pareto"
        )

        if needs_fallback_dimension and not wants_overall:

            dim = detect_dimension(df)

            if dim:
                plan["group_by"] = [dim]

    if plan.get("time_granularity"):

        if detect_date_column(df) is None:
            plan["time_granularity"] = None

    return plan


# ============================================================
# Exported Helpers
# ============================================================

__all__ = [

    "BUSINESS_SYNONYMS",

    "normalize",

    "similarity",

    "all_columns",

    "numeric_columns",

    "categorical_columns",

    "date_columns",

    "semantic_match",

    "resolve_business_column",

    "resolve_measure",

    "resolve_dimension",

    "detect_primary_measure",

    "detect_dimension",

    "detect_date_column",

    "resolve_time_column",

    "validate_plan",

    "sanitize_plan",

    "dataset_profile",

    "suggest_kpis"

]
