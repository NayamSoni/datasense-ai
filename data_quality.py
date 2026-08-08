"""Dataset quality profiling, scoring, and safe cleaning for DataSense AI V2."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_pct(part: float, whole: float) -> float:
    return round((part / whole * 100), 2) if whole else 0.0


def _outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = df.select_dtypes(include="number")

    for column in numeric.columns:
        series = numeric[column].dropna()
        if len(series) < 8 or series.nunique() < 5:
            continue

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count:
            rows.append({
                "Column": column,
                "Outliers": count,
                "Outlier %": _safe_pct(count, len(series)),
                "Lower Bound": round(float(lower), 4),
                "Upper Bound": round(float(upper), 4),
            })

    return pd.DataFrame(rows)


def _type_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    date_hints = ("date", "time", "month", "year", "created", "updated")
    numeric_hints = (
        "sales", "revenue", "profit", "cost", "price", "amount", "quantity",
        "count", "rate", "score", "discount", "age", "income", "salary",
    )

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna().astype(str).str.strip()
        if values.empty:
            continue

        lowered = column.lower()
        numeric_ratio = pd.to_numeric(
            values.str.replace(",", "", regex=False), errors="coerce"
        ).notna().mean()
        date_ratio = (
            pd.to_datetime(values, errors="coerce").notna().mean()
            if any(h in lowered for h in date_hints)
            else 0.0
        )

        # Leading-zero codes such as postal codes should remain text.
        leading_zero_ratio = values.str.match(r"^0\d+").mean()

        if (
            numeric_ratio >= 0.90
            and leading_zero_ratio < 0.20
            and (any(h in lowered for h in numeric_hints) or numeric_ratio >= 0.98)
        ):
            issues.append({
                "Column": column,
                "Current Type": str(df[column].dtype),
                "Suggested Type": "numeric",
                "Convertible %": round(numeric_ratio * 100, 2),
            })
        elif date_ratio >= 0.85 and any(h in lowered for h in date_hints):
            issues.append({
                "Column": column,
                "Current Type": str(df[column].dtype),
                "Suggested Type": "datetime",
                "Convertible %": round(date_ratio * 100, 2),
            })

    return pd.DataFrame(issues)


def profile_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Return a calculation-backed quality report for an arbitrary DataFrame."""
    rows, columns = df.shape
    missing_count = df.isna().sum()
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values,
        "Missing %": [
            _safe_pct(int(missing_count[col]), rows) for col in df.columns
        ],
    })
    missing = missing[missing["Missing Values"] > 0].reset_index(drop=True)

    constants = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    high_cardinality = []
    for column in df.select_dtypes(exclude="number").columns:
        unique = int(df[column].nunique(dropna=True))
        ratio = unique / max(rows, 1)
        if unique >= 20 and ratio >= 0.80:
            high_cardinality.append({
                "Column": column,
                "Unique Values": unique,
                "Cardinality %": round(ratio * 100, 2),
            })

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "missing_columns": int(len(missing)),
        "missing_cells": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers": _outlier_report(df),
        "incorrect_types": _type_issues(df),
        "constant_columns": constants,
        "high_cardinality": pd.DataFrame(high_cardinality),
    }


def calculate_health_score(df: pd.DataFrame, report: dict[str, Any] | None = None) -> dict[str, float]:
    """Score quality dimensions from 0–100 using transparent formulas."""
    report = report or profile_data_quality(df)
    rows, columns = df.shape
    cells = rows * columns

    completeness = 100 - _safe_pct(report["missing_cells"], cells)
    missing_values = 100 - _safe_pct(report["rows_with_missing"], rows)
    duplicates = 100 - _safe_pct(report["duplicate_rows"], rows)

    issue_columns = len(report["incorrect_types"]) + len(report["constant_columns"])
    consistency = 100 - _safe_pct(issue_columns, columns)

    numeric_cells = rows * max(len(df.select_dtypes(include="number").columns), 1)
    outlier_cells = (
        int(report["outliers"]["Outliers"].sum())
        if not report["outliers"].empty else 0
    )
    outliers = 100 - _safe_pct(outlier_cells, numeric_cells)

    components = {
        "Completeness": max(0.0, completeness),
        "Consistency": max(0.0, consistency),
        "Duplicates": max(0.0, duplicates),
        "Missing Values": max(0.0, missing_values),
        "Outliers": max(0.0, outliers),
    }
    weights = {
        "Completeness": 0.25,
        "Consistency": 0.20,
        "Duplicates": 0.20,
        "Missing Values": 0.20,
        "Outliers": 0.15,
    }
    overall = sum(components[name] * weights[name] for name in components)
    return {"Overall": round(overall, 1), **{k: round(v, 1) for k, v in components.items()}}


def cleaning_suggestions(df: pd.DataFrame, report: dict[str, Any]) -> dict[str, list[str]]:
    """Suggest safe choices per incomplete column; the user remains in control."""
    suggestions = {}
    for column in report["missing"]["Column"].tolist():
        if df[column].notna().sum() == 0:
            suggestions[column] = ["Ignore", "Remove rows"]
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            suggestions[column] = ["Fill with median", "Remove rows", "Ignore"]
        else:
            suggestions[column] = ["Fill with mode", "Remove rows", "Ignore"]
    return suggestions


def apply_cleaning(
    df: pd.DataFrame,
    missing_actions: dict[str, str] | None = None,
    remove_duplicates: bool = False,
    convert_types: bool = False,
    outlier_action: str = "Ignore",
    report: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply only explicitly selected cleaning actions and return an audit log."""
    cleaned = df.copy()
    report = report or profile_data_quality(cleaned)
    actions = []

    for column, action in (missing_actions or {}).items():
        if column not in cleaned.columns or action == "Ignore":
            continue
        if action == "Remove rows":
            before = len(cleaned)
            cleaned = cleaned.dropna(subset=[column])
            actions.append(f"Removed {before - len(cleaned):,} rows missing {column}.")
        elif action == "Fill with median":
            value = cleaned[column].median()
            count = int(cleaned[column].isna().sum())
            cleaned[column] = cleaned[column].fillna(value)
            actions.append(f"Filled {count:,} values in {column} with median {value:,.2f}.")
        elif action == "Fill with mode":
            mode = cleaned[column].mode(dropna=True)
            if not mode.empty:
                count = int(cleaned[column].isna().sum())
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])
                actions.append(f"Filled {count:,} values in {column} with the mode.")

    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        actions.append(f"Removed {before - len(cleaned):,} duplicate rows.")

    if convert_types:
        for issue in report["incorrect_types"].to_dict("records"):
            column = issue["Column"]
            if issue["Suggested Type"] == "numeric":
                cleaned[column] = pd.to_numeric(
                    cleaned[column].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
            else:
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
            actions.append(f"Converted {column} to {issue['Suggested Type']}.")

    if outlier_action in ("Cap at IQR bounds", "Remove outlier rows"):
        for item in report["outliers"].to_dict("records"):
            column = item["Column"]
            lower, upper = item["Lower Bound"], item["Upper Bound"]
            mask = cleaned[column].notna() & (
                (cleaned[column] < lower) | (cleaned[column] > upper)
            )
            count = int(mask.sum())
            if outlier_action == "Cap at IQR bounds":
                cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
                actions.append(f"Capped {count:,} outliers in {column}.")
            else:
                cleaned = cleaned.loc[~mask]
                actions.append(f"Removed {count:,} outlier rows using {column}.")

    return cleaned.reset_index(drop=True), actions


__all__ = [
    "apply_cleaning",
    "calculate_health_score",
    "cleaning_suggestions",
    "profile_data_quality",
]
