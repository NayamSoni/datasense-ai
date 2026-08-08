"""Grounded business insight and recommendation engine for DataSense AI V2."""

from __future__ import annotations

import calendar
from html import escape
import re

import pandas as pd

from utils import detect_date_column, detect_primary_measure


def _named_column(df, candidates):
    lookup = {str(col).lower().replace("_", " "): col for col in df.columns}
    for candidate in candidates:
        for normalized, original in lookup.items():
            if candidate == normalized or candidate in normalized:
                return original
    return None


def _dimension_column(df, candidates):
    """Resolve only categorical columns with explicit business-name evidence."""
    categorical = df.select_dtypes(exclude="number").columns
    for candidate in candidates:
        for column in categorical:
            normalized = str(column).lower().replace("_", " ").replace("-", " ")
            if candidate == normalized or candidate in normalized.split():
                return column
    return None


def _primary_dimension(df, date_col=None):
    preferred = _dimension_column(
        df,
        ["category", "product", "segment", "region", "customer", "state", "city", "country", "channel", "brand"],
    )
    if preferred:
        return preferred

    # Conservative fallback: a reusable categorical group, never a constant,
    # date, or almost-unique identifier/text field.
    for column in df.select_dtypes(exclude="number").columns:
        unique = df[column].nunique(dropna=True)
        if column != date_col and 2 <= unique <= min(50, max(2, len(df) // 2)):
            return column
    return None


def _fmt(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


def generate_business_insights(df: pd.DataFrame, limit: int = 5) -> list[dict]:
    """Generate concise insights only from values calculated from the dataset."""
    insights = []
    if df.empty:
        return insights

    measure = detect_primary_measure(df)
    date_col = detect_date_column(df)
    dimension = _primary_dimension(df, date_col)

    if measure and date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        work = df.loc[dates.notna(), [measure]].copy()
        work["_date"] = dates[dates.notna()]
        work["_quarter"] = work["_date"].dt.to_period("Q")
        quarterly = work.groupby("_quarter")[measure].sum().sort_index()
        if len(quarterly) >= 2 and quarterly.iloc[-2] != 0:
            growth = (quarterly.iloc[-1] / quarterly.iloc[-2] - 1) * 100
            insights.append({
                "type": "period_growth",
                "text": (
                    f"{measure} {'increased' if growth >= 0 else 'decreased'} "
                    f"{abs(growth):.1f}% in {quarterly.index[-1]} versus {quarterly.index[-2]}."
                ),
                "evidence": {"growth_pct": round(growth, 2), "measure": measure},
            })

        work["_month"] = work["_date"].dt.month
        monthly = work.groupby("_month")[measure].sum()
        if not monthly.empty and monthly.sum() != 0:
            peak_month = int(monthly.idxmax())
            share = monthly.max() / monthly.sum() * 100
            insights.append({
                "type": "seasonality",
                "text": f"{measure} is highest in {calendar.month_name[peak_month]}, contributing {share:.1f}% of the monthly total.",
                "evidence": {"month": calendar.month_name[peak_month], "share_pct": round(share, 2)},
            })

    if measure and dimension and dimension != date_col:
        grouped = df.groupby(dimension, dropna=False)[measure].sum().sort_values(ascending=False)
        if not grouped.empty and grouped.sum() != 0:
            top_name, top_value = grouped.index[0], grouped.iloc[0]
            contribution = top_value / grouped.sum() * 100
            insights.append({
                "type": "concentration",
                "text": f"{top_name} leads {dimension} with {_fmt(top_value)} {measure}, contributing {contribution:.1f}%.",
                "evidence": {"dimension": dimension, "leader": str(top_name), "share_pct": round(contribution, 2)},
            })

    revenue = _named_column(df, ["revenue", "sales", "net sales"])
    profit = _named_column(df, ["profit", "gross profit", "net profit"])
    category = _dimension_column(df, ["category"])
    geography = _dimension_column(df, ["state", "region", "city", "country"])

    if profit and geography:
        geo = df.groupby(geography, dropna=False)[profit].sum().sort_values(ascending=False)
        if not geo.empty:
            insights.append({
                "type": "top_geography",
                "text": f"{geo.index[0]} has the highest {profit} at {_fmt(geo.iloc[0])}.",
                "evidence": {"geography": geography, "leader": str(geo.index[0]), "value": float(geo.iloc[0])},
            })

    if revenue and profit and category and revenue != profit:
        margin = df.groupby(category)[[revenue, profit]].sum()
        margin = margin[margin[revenue] != 0]
        if not margin.empty:
            margin["_margin"] = margin[profit] / margin[revenue] * 100
            lowest = margin["_margin"].idxmin()
            insights.append({
                "type": "low_margin",
                "text": f"{lowest} has the lowest {category} margin at {margin.loc[lowest, '_margin']:.1f}%.",
                "evidence": {"category": category, "lowest": str(lowest), "margin_pct": round(float(margin.loc[lowest, "_margin"]), 2)},
            })

    discount = _named_column(df, ["discount", "discount rate"])
    if discount and profit and discount != profit:
        pair = df[[discount, profit]].dropna()
        if len(pair) >= 3 and pair[discount].nunique() > 1 and pair[profit].nunique() > 1:
            corr = pair[discount].corr(pair[profit])
            if pd.notna(corr) and abs(corr) >= 0.20:
                direction = "negative" if corr < 0 else "positive"
                insights.append({
                    "type": "discount_profit",
                    "text": f"{discount} and {profit} have a {direction} correlation of {corr:.2f}.",
                    "evidence": {"correlation": round(float(corr), 3)},
                })

    return insights[:limit]


def generate_recommendations(insights: list[dict], limit: int = 4) -> list[str]:
    """Translate grounded findings into cautious, consultant-style actions."""
    recommendations = []
    for insight in insights:
        kind, evidence = insight["type"], insight.get("evidence", {})
        if kind == "period_growth":
            if evidence.get("growth_pct", 0) >= 0:
                recommendations.append("Plan capacity and inventory around the latest period's demonstrated growth.")
            else:
                recommendations.append("Review segment and product drivers behind the latest period's decline.")
        elif kind == "seasonality":
            recommendations.append(f"Prepare inventory, staffing, and campaigns ahead of {evidence.get('month')}.")
        elif kind == "concentration":
            recommendations.append(f"Protect performance in {evidence.get('leader')} while reducing over-concentration risk.")
        elif kind == "top_geography":
            recommendations.append(f"Test premium or retention campaigns in {evidence.get('leader')}, the strongest geography.")
        elif kind == "low_margin":
            recommendations.append(f"Review pricing, discounting, and costs for {evidence.get('lowest')} before pursuing more volume.")
        elif kind == "discount_profit" and evidence.get("correlation", 0) < 0:
            recommendations.append("Review discount thresholds and require margin checks before deeper discounting.")

    # De-duplicate while keeping the evidence-driven priority order.
    return list(dict.fromkeys(recommendations))[:limit]


# =====================================================================
# Decision-focused business report
# =====================================================================

def _normal_name(column) -> str:
    return str(column).lower().replace("_", " ").replace("-", " ").strip()


def _business_metrics(df: pd.DataFrame) -> list:
    """Return useful numeric metrics while excluding obvious identifiers."""
    excluded = (" id", "id ", "row id", "index", "postal", "zip", "code")
    metrics = []
    for column in df.select_dtypes(include="number").columns:
        name = f" {_normal_name(column)} "
        if any(token in name for token in excluded):
            continue
        metrics.append(column)
    return metrics


def _important_dimensions(df: pd.DataFrame, date_col=None) -> list:
    preferred_words = (
        "category", "product", "segment", "region", "state", "city",
        "country", "channel", "customer", "employee", "agent", "brand",
    )
    candidates = []
    for column in df.select_dtypes(exclude="number").columns:
        if column == date_col:
            continue
        unique = int(df[column].nunique(dropna=True))
        if not 2 <= unique <= min(100, max(2, len(df) // 2)):
            continue
        name = _normal_name(column)
        score = next((20 - i for i, word in enumerate(preferred_words) if word in name), 0)
        candidates.append((score, -unique, str(column), column))
    candidates.sort(reverse=True)
    return [column for _, _, _, column in candidates]


def _is_rank_metric(column) -> bool:
    return "rank" in _normal_name(column)


def _is_additive_metric(column) -> bool:
    name = _normal_name(column)
    return any(word in name for word in (
        "sales", "revenue", "profit", "cost", "expense", "amount",
        "quantity", "units", "volume", "orders", "count", "income",
    ))


def _format_value(value, column=None, decimals=1) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    value = float(value)
    name = _normal_name(column) if column is not None else ""
    if any(word in name for word in ("rate", "percent", "%", "margin")):
        return f"{value:.{decimals}f}%"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def _format_month(period) -> str:
    """Turn a monthly Period/string into a reader-friendly month label."""
    try:
        return pd.Period(period, freq="M").start_time.strftime("%B %Y")
    except (TypeError, ValueError):
        return str(period)


def _time_evidence(df, date_col, metric):
    if not date_col or not metric or _is_rank_metric(metric):
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[metric], errors="coerce")
    work = pd.DataFrame({"Date": dates, "Value": values}).dropna()
    if work.empty:
        return None

    work["Period"] = work["Date"].dt.to_period("M")
    aggregation = "sum" if _is_additive_metric(metric) else "mean"
    monthly = work.groupby("Period")["Value"].agg(aggregation).sort_index()
    if len(monthly) < 2:
        return {"monthly": monthly, "comparison_monthly": monthly, "growth": None}

    # Avoid treating an obviously incomplete final month as a decline. A month
    # is considered covered when the latest date reaches at least 90% of it.
    latest_period_in_data = monthly.index[-1]
    latest_date = work["Date"].max()
    coverage = latest_date.day / latest_date.days_in_month
    dates_per_month = work.groupby("Period")["Date"].nunique().sort_index()
    typical_month_dates = float(dates_per_month.iloc[:-1].median()) if len(dates_per_month) > 1 else 0
    comparison_monthly = monthly
    excluded_partial_period = None
    if typical_month_dates >= 5 and coverage < 0.90 and len(monthly) >= 3:
        comparison_monthly = monthly.iloc[:-1]
        excluded_partial_period = latest_period_in_data

    previous_period = comparison_monthly.index[-2]
    latest_period = comparison_monthly.index[-1]
    previous = float(comparison_monthly.iloc[-2])
    latest = float(comparison_monthly.iloc[-1])
    growth = None if previous == 0 else (latest / previous - 1) * 100

    year_ago_period = latest_period - 12
    year_ago_value = (
        float(comparison_monthly.loc[year_ago_period])
        if year_ago_period in comparison_monthly.index else None
    )
    year_over_year_growth = (
        None if year_ago_value in (None, 0)
        else (latest / year_ago_value - 1) * 100
    )

    return {
        "monthly": monthly,
        "comparison_monthly": comparison_monthly,
        "growth": growth,
        "latest_period": latest_period,
        "previous_period": previous_period,
        "latest_label": _format_month(latest_period),
        "previous_label": _format_month(previous_period),
        "latest_value": latest,
        "previous_value": previous,
        "year_ago_period": year_ago_period if year_ago_value is not None else None,
        "year_ago_label": _format_month(year_ago_period) if year_ago_value is not None else None,
        "year_ago_value": year_ago_value,
        "year_over_year_growth": year_over_year_growth,
        "excluded_partial_period": excluded_partial_period,
    }


def generate_decision_report(
    df: pd.DataFrame,
    quality_report: dict | None = None,
    max_findings: int = 7,
    max_recommendations: int = 5,
) -> dict:
    """Create a concise report from calculated evidence only."""
    rows, columns = df.shape
    date_col = detect_date_column(df)
    metrics = _business_metrics(df)
    dimensions = _important_dimensions(df, date_col)

    preferred_metric = _named_column(
        df,
        ["revenue", "sales", "profit", "amount", "quantity", "units", "rank"],
    )
    if preferred_metric not in metrics:
        preferred_metric = metrics[0] if metrics else None
    primary_dimension = dimensions[0] if dimensions else None

    valid_dates = (
        pd.to_datetime(df[date_col], errors="coerce").dropna()
        if date_col else pd.Series(dtype="datetime64[ns]")
    )
    if not valid_dates.empty:
        time_period = (
            f"{valid_dates.min().strftime('%d %B %Y')} to "
            f"{valid_dates.max().strftime('%d %B %Y')}"
        )
    else:
        time_period = "No reliable date range is available"

    overview = {
        "records": rows,
        "columns": columns,
        "date_column": date_col,
        "time_period": time_period,
        "dimensions": dimensions[:5],
        "metrics": metrics[:6],
        "note": (
            "Monetary currency is not specified; monetary values are shown in dataset units."
            if any(any(word in _normal_name(metric) for word in ("sales", "revenue", "profit", "cost", "amount")) for metric in metrics)
            else None
        ),
    }

    revenue = _named_column(df, ["revenue", "sales", "net sales"])
    profit = _named_column(df, ["profit", "net profit", "gross profit"])
    quantity = _named_column(df, ["quantity", "units sold", "units", "volume"])
    order_id = _named_column(df, ["order id", "order number", "transaction id", "invoice id"])

    # KPI cards contain business performance only. Dataset metadata belongs in
    # Business Overview, not in the KPI scorecard.
    kpis = []
    added_metrics = set()
    for metric in (revenue, profit, quantity):
        if metric in metrics and metric not in added_metrics:
            numeric = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not numeric.empty:
                kpis.append({
                    "name": f"Total {metric}",
                    "value": _format_value(numeric.sum(), metric),
                    "context": "Across the available dataset period",
                })
                added_metrics.add(metric)

    if revenue in metrics and profit in metrics and revenue != profit:
        total_revenue = pd.to_numeric(df[revenue], errors="coerce").sum()
        total_profit = pd.to_numeric(df[profit], errors="coerce").sum()
        if total_revenue:
            margin = total_profit / total_revenue * 100
            kpis.append({
                "name": "Profit Margin",
                "value": f"{margin:.1f}%",
                "context": f"{_format_value(total_profit, profit)} {profit} on {_format_value(total_revenue, revenue)} {revenue}",
            })

    if revenue in metrics and order_id in df.columns:
        distinct_orders = int(df[order_id].nunique(dropna=True))
        total_revenue = pd.to_numeric(df[revenue], errors="coerce").sum()
        if distinct_orders:
            kpis.append({
                "name": f"Average {revenue} per Order",
                "value": _format_value(total_revenue / distinct_orders, revenue),
                "context": f"Based on {distinct_orders:,} distinct orders",
            })

    if not kpis and preferred_metric:
        numeric = pd.to_numeric(df[preferred_metric], errors="coerce").dropna()
        if not numeric.empty:
            aggregation_label = "Total" if _is_additive_metric(preferred_metric) else "Average"
            value = numeric.sum() if _is_additive_metric(preferred_metric) else numeric.mean()
            kpis.append({
                "name": f"{aggregation_label} {preferred_metric}",
                "value": _format_value(value, preferred_metric),
                "context": "Primary supported performance metric",
            })

    time_evidence = _time_evidence(df, date_col, preferred_metric)
    if time_evidence and time_evidence.get("growth") is not None:
        growth = time_evidence["growth"]
        kpis.append({
            "name": f"Latest Monthly {preferred_metric} Change",
            "value": f"{growth:+.1f}%",
            "context": f"{time_evidence['latest_label']} versus {time_evidence['previous_label']}",
        })

    findings = []

    if time_evidence and time_evidence.get("growth") is not None:
        growth = time_evidence["growth"]
        direction = "increased" if growth > 0 else "decreased" if growth < 0 else "was unchanged"
        yoy = time_evidence.get("year_over_year_growth")
        yoy_sentence = ""
        if yoy is not None:
            yoy_direction = "up" if yoy >= 0 else "down"
            yoy_sentence = (
                f" It was {yoy_direction} {abs(yoy):.1f}% from "
                f"{time_evidence['year_ago_label']} "
                f"({_format_value(time_evidence['year_ago_value'], preferred_metric)})."
            )

        if yoy is not None and growth < 0 < yoy:
            interpretation = (
                f"{preferred_metric} softened from the previous month but remained above the same month last year."
            )
        elif yoy is not None and growth < 0 and yoy < 0:
            interpretation = (
                f"{preferred_metric} declined both month over month and year over year, making the weakness more material."
            )
        elif yoy is not None and growth > 0 and yoy > 0:
            interpretation = (
                f"{preferred_metric} improved both month over month and year over year, indicating broader momentum."
            )
        else:
            interpretation = (
                f"This is a one-month movement; compare {preferred_metric} by {primary_dimension or 'business segment'} to identify the driver."
            )

        if time_evidence.get("excluded_partial_period") is not None:
            interpretation += (
                f" {_format_month(time_evidence['excluded_partial_period'])} was excluded because the dataset covers less than 90% of that month."
            )

        findings.append({
            "section": "Trends Over Time",
            "title": (
                f"{time_evidence['latest_label']} {preferred_metric} "
                f"{direction} {abs(growth):.1f}% month over month"
            ),
            "evidence": (
                f"{preferred_metric} totalled {_format_value(time_evidence['latest_value'], preferred_metric)} "
                f"in {time_evidence['latest_label']}, compared with "
                f"{_format_value(time_evidence['previous_value'], preferred_metric)} "
                f"in {time_evidence['previous_label']} ({growth:+.1f}%)."
                f"{yoy_sentence}"
            ),
            "meaning": interpretation,
            "type": "decline" if growth < -2 else "growth" if growth > 2 else "stable",
        })

        monthly = time_evidence["comparison_monthly"]
        if len(monthly) >= 3:
            years = int(monthly.index.year.nunique())
            if years >= 2:
                month_average = monthly.groupby(monthly.index.month).mean()
                peak_month, weak_month = int(month_average.idxmax()), int(month_average.idxmin())
                peak_value = float(month_average.loc[peak_month])
                weak_value = float(month_average.loc[weak_month])
                spread = None if weak_value == 0 else (peak_value / weak_value - 1) * 100
                spread_text = f", {spread:.1f}% higher" if spread is not None else ""
                seasonal_title = f"{calendar.month_name[peak_month]} is the strongest month for {preferred_metric}"
                seasonal_evidence = (
                    f"Across {years} years, average monthly {preferred_metric} was "
                    f"{_format_value(peak_value, preferred_metric)} in {calendar.month_name[peak_month]}, "
                    f"versus {_format_value(weak_value, preferred_metric)} in {calendar.month_name[weak_month]}"
                    f"{spread_text}."
                )
                seasonal_meaning = (
                    f"The repeated pattern suggests {preferred_metric} is seasonally strongest in "
                    f"{calendar.month_name[peak_month]}; plan capacity or campaigns ahead of that month."
                )
            else:
                peak, weak = monthly.idxmax(), monthly.idxmin()
                seasonal_title = f"Highest and lowest monthly {preferred_metric}"
                seasonal_evidence = (
                    f"Monthly {preferred_metric} was highest in {_format_month(peak)} at "
                    f"{_format_value(monthly.loc[peak], preferred_metric)} and lowest in "
                    f"{_format_month(weak)} at {_format_value(monthly.loc[weak], preferred_metric)}."
                )
                seasonal_meaning = (
                    "Only one year is available, so this identifies the monthly range but does not establish seasonality."
                )
            findings.append({
                "section": "Trends Over Time",
                "title": seasonal_title,
                "evidence": seasonal_evidence,
                "meaning": seasonal_meaning,
                "type": "seasonality",
                "peak_month": calendar.month_name[peak_month] if years >= 2 else None,
            })

    concentration_share = None
    if preferred_metric and primary_dimension:
        method = "sum" if _is_additive_metric(preferred_metric) else "mean"
        grouped = (
            df.groupby(primary_dimension, dropna=False)[preferred_metric]
            .agg(method)
            .dropna()
        )
        if len(grouped) >= 2:
            ascending = _is_rank_metric(preferred_metric)
            ordered = grouped.sort_values(ascending=ascending)
            best_name, worst_name = ordered.index[0], ordered.index[-1]
            best_value, worst_value = float(ordered.iloc[0]), float(ordered.iloc[-1])
            if method == "sum" and grouped.sum() != 0:
                concentration_share = best_value / grouped.sum() * 100
            share_text = (
                f", representing {concentration_share:.1f}% of the total"
                if concentration_share is not None else ""
            )
            findings.append({
                "section": "Top and Bottom Performers",
                "title": f"{best_name} leads {primary_dimension} in {preferred_metric}",
                "evidence": (
                    f"{best_name} generated {_format_value(best_value, preferred_metric)} in {preferred_metric}{share_text}; "
                    f"{worst_name} generated {_format_value(worst_value, preferred_metric)}."
                ),
                "meaning": (
                    f"{best_name}'s contribution is material"
                    + (" and creates concentration risk." if concentration_share and concentration_share >= 50
                       else ", but the business is not dependent on this segment alone.")
                ),
                "type": "concentration" if concentration_share and concentration_share >= 30 else "segment_gap",
                "leader": str(best_name),
            })

    discount = _named_column(df, ["discount", "discount rate"])
    if discount in metrics and profit in metrics and discount != profit:
        pair = df[[discount, profit]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 12 and pair[discount].nunique() > 1 and pair[profit].nunique() > 1:
            corr = float(pair[discount].corr(pair[profit]))
            if pd.notna(corr):
                strength = "strong" if abs(corr) >= .7 else "moderate" if abs(corr) >= .4 else "weak"
                findings.append({
                    "section": "Key Drivers and Relationships",
                    "title": f"{discount} and {profit} relationship",
                    "evidence": f"Their Pearson correlation is {corr:.2f} across {len(pair):,} complete records ({strength}, {'negative' if corr < 0 else 'positive'}).",
                    "meaning": (
                        f"The relationship is {strength}, so {discount} should not be used alone to explain or predict {profit}. "
                        "Correlation does not prove causation."
                    ),
                    "type": "negative_correlation" if corr < 0 else "correlation",
                })

    if profit in metrics:
        profit_values = pd.to_numeric(df[profit], errors="coerce")
        loss_mask = profit_values < 0
        if loss_mask.any():
            loss_rows = int(loss_mask.sum())
            losses = float(profit_values[loss_mask].sum())
            positive_profit = float(profit_values[profit_values > 0].sum())
            offset_pct = abs(losses) / positive_profit * 100 if positive_profit else None
            offset_text = (
                f" These losses offset {offset_pct:.1f}% of profit from profitable records."
                if offset_pct is not None else ""
            )
            findings.append({
                "section": "Anomalies and Business Risks",
                "title": "Loss-making records",
                "evidence": (
                    f"{loss_rows:,} records ({loss_rows / max(len(df), 1) * 100:.1f}%) have negative {profit}, "
                    f"totalling {_format_value(losses, profit)}.{offset_text}"
                ),
                "meaning": f"Losses are directly reducing overall {profit}; prioritize the segments and discount levels responsible for them.",
                "type": "loss_risk",
            })

    if quality_report:
        issue_parts = []
        if quality_report.get("missing_cells"):
            issue_parts.append(f"{quality_report['missing_cells']:,} missing cells")
        if quality_report.get("duplicate_rows"):
            issue_parts.append(f"{quality_report['duplicate_rows']:,} duplicate rows")
        outliers = quality_report.get("outliers")
        outlier_count = int(outliers["Outliers"].sum()) if isinstance(outliers, pd.DataFrame) and not outliers.empty else 0
        if outlier_count:
            issue_parts.append(f"{outlier_count:,} IQR outliers")
        if issue_parts:
            findings.append({
                "section": "Anomalies and Business Risks",
                "title": "Data-quality risk",
                "evidence": "The quality scan found " + ", ".join(issue_parts) + ".",
                "meaning": "Validate these records before using the report for high-stakes decisions.",
                "type": "data_quality",
            })

    category = _dimension_column(df, ["category", "segment", "product"])
    if revenue in metrics and profit in metrics and category:
        margin = df.groupby(category)[[revenue, profit]].sum().dropna()
        margin = margin[margin[revenue] != 0]
        if len(margin) >= 2:
            margin["Margin"] = margin[profit] / margin[revenue] * 100
            low, high = margin["Margin"].idxmin(), margin["Margin"].idxmax()
            findings.append({
                "section": "Business Opportunities",
                "title": "Margin improvement opportunity",
                "evidence": f"{low} has the lowest margin at {margin.loc[low, 'Margin']:.1f}%, versus {high} at {margin.loc[high, 'Margin']:.1f}%.",
                "meaning": "The low-margin segment is a candidate for pricing, discount, mix, and cost review rather than volume growth alone.",
                "type": "low_margin",
                "segment": str(low),
            })

    # Keep the report bounded by business importance, then restore the requested
    # A-F reading order. This prevents a material risk/opportunity from being
    # silently pushed out by lower-value descriptive observations.
    finding_priority = {
        "loss_risk": 100,
        "decline": 95,
        "low_margin": 90,
        "concentration": 85,
        "growth": 80,
        "negative_correlation": 75,
        "data_quality": 70,
        "seasonality": 60,
        "segment_gap": 55,
        "stable": 45,
        "correlation": 40,
    }
    section_order = {
        "Trends Over Time": 1,
        "Top and Bottom Performers": 2,
        "Key Drivers and Relationships": 3,
        "Anomalies and Business Risks": 4,
        "Business Opportunities": 5,
    }
    findings = sorted(
        sorted(
            findings,
            key=lambda item: finding_priority.get(item["type"], 0),
            reverse=True,
        )[:max_findings],
        key=lambda item: section_order.get(item["section"], 99),
    )

    recommendations = []
    for finding in findings:
        kind = finding["type"]
        if kind == "decline":
            recommendations.append({
                "priority": "High", "action": "Diagnose the latest-period decline by the most important dimensions.",
                "evidence": finding["evidence"], "impact": "Identifies the segments responsible for the decline before resources are changed.",
                "next_step": f"Break {preferred_metric} down by {primary_dimension or 'the available business dimensions'} for the two latest periods.",
                "kpi": f"Monthly {preferred_metric}",
            })
        elif kind == "loss_risk":
            recommendations.append({
                "priority": "High", "action": "Review loss-making records before pursuing additional volume.",
                "evidence": finding["evidence"], "impact": "Supports profitability improvement by isolating avoidable losses.",
                "next_step": "Rank negative-profit records by category, customer, geography, and discount.",
                "kpi": f"Negative-{profit} rate and total losses",
            })
        elif kind == "low_margin":
            recommendations.append({
                "priority": "High", "action": f"Review pricing, discounting, and costs for {finding.get('segment')}.",
                "evidence": finding["evidence"], "impact": "May improve margin without relying only on higher sales volume.",
                "next_step": "Compare price, discount, product mix, and costs with the highest-margin segment.",
                "kpi": "Segment profit margin",
            })
        elif kind == "concentration":
            recommendations.append({
                "priority": "Medium", "action": "Protect the leading segment while testing diversification opportunities.",
                "evidence": finding["evidence"], "impact": "Reduces dependence on one segment while protecting current contribution.",
                "next_step": "Measure the next five segments by contribution, growth, and profitability.",
                "kpi": "Top-segment contribution %",
            })
        elif kind == "negative_correlation":
            recommendations.append({
                "priority": "Medium", "action": "Test discount thresholds against profitability at a finer segment level.",
                "evidence": finding["evidence"], "impact": "Helps identify discounting patterns associated with weak profit while avoiding causal overstatement.",
                "next_step": "Compare profit margin across discount bands while controlling for category and order size.",
                "kpi": "Profit margin by discount band",
            })
        elif kind == "data_quality":
            recommendations.append({
                "priority": "Medium", "action": "Resolve material data-quality issues before operationalizing the report.",
                "evidence": finding["evidence"], "impact": "Improves confidence in KPIs, comparisons, and downstream decisions.",
                "next_step": "Review the Data Quality page and document accepted cleaning actions.",
                "kpi": "Dataset health score",
            })
        elif kind == "seasonality":
            peak_month = finding.get("peak_month")
            recommendations.append({
                "priority": "Medium",
                "action": (
                    f"Prepare capacity, inventory, and campaigns ahead of {peak_month}."
                    if peak_month else "Plan for the strongest observed month after validating the pattern."
                ),
                "evidence": finding["evidence"],
                "impact": "Aligns resources with the recurring period of highest observed demand.",
                "next_step": (
                    f"Confirm the {peak_month} pattern by category and year, then set the operating plan."
                    if peak_month else "Compare monthly performance by year and segment before changing plans."
                ),
                "kpi": f"Monthly {preferred_metric}",
            })

    # De-duplicate actions and keep no more than the requested number.
    unique_recommendations = []
    seen_actions = set()
    for item in recommendations:
        if item["action"] not in seen_actions:
            seen_actions.add(item["action"])
            unique_recommendations.append(item)
    recommendations = unique_recommendations[:max_recommendations]

    if not recommendations:
        recommendations.append({
            "priority": "Low",
            "action": "Define the management decision and benchmark for the next analysis.",
            "evidence": "The dataset supports descriptive analysis, but no target or external benchmark is supplied.",
            "impact": "Prevents generic recommendations and focuses analysis on a decision.",
            "next_step": "Document the target KPI, comparison period, and decision owner.",
            "kpi": preferred_metric or "Selected business KPI",
        })

    questions = []
    if time_evidence and primary_dimension:
        questions.append(f"Which {primary_dimension} values explain the latest change in {preferred_metric}?")
    if revenue in metrics and profit in metrics and primary_dimension:
        questions.append(f"Which {primary_dimension} values combine high {revenue} with strong profit margin?")
    if discount in metrics and profit in metrics:
        questions.append(f"How does {profit} change across {discount} bands after controlling for category or segment?")
    if date_col:
        questions.append("Do the strongest and weakest periods repeat across years, or are they one-time events?")
    if quality_report and (quality_report.get("missing_cells") or quality_report.get("duplicate_rows")):
        questions.append("How materially do the reported KPIs change after approved data cleaning?")
    if primary_dimension and preferred_metric:
        questions.append(f"What separates the top and bottom {primary_dimension} values on {preferred_metric}?")
    questions = list(dict.fromkeys(questions))[:5]

    return {
        "overview": overview,
        "kpis": kpis[:6],
        "findings": findings,
        "recommendations": recommendations,
        "questions": questions,
        "primary_metric": preferred_metric,
        "primary_dimension": primary_dimension,
    }


def report_to_markdown(report: dict, title: str = "DataSense AI Business Insights Report") -> str:
    """Convert the grounded report into editable Markdown."""
    overview = report["overview"]
    lines = [f"# {title}", "", "## 1. Business Overview", ""]
    lines.append(f"- **Dataset size:** {overview['records']:,} records and {overview['columns']} columns")
    lines.append(f"- **Available period:** {overview['time_period']}")
    lines.append("- **Important dimensions:** " + (", ".join(map(str, overview["dimensions"])) or "No reliable business dimensions identified"))
    lines.append("- **Business metrics:** " + (", ".join(map(str, overview["metrics"])) or "No numeric business metrics identified"))
    if overview.get("note"):
        lines.append(f"- **Note:** {overview['note']}")

    lines.extend(["", "## 2. KPI Summary", "", "| KPI | Value | Context |", "|---|---:|---|"])
    for kpi in report["kpis"]:
        lines.append(f"| {kpi['name']} | {kpi['value']} | {kpi['context']} |")

    lines.extend(["", "## 3. Key Findings & Analysis", ""])
    current_section = None
    section_number = 0
    for finding in report["findings"]:
        if finding["section"] != current_section:
            current_section = finding["section"]
            section_number += 1
            lines.extend([f"### 3.{section_number} {current_section}", ""])
        lines.append(f"- **{finding['title']}**")
        lines.append(f"  {finding['evidence']}")
        if finding.get("meaning"):
            lines.append(f"  **Why it matters:** {finding['meaning']}")

    lines.extend([
        "", "## 4. Recommendations & Next Steps", "",
        "| Priority | Recommended action | Supporting evidence | Expected impact | Specific next step | KPI to monitor |",
        "|---|---|---|---|---|---|",
    ])
    for item in report["recommendations"]:
        cells = [
            item["priority"], item["action"], item["evidence"], item["impact"],
            item["next_step"], item["kpi"],
        ]
        cells = [str(cell).replace("|", "/").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## 5. Questions for Further Analysis", ""])
    for question in report["questions"]:
        lines.append(f"- {question}")

    lines.extend([
        "",
        "> Recommendations are evidence-based hypotheses. Validate business context, causality, targets, and operational constraints before acting.",
    ])
    return "\n".join(lines)


def _inline_markdown(text: str) -> str:
    safe = escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    return safe


def _markdown_body_to_html(markdown_text: str) -> str:
    """Render the generated editable Markdown without an extra dependency."""
    lines = markdown_text.splitlines()
    output = []
    index = 0
    list_kind = None

    def close_list():
        nonlocal list_kind
        if list_kind:
            output.append(f"</{list_kind}>")
            list_kind = None

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if stripped.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if separator.startswith("|") and set(separator.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")) == set():
                close_list()
                table_lines = [line]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index].strip())
                    index += 1
                headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                output.append('<div class="table-wrap"><table><thead><tr>')
                output.extend(f"<th>{_inline_markdown(cell)}</th>" for cell in headers)
                output.append("</tr></thead><tbody>")
                for row in table_lines[1:]:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    output.append("<tr>")
                    output.extend(f"<td>{_inline_markdown(cell)}</td>" for cell in cells)
                    output.append("</tr>")
                output.append("</tbody></table></div>")
                continue

        if not stripped:
            close_list()
        elif stripped.startswith("### "):
            close_list()
            output.append(f"<h3>{_inline_markdown(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_list()
            output.append(f"<h2>{_inline_markdown(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_list()
            output.append(f"<h1>{_inline_markdown(stripped[2:])}</h1>")
        elif stripped.startswith("> "):
            close_list()
            output.append(f"<blockquote>{_inline_markdown(stripped[2:])}</blockquote>")
        elif stripped.startswith("- ") or stripped.startswith("   - "):
            if list_kind != "ul":
                close_list()
                list_kind = "ul"
                output.append("<ul>")
            output.append(f"<li>{_inline_markdown(stripped[2:])}</li>")
        elif re.match(r"^\d+\.\s", stripped):
            if list_kind != "ol":
                close_list()
                list_kind = "ol"
                output.append("<ol>")
            content = re.sub(r"^\d+\.\s+", "", stripped)
            output.append(f"<li>{_inline_markdown(content)}</li>")
        else:
            close_list()
            output.append(f"<p>{_inline_markdown(stripped)}</p>")
        index += 1

    close_list()
    return "\n".join(output)


def report_to_html(markdown_text: str, title: str = "DataSense AI Business Insights Report") -> str:
    """Create a formatted, portable HTML download of the edited report."""
    body = _markdown_body_to_html(markdown_text)
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{escape(title)}</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{max-width:1120px;margin:0 auto;padding:48px 32px 80px;background:#07101f;color:#e8edf8;font:15px/1.65 Inter,Arial,sans-serif}}
h1{{font-size:36px;line-height:1.2;color:#f8fafc;margin:0 0 30px}}h2{{margin:38px 0 15px;color:#a78bfa;border-bottom:1px solid #27345b;padding-bottom:9px}}h3{{margin:26px 0 10px;color:#67e8f9}}p,li{{color:#d7deed}}strong{{color:#fff}}code{{background:#111c35;padding:2px 5px;border-radius:4px}}
.table-wrap{{overflow-x:auto;margin:18px 0 26px;border:1px solid #27345b;border-radius:12px}}table{{width:100%;border-collapse:collapse;min-width:760px}}th,td{{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid #27345b}}th{{background:#111a33;color:#c4b5fd}}tr:last-child td{{border-bottom:0}}blockquote{{margin:30px 0;padding:14px 18px;background:#0d1730;border-left:4px solid #7c5cff;border-radius:0 10px 10px 0;color:#cbd5e1}}
@media print{{:root{{color-scheme:light}}body{{background:#fff;color:#111;padding:20px}}p,li,strong{{color:#111}}h2,h3{{color:#4c1d95}}.table-wrap{{border-color:#bbb}}th,td{{border-color:#ccc}}th{{background:#eee;color:#111}}blockquote{{background:#f5f5f5;color:#222}}}}
</style>
</head><body>{body}</body></html>"""


def generate_report_chart_specs(df: pd.DataFrame, limit: int = 10) -> list[dict]:
    """Return optional chart specifications backed by displayed data."""
    specs = []
    date_col = detect_date_column(df)
    metrics = _business_metrics(df)
    dimensions = _important_dimensions(df, date_col)
    metric = _named_column(df, ["revenue", "sales", "profit", "amount", "quantity", "units", "rank"])
    if metric not in metrics:
        metric = metrics[0] if metrics else None

    time_evidence = _time_evidence(df, date_col, metric)
    if time_evidence and len(time_evidence["comparison_monthly"]) >= 2:
        trend = time_evidence["comparison_monthly"].reset_index()
        trend["Period"] = trend["Period"].map(_format_month)
        trend.columns = ["Period", "Value"]
        specs.append({
            "id": "trend", "title": f"Monthly {metric}", "type": "line",
            "data": trend.tail(36), "x": "Period", "y": "Value", "y_title": str(metric),
        })

    if metric and dimensions:
        dimension = dimensions[0]
        method = "sum" if _is_additive_metric(metric) else "mean"
        grouped = df.groupby(dimension, dropna=False)[metric].agg(method).dropna()
        ascending = _is_rank_metric(metric)
        grouped = grouped.sort_values(ascending=ascending).head(limit).reset_index()
        grouped.columns = ["Segment", "Value"]
        specs.append({
            "id": "segments", "title": f"Top {dimension} by {metric}", "type": "bar",
            "data": grouped, "x": "Value", "y": "Segment", "x_title": str(metric),
        })

    profit = _named_column(df, ["profit", "net profit", "gross profit"])
    discount = _named_column(df, ["discount", "discount rate"])
    if profit in metrics and discount in metrics and profit != discount:
        relationship = df[[discount, profit]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(relationship) >= 12:
            if len(relationship) > 3000:
                relationship = relationship.sample(3000, random_state=42)
            relationship.columns = ["X", "Y"]
            specs.append({
                "id": "relationship", "title": f"{discount} vs {profit}", "type": "scatter",
                "data": relationship, "x": "X", "y": "Y",
                "x_title": str(discount), "y_title": str(profit),
            })

    return specs


__all__ = [
    "generate_business_insights", "generate_recommendations",
    "generate_decision_report", "generate_report_chart_specs",
    "report_to_html", "report_to_markdown",
]