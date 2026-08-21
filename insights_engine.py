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


def _first_sentence(text: str) -> str:
    """Return one complete sentence without cutting a decimal percentage."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(text).strip())
    return sentences[0].strip() if sentences else ""


def _dashboard_insight_summary(finding: dict, max_chars: int = 180) -> str:
    """Create a compact, evidence-backed preview for the workspace card.

    The full finding remains available on the Insights page. This summary is
    deliberately deterministic so the dashboard cannot invent or paraphrase a
    number differently from the calculated report.
    """
    title = str(finding.get("title", "")).strip()
    evidence = str(finding.get("evidence", "")).strip()
    finding_type = finding.get("type")

    if finding_type == "calendar_effect":
        evidence_sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            evidence,
        )
        adjusted_sentence = next(
            (
                sentence.strip()
                for sentence in evidence_sentences
                if "after dividing" in sentence.lower()
                or "adjust" in sentence.lower()
            ),
            _first_sentence(evidence),
        )
        summary = f"{title}. {adjusted_sentence}"
    elif finding_type == "segment_driver":
        # The evidence already contains the segment, change and contribution;
        # repeating the long title adds little value in a narrow card.
        summary = _first_sentence(evidence) or title
    else:
        evidence_sentence = _first_sentence(evidence)
        summary = f"{title}. {evidence_sentence}" if evidence_sentence else title

    summary = re.sub(r"\bmonth over month\b", "MoM", summary, flags=re.I)
    summary = re.sub(r"\byear over year\b", "YoY", summary, flags=re.I)
    summary = re.sub(r"\s+", " ", summary).strip()

    if len(summary) <= max_chars:
        return summary

    shortened = summary[: max_chars - 1].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{shortened}…"


def generate_business_insights(df: pd.DataFrame, limit: int = 5) -> list[dict]:
    """Return the same guarded findings used by the full decision report."""
    if df.empty:
        return []

    report = generate_decision_report(
        df,
        max_findings=limit,
        max_recommendations=limit,
    )
    recommendation_by_evidence = {
        item["evidence"]: item["action"]
        for item in report["recommendations"]
    }
    return [
        {
            "type": finding["type"],
            "summary": _dashboard_insight_summary(finding),
            "text": (
                f"{finding['title']}. {finding['evidence']} "
                f"{finding.get('meaning', '')}"
            ).strip(),
            "evidence": finding,
            "recommendation": recommendation_by_evidence.get(finding["evidence"]),
        }
        for finding in report["findings"][:limit]
    ]


def generate_recommendations(insights: list[dict], limit: int = 4) -> list[str]:
    """Return only recommendations that passed the decision-report guardrails."""
    recommendations = [
        insight.get("recommendation")
        for insight in insights
        if insight.get("recommendation")
    ]
    return list(dict.fromkeys(recommendations))[:limit]


# =====================================================================
# Decision-focused business report
# =====================================================================

def _normal_name(column) -> str:
    return str(column).lower().replace("_", " ").replace("-", " ").strip()


def _business_metrics(df: pd.DataFrame) -> list:
    """Return numeric business measures while excluding identifier-like fields."""
    excluded = (
        " id", "id ", "identifier", "row id", "index", "postal", "zip",
        "code", "phone", "account number", "order number", "invoice number",
        "transaction number", "room number", "rank",
    )
    metrics = []
    for column in df.select_dtypes(include="number").columns:
        name = f" {_normal_name(column)} "
        if any(token in name for token in excluded):
            continue
        metrics.append(column)
    return metrics


def _important_dimensions(df: pd.DataFrame, date_col=None) -> list:
    preferred_words = (
        "category", "segment", "product", "admission type", "medical condition",
        "insurance provider", "region", "state", "country", "city", "channel",
        "status", "type", "department", "brand", "test result", "medication",
        "gender", "customer", "employee", "agent",
    )
    candidates = []
    for column in df.select_dtypes(exclude="number").columns:
        if column == date_col:
            continue
        unique = int(df[column].nunique(dropna=True))
        if not 2 <= unique <= min(100, max(2, len(df) // 2)):
            continue
        name = _normal_name(column)
        score = next((100 - i for i, word in enumerate(preferred_words) if word in name), 0)
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


def _entity_identifier(df: pd.DataFrame):
    """Return a stable entity/transaction identifier when its name proves it."""
    exact = (
        "order id", "transaction id", "invoice id", "patient id", "customer id",
        "user id", "account id", "employee id", "ticket id", "booking id",
        "claim id", "record id", "medical record number", "mrn",
    )
    lookup = {_normal_name(column): column for column in df.columns}
    for name in exact:
        if name in lookup:
            return lookup[name]
    for normalized, column in lookup.items():
        tokens = normalized.split()
        if "id" in tokens and len(tokens) >= 2:
            return column
    return None


def _pct_change(current, previous):
    if previous is None or pd.isna(previous) or float(previous) == 0:
        return None
    return (float(current) / float(previous) - 1) * 100


def _material_direction(value, threshold=2.0):
    if value is None or pd.isna(value) or abs(float(value)) < threshold:
        return "stable"
    return "up" if value > 0 else "down"


def _normalized_hhi(shares: pd.Series) -> float:
    """Return 0 for an equal split and 1 for full concentration."""
    shares = pd.to_numeric(shares, errors="coerce").dropna()
    shares = shares[shares >= 0]
    if len(shares) < 2 or shares.sum() == 0:
        return 0.0
    proportions = shares / shares.sum()
    hhi = float((proportions ** 2).sum())
    equal_hhi = 1 / len(proportions)
    return max(0.0, min(1.0, (hhi - equal_hhi) / (1 - equal_hhi)))


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
    monthly_frame = work.groupby("Period").agg(
        Value=("Value", aggregation),
        Records=("Value", "size"),
        Average=("Value", "mean"),
        FirstDate=("Date", "min"),
        LastDate=("Date", "max"),
        ActiveDates=("Date", "nunique"),
    ).sort_index()
    monthly = monthly_frame["Value"]
    if len(monthly_frame) < 2:
        return {
            "monthly": monthly,
            "comparison_monthly": monthly,
            "monthly_frame": monthly_frame,
            "comparison_frame": monthly_frame,
            "growth": None,
        }

    # Avoid treating an obviously incomplete final month as a decline. A month
    # is considered covered when the latest date reaches at least 90% of it.
    latest_period_in_data = monthly_frame.index[-1]
    latest_date = work["Date"].max()
    coverage = latest_date.day / latest_date.days_in_month
    dates_per_month = monthly_frame["ActiveDates"]
    typical_month_dates = float(dates_per_month.iloc[:-1].median()) if len(dates_per_month) > 1 else 0
    comparison_frame = monthly_frame
    excluded_partial_period = None
    if typical_month_dates >= 5 and coverage < 0.90 and len(monthly_frame) >= 3:
        comparison_frame = monthly_frame.iloc[:-1]
        excluded_partial_period = latest_period_in_data

    comparison_monthly = comparison_frame["Value"]

    previous_period = comparison_frame.index[-2]
    latest_period = comparison_frame.index[-1]
    previous = float(comparison_frame.iloc[-2]["Value"])
    latest = float(comparison_frame.iloc[-1]["Value"])
    growth = _pct_change(latest, previous)

    previous_records = int(comparison_frame.iloc[-2]["Records"])
    latest_records = int(comparison_frame.iloc[-1]["Records"])
    record_growth = _pct_change(latest_records, previous_records)
    previous_average = float(comparison_frame.iloc[-2]["Average"])
    latest_average = float(comparison_frame.iloc[-1]["Average"])
    average_growth = _pct_change(latest_average, previous_average)

    year_ago_period = latest_period - 12
    year_ago_value = (
        float(comparison_frame.loc[year_ago_period, "Value"])
        if year_ago_period in comparison_frame.index else None
    )
    year_over_year_growth = _pct_change(latest, year_ago_value)

    return {
        "monthly": monthly,
        "comparison_monthly": comparison_monthly,
        "monthly_frame": monthly_frame,
        "comparison_frame": comparison_frame,
        "aggregation": aggregation,
        "growth": growth,
        "latest_period": latest_period,
        "previous_period": previous_period,
        "latest_label": _format_month(latest_period),
        "previous_label": _format_month(previous_period),
        "latest_value": latest,
        "previous_value": previous,
        "latest_records": latest_records,
        "previous_records": previous_records,
        "record_growth": record_growth,
        "latest_average": latest_average,
        "previous_average": previous_average,
        "average_growth": average_growth,
        "year_ago_period": year_ago_period if year_ago_value is not None else None,
        "year_ago_label": _format_month(year_ago_period) if year_ago_value is not None else None,
        "year_ago_value": year_ago_value,
        "year_over_year_growth": year_over_year_growth,
        "excluded_partial_period": excluded_partial_period,
    }


def _movement_interpretation(time_evidence, metric):
    """Explain whether an additive movement came from volume or value."""
    growth = time_evidence.get("growth")
    if growth is None:
        return "No comparable prior month is available."

    if time_evidence.get("aggregation") != "sum":
        return (
            f"The latest monthly average changed by {growth:+.1f}%; compare it "
            "with a target or business benchmark before treating it as good or bad."
        )

    volume = time_evidence.get("record_growth")
    intensity = time_evidence.get("average_growth")
    if volume is None or intensity is None:
        return "The movement is verified, but its volume and per-record drivers are unavailable."

    volume_direction = _material_direction(volume)
    intensity_direction = _material_direction(intensity)
    if volume_direction == "up" and intensity_direction == "down":
        driver = (
            "The increase was volume-led while value per record weakened."
            if growth >= 0 else
            "Higher record volume partially offset weaker value per record."
        )
    elif volume_direction == "down" and intensity_direction == "up":
        driver = (
            "Higher value per record offset lower record volume."
            if growth >= 0 else
            "The decline was volume-led despite stronger value per record."
        )
    elif volume_direction == "up" and intensity_direction == "up":
        driver = "Both record volume and value per record improved, so the movement was broad-based."
    elif volume_direction == "down" and intensity_direction == "down":
        driver = "Both record volume and value per record declined, making the weakness broad-based."
    elif abs(volume) >= abs(intensity):
        driver = "Record volume was the larger observed driver of the movement."
    else:
        driver = f"Average {metric} per record was the larger observed driver of the movement."

    return (
        f"{driver} Record volume changed {volume:+.1f}% and average {metric} "
        f"per record changed {intensity:+.1f}%."
    )


def _seasonality_finding(time_evidence, metric):
    """Return only material, repeated seasonality or a material calendar effect."""
    if not time_evidence:
        return None
    frame = time_evidence.get("comparison_frame")
    if frame is None or len(frame) < 24:
        return None

    frame = frame.copy()
    first_period = frame.index[0]
    first_date = frame.iloc[0]["FirstDate"]
    if first_date.day > 3:
        frame = frame.iloc[1:]
    if len(frame) < 24:
        return None

    frame["MonthNumber"] = frame.index.month
    frame["Year"] = frame.index.year
    frame["Days"] = frame.index.days_in_month
    frame["NormalizedValue"] = (
        frame["Value"] / frame["Days"]
        if time_evidence.get("aggregation") == "sum"
        else frame["Value"]
    )

    raw_summary = frame.groupby("MonthNumber")["Value"].agg(["mean", "count"])
    normalized_summary = frame.groupby("MonthNumber")["NormalizedValue"].agg(["mean", "count"])
    eligible = normalized_summary[normalized_summary["count"] >= 3]
    if len(eligible) < 2:
        return None

    raw_eligible = raw_summary.loc[eligible.index]
    raw_peak = int(raw_eligible["mean"].idxmax())
    raw_weak = int(raw_eligible["mean"].idxmin())
    raw_spread = _pct_change(raw_eligible.loc[raw_peak, "mean"], raw_eligible.loc[raw_weak, "mean"])
    normalized_raw_spread = _pct_change(
        normalized_summary.loc[raw_peak, "mean"],
        normalized_summary.loc[raw_weak, "mean"],
    )

    if (
        time_evidence.get("aggregation") == "sum"
        and raw_spread is not None
        and normalized_raw_spread is not None
        and raw_spread >= 10
        and normalized_raw_spread < 10
    ):
        occurrences = int(min(raw_eligible.loc[raw_peak, "count"], raw_eligible.loc[raw_weak, "count"]))
        return {
            "section": "Trends and Movement Drivers",
            "title": "Calendar length explains most of the apparent monthly peak",
            "evidence": (
                f"Raw monthly {metric} averaged {_format_value(raw_eligible.loc[raw_peak, 'mean'], metric)} "
                f"in {calendar.month_name[raw_peak]} versus "
                f"{_format_value(raw_eligible.loc[raw_weak, 'mean'], metric)} in "
                f"{calendar.month_name[raw_weak]} ({raw_spread:.1f}% higher). After dividing by "
                f"calendar days, the difference falls to {normalized_raw_spread:.1f}% across "
                f"{occurrences} comparable observations per month."
            ),
            "meaning": (
                "The raw monthly ranking is not strong evidence of seasonality; month length and "
                "record volume must be separated before operational planning."
            ),
            "type": "calendar_effect",
            "confidence": "High",
            "materiality": abs(raw_spread - normalized_raw_spread),
        }

    peak_month = int(eligible["mean"].idxmax())
    weak_month = int(eligible["mean"].idxmin())
    spread = _pct_change(eligible.loc[peak_month, "mean"], eligible.loc[weak_month, "mean"])
    paired = frame[frame["MonthNumber"].isin([peak_month, weak_month])].pivot_table(
        index="Year",
        columns="MonthNumber",
        values="NormalizedValue",
        aggfunc="mean",
    ).dropna()
    repeat_rate = (
        float((paired[peak_month] > paired[weak_month]).mean())
        if len(paired) >= 3 else 0.0
    )
    if spread is None or spread < 12 or repeat_rate < 0.67:
        return None

    unit = "average per calendar day" if time_evidence.get("aggregation") == "sum" else "monthly average"
    return {
        "section": "Trends and Movement Drivers",
        "title": f"A repeated {calendar.month_name[peak_month]} seasonal signal is visible",
        "evidence": (
            f"The {unit} for {metric} was {spread:.1f}% higher in "
            f"{calendar.month_name[peak_month]} than {calendar.month_name[weak_month]}, and the "
            f"relationship repeated in {repeat_rate:.0%} of {len(paired)} comparable years."
        ),
        "meaning": (
            "This is a repeatable descriptive pattern, but business events and operational "
            "context are still required before treating it as causal."
        ),
        "type": "seasonality",
        "peak_month": calendar.month_name[peak_month],
        "confidence": "Medium",
        "materiality": spread,
    }


def _segment_structure_finding(df, metric, dimension):
    if not metric or not dimension or dimension not in df.columns:
        return None
    work = pd.DataFrame({
        "Dimension": df[dimension],
        "Value": pd.to_numeric(df[metric], errors="coerce"),
    }).dropna()
    if work.empty or not 2 <= work["Dimension"].nunique() <= 50:
        return None

    grouped = work.groupby("Dimension").agg(
        Total=("Value", "sum"),
        Records=("Value", "size"),
        Average=("Value", "mean"),
    )
    min_records = max(5, int(len(work) * 0.005))
    reliable = grouped[grouped["Records"] >= min_records]
    if len(reliable) < 2:
        return None

    if _is_additive_metric(metric) and reliable["Total"].sum() > 0:
        shares = reliable["Total"] / reliable["Total"].sum() * 100
        ordered_shares = shares.sort_values(ascending=False)
        leader = ordered_shares.index[0]
        concentration = _normalized_hhi(reliable["Total"])
        if concentration >= 0.25:
            return {
                "section": "Segments and Concentration",
                "title": f"{metric} is materially concentrated in {leader}",
                "evidence": (
                    f"{leader} represents {ordered_shares.iloc[0]:.1f}% of {metric}; the "
                    f"normalized concentration score is {concentration:.2f} across "
                    f"{len(reliable)} reliable {dimension} groups."
                ),
                "meaning": "Performance depends disproportionately on one segment, creating dependency risk.",
                "type": "concentration",
                "leader": str(leader),
                "dimension": str(dimension),
                "confidence": "High",
                "materiality": ordered_shares.iloc[0],
            }

    ordered_average = reliable["Average"].sort_values(ascending=False)
    top_name, low_name = ordered_average.index[0], ordered_average.index[-1]
    average_gap = _pct_change(ordered_average.iloc[0], ordered_average.iloc[-1])
    if average_gap is not None and average_gap >= 10:
        return {
            "section": "Segments and Concentration",
            "title": f"Average {metric} differs materially across {dimension}",
            "evidence": (
                f"{top_name} averages {_format_value(ordered_average.iloc[0], metric)} per record, "
                f"versus {_format_value(ordered_average.iloc[-1], metric)} for {low_name} "
                f"({average_gap:.1f}% higher); both groups meet the minimum sample threshold."
            ),
            "meaning": "The difference is material, but mix and context should be checked before assigning causality.",
            "type": "segment_gap",
            "leader": str(top_name),
            "dimension": str(dimension),
            "confidence": "Medium",
            "materiality": average_gap,
        }

    if _is_additive_metric(metric):
        shares = reliable["Total"] / reliable["Total"].sum() * 100
        share_range = float(shares.max() - shares.min())
        average_range = float(
            (reliable["Average"].max() / reliable["Average"].min() - 1) * 100
        ) if reliable["Average"].min() != 0 else None
        if share_range <= 5 and average_range is not None and average_range < 5:
            return {
                "section": "Segments and Concentration",
                "title": f"{metric} is broadly balanced across {dimension}",
                "evidence": (
                    f"Group contributions range from {shares.min():.1f}% to {shares.max():.1f}%, "
                    f"while average {metric} per record differs by only {average_range:.1f}%."
                ),
                "meaning": "No segment is large or different enough to justify a concentration-risk claim.",
                "type": "balanced_segments",
                "dimension": str(dimension),
                "confidence": "High",
                "materiality": 5 - share_range,
            }
    return None


def _latest_segment_driver(df, date_col, metric, dimension, time_evidence):
    if (
        not time_evidence or time_evidence.get("growth") is None
        or not _is_additive_metric(metric) or not dimension
    ):
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[metric], errors="coerce")
    work = pd.DataFrame({"Date": dates, "Value": values, "Dimension": df[dimension]}).dropna()
    work["Period"] = work["Date"].dt.to_period("M")
    periods = [time_evidence["previous_period"], time_evidence["latest_period"]]
    cut = work[work["Period"].isin(periods)]
    if cut.empty:
        return None
    pivot = cut.pivot_table(
        index="Dimension",
        columns="Period",
        values="Value",
        aggfunc="sum",
        fill_value=0,
    )
    if not all(period in pivot.columns for period in periods):
        return None
    delta = pivot[periods[1]] - pivot[periods[0]]
    if delta.abs().sum() == 0:
        return None
    total_delta = float(delta.sum())
    same_direction = delta[delta * total_delta > 0] if total_delta != 0 else delta
    if same_direction.empty:
        return None
    driver = same_direction.abs().idxmax()
    driver_delta = float(delta.loc[driver])
    gross_share = abs(driver_delta) / float(delta.abs().sum()) * 100
    return {
        "section": "Key Drivers and Relationships",
        "title": f"{driver} was the largest observed {dimension} contributor to the latest movement",
        "evidence": (
            f"{driver}'s {metric} changed by {_format_value(driver_delta, metric)} from "
            f"{time_evidence['previous_label']} to {time_evidence['latest_label']}, representing "
            f"{gross_share:.1f}% of gross absolute movement across {dimension}."
        ),
        "meaning": "This identifies where the movement occurred; it does not by itself establish why it occurred.",
        "type": "segment_driver",
        "segment": str(driver),
        "dimension": str(dimension),
        "confidence": "High",
        "materiality": gross_share,
    }


def _generate_decision_report_legacy(
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
    recommendation_priority = {"High": 3, "Medium": 2, "Low": 1}
    recommendations = sorted(
        unique_recommendations,
        key=lambda item: recommendation_priority.get(item["priority"], 0),
        reverse=True,
    )[:max_recommendations]

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


def generate_decision_report(
    df: pd.DataFrame,
    quality_report: dict | None = None,
    max_findings: int = 7,
    max_recommendations: int = 5,
) -> dict:
    """Build a materiality-gated, driver-aware report for an arbitrary table."""
    rows, columns = df.shape
    date_col = detect_date_column(df)
    metrics = _business_metrics(df)
    dimensions = _important_dimensions(df, date_col)
    entity_id = _entity_identifier(df)

    preferred_metric = _named_column(
        df,
        ["revenue", "sales", "profit", "billing amount", "amount", "quantity", "units"],
    )
    if preferred_metric not in metrics:
        preferred_metric = metrics[0] if metrics else None
    primary_dimension = dimensions[0] if dimensions else None

    valid_dates = (
        pd.to_datetime(df[date_col], errors="coerce").dropna()
        if date_col else pd.Series(dtype="datetime64[ns]")
    )
    time_period = (
        f"{valid_dates.min().strftime('%d %B %Y')} to "
        f"{valid_dates.max().strftime('%d %B %Y')}"
        if not valid_dates.empty else "No reliable date range is available"
    )

    monetary_metrics = [
        metric for metric in metrics
        if any(word in _normal_name(metric) for word in (
            "sales", "revenue", "profit", "cost", "amount", "income", "price"
        ))
    ]
    overview = {
        "records": rows,
        "columns": columns,
        "date_column": date_col,
        "time_period": time_period,
        "dimensions": dimensions[:6],
        "metrics": metrics[:6],
        "grain": (
            f"One row per observed record; `{entity_id}` is the available identifier."
            if entity_id else
            "One row is treated as one record; no reliable entity identifier was detected."
        ),
        "note": (
            "Monetary currency is not specified; monetary values are shown in dataset units."
            if monetary_metrics else None
        ),
    }

    revenue = _named_column(df, ["revenue", "sales", "net sales"])
    profit = _named_column(df, ["profit", "net profit", "gross profit"])
    quantity = _named_column(df, ["quantity", "units sold", "units", "volume"])
    order_id = _named_column(df, ["order id", "order number", "transaction id", "invoice id"])

    # KPI summary: totals for additive measures, averages for intensity, and a
    # count whose basis is explicit. This avoids treating identifiers as KPIs.
    kpis = []
    added_metrics = set()
    for metric in (revenue, profit, quantity, preferred_metric):
        if metric not in metrics or metric in added_metrics:
            continue
        numeric = pd.to_numeric(df[metric], errors="coerce").dropna()
        if numeric.empty:
            continue
        if _is_additive_metric(metric):
            kpis.append({
                "name": f"Total {metric}",
                "value": _format_value(numeric.sum(), metric),
                "context": "Before any unapproved cleaning adjustments",
            })
            kpis.append({
                "name": f"Average {metric} per Record",
                "value": _format_value(numeric.mean(), metric),
                "context": f"Based on {len(numeric):,} non-null records",
            })
        else:
            kpis.append({
                "name": f"Average {metric}",
                "value": _format_value(numeric.mean(), metric),
                "context": f"Based on {len(numeric):,} non-null records",
            })
        added_metrics.add(metric)

    if revenue in metrics and profit in metrics and revenue != profit:
        total_revenue = pd.to_numeric(df[revenue], errors="coerce").sum()
        total_profit = pd.to_numeric(df[profit], errors="coerce").sum()
        if total_revenue:
            kpis.append({
                "name": "Profit Margin",
                "value": f"{total_profit / total_revenue * 100:.1f}%",
                "context": f"Total {profit} divided by total {revenue}",
            })

    if revenue in metrics and order_id in df.columns:
        distinct_orders = int(df[order_id].nunique(dropna=True))
        if distinct_orders:
            total_revenue = pd.to_numeric(df[revenue], errors="coerce").sum()
            kpis.append({
                "name": f"Average {revenue} per Order",
                "value": _format_value(total_revenue / distinct_orders, revenue),
                "context": f"Based on {distinct_orders:,} distinct {order_id} values",
            })

    time_evidence = _time_evidence(df, date_col, preferred_metric)
    if time_evidence and time_evidence.get("growth") is not None:
        kpis.append({
            "name": f"Latest Monthly {preferred_metric} Change",
            "value": f"{time_evidence['growth']:+.1f}%",
            "context": f"{time_evidence['latest_label']} versus {time_evidence['previous_label']}",
        })

    findings = []

    if time_evidence and time_evidence.get("growth") is not None:
        growth = float(time_evidence["growth"])
        yoy = time_evidence.get("year_over_year_growth")
        yoy_sentence = (
            f" It was {'up' if yoy >= 0 else 'down'} {abs(yoy):.1f}% from "
            f"{time_evidence['year_ago_label']}."
            if yoy is not None else ""
        )
        interpretation = _movement_interpretation(time_evidence, preferred_metric)
        if time_evidence.get("excluded_partial_period") is not None:
            interpretation += (
                f" {_format_month(time_evidence['excluded_partial_period'])} was excluded "
                "because the available dates cover less than 90% of that month."
            )
        findings.append({
            "section": "Trends and Movement Drivers",
            "title": (
                f"{time_evidence['latest_label']} {preferred_metric} "
                f"{'increased' if growth > 0 else 'decreased' if growth < 0 else 'was unchanged'} "
                f"{abs(growth):.1f}% month over month"
            ),
            "evidence": (
                f"{preferred_metric} was {_format_value(time_evidence['latest_value'], preferred_metric)} "
                f"versus {_format_value(time_evidence['previous_value'], preferred_metric)} in "
                f"{time_evidence['previous_label']} ({growth:+.1f}%).{yoy_sentence}"
            ),
            "meaning": interpretation,
            "type": "decline" if growth < -2 else "growth" if growth > 2 else "stable",
            "confidence": "High",
            "materiality": abs(growth),
            "average_growth": time_evidence.get("average_growth"),
            "record_growth": time_evidence.get("record_growth"),
        })

        seasonal = _seasonality_finding(time_evidence, preferred_metric)
        if seasonal:
            findings.append(seasonal)

        driver = _latest_segment_driver(
            df,
            date_col,
            preferred_metric,
            primary_dimension,
            time_evidence,
        )
        if driver:
            findings.append(driver)

    segment_finding = _segment_structure_finding(df, preferred_metric, primary_dimension)
    if segment_finding:
        findings.append(segment_finding)

    discount = _named_column(df, ["discount", "discount rate"])
    if discount in metrics and profit in metrics and discount != profit:
        pair = df[[discount, profit]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 30 and pair[discount].nunique() > 1 and pair[profit].nunique() > 1:
            correlation = float(pair[discount].corr(pair[profit]))
            if pd.notna(correlation) and abs(correlation) >= 0.20:
                strength = "strong" if abs(correlation) >= .7 else "moderate" if abs(correlation) >= .4 else "weak"
                findings.append({
                    "section": "Key Drivers and Relationships",
                    "title": f"{discount} and {profit} show a {strength} relationship",
                    "evidence": (
                        f"Pearson correlation is {correlation:.2f} across {len(pair):,} complete records."
                    ),
                    "meaning": "This is an association, not proof that changing one field will cause the other to change.",
                    "type": "negative_correlation" if correlation < 0 else "correlation",
                    "confidence": "Medium",
                    "materiality": abs(correlation) * 100,
                })

    # Negative values are handled according to the metric's meaning. Profit
    # can legitimately be negative; other additive measures need a business-
    # rule check for returns, credits, adjustments, or invalid records.
    if preferred_metric in metrics and _is_additive_metric(preferred_metric):
        primary_values = pd.to_numeric(df[preferred_metric], errors="coerce")
        negative_mask = primary_values < 0
        if negative_mask.any():
            negative_rows = int(negative_mask.sum())
            negative_total = float(primary_values[negative_mask].sum())
            is_profit = preferred_metric == profit
            findings.append({
                "section": "Anomalies and Data Trust",
                "title": "Loss-making records" if is_profit else f"Negative {preferred_metric} values require validation",
                "evidence": (
                    f"{negative_rows:,} records ({negative_rows / max(rows, 1) * 100:.2f}%) contain "
                    f"negative {preferred_metric}, totalling {_format_value(negative_total, preferred_metric)}."
                ),
                "meaning": (
                    "These records directly reduce reported profit and should be segmented by their business drivers."
                    if is_profit else
                    "Confirm whether these are valid returns, refunds, credits, or adjustments; otherwise they may be data errors."
                ),
                "type": "loss_risk" if is_profit else "negative_values",
                "confidence": "High",
                "materiality": negative_rows / max(rows, 1) * 100,
            })

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        duplicate_rate = duplicate_rows / max(rows, 1) * 100
        if preferred_metric in metrics and _is_additive_metric(preferred_metric):
            raw_total = pd.to_numeric(df[preferred_metric], errors="coerce").sum()
            deduplicated = df.drop_duplicates()
            clean_total = pd.to_numeric(deduplicated[preferred_metric], errors="coerce").sum()
            impact = float(raw_total - clean_total)
            impact_pct = abs(impact / clean_total * 100) if clean_total else None
            impact_text = (
                f" They change total {preferred_metric} by {_format_value(impact, preferred_metric)} "
                f"({impact_pct:.2f}% versus the deduplicated total)."
                if impact_pct is not None else ""
            )
        else:
            impact_pct = duplicate_rate
            impact_text = ""
        findings.append({
            "section": "Anomalies and Data Trust",
            "title": "Exact duplicates affect reported results",
            "evidence": (
                f"The dataset contains {duplicate_rows:,} duplicate copies ({duplicate_rate:.2f}% of rows)."
                f"{impact_text}"
            ),
            "meaning": "Approve a deduplication rule and rerun the report before using totals operationally.",
            "type": "duplicate_impact",
            "confidence": "High",
            "materiality": impact_pct,
        })

    if quality_report:
        missing_cells = int(quality_report.get("missing_cells") or 0)
        type_issues = quality_report.get("incorrect_types")
        type_issue_count = len(type_issues) if isinstance(type_issues, pd.DataFrame) else 0
        if missing_cells or type_issue_count:
            parts = []
            if missing_cells:
                parts.append(f"{missing_cells:,} missing cells")
            if type_issue_count:
                parts.append(f"{type_issue_count} likely type issues")
            findings.append({
                "section": "Anomalies and Data Trust",
                "title": "Completeness or type issues may affect analysis coverage",
                "evidence": "The quality scan found " + " and ".join(parts) + ".",
                "meaning": "Review affected KPI and dimension columns before interpreting comparisons.",
                "type": "data_quality",
                "confidence": "High",
                "materiality": missing_cells / max(rows * columns, 1) * 100,
            })

    category = _dimension_column(df, ["category", "segment", "product"])
    if revenue in metrics and profit in metrics and category:
        margin = df.groupby(category)[[revenue, profit]].sum().dropna()
        margin = margin[margin[revenue] != 0]
        if len(margin) >= 2:
            margin["Margin"] = margin[profit] / margin[revenue] * 100
            low, high = margin["Margin"].idxmin(), margin["Margin"].idxmax()
            gap = float(margin.loc[high, "Margin"] - margin.loc[low, "Margin"])
            if gap >= 5:
                findings.append({
                    "section": "Business Opportunities",
                    "title": "A material segment margin gap is visible",
                    "evidence": (
                        f"{low} has {margin.loc[low, 'Margin']:.1f}% margin versus "
                        f"{high} at {margin.loc[high, 'Margin']:.1f}% ({gap:.1f} percentage-point gap)."
                    ),
                    "meaning": "Pricing, discount, cost, and mix differences should be compared before pursuing more volume.",
                    "type": "low_margin",
                    "segment": str(low),
                    "confidence": "High",
                    "materiality": gap,
                })

    priority = {
        "duplicate_impact": 110,
        "loss_risk": 105,
        "negative_values": 100,
        "decline": 95,
        "low_margin": 92,
        "segment_driver": 90,
        "growth": 85,
        "concentration": 82,
        "data_quality": 80,
        "calendar_effect": 78,
        "seasonality": 75,
        "segment_gap": 70,
        "negative_correlation": 68,
        "stable": 50,
        "balanced_segments": 45,
        "correlation": 40,
    }
    section_order = {
        "Trends and Movement Drivers": 1,
        "Segments and Concentration": 2,
        "Key Drivers and Relationships": 3,
        "Anomalies and Data Trust": 4,
        "Business Opportunities": 5,
    }
    findings = sorted(
        sorted(
            findings,
            key=lambda item: (priority.get(item["type"], 0), item.get("materiality") or 0),
            reverse=True,
        )[:max_findings],
        key=lambda item: section_order.get(item["section"], 99),
    )

    recommendations = []
    for finding in findings:
        kind = finding["type"]
        if kind == "duplicate_impact":
            recommendations.append({
                "priority": "High",
                "action": "Approve and apply an exact-duplicate handling rule before publishing KPIs.",
                "evidence": finding["evidence"],
                "impact": "Removes a quantified source of KPI inflation or double counting.",
                "next_step": "Review duplicate pairs, confirm whether they are repeated loads or valid events, then regenerate the report.",
                "kpi": f"Deduplicated {preferred_metric}" if preferred_metric else "Duplicate row rate",
                "confidence": "High",
            })
        elif kind in ("negative_values", "loss_risk"):
            recommendations.append({
                "priority": "High" if kind == "loss_risk" else "Medium",
                "action": f"Define and validate the business rule for negative {preferred_metric} values.",
                "evidence": finding["evidence"],
                "impact": "Separates legitimate adjustments or losses from data errors and improves KPI trust.",
                "next_step": f"Classify negative {preferred_metric} records by reason and report them separately from ordinary activity.",
                "kpi": f"Negative-{preferred_metric} record rate",
                "confidence": "High",
            })
        elif kind in ("growth", "decline"):
            average_growth = finding.get("average_growth")
            if average_growth is not None and average_growth < -2:
                action = f"Investigate the decline in average {preferred_metric} per record before treating total growth as improvement."
            else:
                action = f"Validate the latest {preferred_metric} movement by segment and confirm whether it persists."
            recommendations.append({
                "priority": "High" if kind == "decline" else "Medium",
                "action": action,
                "evidence": finding["evidence"],
                "impact": "Distinguishes volume, value, and mix effects before changing forecasts or operations.",
                "next_step": f"Compare record count and average {preferred_metric} per record by {primary_dimension or 'the strongest available dimension'}.",
                "kpi": f"Monthly total and average {preferred_metric} per record",
                "confidence": "High",
            })
        elif kind == "segment_driver":
            recommendations.append({
                "priority": "Medium",
                "action": f"Investigate the business events and mix behind the change in {finding.get('segment')}.",
                "evidence": finding["evidence"],
                "impact": "Focuses follow-up on the segment where the verified movement was largest.",
                "next_step": f"Compare volume and average {preferred_metric} per record for {finding.get('segment')} across the two periods.",
                "kpi": f"{finding.get('segment')} monthly {preferred_metric}",
                "confidence": "Medium",
            })
        elif kind == "concentration":
            recommendations.append({
                "priority": "Medium",
                "action": f"Assess dependency on {finding.get('leader')} and define an acceptable concentration threshold.",
                "evidence": finding["evidence"],
                "impact": "Makes concentration risk measurable instead of assuming that every leading segment is risky.",
                "next_step": "Compare concentration over time and test downside scenarios for the leading segment.",
                "kpi": "Top-segment share and normalized concentration score",
                "confidence": "High",
            })
        elif kind == "low_margin":
            recommendations.append({
                "priority": "High",
                "action": f"Review price, discount, cost, and mix for {finding.get('segment')}.",
                "evidence": finding["evidence"],
                "impact": "Targets a measured margin gap rather than pursuing volume without profitability context.",
                "next_step": "Reconcile segment revenue and profit, then compare unit economics with the highest-margin segment.",
                "kpi": "Segment profit margin",
                "confidence": "High",
            })
        elif kind == "seasonality":
            recommendations.append({
                "priority": "Low",
                "action": "Validate the repeated seasonal signal against business events before using it for planning.",
                "evidence": finding["evidence"],
                "impact": "Prevents calendar patterns from being mistaken for causal demand signals.",
                "next_step": "Compare the pattern by year and material segment, then document known operational causes.",
                "kpi": f"Calendar-normalized monthly {preferred_metric}",
                "confidence": "Medium",
            })
        elif kind == "data_quality":
            recommendations.append({
                "priority": "Medium",
                "action": "Resolve quality issues in fields used by the report before operational use.",
                "evidence": finding["evidence"],
                "impact": "Improves coverage and prevents avoidable calculation bias.",
                "next_step": "Review the Data Quality page and record every accepted cleaning rule.",
                "kpi": "Dataset health score",
                "confidence": "High",
            })

    unique_recommendations = []
    seen_actions = set()
    for item in recommendations:
        if item["action"] not in seen_actions:
            seen_actions.add(item["action"])
            unique_recommendations.append(item)
    recommendation_priority = {"High": 3, "Medium": 2, "Low": 1}
    recommendations = sorted(
        unique_recommendations,
        key=lambda item: recommendation_priority.get(item["priority"], 0),
        reverse=True,
    )[:max_recommendations]
    if not recommendations:
        recommendations = [{
            "priority": "Low",
            "action": "Add a target, benchmark, and business decision before requesting prescriptive recommendations.",
            "evidence": "No material risk, driver, or opportunity passed the automatic evidence thresholds.",
            "impact": "Prevents generic advice from being presented as a data-backed action.",
            "next_step": "Define the outcome, comparison baseline, and decision owner, then regenerate the report.",
            "kpi": preferred_metric or "Selected decision KPI",
            "confidence": "High",
        }]

    limitations = []
    if not entity_id:
        limitations.append(
            "No reliable entity identifier was detected. Counts represent dataset rows and must not be described as unique customers, patients, orders, or other entities."
        )
    if monetary_metrics:
        limitations.append(
            "Currency and accounting meaning are not supplied. Amounts must not automatically be described as collected revenue, cash, or profit."
        )
    high_cardinality = []
    for column in df.select_dtypes(exclude="number").columns:
        unique = int(df[column].nunique(dropna=True))
        if rows and unique >= 50 and unique / rows >= 0.50:
            high_cardinality.append(str(column))
    if high_cardinality:
        limitations.append(
            "High-cardinality fields were excluded from automatic leader claims without stronger identifiers or minimum sample support: "
            + ", ".join(high_cardinality[:5]) + "."
        )
    limitations.append(
        "No external target, benchmark, causal context, or operational constraint was supplied; recommendations are prioritized hypotheses, not proven interventions."
    )

    questions = []
    if time_evidence and primary_dimension:
        questions.append(f"Which {primary_dimension} values explain the latest change in {preferred_metric}, after separating record volume from average value?")
    if duplicate_rows:
        questions.append("How materially do KPIs and rankings change after the approved deduplication rule?")
    if preferred_metric in metrics and (pd.to_numeric(df[preferred_metric], errors="coerce") < 0).any():
        questions.append(f"What business events explain negative {preferred_metric} values?")
    if revenue in metrics and profit in metrics and primary_dimension:
        questions.append(f"Which {primary_dimension} groups combine scale with sustainable profit margin?")
    if date_col:
        questions.append("Does the calendar-normalized pattern repeat by year and material segment?")
    questions = list(dict.fromkeys(questions))[:5]

    return {
        "overview": overview,
        "kpis": kpis[:6],
        "findings": findings,
        "recommendations": recommendations,
        "limitations": limitations,
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
    if overview.get("grain"):
        lines.append(f"- **Counting basis:** {overview['grain']}")
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
            lines.append(f"  **Interpretation:** {finding['meaning']}")
        if finding.get("confidence"):
            lines.append(f"  **Confidence:** {finding['confidence']}")

    lines.extend([
        "", "## 4. Recommendations & Next Steps", "",
        "| Priority | Recommended action | Supporting evidence | Expected impact | Specific next step | KPI to monitor | Confidence |",
        "|---|---|---|---|---|---|---|",
    ])
    for item in report["recommendations"]:
        cells = [
            item["priority"], item["action"], item["evidence"], item["impact"],
            item["next_step"], item["kpi"], item.get("confidence", "Medium"),
        ]
        cells = [str(cell).replace("|", "/").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## 5. Limitations and Trust Notes", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.extend(["", "## 6. Questions for Further Analysis", ""])
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
