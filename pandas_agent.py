import numpy as np
import pandas as pd

print("### PANDAS_AGENT.PY LOADED — VERSION 2026-08-19-v12 (mixed-type relationships) ###")

from utils import detect_date_column


# ======================================================
# Statistical Summary
# ======================================================

def statistical_summary(df):

    response = []

    response.append("## 📊 Dataset Summary")
    response.append(f"- Rows : {len(df):,}")
    response.append(f"- Columns : {len(df.columns)}")

    numeric = df.select_dtypes(include="number")

    if len(numeric.columns):

        response.append("\n## 📈 Numeric Columns")

        summary = numeric.describe().T

        for col in summary.index:

            response.append(f"\n### {col}")
            response.append(f"Count : {summary.loc[col,'count']:.0f}")
            response.append(f"Mean : {summary.loc[col,'mean']:.2f}")
            response.append(f"Min : {summary.loc[col,'min']:.2f}")
            response.append(f"Max : {summary.loc[col,'max']:.2f}")
            response.append(f"Std : {summary.loc[col,'std']:.2f}")

    return "\n".join(response)


# ======================================================
# Aggregations
# ======================================================

AGGREGATIONS = {

    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "nunique": "nunique"

}


# ======================================================
# Whole-Number Display Formatting
# ======================================================

def _round_whole(series):

    return series.round(0).astype("Int64")


def _finalize_measure_columns(result, value_columns):

    result = result.copy()

    for col in value_columns:

        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):

            result[col] = _round_whole(result[col])

    return result


# ======================================================
# Filters
# ======================================================

def apply_filters(df, filters):

    data = df.copy()

    for f in filters:

        column = f.get("column")
        value = f.get("value")

        if column not in data.columns:
            continue

        if value is None:
            continue

        data = data[
            data[column]
            .astype(str)
            .str.strip()
            .str.casefold()
            ==
            str(value).strip().casefold()
        ]

    return data


# ======================================================
# Time Intelligence
# ======================================================

def apply_time_granularity(df, granularity, year_filter=None, date_column=None):
    """
    FIX (root cause of the "Sample Size" / row-count bug): this
    previously parsed the date column and dropped every row with
    an unparseable date UNCONDITIONALLY, before ever checking
    whether the current analysis needed date handling at all. A
    correlation between Discount and Profit has nothing to do with
    dates, but was still losing ~60% of its rows because of this —
    any analysis with no time_granularity and no year_filter was
    silently being put through date parsing/dropping for no reason
    connected to the question asked. Now the date column is only
    touched at all if it's actually going to be used.
    """

    if granularity is None and not year_filter:
        return df, []

    # A confirmed learned correction can explicitly select the date field.
    # Otherwise keep the existing automatic date detection behaviour.
    if date_column not in df.columns:
        date_column = detect_date_column(df)

    if date_column is None:

        return df, []

    data = df.copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data = data.dropna(subset=[date_column])

    if year_filter:

        data = data[data[date_column].dt.year.isin(year_filter)]

    if granularity is None:

        return data, []

    if granularity == "year":

        data["_time"] = data[date_column].dt.year

    elif granularity == "quarter":

        data["_time"] = data[date_column].dt.to_period("Q").astype(str)

    elif granularity == "month":

        data["_time"] = data[date_column].dt.to_period("M").astype(str)

    elif granularity == "week":

        data["_time"] = data[date_column].dt.isocalendar().week

    elif granularity == "day":

        data["_time"] = data[date_column].dt.date

    return data, ["_time"]


# ======================================================
# Aggregate
# ======================================================

def aggregate(df, operation, measure, group_by):

    agg = AGGREGATIONS.get(operation)

    if agg is None:

        raise ValueError(f"Unsupported operation: {operation}")

    if (
        agg not in ("count", "nunique")
        and not pd.api.types.is_numeric_dtype(df[measure])
    ):
        raise ValueError(
            f"'{operation}' requires a numeric measure, but '{measure}' is text. "
            "Use count/nunique or select a numeric column."
        )

    if len(group_by) == 0:

        return pd.DataFrame({

            measure: [

                getattr(df[measure], agg)()

            ]

        })

    return (

        df

        .groupby(

            group_by,

            dropna=False

        )[measure]

        .agg(agg)

        .reset_index()

    )

# ======================================================
# Sorting
# ======================================================

def sort_result(result, measure, sort):

    if sort == "desc":

        return result.sort_values(
            by=measure,
            ascending=False,
            ignore_index=True
        )

    elif sort == "asc":

        return result.sort_values(
            by=measure,
            ascending=True,
            ignore_index=True
        )

    return result


# ======================================================
# Ranking
# ======================================================

def add_rank(result):

    result = result.copy()

    result.insert(

        0,

        "Rank",

        range(1, len(result) + 1)

    )

    return result


# ======================================================
# Contribution %
# ======================================================

def add_contribution(result, measure):

    result = result.copy()

    total = result[measure].sum()

    if total == 0:

        result["Contribution %"] = 0

    else:

        result["Contribution %"] = (

            result[measure]

            / total

            * 100

        ).round(2)

    return result


# ======================================================
# Running Total
# ======================================================

def add_running_total(result, measure):

    result = result.copy()

    result["Running Total"] = (

        result[measure]

        .cumsum()

    )

    return result


# ======================================================
# Business Metrics
# ======================================================

def add_business_metrics(result, measure):

    result = add_rank(result)

    result = add_contribution(result, measure)

    return result


# ======================================================
# Average Order Value
# ======================================================

def average_order_value_analysis(data, sales_column, order_column, group_by):
    """Calculate SUM(Sales) / COUNT(DISTINCT Order ID)."""
    if order_column not in data.columns:
        return f"Column '{order_column}' not found. An Order ID column is required."

    working = data.dropna(subset=[order_column]).copy()
    working[sales_column] = pd.to_numeric(working[sales_column], errors="coerce")
    working = working.dropna(subset=[sales_column])

    if working.empty:
        return "No valid Sales and Order ID values are available to calculate Average Order Value."

    if group_by:
        grouped = working.groupby(group_by, dropna=False)
        result = grouped[sales_column].sum().rename("Total Sales").to_frame()
        result["Distinct Orders"] = grouped[order_column].nunique(dropna=True)
        result = result.reset_index()
    else:
        result = pd.DataFrame({
            "Total Sales": [working[sales_column].sum()],
            "Distinct Orders": [working[order_column].nunique(dropna=True)],
        })

    valid_orders = result["Distinct Orders"].replace(0, np.nan)
    result["Average Order Value"] = (
        result["Total Sales"] / valid_orders
    ).round(2)
    result["Total Sales"] = result["Total Sales"].round(2)
    result["Distinct Orders"] = result["Distinct Orders"].astype("Int64")

    return result


# ======================================================
# Top / Bottom Analysis
# ======================================================

def top_bottom_analysis(

    data,

    operation,

    measure,

    group_by,

    sort,

    limit

):

    result = aggregate(

        data,

        operation,

        measure,

        group_by

    )

    result = sort_result(

        result,

        measure,

        sort

    )

    # Calculate share against every category before applying Top/Bottom N.
    # Otherwise the displayed subset would incorrectly add up to 100%.
    result = add_contribution(result, measure)

    if limit == 1 and not result.empty:

        # A superlative can have more than one valid winner. Keep every tied
        # entity instead of silently presenting the first row as unique.
        winning_value = result.iloc[0][measure]
        result = result[result[measure].eq(winning_value)]

    elif limit:

        result = result.head(limit)

    result = add_rank(result)

    result = _finalize_measure_columns(result, [measure])

    return result.reset_index(drop=True)

# ======================================================
# Time Series Analysis
# ======================================================

def time_series_analysis(

    data,

    operation,

    measure,

    group_by

):

    result = aggregate(

        data,

        operation,

        measure,

        group_by

    )

    if "_time" in result.columns:

        result = result.sort_values(

            "_time"

        )

    result = _finalize_measure_columns(result, [measure])

    return result.reset_index(drop=True)


# ======================================================
# Pivot Analysis
# ======================================================

def pivot_analysis(

    data,

    operation,

    measure,

    group_by

):

    if "_time" not in data.columns:

        raise ValueError(

            "No time column available for pivot."

        )

    dimensions = [

        c

        for c in group_by

        if c != "_time"

    ]

    if len(dimensions) == 0:

        data = data.copy()
        data["_all"] = "All"
        dimensions = ["_all"]

    result = (

        data

        .pivot_table(

            index=dimensions,

            columns="_time",

            values=measure,

            aggfunc=AGGREGATIONS.get(operation, "sum"),

            fill_value=0

        )

        .reset_index()

    )

    if "_all" in result.columns:

        result = result.drop(columns=["_all"])

    result.columns.name = None

    value_columns = [c for c in result.columns if c not in dimensions]

    result = _finalize_measure_columns(result, value_columns)

    return result


# ======================================================
# Month over Month Analysis
# ======================================================

def mom_analysis(

    data,

    operation,

    measure,

    group_by

):

    dimensions = [c for c in group_by if c != "_time"]

    if len(dimensions) == 0:

        result = aggregate(

            data,

            operation,

            measure,

            ["_time"]

        )

        result = result.sort_values("_time").reset_index(drop=True)

        result = result.rename(columns={"_time": "Month"})

        result["MoM % (vs previous month)"] = (

            result[measure]

            .pct_change()

            * 100

        ).round(2)

        result = _finalize_measure_columns(result, [measure])

        return result

    result = pivot_analysis(

        data,

        operation,

        measure,

        group_by

    )

    month_columns = [

        c

        for c in result.columns

        if c not in group_by

    ]

    if len(month_columns) >= 2:

        current = month_columns[-1]

        previous = month_columns[-2]

        result["MoM % (latest vs previous month)"] = (

            (

                result[current]

                - result[previous]

            )

            /

            result[previous]

            .replace(0, np.nan)

            * 100

        ).round(2)

    return result

# ======================================================
# Year over Year Analysis
# ======================================================

def yoy_analysis(

    data,

    operation,

    measure,

    group_by

):

    dimensions = [c for c in group_by if c != "_time"]

    if len(dimensions) == 0:

        result = aggregate(

            data,

            operation,

            measure,

            ["_time"]

        )

        result = result.sort_values("_time").reset_index(drop=True)

        result = result.rename(columns={"_time": "Year"})

        result["YoY % (vs previous year)"] = (

            result[measure]

            .pct_change()

            * 100

        ).round(2)

        result = _finalize_measure_columns(result, [measure])

        return result

    result = pivot_analysis(

        data,

        operation,

        measure,

        group_by

    )

    year_columns = [

        c

        for c in result.columns

        if c not in group_by

    ]

    if len(year_columns) >= 2:

        current = year_columns[-1]

        previous = year_columns[-2]

        result["YoY % (latest vs previous year)"] = (

            (

                result[current]

                - result[previous]

            )

            /

            result[previous]

            .replace(0, np.nan)

            * 100

        ).round(2)

    return result


# ======================================================
# Correlation Analysis
# ======================================================

def _correlation_strength(corr):

    value = abs(corr)

    if value >= 0.7:
        return "Strong"
    elif value >= 0.4:
        return "Moderate"
    elif value >= 0.2:
        return "Weak"

    return "Very Weak / None"


def correlation_analysis(data, measure, measure2):

    if measure2 is None or measure2 not in data.columns:

        return "A second numeric column is required to compute a correlation."

    if measure == measure2:

        return "Correlation requires two different numeric columns."

    # Build two explicitly named Series so even unusual duplicate dataset
    # headers cannot turn clean[measure] into a DataFrame.
    series_a = data.loc[:, measure]
    series_b = data.loc[:, measure2]

    if isinstance(series_a, pd.DataFrame):
        series_a = series_a.iloc[:, 0]
    if isinstance(series_b, pd.DataFrame):
        series_b = series_b.iloc[:, 0]

    clean = pd.DataFrame({
        "Metric A Values": pd.to_numeric(series_a, errors="coerce"),
        "Metric B Values": pd.to_numeric(series_b, errors="coerce"),
    }).dropna()

    if len(clean) < 3:

        return (
            "Correlation requires two numeric columns. If one field is a "
            "category, compare the numeric distribution across its groups instead."
        )

    corr = clean["Metric A Values"].corr(clean["Metric B Values"])

    direction = "Positive" if corr >= 0 else "Negative"

    return pd.DataFrame({

        "Metric A": [measure],

        "Metric B": [measure2],

        "Correlation": [round(corr, 3)],

        "Direction": [direction],

        "Strength": [_correlation_strength(corr)],

        "Sample Size": [len(clean)]

    })


# ======================================================
# Numeric-to-categorical relationship analysis
# ======================================================

def _eta_squared_strength(value):
    """Describe eta-squared using conventional descriptive thresholds."""
    if value >= 0.14:
        return "Large"
    if value >= 0.06:
        return "Moderate"
    if value >= 0.01:
        return "Small"
    return "Negligible"


def categorical_relationship_analysis(data, measure, category):
    """Compare a numeric measure across groups and report eta-squared.

    Pearson correlation is undefined for an unordered category. Eta-squared
    measures the share of numeric variance associated with differences between
    the category-group means, while the group table keeps the calculation
    inspectable.
    """
    if category is None or category not in data.columns:
        return "A categorical column is required for this relationship analysis."

    clean = pd.DataFrame({
        category: data[category],
        measure: pd.to_numeric(data[measure], errors="coerce"),
    }).dropna()

    if len(clean) < 3 or clean[category].nunique() < 2:
        return "Not enough valid groups to compare this numeric-to-category relationship."

    grouped = (
        clean.groupby(category, dropna=False)[measure]
        .agg(Records="size", **{
            f"Average {measure}": "mean",
            f"Median {measure}": "median",
            f"Std Dev {measure}": "std",
        })
        .reset_index()
    )

    overall_average = float(clean[measure].mean())
    total_sum_squares = float(((clean[measure] - overall_average) ** 2).sum())
    between_sum_squares = 0.0
    average_column = f"Average {measure}"
    for _, row in grouped.iterrows():
        between_sum_squares += float(row["Records"]) * (
            float(row[average_column]) - overall_average
        ) ** 2
    eta_squared = (
        between_sum_squares / total_sum_squares
        if total_sum_squares > 0 else 0.0
    )

    grouped["Difference from Overall %"] = (
        (grouped[average_column] / overall_average - 1) * 100
        if overall_average != 0 else 0.0
    )
    grouped["Association (η²)"] = round(eta_squared, 6)
    grouped["Association Strength"] = _eta_squared_strength(eta_squared)
    grouped = grouped.sort_values(average_column, ascending=False)

    numeric_columns = [
        average_column,
        f"Median {measure}",
        f"Std Dev {measure}",
        "Difference from Overall %",
    ]
    grouped[numeric_columns] = grouped[numeric_columns].round(2)
    return grouped.reset_index(drop=True)


# ======================================================
# Outlier Analysis (IQR method)
# ======================================================

def outlier_analysis(data, measure):

    series = pd.to_numeric(data[measure], errors="coerce")

    valid = series.dropna()

    if len(valid) < 4:

        return "Not enough data to detect outliers."

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (series < lower_bound) | (series > upper_bound)

    result = data[mask.fillna(False)].copy()

    if result.empty:

        return f"No outliers detected in '{measure}' (IQR method, 1.5x bounds)."

    result = result.sort_values(measure, ascending=False)

    return result.reset_index(drop=True)


# ======================================================
# Distribution Analysis
# ======================================================

def distribution_analysis(data, measure, bins=10):

    series = pd.to_numeric(data[measure], errors="coerce").dropna()

    if series.empty:

        return f"No numeric data available in '{measure}' to build a distribution."

    normalized_measure = str(measure).lower().replace("_", " ").strip()
    is_age = "age" in normalized_measure.split()

    if is_age:
        # Human-friendly decade buckets are more useful for ages than the
        # long floating-point ranges produced by generic equal-width bins.
        lower = int(np.floor(series.min() / 10) * 10)
        upper = int(np.ceil((series.max() + 1) / 10) * 10)
        edges = np.arange(lower, upper + 1, 10)
        if len(edges) < 2:
            edges = np.array([lower, lower + 10])
        labels = [
            f"{int(left)}–{int(right - 1)}"
            for left, right in zip(edges[:-1], edges[1:])
        ]
        binned = pd.cut(
            series,
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )
    else:
        binned = pd.cut(series, bins=bins, duplicates="drop")

    counts = binned.value_counts().sort_index()

    result = pd.DataFrame({

        "Range": counts.index.astype(str),

        "Count": counts.values

    })

    result["Percent"] = (

        result["Count"] / result["Count"].sum() * 100

    ).round(2)

    return result


# ======================================================
# Pareto Analysis
# ======================================================

def pareto_analysis(

    data,

    measure,

    group_by

):

    result = aggregate(

        data,

        "sum",

        measure,

        group_by

    )

    result = result.sort_values(

        measure,

        ascending=False

    ).reset_index(drop=True)

    total = result[measure].sum()

    contribution = result[measure] / total * 100

    result["Contribution %"] = contribution.round(2)

    result["Cumulative %"] = (

        contribution

        .cumsum()

    ).round(2)

    result["ABC"] = np.where(

        result["Cumulative %"] <= 80,

        "A",

        np.where(

            result["Cumulative %"] <= 95,

            "B",

            "C"

        )

    )

    result = _finalize_measure_columns(result, [measure])

    return result


# ======================================================
# Main Calculation Engine
# ======================================================

def build_calculation_audit(df, plan):
    """Describe the deterministic calculation path shown to the user."""
    source_rows = len(df)
    filters = plan.get("filters", [])
    data = apply_filters(df, filters)
    rows_after_filters = len(data)

    data, time_group = apply_time_granularity(
        data,
        plan.get("time_granularity"),
        year_filter=plan.get("year_filter"),
        date_column=plan.get("date_column"),
    )

    analysis = plan.get("analysis_type", "aggregation")
    measure = plan.get("measure")
    operation_key = str(plan.get("operation", "sum")).lower()
    count_basis = None

    if analysis == "patient_count":
        patient_column = plan.get("patient_column")
        if patient_column in data.columns:
            valid_measure_rows = int(data[patient_column].notna().sum())
            formula = f"COUNT(DISTINCT {patient_column})"
            count_basis = f"Distinct {patient_column}"
        else:
            valid_measure_rows = int(len(data))
            formula = "COUNT(admission rows)"
            count_basis = "Admission rows (no patient identifier found)"
    elif analysis == "distribution":
        valid_measure_rows = (
            int(pd.to_numeric(data[measure], errors="coerce").notna().sum())
            if measure in data.columns else 0
        )
        formula = f"COUNT(records) grouped into {measure} buckets"
        count_basis = f"Rows with a non-null numeric {measure}"
    elif analysis == "categorical_relationship":
        category = next(
            (column for column in plan.get("group_by", []) if column in data.columns),
            None,
        )
        if category:
            joint_valid = (
                data[category].notna()
                & pd.to_numeric(data[measure], errors="coerce").notna()
            )
            valid_measure_rows = int(joint_valid.sum())
            formula = f"GROUP_MEAN_MEDIAN({measure}) + ETA_SQUARED BY {category}"
            count_basis = f"Rows with non-null {category} and numeric {measure}"
        else:
            valid_measure_rows = 0
            formula = f"CATEGORICAL_RELATIONSHIP({measure})"
            count_basis = "No valid categorical field"
    else:
        if measure in data.columns:
            if operation_key in ("count", "nunique"):
                valid_measure_rows = int(data[measure].notna().sum())
            else:
                valid_measure_rows = int(
                    pd.to_numeric(data[measure], errors="coerce").notna().sum()
                )
        else:
            valid_measure_rows = 0

        operation = operation_key.upper()
        formula = f"{operation}({measure})"
    group_by = [
        column
        for column in (time_group + plan.get("group_by", []))
        if column in data.columns
    ]
    if analysis == "distribution" and measure in data.columns:
        distribution_result = distribution_analysis(data, measure)
        groups_evaluated = (
            int(len(distribution_result))
            if isinstance(distribution_result, pd.DataFrame)
            else 0
        )
    else:
        groups_evaluated = (
            int(data[group_by].drop_duplicates().shape[0])
            if group_by
            else 1
        )

    if group_by and analysis != "categorical_relationship":
        formula += f" grouped by {', '.join(map(str, group_by))}"

    filter_text = " AND ".join(
        f"{item.get('column')} = {item.get('value')}"
        for item in filters
    ) or "None"

    return {
        "source_rows": source_rows,
        "rows_after_filters": rows_after_filters,
        "valid_measure_rows": valid_measure_rows,
        "groups_evaluated": groups_evaluated,
        "formula": formula,
        "count_basis": count_basis,
        "filters": filter_text,
        "sort": plan.get("sort") or "Not applied",
        "limit": plan.get("limit"),
    }

def calculate(df, plan):

    if plan.get("no_data_message"):

        return plan["no_data_message"]

    data = df.copy()

    operation = plan.get("operation", "sum")

    measure = plan.get("measure")

    group_by = plan.get("group_by", [])

    filters = plan.get("filters", [])

    sort = plan.get("sort")

    limit = plan.get("limit")

    time = plan.get("time_granularity")

    analysis = plan.get("analysis_type", "aggregation")

    pivot = plan.get("pivot", False)

    year_filter = plan.get("year_filter")

    date_column = plan.get("date_column")

    if analysis != "patient_count" and measure not in data.columns:

        return f"Column '{measure}' not found."

    data = apply_filters(

        data,

        filters

    )

    data, time_group = apply_time_granularity(

        data,

        time,

        year_filter=year_filter,

        date_column=date_column

    )

    if time_group:

        date_col = date_column if date_column in data.columns else detect_date_column(data)

        group_by = [

            c

            for c in group_by

            if c != date_col

        ]

        group_by = time_group + group_by

    if analysis == "patient_count":
        patient_column = plan.get("patient_column")
        if patient_column in data.columns:
            value = data[patient_column].nunique(dropna=True)
            label = plan.get("metric_label") or "Distinct Patients"
        else:
            value = len(data)
            label = plan.get("metric_label") or "Patient Admission Records"

        return pd.DataFrame({label: [int(value)]})

    if analysis == "average_order_value":

        return average_order_value_analysis(

            data,

            measure,

            plan.get("order_column"),

            group_by

        )

    if analysis == "top_bottom":

        return top_bottom_analysis(

            data,

            operation,

            measure,

            group_by,

            sort,

            limit

        )

    if analysis == "time_series":

        return time_series_analysis(

            data,

            operation,

            measure,

            group_by

        )

    if analysis == "mom":

        return mom_analysis(

            data,

            operation,

            measure,

            group_by

        )

    if analysis == "yoy":

        return yoy_analysis(

            data,

            operation,

            measure,

            group_by

        )

    if analysis == "pareto":

        return pareto_analysis(

            data,

            measure,

            group_by

        )

    if analysis == "correlation":

        return correlation_analysis(

            data,

            measure,

            plan.get("measure2")

        )

    if analysis == "categorical_relationship":

        category = next((column for column in group_by if column in data.columns), None)

        return categorical_relationship_analysis(

            data,

            measure,

            category

        )

    if analysis == "outlier":

        return outlier_analysis(

            data,

            measure

        )

    if analysis == "distribution":

        return distribution_analysis(

            data,

            measure

        )

    # A pivot requires a derived time axis. If an upstream planner ever sends
    # pivot=True without time context, safely execute the requested ordinary
    # category aggregation instead of raising a user-facing exception.
    if pivot and "_time" in data.columns:

        return pivot_analysis(

            data,

            operation,

            measure,

            group_by

        )

    result = aggregate(

        data,

        operation,

        measure,

        group_by

    )

    result = sort_result(

        result,

        measure,

        sort

    )

    # Calculate contribution against the complete breakdown before applying
    # Top/Bottom N, then rank only the rows that will be displayed.
    if group_by:

        result = add_contribution(result, measure)

    if limit:

        result = result.head(limit)

    # Rank and contribution make an ordinary category breakdown easier to
    # interpret. Running totals are reserved for explicit Pareto analysis,
    # where the engine returns Cumulative % instead.
    if group_by:

        result = add_rank(result)

    result = _finalize_measure_columns(result, [measure])

    return result
