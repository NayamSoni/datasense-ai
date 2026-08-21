import pandas as pd
import altair as alt
import streamlit as st


LABEL_COLOR = "#f8fafc"
GRID_COLOR = "#2a3556"
BAR_COLOR = "#7c5cff"
LINE_COLOR = "#38bdf8"


# ======================================================
# Derived / helper column detection
# ======================================================

_DERIVED_EXACT_NAMES = {"Rank", "Running Total", "ABC"}


def _is_derived_column(col):

    if col in _DERIVED_EXACT_NAMES:
        return True

    if "%" in str(col):
        return True

    return False


# ======================================================
# Shape Detection
# ======================================================

def _is_pivot_shaped(plan):

    real_group_by = [
        c for c in (plan.get("group_by") or [])
        if c != "_time"
    ]

    if plan.get("analysis_type") in ("mom", "yoy"):
        return bool(real_group_by)

    return bool(plan.get("pivot"))


# ======================================================
# Pivot-shaped charts (mom / yoy with a dimension, or pivot=true)
# ======================================================

def _render_pivot_chart(result, group_by, chart_type):

    dim_cols = [c for c in group_by if c in result.columns]

    id_col = dim_cols[0] if dim_cols else result.columns[0]

    value_cols = [
        c for c in result.columns
        if c != id_col
        and not _is_derived_column(c)
        and pd.api.types.is_numeric_dtype(result[c])
    ]

    if len(value_cols) < 2:
        return

    melted = result.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="Period",
        value_name="Value"
    )

    tooltip = [id_col, "Period", "Value"]

    if chart_type == "heatmap":

        chart = alt.Chart(melted).mark_rect().encode(
            x=alt.X("Period:N", title=None),
            y=alt.Y(f"{id_col}:N", title=None),
            color=alt.Color("Value:Q"),
            tooltip=tooltip
        )

    elif chart_type == "area":

        chart = alt.Chart(melted).mark_area().encode(
            x=alt.X("Period:N", title=None),
            y=alt.Y("Value:Q"),
            color=alt.Color(f"{id_col}:N", title=None),
            tooltip=tooltip
        )

    elif chart_type == "bar":

        bars = alt.Chart(melted).mark_bar().encode(
            x=alt.X("Period:N", title=None),
            y=alt.Y("Value:Q"),
            color=alt.Color(f"{id_col}:N", title=None),
            tooltip=tooltip
        )

        labels = alt.Chart(melted).mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            color=LABEL_COLOR,
            fontSize=11,
        ).encode(
            x=alt.X("Period:N", title=None),
            y=alt.Y("Value:Q"),
            detail=f"{id_col}:N",
            text=alt.Text("Value:Q", format=",.0f")
        )

        chart = bars + labels

    else:

        base = alt.Chart(melted)
        lines = base.mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Period:N", title=None),
            y=alt.Y("Value:Q"),
            color=alt.Color(f"{id_col}:N", title=None),
            tooltip=tooltip
        )

        if len(melted) <= 24:
            labels = base.mark_text(
                dy=-9,
                color=LABEL_COLOR,
                fontSize=10,
            ).encode(
                x=alt.X("Period:N", title=None),
                y=alt.Y("Value:Q"),
                detail=f"{id_col}:N",
                text=alt.Text("Value:Q", format=",.0f"),
            )
            chart = lines + labels
        else:
            chart = lines

    st.altair_chart(chart, use_container_width=True)


# ======================================================
# Distribution charts (Range / Count table)
# ======================================================

def _render_distribution_chart(result, measure=None):

    base = alt.Chart(result)
    bucket_title = f"{measure} bucket" if measure else "Range"
    bars = base.mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Range:N", sort=None, title=bucket_title),
        y=alt.Y("Count:Q", title="Record count"),
        tooltip=["Range", "Count"] + (
            ["Percent"] if "Percent" in result.columns else []
        )
    )

    labels = base.mark_text(
        dy=-6,
        color=LABEL_COLOR,
        fontSize=11,
    ).encode(
        x=alt.X("Range:N", sort=None, title=bucket_title),
        y=alt.Y("Count:Q", title="Record count"),
        text=alt.Text("Count:Q", format=",.0f"),
    )

    chart = bars + labels

    st.altair_chart(chart, use_container_width=True)


# ======================================================
# Simple shape charts (aggregation / top_bottom / time_series /
# pareto / mom-yoy-without-a-dimension)
# ======================================================

def _pick_axes(result, plan):

    group_by = plan.get("group_by") or []
    measure = plan.get("measure")

    if "_time" in result.columns:
        x_col = "_time"
    elif "Month" in result.columns:
        x_col = "Month"
    elif "Year" in result.columns:
        x_col = "Year"
    elif group_by and group_by[0] in result.columns:
        x_col = group_by[0]
    else:
        non_numeric = [
            c for c in result.columns
            if not pd.api.types.is_numeric_dtype(result[c])
        ]
        x_col = non_numeric[0] if non_numeric else None

    if measure in result.columns:
        y_col = measure
    else:
        numeric_cols = [
            c for c in result.columns
            if pd.api.types.is_numeric_dtype(result[c])
            and not _is_derived_column(c)
        ]
        y_col = numeric_cols[0] if numeric_cols else None

    return x_col, y_col


def _render_simple_chart(result, plan, chart_type):

    x_col, y_col = _pick_axes(result, plan)

    if x_col is None or y_col is None or x_col == y_col:
        return

    data = result[[x_col, y_col]].copy()

    tooltip = [x_col, y_col]

    if chart_type == "pie":

        chart = alt.Chart(data).mark_arc().encode(
            theta=alt.Theta(f"{y_col}:Q"),
            color=alt.Color(f"{x_col}:N", title=None),
            tooltip=tooltip
        )

    elif chart_type == "line":

        base = alt.Chart(data)
        line = base.mark_line(
            point=alt.OverlayMarkDef(color=LINE_COLOR, filled=True, size=70),
            color=LINE_COLOR,
            strokeWidth=3,
        ).encode(
            x=alt.X(f"{x_col}:N", sort=None, title=None),
            y=alt.Y(f"{y_col}:Q"),
            tooltip=tooltip
        )

        if len(data) <= 24:
            labels = base.mark_text(
                dy=-10,
                color=LABEL_COLOR,
                fontSize=11,
            ).encode(
                x=alt.X(f"{x_col}:N", sort=None, title=None),
                y=alt.Y(f"{y_col}:Q"),
                text=alt.Text(f"{y_col}:Q", format=",.0f"),
            )
            chart = line + labels
        else:
            chart = line

    elif chart_type == "area":

        chart = alt.Chart(data).mark_area().encode(
            x=alt.X(f"{x_col}:N", sort=None, title=None),
            y=alt.Y(f"{y_col}:Q"),
            tooltip=tooltip
        )

    elif chart_type == "scatter":

        chart = alt.Chart(data).mark_circle(size=80).encode(
            x=alt.X(f"{x_col}:N", title=None),
            y=alt.Y(f"{y_col}:Q"),
            tooltip=tooltip
        )

    elif chart_type == "histogram":

        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{y_col}:Q", bin=True),
            y="count()"
        )

    else:
        base = alt.Chart(data)
        long_categories = (
            len(data) > 6
            or data[x_col].astype(str).str.len().max() > 14
        )

        if long_categories:
            bars = base.mark_bar(
                color=BAR_COLOR,
                cornerRadiusEnd=4,
            ).encode(
                y=alt.Y(
                    f"{x_col}:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=300),
                ),
                x=alt.X(f"{y_col}:Q", title=y_col),
                tooltip=tooltip,
            )
            labels = base.mark_text(
                align="left",
                baseline="middle",
                dx=6,
                color=LABEL_COLOR,
                fontSize=11,
            ).encode(
                y=alt.Y(f"{x_col}:N", sort="-x", title=None),
                x=alt.X(f"{y_col}:Q"),
                text=alt.Text(f"{y_col}:Q", format=",.0f"),
            )
        else:
            bars = base.mark_bar(
                color=BAR_COLOR,
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
            ).encode(
                x=alt.X(f"{x_col}:N", sort="-y", title=None),
                y=alt.Y(f"{y_col}:Q"),
                tooltip=tooltip,
            )
            labels = base.mark_text(
                align="center",
                baseline="bottom",
                dy=-6,
                color=LABEL_COLOR,
                fontSize=11,
            ).encode(
                x=alt.X(f"{x_col}:N", sort="-y", title=None),
                y=alt.Y(f"{y_col}:Q"),
                text=alt.Text(f"{y_col}:Q", format=",.0f"),
            )

        chart = bars + labels

    st.altair_chart(chart, use_container_width=True)


# ======================================================
# True rotatable 3D bars (Plotly Mesh3d)
# ======================================================

def _render_3d_bar(result, dimension, measure, color_by=None, show_install_error=True):
    """Render each category value as a real 3D cuboid."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        if show_install_error:
            st.error(
                "3D charts require Plotly. Run `pip install plotly`, then restart Streamlit."
            )
        return False

    data = result.dropna(subset=[dimension, measure]).copy()

    if data.empty:
        st.info("No data is available for this 3D chart.")
        return False

    x_values = list(dict.fromkeys(data[dimension].astype(str)))
    x_lookup = {value: index for index, value in enumerate(x_values)}

    has_depth = bool(color_by) and color_by in data.columns
    depth_values = (
        list(dict.fromkeys(data[color_by].astype(str))) if has_depth else ["All"]
    )
    depth_lookup = {value: index for index, value in enumerate(depth_values)}

    palette = [
        "#7c5cff", "#22d3ee", "#a78bfa", "#38bdf8", "#f472b6",
        "#34d399", "#fb7185", "#60a5fa", "#c084fc", "#2dd4bf",
    ]
    faces_i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    faces_j = [2, 3, 5, 6, 1, 5, 7, 6, 4, 7, 2, 6]
    faces_k = [1, 2, 6, 7, 5, 4, 6, 2, 7, 3, 6, 5]

    figure = go.Figure()
    label_x, label_y, label_z, label_text = [], [], [], []
    max_abs_value = max(abs(float(value)) for value in data[measure]) or 1.0
    label_offset = max_abs_value * 0.025

    for row_number, (_, row) in enumerate(data.iterrows()):
        category = str(row[dimension])
        depth = str(row[color_by]) if has_depth else "All"
        value = float(row[measure])

        x_center = x_lookup[category]
        y_center = depth_lookup[depth]
        x0, x1 = x_center - 0.31, x_center + 0.31
        y0, y1 = y_center - 0.20, y_center + 0.20
        z0, z1 = (value, 0.0) if value < 0 else (0.0, value)

        vertices_x = [x0, x1, x1, x0, x0, x1, x1, x0]
        vertices_y = [y0, y0, y1, y1, y0, y0, y1, y1]
        vertices_z = [z0, z0, z0, z0, z1, z1, z1, z1]
        color_index = depth_lookup[depth] if has_depth else row_number

        hover_lines = [
            f"{dimension}: {category}",
            f"{measure}: {value:,.2f}",
        ]
        if has_depth:
            hover_lines.insert(1, f"{color_by}: {depth}")

        figure.add_trace(
            go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=faces_i,
                j=faces_j,
                k=faces_k,
                color=palette[color_index % len(palette)],
                opacity=0.94,
                flatshading=False,
                lighting={
                    "ambient": 0.62,
                    "diffuse": 0.72,
                    "specular": 0.24,
                    "roughness": 0.56,
                    "fresnel": 0.10,
                },
                lightposition={"x": 120, "y": 80, "z": 180},
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                name=category,
                showlegend=False,
            )
        )

        label_x.append(x_center)
        label_y.append(y_center)
        label_z.append(value + label_offset if value >= 0 else value - label_offset)
        label_text.append(f"{value:,.0f}")

    if len(data) <= 20:
        figure.add_trace(
            go.Scatter3d(
                x=label_x,
                y=label_y,
                z=label_z,
                mode="text",
                text=label_text,
                textfont={"color": "#eef2ff", "size": 11},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    y_axis = {
        "title": color_by if has_depth else "",
        "tickvals": list(range(len(depth_values))) if has_depth else [],
        "ticktext": depth_values if has_depth else [],
        "showticklabels": has_depth,
        "showgrid": has_depth,
        "backgroundcolor": "rgba(8,12,27,0.82)",
        "gridcolor": "rgba(129,112,210,0.18)",
        "zerolinecolor": "rgba(255,255,255,0.15)",
    }

    figure.update_layout(
        height=540,
        margin={"l": 0, "r": 0, "t": 25, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbe3f7", "family": "Inter"},
        scene={
            "bgcolor": "rgba(6,10,24,0.92)",
            "camera": {
                "eye": {"x": 1.35, "y": -2.05, "z": 1.20},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "dragmode": "turntable",
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(1.6, min(3.2, len(x_values) / 3.2)),
                "y": max(0.42, min(1.4, len(depth_values) / 2.5)),
                "z": 1.12,
            },
            "xaxis": {
                "title": dimension,
                "tickvals": list(range(len(x_values))),
                "ticktext": [
                    value if len(value) <= 18 else value[:17] + "â€¦"
                    for value in x_values
                ],
                "backgroundcolor": "rgba(8,12,27,0.82)",
                "gridcolor": "rgba(129,112,210,0.18)",
                "zerolinecolor": "rgba(255,255,255,0.15)",
            },
            "yaxis": y_axis,
            "zaxis": {
                "title": measure,
                "backgroundcolor": "rgba(8,12,27,0.82)",
                "gridcolor": "rgba(129,112,210,0.20)",
                "zerolinecolor": "rgba(255,255,255,0.18)",
            },
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    return True


def _render_correlation_scatter(data, plan):
    """Plot the observation-level relationship behind a correlation result."""
    measure = plan.get("measure")
    measure2 = plan.get("measure2")

    if not measure or not measure2 or measure not in data.columns or measure2 not in data.columns:
        return False

    clean = data[[measure, measure2]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        return False

    # Plotting every point in a large dataset makes the browser slow without
    # improving the visible relationship. Sampling is only for the chart; the
    # correlation table continues to use the complete dataset.
    plotted = clean.sample(5000, random_state=42) if len(clean) > 5000 else clean
    base = alt.Chart(plotted)

    points = base.mark_circle(
        size=54,
        opacity=0.38,
        color="#38bdf8",
    ).encode(
        x=alt.X(f"{measure}:Q", title=measure),
        y=alt.Y(f"{measure2}:Q", title=measure2),
        tooltip=[
            alt.Tooltip(f"{measure}:Q", format=",.2f"),
            alt.Tooltip(f"{measure2}:Q", format=",.2f"),
        ],
    )

    trend = base.transform_regression(measure, measure2).mark_line(
        color="#f59e0b",
        strokeWidth=3,
    ).encode(
        x=alt.X(f"{measure}:Q"),
        y=alt.Y(f"{measure2}:Q"),
    )

    chart = (points + trend).properties(height=460)
    st.altair_chart(chart, use_container_width=True)
    return True


def _render_categorical_relationship_box(data, plan):
    """Show a numeric distribution across an unordered category."""
    measure = plan.get("measure")
    category = next(
        (column for column in (plan.get("group_by") or []) if column in data.columns),
        None,
    )
    if not measure or measure not in data.columns or not category:
        return False

    clean = data[[category, measure]].copy()
    clean[measure] = pd.to_numeric(clean[measure], errors="coerce")
    clean = clean.dropna()
    if len(clean) < 3 or clean[category].nunique() < 2:
        return False

    plotted = clean.sample(10000, random_state=42) if len(clean) > 10000 else clean
    chart = (
        alt.Chart(plotted)
        .mark_boxplot(
            extent=1.5,
            color=BAR_COLOR,
            median={"color": "#f8fafc"},
        )
        .encode(
            x=alt.X(f"{category}:N", title=category, sort=None),
            y=alt.Y(f"{measure}:Q", title=measure, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip(f"{category}:N", title=category)],
        )
        .properties(height=460)
    )
    st.altair_chart(chart, use_container_width=True)
    return True


# ======================================================
# Public Entry Point (used by the AI chat / execution-plan flow)
# ======================================================

def render_chart(result, plan):

    if not isinstance(result, pd.DataFrame) or result.empty:
        return

    chart_type = (plan.get("chart") or "table").lower()

    if chart_type == "table":
        return

    if chart_type in ("3d bar", "3d_bar"):
        x_col, y_col = _pick_axes(result, plan)
        if x_col and y_col:
            _render_3d_bar(result, x_col, y_col)
        return

    analysis_type = plan.get("analysis_type", "aggregation")
    group_by = plan.get("group_by") or []

    if analysis_type == "correlation":
        if chart_type == "scatter":
            _render_correlation_scatter(result, plan)
        return

    if analysis_type == "categorical_relationship":
        if chart_type == "box":
            _render_categorical_relationship_box(result, plan)
        return

    if chart_type in ("box", "treemap"):
        return

    try:

        if _is_pivot_shaped(plan):
            _render_pivot_chart(result, group_by, chart_type)
            return

        if analysis_type == "distribution" and {"Range", "Count"}.issubset(result.columns):
            _render_distribution_chart(result, plan.get("measure"))
            return

        _render_simple_chart(result, plan, chart_type)

    except Exception:
        return


# ======================================================
# Manual Chart Builder (used by the Visualisation page)
# ======================================================

_AGG_FUNCS = {
    "Sum": "sum",
    "Average": "mean",
    "Count": "count",
    "Min": "min",
    "Max": "max",
    "Median": "median"
}


def render_manual_chart(df, dimension, measure, operation, chart_type, color_by=None, top_n=None):

    agg_func = _AGG_FUNCS.get(operation, "sum")

    has_color = bool(color_by) and color_by != "None"

    group_cols = [dimension, color_by] if has_color else [dimension]

    if agg_func == "count":
        result = df.groupby(group_cols, dropna=False)[measure].count().reset_index()
    else:
        result = df.groupby(group_cols, dropna=False)[measure].agg(agg_func).reset_index()

    if top_n:
        result = result.sort_values(measure, ascending=False).head(top_n)

    chart_type_lower = chart_type.lower()

    if chart_type_lower == "3d bar":
        _render_3d_bar(
            result,
            dimension,
            measure,
            color_by=color_by if has_color else None,
        )
        return result

    tooltip = group_cols + [measure]

    color_enc = alt.Color(f"{color_by}:N", title=None) if has_color else alt.value("#8FCBFA")

    if chart_type_lower == "bar":
        base = alt.Chart(result)
        long_categories = (
            len(result) > 6
            or result[dimension].astype(str).str.len().max() > 14
        )

        if long_categories and not has_color:
            bars = base.mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y(
                    f"{dimension}:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=320),
                ),
                x=alt.X(f"{measure}:Q", title=measure),
                color=color_enc,
                tooltip=tooltip,
            )
        else:
            bars = base.mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
            ).encode(
                x=alt.X(f"{dimension}:N", sort="-y", title=None),
                y=alt.Y(f"{measure}:Q"),
                color=color_enc,
                tooltip=tooltip,
            )

        if has_color:

            chart = bars

        else:

            if long_categories:
                labels = base.mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color=LABEL_COLOR,
                    fontSize=11,
                ).encode(
                    y=alt.Y(f"{dimension}:N", sort="-x", title=None),
                    x=alt.X(f"{measure}:Q"),
                    text=alt.Text(f"{measure}:Q", format=",.0f"),
                )
            else:
                labels = base.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-6,
                    color=LABEL_COLOR,
                    fontSize=11,
                ).encode(
                    x=alt.X(f"{dimension}:N", sort="-y", title=None),
                    y=alt.Y(f"{measure}:Q"),
                    text=alt.Text(f"{measure}:Q", format=",.0f"),
                )

            chart = bars + labels

    elif chart_type_lower == "line":
        base = alt.Chart(result)
        line = base.mark_line(
            point=alt.OverlayMarkDef(filled=True, size=70),
            strokeWidth=3,
        ).encode(
            x=alt.X(f"{dimension}:N", sort=None, title=None),
            y=alt.Y(f"{measure}:Q"),
            color=color_enc,
            tooltip=tooltip
        )

        if not has_color and len(result) <= 24:
            labels = base.mark_text(
                dy=-10,
                color=LABEL_COLOR,
                fontSize=11,
            ).encode(
                x=alt.X(f"{dimension}:N", sort=None, title=None),
                y=alt.Y(f"{measure}:Q"),
                text=alt.Text(f"{measure}:Q", format=",.0f"),
            )
            chart = line + labels
        else:
            chart = line

    elif chart_type_lower == "area":

        chart = alt.Chart(result).mark_area().encode(
            x=alt.X(f"{dimension}:N", sort=None, title=None),
            y=alt.Y(f"{measure}:Q"),
            color=color_enc,
            tooltip=tooltip
        )

    elif chart_type_lower == "pie":

        chart = alt.Chart(result).mark_arc().encode(
            theta=alt.Theta(f"{measure}:Q"),
            color=alt.Color(f"{dimension}:N", title=None),
            tooltip=tooltip
        )

    elif chart_type_lower == "scatter":

        chart = alt.Chart(result).mark_circle(size=80).encode(
            x=alt.X(f"{dimension}:N", title=None),
            y=alt.Y(f"{measure}:Q"),
            color=color_enc,
            tooltip=tooltip
        )

    else:

        chart = alt.Chart(result).mark_bar().encode(
            x=alt.X(f"{dimension}:N", sort="-y", title=None),
            y=alt.Y(f"{measure}:Q"),
            tooltip=tooltip
        )

    st.altair_chart(chart, use_container_width=True)

    return result


# ======================================================
# Decision report charts
# ======================================================

def render_report_chart(spec):
    """Render one evidence-backed chart used by the Insights report."""
    data = spec.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.info("This chart does not have enough supported data to display.")
        return

    chart_type = spec.get("type")
    title = spec.get("title", "Report chart")

    if chart_type == "line":
        x, y = spec["x"], spec["y"]
        base = alt.Chart(data).encode(
            x=alt.X(f"{x}:O", sort=None, title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y(f"{y}:Q", title=spec.get("y_title", y)),
            tooltip=[alt.Tooltip(f"{x}:O"), alt.Tooltip(f"{y}:Q", format=",.2f")],
        )
        line = base.mark_line(
            color=LINE_COLOR,
            strokeWidth=3,
            point=alt.OverlayMarkDef(color="#a78bfa", filled=True, size=64),
        )
        if len(data) <= 18:
            labels = base.mark_text(
                dy=-12,
                color=LABEL_COLOR,
                fontSize=11,
            ).encode(text=alt.Text(f"{y}:Q", format="~s"))
            chart = line + labels
        else:
            chart = line

    elif chart_type == "bar":
        x, y = spec["x"], spec["y"]
        base = alt.Chart(data)
        bars = base.mark_bar(
            color=BAR_COLOR,
            cornerRadiusEnd=5,
        ).encode(
            y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340)),
            x=alt.X(f"{x}:Q", title=spec.get("x_title", x)),
            tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=",.2f")],
        )
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=7,
            color=LABEL_COLOR,
            fontSize=11,
        ).encode(
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            x=alt.X(f"{x}:Q"),
            text=alt.Text(f"{x}:Q", format="~s"),
        )
        chart = bars + labels

    elif chart_type == "scatter":
        x, y = spec["x"], spec["y"]
        points = alt.Chart(data).mark_circle(
            color="#38bdf8",
            opacity=0.42,
            size=55,
        ).encode(
            x=alt.X(f"{x}:Q", title=spec.get("x_title", x)),
            y=alt.Y(f"{y}:Q", title=spec.get("y_title", y)),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title=spec.get("x_title", x), format=",.3f"),
                alt.Tooltip(f"{y}:Q", title=spec.get("y_title", y), format=",.2f"),
            ],
        )
        trend = points.transform_regression(x, y).mark_line(
            color="#f472b6",
            strokeWidth=3,
        )
        chart = points + trend

    else:
        st.info("This chart type is not available in the report workspace.")
        return

    chart = (
        chart.properties(title=title, height=360)
        .configure_title(color=LABEL_COLOR, fontSize=18, anchor="start", offset=18)
        .configure_axis(
            labelColor=LABEL_COLOR,
            titleColor=LABEL_COLOR,
            gridColor=GRID_COLOR,
            domainColor=GRID_COLOR,
            tickColor=GRID_COLOR,
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)
