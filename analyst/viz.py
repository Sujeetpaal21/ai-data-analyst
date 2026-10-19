"""Intelligent chart recommendation + construction (Plotly).

Data preparation (``prepare_chart_data``) is pure pandas and independent from Plotly, so the
aggregation logic is testable without a rendering library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .analytics import ALLOWED_AGGS, auto_freq, norm_freq, numeric_series, period_start
from .profiling import column_kind
from .util import AnalystError, ensure_datetime, is_datetime, resolve_column

CHART_TYPES = ["auto", "bar", "grouped_bar", "stacked_bar", "line", "area", "histogram", "box", "scatter", "bubble",
               "pie", "donut", "heatmap", "correlation_heatmap", "kpi", "table"]
AGG_TYPES = {"bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut"}
MAX_POINTS = 20_000


@dataclass
class ChartResult:
    figure: Any
    data: pd.DataFrame
    chart_type: str
    reason: str
    notes: list[str] = field(default_factory=list)
    title: str = ""


def _kind(df, c):
    if c is None:
        return None
    k = column_kind(df[c])
    return "datetime" if is_datetime(df[c]) else k


def recommend_chart(df: pd.DataFrame, x: str | None = None, y: str | None = None, color: str | None = None,
                    size: str | None = None, intent: str | None = None) -> tuple[str, str]:
    """Return (chart_type, plain-language reason) based on column types and intent."""
    intent = (intent or "").lower()
    kx, ky = _kind(df, x), _kind(df, y)
    if "correl" in intent:
        return "correlation_heatmap", "A correlation heatmap shows the relationship between all numeric columns at once."
    if kx == "datetime" and ky == "numeric":
        return "line", "A line chart is best for a numeric measure over time because it shows trend and seasonality."
    if kx == "datetime" and y is None:
        return "line", "Counting records per time period as a line shows activity over time."
    if kx in {"categorical", "boolean"} and ky == "numeric":
        n = df[x].nunique()
        if any(w in intent for w in ("share", "proportion", "percent", "composition")) and n <= 6:
            return "donut", f"A donut chart shows each of the {n} categories' share of the total."
        if color and column_kind(df[color]) in {"categorical", "boolean"}:
            return "grouped_bar", "A grouped bar chart compares a measure across two categorical dimensions."
        return "bar", "A bar chart compares a numeric measure across categories; bars start at zero so lengths are comparable."
    if kx == "numeric" and ky == "numeric":
        return ("bubble", "A bubble chart adds a third numeric dimension (size) to the scatter plot.") if size else \
               ("scatter", "A scatter plot shows the relationship between two numeric variables and reveals clusters/outliers.")
    if kx == "numeric" and y is None:
        return "histogram", "A histogram shows the distribution (shape, spread, skew) of one numeric column."
    if kx in {"categorical", "boolean"} and ky in {"categorical", "boolean"}:
        return "heatmap", "A heatmap of counts shows how two categorical columns co-occur."
    if kx in {"categorical", "boolean"} and y is None:
        return "bar", "A bar chart of counts shows the frequency of each category."
    if kx in {"categorical", "boolean"} and ky == "numeric":
        return "box", "A box plot compares the distribution of a number across groups."
    if ky == "numeric" and x is None:
        return "histogram", "A histogram shows the distribution of one numeric column."
    return "table", "No clear chart fits this combination of columns, so a table is shown."


def prepare_chart_data(df: pd.DataFrame, chart_type: str, x: str | None, y: str | None, color: str | None = None,
                       size: str | None = None, agg: str = "sum", top_n: int | None = None, freq: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Aggregate/limit data for a chart. Returns (data, meta) where meta has x, y, color, notes."""
    notes: list[str] = []
    agg = (agg or "sum").lower()
    if agg not in ALLOWED_AGGS:
        raise AnalystError(f"Aggregation '{agg}' not allowed.")
    meta = {"x": x, "y": y, "color": color, "size": size, "agg": agg, "notes": notes}
    if chart_type in AGG_TYPES:
        if x is None:
            raise AnalystError("Choose an X column (category or date).")
        work = df.copy()
        keys = [x] + ([color] if color and color != x else [])
        if is_datetime(work[x]) or column_kind(work[x]) == "datetime":
            d = ensure_datetime(work[x])
            f = norm_freq(freq) or auto_freq(d)
            work[x] = period_start(d, f)
            meta["freq"] = f
            notes.append(f"Dates aggregated by period ({f}).")
        if y is None or agg == "count" and y in (None, "*"):
            data = work.groupby(keys, observed=True, dropna=True).size().reset_index(name="count")
            meta.update(y="count", agg="count")
        else:
            work[y] = numeric_series(work[y])
            if agg in {"sum", "mean", "median", "std", "min", "max"} and work[y].notna().sum() == 0:
                raise AnalystError(f"'{y}' is not numeric; cannot {agg} it.")
            name = f"{agg}_{y}"
            data = work.groupby(keys, observed=True, dropna=True)[y].agg(agg).reset_index(name=name)
            meta["y"] = name
        yv = meta["y"]
        if x in data.columns and not is_datetime(data[x]):
            limit = top_n or (8 if chart_type in {"pie", "donut"} else 20)
            order = data.groupby(x)[yv].sum().sort_values(ascending=False)
            if len(order) > limit:
                keep = order.head(limit).index
                if chart_type in {"pie", "donut"}:
                    other = data[~data[x].isin(keep)][yv].sum()
                    data = pd.concat([data[data[x].isin(keep)], pd.DataFrame({x: ["Other"], yv: [other]})], ignore_index=True)
                    notes.append(f"Smallest {len(order) - limit} categories grouped as 'Other'.")
                else:
                    data = data[data[x].isin(keep)]
                    notes.append(f"Showing top {limit} of {len(order)} categories.")
            data = data.assign(_o=data[x].map(order)).sort_values("_o", ascending=False).drop(columns="_o") if chart_type != "line" else data.sort_values(x)
        else:
            data = data.sort_values(x)
        if chart_type in {"pie", "donut"} and (data[yv] < 0).any():
            raise AnalystError("Pie/donut charts cannot show negative values; use a bar chart.")
        return data.reset_index(drop=True), meta
    if chart_type in {"histogram", "box", "scatter", "bubble"}:
        cols = [c for c in {x, y, color, size} if c]
        data = df[cols].copy()
        for c in (x, y, size):
            if c and column_kind(df[c]) == "numeric":
                data[c] = numeric_series(df[c])
        if size:
            data[size] = data[size].clip(lower=0)
        if len(data) > MAX_POINTS:
            data = data.sample(MAX_POINTS, random_state=0)
            notes.append(f"Plotted a random sample of {MAX_POINTS:,} of {len(df):,} rows for responsiveness.")
        return data.dropna(subset=[c for c in (x, y) if c]), meta
    if chart_type == "heatmap":
        if not x or not y:
            raise AnalystError("A heatmap needs two columns (x and y).")
        a, b = df[x], df[y]
        for col, ser in ((x, a), (y, b)):
            if ser.nunique() > 30:
                raise AnalystError(f"'{col}' has {ser.nunique()} categories; a heatmap needs at most 30 per axis.")
        return pd.crosstab(b, a), meta
    return df, meta


def _px():
    try:
        import plotly.express as px
        return px
    except ImportError as e:  # pragma: no cover
        raise AnalystError("Plotly is not installed. Run: pip install -r requirements.txt") from e


def build_chart(df: pd.DataFrame, chart_type: str = "auto", x: str | None = None, y: str | None = None, color: str | None = None,
                size: str | None = None, agg: str = "sum", top_n: int | None = None, freq: str | None = None,
                title: str | None = None, intent: str | None = None) -> ChartResult:
    x = resolve_column(df, x) if x else None
    y = resolve_column(df, y) if y and y != "*" else None
    color = resolve_column(df, color) if color else None
    size = resolve_column(df, size) if size else None
    ct = (chart_type or "auto").lower()
    if ct not in CHART_TYPES:
        raise AnalystError(f"Unknown chart type '{chart_type}'. Options: {CHART_TYPES}")
    if x is None and y is not None and ct in {"auto", "histogram", "box"}:
        x, y = (None, y) if ct == "box" else (y, None)
    reason = "Chart type chosen by the user."
    if ct == "auto":
        ct, reason = recommend_chart(df, x, y, color, size, intent)
    if ct == "correlation_heatmap":
        return _corr_heatmap(df, reason)
    if ct == "kpi":
        return _kpi(df, y or x, agg, reason)
    if ct == "table":
        data = df[[c for c in (x, y, color) if c] or list(df.columns)[:8]].head(200)
        go = _go()
        fig = go.Figure(go.Table(header=dict(values=list(data.columns)), cells=dict(values=[data[c] for c in data.columns])))
        return ChartResult(fig, data, "table", reason, [], title or "Data table")
    if ct == "grouped_bar" and not color:
        ct, reason = "bar", reason + " (No second dimension chosen, so a simple bar chart is used.)"
    data, meta = prepare_chart_data(df, ct, x, y, color, size, agg, top_n, freq)
    notes, yv = meta["notes"], meta["y"]
    px = _px()
    kw: dict[str, Any] = {}
    if ct in {"bar", "grouped_bar", "stacked_bar"}:
        horizontal = (not is_datetime(data[x])) and data[x].nunique() > 12
        if horizontal:
            fig = px.bar(data, x=yv, y=x, color=color, orientation="h", barmode="stack" if ct == "stacked_bar" else "group")
            fig.update_yaxes(categoryorder="total ascending")
        else:
            fig = px.bar(data, x=x, y=yv, color=color, barmode="stack" if ct == "stacked_bar" else "group")
    elif ct == "line":
        fig = px.line(data, x=x, y=yv, color=color, markers=True)
    elif ct == "area":
        fig = px.area(data, x=x, y=yv, color=color)
    elif ct in {"pie", "donut"}:
        fig = px.pie(data, names=x, values=yv, hole=0.45 if ct == "donut" else 0)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif ct == "histogram":
        if x is None:
            raise AnalystError("Choose a numeric column for the histogram.")
        fig = px.histogram(data, x=x, color=color, marginal="box")
    elif ct == "box":
        fig = px.box(data, x=x if x and y else None, y=y or x, color=color, points="outliers")
    elif ct in {"scatter", "bubble"}:
        if not (x and y):
            raise AnalystError("Scatter charts need both X and Y numeric columns.")
        fig = px.scatter(data, x=x, y=y, color=color, size=size if ct == "bubble" else None, opacity=0.7)
        sub = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 3 and sub[x].nunique() > 1 and not color:
            m, b = np.polyfit(sub[x], sub[y], 1)
            xs = np.array([sub[x].min(), sub[x].max()])
            fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Linear fit", line=dict(dash="dash"))
    elif ct == "heatmap":
        fig = px.imshow(data, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig.update_xaxes(title=x)
        fig.update_yaxes(title=y)
    else:
        raise AnalystError(f"Chart type '{ct}' could not be built.")
    auto_title = title or _auto_title(ct, x, y, yv, meta)
    fig.update_layout(title=auto_title, legend_title_text=color or "", margin=dict(l=40, r=20, t=60, b=40), hovermode="closest")
    if ct in AGG_TYPES - {"pie", "donut"} and x:
        pass
    return ChartResult(fig, data, ct, reason, notes, auto_title)


def _auto_title(ct, x, y, yv, meta) -> str:
    if ct == "histogram":
        return f"Distribution of {x}"
    if ct == "heatmap":
        return f"{x} vs {y} (record counts)"
    if ct in {"scatter", "bubble"}:
        return f"{y} vs {x}"
    if ct == "box":
        return f"Distribution of {y or x}" + (f" by {x}" if x and y else "")
    if yv == "count":
        return f"Number of records by {x}"
    return f"{meta['agg'].title()} of {y} by {x}"


def _go():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:  # pragma: no cover
        raise AnalystError("Plotly is not installed. Run: pip install -r requirements.txt") from e


def _corr_heatmap(df: pd.DataFrame, reason: str) -> ChartResult:
    from .analytics import correlation_analysis

    res = correlation_analysis(df)
    px = _px()
    fig = px.imshow(res["matrix"], zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
    fig.update_layout(title="Correlation matrix (" + res["method"] + ")", margin=dict(l=40, r=20, t=60, b=40))
    return ChartResult(fig, res["matrix"].reset_index().rename(columns={"index": "column"}), "correlation_heatmap", reason,
                       ["Correlation does not imply causation."], "Correlation matrix")


def _kpi(df: pd.DataFrame, col: str | None, agg: str, reason: str) -> ChartResult:
    go = _go()
    if col is None:
        val, label = float(len(df)), "Records"
    else:
        s = numeric_series(df[col])
        val, label = float(s.agg(agg if agg in ALLOWED_AGGS else "sum")), f"{agg} of {col}"
    fig = go.Figure(go.Indicator(mode="number", value=val, title={"text": label}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=60, b=20))
    return ChartResult(fig, pd.DataFrame({"kpi": [label], "value": [val]}), "kpi", reason, [], label)


def fig_to_html(fig, title: str = "chart") -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displaylogo": False, "toImageButtonOptions": {"filename": title}})
