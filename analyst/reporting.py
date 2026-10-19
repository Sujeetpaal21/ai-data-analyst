"""Insights, reports (Markdown / HTML / PDF) and exports (CSV / Excel) built from real computed results."""
from __future__ import annotations

import datetime as _dt
import html
import io
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

from . import analytics as A
from .profiling import column_kind, data_dictionary, profile_dataset
from .util import AnalystError, fmt_num, is_text

_METRIC_HINT = ("revenue", "sales", "amount", "profit", "income", "total", "price", "cost", "value")


def main_metric(df: pd.DataFrame) -> str | None:
    nums = A.analysable_numeric_cols(df, include_boolean=False)
    for h in _METRIC_HINT:
        for c in nums:
            if h in str(c).lower():
                return c
    return nums[0] if nums else None


def generate_insights(df: pd.DataFrame, prof: dict | None = None, max_items: int = 12) -> list[dict]:
    """Computed observations. Each has a label: Fact (directly computed) or Hypothesis (needs verification)."""
    prof = prof or profile_dataset(df)
    out: list[dict] = []

    def add(label, text):
        out.append({"label": label, "text": text})

    n = len(df)
    q = prof["quality"]
    add("Fact", f"The dataset has {n:,} rows and {df.shape[1]} columns; data-quality score is {q['score']}/100.")
    heavy = [c for c in prof["columns_profile"] if c["missing_pct"] > 5]
    if heavy:
        add("Fact", "Columns with more than 5% missing values: " + ", ".join(f"{c['name']} ({c['missing_pct']}%)" for c in heavy[:5]) + ".")
    if prof["duplicate_rows"]:
        add("Fact", f"{prof['duplicate_rows']} duplicate rows ({prof['duplicate_pct']}%) may inflate totals.")
    metric = main_metric(df)
    cats = [c for c in df.columns if column_kind(df[c]) == "categorical" and 2 <= df[c].nunique() <= 30][:3]
    dates = [c for c in df.columns if column_kind(df[c]) == "datetime"]
    if metric:
        s = A.numeric_series(df[metric]).dropna()
        st = A.numeric_stats(s)
        add("Fact", f"'{metric}': total {fmt_num(st['sum'])}, mean {fmt_num(st['mean'])}, median {fmt_num(st['median'])} (n={st['count']:,}); distribution is {st['distribution']}.")
        for c in cats[:2]:
            g, _ = A.group_and_aggregate(df, [c], {metric: "sum"}, top_n=None)
            col = g.columns[-1]
            tot = g[col].sum()
            if tot:
                top = g.iloc[0]
                add("Fact", f"'{top[c]}' is the top '{c}' by total {metric}: {fmt_num(top[col])} ({100 * top[col] / tot:.1f}% of the total across {len(g)} groups).")
        if dates:
            try:
                ts = A.time_series_analysis(df, dates[0], metric, None, "sum")
                if "trend" in ts:
                    t = ts["trend"]
                    add("Fact", f"Over {ts['periods']} {ts['freq']}-periods, {metric} shows: {t['direction']} (p={t['p_value']:.3g}).")
                lv = ts.get("latest_vs_previous")
                if lv and lv["pct_change"] is not None:
                    partial = " (latest period may be incomplete)" if ts["last_period_may_be_partial"] else ""
                    add("Fact", f"Latest period vs previous: {lv['pct_change']:+.1f}%{partial}.")
                if lv and lv["pct_change"] is not None and lv["pct_change"] < -10:
                    add("Hypothesis", "A large drop in the latest period may reflect seasonality or an incomplete period - verify before drawing conclusions.")
            except AnalystError:
                pass
    try:
        cr = A.correlation_analysis(df, top_n=3)["pairs"]
        for _, r in cr.iterrows():
            if abs(r["correlation"]) >= 0.5 and r["significant_5pct"]:
                add("Fact", f"'{r['column_a']}' and '{r['column_b']}' have a {r['strength']} correlation (r={r['correlation']:.2f}, n={r['n']:,}); this is association, not causation.")
                break
    except AnalystError:
        pass
    outs = [i for i in prof["issues"] if i["code"] == "outliers"]
    if outs:
        add("Fact", f"Potential outliers detected in: {', '.join(str(i['column']) for i in outs[:4])} (1.5xIQR rule). They may be genuine values.")
    return out[:max_items]


def recommendations(prof: dict) -> list[str]:
    recs, codes = [], {i["code"] for i in prof["issues"]}
    if "duplicates" in codes:
        recs.append("Review and remove duplicate rows if they are not legitimate repeats.")
    if "missing" in codes:
        recs.append("Decide a missing-value policy per column (fill, drop or keep) based on why values are missing.")
    if codes & {"numeric_as_text", "date_as_text"}:
        recs.append("Convert numeric/date columns stored as text so they can be aggregated and sorted correctly.")
    if codes & {"whitespace", "label_variants"}:
        recs.append("Trim whitespace and standardize category labels before grouping.")
    if "outliers" in codes:
        recs.append("Inspect outliers with the business owner before capping or removing them.")
    recs.append("Treat all findings as descriptive; correlations do not establish causes and models give estimates, not facts.")
    return recs


# ------------------------------------------------------------------------------ report structure
def build_sections(session, include_charts: bool = False) -> list[tuple[str, list[Any]]]:
    df, prof = session.view(), None
    prof = session.cached("profile", lambda: profile_dataset(session.view()))
    q = prof["quality"]
    secs: list[tuple[str, list[Any]]] = []
    filt = f" Active filters: {session.filters}." if session.filters else ""
    secs.append(("1. Dataset overview", [
        f"Dataset **{session.name}**: {prof['rows']:,} rows x {prof['columns']} columns, {prof['memory_mb']} MB in memory. "
        f"Missing cells: {prof['missing_pct']}%. Duplicate rows: {prof['duplicate_rows']}.{filt}",
        data_dictionary(df, prof)]))
    iss = pd.DataFrame(prof["issues"])[["severity", "column", "issue", "why_it_matters", "suggestion"]] if prof["issues"] else None
    secs.append(("2. Data-quality findings", [f"Quality score: **{q['score']}/100**. {q['explanation']}",
                                              iss if iss is not None else "No data-quality issues detected."]))
    secs.append(("3. Cleaning operations performed", [session.log_df().drop(columns=["params"]) if session.log else "No cleaning operations were applied."]))
    st = A.stats_table(A.calculate_statistics(df))
    secs.append(("4. Key statistics", [st if not st.empty else "No numeric columns."]))
    secs.append(("5. Important patterns and insights", [f"[{i['label']}] {i['text']}" for i in generate_insights(df, prof)]))
    if include_charts:
        secs.append(("6. Visualizations", ["__CHARTS__"]))
    else:
        secs.append(("6. Visualizations", [f"Saved charts: {', '.join(t for t, _ in session.charts) or 'none'} (see the interactive charts export)."]))
    secs.append(("7. Limitations and recommendations", recommendations(prof) + [
        "Analysis covers the rows currently in view; results depend on the cleaning steps listed above."]))
    return secs


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    d = df.head(max_rows).copy()
    d.columns = [str(c) for c in d.columns]
    rows = [d.columns.tolist()] + [[_cell(v) for v in r] for r in d.itertuples(index=False)]
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _cell(v) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:,.4g}"
    return "" if v is None or v is pd.NaT else str(v)


def report_markdown(session) -> str:
    out = [f"# Data Analysis Report - {session.name}", f"_Generated {_dt.datetime.now():%Y-%m-%d %H:%M}_", ""]
    for title, blocks in build_sections(session):
        out.append(f"## {title}")
        for b in blocks:
            out.append(_md_table(b) if isinstance(b, pd.DataFrame) else (f"- {b}" if title.startswith(("5", "7")) else str(b)))
            out.append("")
    return "\n".join(out)


def _html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    d = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_cell(v))}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False))
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _inline_md(t: str) -> str:
    t = html.escape(t)
    parts = t.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def report_html(session) -> str:
    body = []
    for title, blocks in build_sections(session, include_charts=True):
        body.append(f"<h2>{html.escape(title)}</h2>")
        listy = title.startswith(("5", "7"))
        if listy:
            body.append("<ul>" + "".join(f"<li>{_inline_md(str(b))}</li>" for b in blocks if not isinstance(b, pd.DataFrame)) + "</ul>")
            continue
        for b in blocks:
            if isinstance(b, pd.DataFrame):
                body.append(_html_table(b))
            elif b == "__CHARTS__":
                charts = list(session.charts)
                if not charts:
                    charts = auto_charts(session.view())
                if not charts:
                    body.append("<p>No charts available (create charts in Chart Studio and save them).</p>")
                for i, (t, fig) in enumerate(charts):
                    body.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
            else:
                body.append(f"<p>{_inline_md(str(b))}</p>")
    css = ("body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}"
           "h1{border-bottom:3px solid #4f46e5;padding-bottom:.3rem}h2{margin-top:2rem;color:#3730a3}"
           ".tw{overflow-x:auto}table{border-collapse:collapse;font-size:.85rem}th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}"
           "th{background:#eef2ff}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Report - {html.escape(session.name)}</title><style>{css}</style></head><body>"
            f"<h1>Data Analysis Report - {html.escape(session.name)}</h1><p><i>Generated {_dt.datetime.now():%Y-%m-%d %H:%M}</i></p>" + "".join(body) + "</body></html>")


def auto_charts(df: pd.DataFrame, max_charts: int = 3) -> list[tuple[str, Any]]:
    """A few sensible default charts for reports; silently returns [] if plotly is unavailable."""
    from . import viz

    res = []
    metric = main_metric(df)
    cats = [c for c in df.columns if column_kind(df[c]) == "categorical" and 2 <= df[c].nunique() <= 30]
    dates = [c for c in df.columns if column_kind(df[c]) == "datetime"]
    specs = []
    if dates and metric:
        specs.append(dict(chart_type="line", x=dates[0], y=metric))
    if cats and metric:
        specs.append(dict(chart_type="bar", x=cats[0], y=metric))
    if metric:
        specs.append(dict(chart_type="histogram", x=metric))
    for sp in specs[:max_charts]:
        try:
            r = viz.build_chart(df, **sp)
            res.append((r.title, r.figure))
        except Exception:  # noqa: BLE001
            continue
    return res


def report_pdf(session) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as e:
        raise AnalystError("PDF export needs 'reportlab' (pip install reportlab).") from e
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    ss = getSampleStyleSheet()
    story = [Paragraph(xml_escape(f"Data Analysis Report - {session.name}"), ss["Title"]), Spacer(1, 8)]
    for title, blocks in build_sections(session):
        story.append(Paragraph(xml_escape(title), ss["Heading2"]))
        for b in blocks:
            if isinstance(b, pd.DataFrame):
                d = b.iloc[:15, :8]
                data = [[xml_escape(str(c))[:22] for c in d.columns]] + [[xml_escape(_cell(v))[:40] for v in r] for r in d.itertuples(index=False)]
                t = Table([[Paragraph(c, ss["BodyText"]) for c in row] for row in data], repeatRows=1)
                t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                                       ("FONTSIZE", (0, 0), (-1, -1), 7)]))
                story += [t, Spacer(1, 6)]
            else:
                story.append(Paragraph(xml_escape(str(b)).replace("**", ""), ss["BodyText"]))
    doc.build(story)
    return buf.getvalue()


# ------------------------------------------------------------------------------ exports
def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection (cells starting with = + - @ in text columns)."""
    out = df.copy()
    for c in out.columns:
        if is_text(out[c]) and not isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+-@" and len(v) > 1 else v)
    return out


def to_csv_bytes(df: pd.DataFrame, sanitize: bool = True) -> bytes:
    return (_csv_safe(df) if sanitize else df).to_csv(index=False).encode("utf-8-sig")


def _excel_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.head(1_048_000).copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
        elif out[c].dtype == object:
            out[c] = out[c].map(lambda v: v.replace(tzinfo=None) if isinstance(v, (_dt.datetime, pd.Timestamp)) and v.tzinfo else v)
    return out


def build_excel(session) -> bytes:
    prof = session.cached("profile", lambda: profile_dataset(session.view()))
    df = session.view()
    findings = pd.DataFrame([{"type": i["label"], "finding": i["text"]} for i in generate_insights(df, prof)])
    st = A.stats_table(A.calculate_statistics(df))
    sheets = {"Cleaned Data": session.working, "Data Dictionary": data_dictionary(df, prof), "Statistics": st if not st.empty else pd.DataFrame({"info": ["no numeric columns"]}),
              "Findings": findings, "Data Quality": pd.DataFrame(prof["issues"]) if prof["issues"] else pd.DataFrame({"info": ["no issues"]}),
              "Cleaning Log": session.log_df() if session.log else pd.DataFrame({"info": ["no cleaning operations"]})}
    if session.filters:
        sheets["Filtered Data"] = df
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d = _excel_ready(d)
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for j, col in enumerate(d.columns, start=1):
                width = min(max(len(str(col)), int(d[col].astype(str).head(200).str.len().max() or 0) if len(d) else 0) + 2, 60)
                ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
                if is_text(d[col]) and len(d):  # never let text starting with '=' execute as a formula
                    mask = d[col].map(lambda v: isinstance(v, str) and v.startswith("=")).to_numpy()
                    for i in np.flatnonzero(mask)[:50_000]:
                        ws.cell(row=int(i) + 2, column=j).data_type = "s"
    return buf.getvalue()


def charts_html(session) -> bytes:
    charts = session.charts or auto_charts(session.view())
    if not charts:
        raise AnalystError("No charts to export. Build a chart in Chart Studio and click 'Save chart'.")
    parts = [f"<h2>{html.escape(t)}</h2>" + f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, (t, f) in enumerate(charts)]
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Charts</title></head><body>" + "".join(parts) + "</body></html>").encode()


EXPORT_KINDS = {
    "cleaned_csv": ("cleaned_data.csv", "text/csv"), "filtered_csv": ("filtered_data.csv", "text/csv"),
    "excel": ("analysis_workbook.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "data_dictionary": ("data_dictionary.csv", "text/csv"), "statistics": ("statistics.csv", "text/csv"),
    "report_md": ("analysis_report.md", "text/markdown"), "report_html": ("analysis_report.html", "text/html"),
    "report_pdf": ("analysis_report.pdf", "application/pdf"), "charts_html": ("charts.html", "text/html"),
    "cleaning_log": ("cleaning_log.csv", "text/csv"),
}


def make_export(session, kind: str, sanitize: bool = True) -> tuple[str, bytes, str]:
    if kind not in EXPORT_KINDS:
        raise AnalystError(f"Unknown export '{kind}'. Options: {list(EXPORT_KINDS)}")
    fname, mime = EXPORT_KINDS[kind]
    df = session.view()
    if kind == "cleaned_csv":
        data = to_csv_bytes(session.working, sanitize)
    elif kind == "filtered_csv":
        data = to_csv_bytes(df, sanitize)
    elif kind == "excel":
        data = build_excel(session)
    elif kind == "data_dictionary":
        data = data_dictionary(df).to_csv(index=False).encode("utf-8-sig")
    elif kind == "statistics":
        data = A.stats_table(A.calculate_statistics(df)).to_csv(index=False).encode("utf-8-sig")
    elif kind == "report_md":
        data = report_markdown(session).encode("utf-8")
    elif kind == "report_html":
        data = report_html(session).encode("utf-8")
    elif kind == "report_pdf":
        data = report_pdf(session)
    elif kind == "charts_html":
        data = charts_html(session)
    else:
        data = session.log_df().to_csv(index=False).encode("utf-8-sig")
    session.exports[fname] = (data, mime)
    return fname, data, mime
