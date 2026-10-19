"""Streamlit pages. All analytics are delegated to the tested `analyst` package."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from analyst import analytics as A
from analyst import cleaning, reporting, viz
from analyst.agent import Agent
from analyst.config import Settings
from analyst.filtering import OPERATORS, describe_conditions
from analyst.ingestion import IngestionError, SUPPORTED_EXTENSIONS, load_file
from analyst.profiling import column_kind, data_dictionary, find_duplicates, find_missing, profile_dataset
from analyst.util import AnalystError
from . import state
from .state import show_df

EXAMPLES = ["What are the most important insights in this dataset?", "Top 10 products by revenue", "Monthly sales trend",
            "Is data mein sabse zyada revenue kis product se aaya?", "Missing values ko median se fill karo", "Why did revenue decrease?",
            "Find outliers", "Is data ka complete report banao"]


def _need_data():
    s = state.current()
    if s is None:
        st.info("Upload a file (or load the sample dataset) on the **Upload & Overview** page first.")
    return s


def _profile(s):
    return s.cached("profile", lambda: profile_dataset(s.view()))


def _fig(fig, key):
    st.plotly_chart(fig, key=key)


# ============================================================================ upload & overview
def page_upload(settings: Settings):
    st.header("📁 Upload & Overview")
    st.caption(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))} · max {settings.max_upload_mb} MB · your original data is never modified.")
    files = st.file_uploader("Upload one or more data files", accept_multiple_files=True,
                             type=[e.strip(".") for e in sorted(SUPPORTED_EXTENSIONS)])
    for f in files or []:
        fid = getattr(f, "file_id", None) or (f.name, f.size)
        if fid in st.session_state["seen_uploads"]:
            continue
        st.session_state["seen_uploads"].add(fid)
        try:
            with st.spinner(f"Reading {f.name} ..."):
                lf = load_file(f.name, f.getvalue(), settings.max_upload_mb, settings.max_rows)
            added = state.add_loaded(lf, settings)
            st.success(f"Loaded {f.name}: {len(added)} dataset(s).")
        except IngestionError as e:
            st.error(f"❌ {f.name}: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Unexpected error while reading {f.name}: {e}")
    c1, c2 = st.columns([1, 3])
    if c1.button("Load sample sales dataset"):
        p = state.SAMPLE_DIR / "sales_sample.csv"
        try:
            lf = load_file(p.name, p.read_bytes(), settings.max_upload_mb, settings.max_rows)
            state.add_loaded(lf, settings)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Sample file problem: {e}")
    s = state.current()
    if s is None:
        return
    info = st.session_state["file_info"].get(s.name, {})
    if info:
        st.markdown(f"**File:** {info['file']} · **Format:** {info['format']} · **Size:** {info['size_kb']} KB · "
                    f"**Datasets/sheets in file:** {', '.join(info['sheets'])}")
        for w in info["warnings"]:
            st.warning(w)
    for n in s.notices:
        st.warning(n)
    prof = _profile(s)
    q = prof["quality"]
    cols = st.columns(6)
    cols[0].metric("Rows (in view)", f"{prof['rows']:,}")
    cols[1].metric("Columns", prof["columns"])
    cols[2].metric("Missing cells", f"{prof['missing_pct']}%")
    cols[3].metric("Duplicate rows", prof["duplicate_rows"])
    cols[4].metric("Memory", f"{prof['memory_mb']} MB")
    cols[5].metric("Quality score", f"{q['score']}/100")
    with st.expander("How is the quality score calculated?"):
        st.write(q["explanation"])
        st.json(q["components"])
    if s.filters:
        st.info(f"Active filter: {describe_conditions(s.filters, s.filter_combine)} → {len(s.view()):,} of {len(s.working):,} rows.")
    st.subheader("Key insights (computed from the data)")
    for i in reporting.generate_insights(s.view(), prof):
        st.markdown(f"- **{i['label']}:** {i['text']}")
    st.subheader("Columns")
    show_df(pd.DataFrame([{"column": p["name"], "type": p["kind"], "dtype": p["dtype"], "missing %": p["missing_pct"], "unique": p["unique"]}
                          for p in prof["columns_profile"]]), hide_index=True)
    st.subheader("Automatic charts")
    charts = s.cached("auto_charts", lambda: reporting.auto_charts(s.view()))
    if not charts:
        st.caption("No automatic charts could be drawn (or Plotly is not installed).")
    for i, (t, fig) in enumerate(charts):
        _fig(fig, f"auto_{i}")


# ============================================================================ preview
def page_preview(settings: Settings):
    st.header("🔎 Data Preview")
    s = _need_data()
    if not s:
        return
    src = st.radio("Show", ["Working copy (with active filters)", "Working copy (all rows)", "Original upload"], horizontal=True)
    df = s.view() if src.startswith("Working copy (with") else s.working if "all rows" in src else s.original
    size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)
    pages = max(1, -(-len(df) // size))
    page = st.number_input("Page", 1, pages, 1)
    st.caption(f"Rows {(page - 1) * size + 1:,}–{min(page * size, len(df)):,} of {len(df):,} · {df.shape[1]} columns")
    show_df(df.iloc[(page - 1) * size: page * size])


# ============================================================================ quality
def page_quality(settings: Settings):
    st.header("🩺 Data Quality & Statistics")
    s = _need_data()
    if not s:
        return
    df, prof = s.view(), _profile(s)
    st.metric("Data-quality score", f"{prof['quality']['score']}/100")
    st.caption(prof["quality"]["explanation"])
    t1, t2, t3, t4, t5 = st.tabs(["Issues", "Missing & duplicates", "Statistics", "Correlations", "Data dictionary"])
    with t1:
        if not prof["issues"]:
            st.success("No data-quality issues detected.")
        sev = st.multiselect("Severity", ["high", "warning", "info"], default=["high", "warning", "info"])
        iss = [i for i in prof["issues"] if i["severity"] in sev]
        if iss:
            show_df(pd.DataFrame(iss)[["severity", "column", "issue", "why_it_matters", "suggestion"]], hide_index=True)
        st.caption("Unusual values (e.g. outliers, negatives) are flagged for review - they are not assumed to be errors.")
    with t2:
        show_df(find_missing(df), hide_index=True)
        n, sample = find_duplicates(df)
        st.write(f"**Duplicate rows:** {n}")
        if n:
            show_df(sample)
    with t3:
        st_ = A.calculate_statistics(df)
        tab = A.stats_table(st_)
        if not tab.empty:
            show_df(tab, hide_index=True)
        for c, v in st_["categorical"].items():
            with st.expander(f"Frequency table: {c}"):
                show_df(pd.DataFrame(v["frequency"]), hide_index=True)
        if st_["skipped"]:
            with st.expander("Columns skipped (statistics not meaningful)"):
                st.write(st_["skipped"])
    with t4:
        try:
            r = A.correlation_analysis(df, method=st.selectbox("Method", ["pearson", "spearman", "kendall"]))
            show_df(r["pairs"], hide_index=True)
            res = viz.build_chart(df, "correlation_heatmap")
            _fig(res.figure, "corr_q")
            st.caption(r["note"])
        except AnalystError as e:
            st.info(str(e))
    with t5:
        show_df(data_dictionary(df, prof), hide_index=True)


# ============================================================================ cleaning
def _parse_mapping(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            a, b = line.split("=", 1)
            out[a.strip()] = b.strip()
    return out


def render_params(op_name: str, df: pd.DataFrame) -> dict:
    op, params, cols = cleaning.OPERATIONS[op_name], {}, list(df.columns)
    for p in op.params:
        k, n, kind = f"{op_name}_{p['name']}", p["name"], p["kind"]
        label = n.replace("_", " ").title() + (" (optional)" if p.get("optional") else "")
        if kind == "column":
            params[n] = st.selectbox(label, cols, key=k)
        elif kind == "columns":
            v = st.multiselect(label, cols, key=k)
            if v or not p.get("optional"):
                params[n] = v
        elif kind == "choice":
            params[n] = st.selectbox(label, p["choices"], index=p["choices"].index(p["default"]), key=k)
        elif kind == "text":
            v = st.text_input(label, key=k)
            if v:
                params[n] = v
        elif kind == "number":
            v = st.text_input(label + " (blank = default)", key=k)
            if v.strip():
                try:
                    params[n] = float(v)
                except ValueError:
                    st.warning(f"{label}: not a number")
        elif kind == "bool":
            params[n] = st.checkbox(label, value=p.get("default", False), key=k)
        elif kind == "mapping":
            m = _parse_mapping(st.text_area(label + " - one 'old=new' per line", key=k))
            if m or not p.get("optional"):
                params[n] = m
        elif kind == "conditions":
            c1, c2, c3 = st.columns(3)
            col = c1.selectbox("Condition column", cols, key=k + "_c")
            opr = c2.selectbox("Operator", OPERATORS, key=k + "_o")
            val = c3.text_input("Value (comma-separated for in/between)", key=k + "_v")
            params[n] = [{"column": col, "op": opr, "value": val}]
    return params


def page_cleaning(settings: Settings):
    st.header("🧹 Cleaning Workspace")
    s = _need_data()
    if not s:
        return
    st.caption("Edits are applied to a separate working copy. Risky operations (removing data or altering values) need your confirmation. Everything is logged and can be undone.")
    if s.pending:
        st.warning(f"⏳ Pending operation from the AI chat: **{s.pending['description']}**")
        c1, c2 = st.columns(2)
        if c1.button("✅ Confirm and apply"):
            r = cleaning.run_operation(s, s.pending["operation"], s.pending["params"], confirmed=True)
            (st.success if r["status"] == "applied" else st.error)(r["message"])
            st.rerun()
        if c2.button("✖ Cancel"):
            s.pending = None
            st.rerun()
    ops = list(cleaning.OPERATIONS)
    op_name = st.selectbox("Operation", ops, format_func=lambda o: cleaning.OPERATIONS[o].label)
    st.caption(cleaning.OPERATIONS[op_name].help)
    params = render_params(op_name, s.working)
    res = cleaning.run_operation(s, op_name, params, dry_run=True)
    if res["status"] == "error":
        st.info(f"Preview unavailable: {res['message']}")
    else:
        r = res["result"]
        st.markdown(f"**Proposed change:** {r.description}")
        for w in r.warnings:
            st.warning(w)
        prev = cleaning.diff_preview(s.working, r.df)
        a, b = st.columns(2)
        a.metric("Rows", f"{prev['shape_after'][0]:,}", delta=prev["shape_after"][0] - prev["shape_before"][0])
        b.metric("Columns", prev["shape_after"][1], delta=prev["shape_after"][1] - prev["shape_before"][1])
        st.caption(f"Cells changed: {r.cells_changed:,}")
        if not prev["changed_cells_sample"].empty:
            st.write("Before → after (sample of changed cells):")
            show_df(prev["changed_cells_sample"], hide_index=True)
        if not prev["removed_rows_sample"].empty:
            st.write("Rows that would be removed (sample):")
            show_df(prev["removed_rows_sample"])
        ok = True
        if r.risky:
            st.error("⚠️ This operation is potentially destructive.")
            ok = st.checkbox("I understand and confirm this change", key=f"confirm_{op_name}")
        if st.button("Apply to working copy", disabled=not ok):
            out = cleaning.run_operation(s, op_name, params, confirmed=ok)
            (st.success if out["status"] == "applied" else st.error)(out["message"])
            st.rerun()
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("↩ Undo last"):
        st.toast("Undone" if s.undo() else "Nothing to undo")
        st.rerun()
    if c2.button("⟲ Reset to original"):
        s.reset()
        st.rerun()
    st.subheader("Change log")
    if s.log:
        show_df(s.log_df(), hide_index=True)
    else:
        st.caption("No changes yet.")
    st.subheader("Current working copy (first 100 rows)")
    show_df(s.working.head(100))


# ============================================================================ filters
def page_filters(settings: Settings):
    st.header("🎯 Filter Builder")
    s = _need_data()
    if not s:
        return
    st.caption("Filters define the *view* that all analyses, charts and exports use. Your data is not changed unless you choose 'make permanent'.")
    cols = list(s.working.columns)
    c1, c2, c3 = st.columns([2, 1, 3])
    col = c1.selectbox("Column", cols)
    opr = c2.selectbox("Operator", OPERATORS)
    kind = column_kind(s.working[col])
    if kind in {"categorical", "boolean"} and s.working[col].nunique() <= 200 and opr in {"==", "!=", "in", "not in"}:
        vals = c3.multiselect("Value(s)", sorted(s.working[col].dropna().astype(str).unique()))
        value = vals if opr in {"in", "not in"} else (vals[0] if vals else "")
    else:
        value = c3.text_input("Value (comma-separated for 'in' / 'between')")
    if st.button("➕ Add condition"):
        s.filters.append({"column": col, "op": opr, "value": value})
        try:
            s.set_filters(s.filters, s.filter_combine)
        except AnalystError as e:
            s.filters.pop()
            st.error(str(e))
        st.rerun()
    s.filter_combine = st.radio("Combine conditions with", ["and", "or"], horizontal=True, index=0 if s.filter_combine == "and" else 1)
    if s.filters:
        st.write("**Active conditions:** " + describe_conditions(s.filters, s.filter_combine))
        a, b = st.columns(2)
        if a.button("Clear filters"):
            s.filters = []
            s.version += 1
            st.rerun()
        if b.button("Make filter permanent (removes other rows from working copy)"):
            r = cleaning.run_operation(s, "filter_rows", {"conditions": s.filters, "combine": s.filter_combine}, confirmed=False)
            st.info(r["message"] + " Confirm it in the Cleaning Workspace.") if r["status"] == "needs_confirmation" else st.success(r["message"])
    view = s.view()
    st.metric("Rows matching", f"{len(view):,}", delta=f"{len(view) - len(s.working):,} vs all rows")
    show_df(view.head(200))
    if len(view):
        st.subheader("Visualization of the filtered data")
        pick = st.selectbox("Column to chart", cols, key="filter_chart_col")
        try:
            res = viz.build_chart(view, "auto", x=pick)
            _fig(res.figure, "filter_chart")
            st.caption(res.reason)
        except AnalystError as e:
            st.info(str(e))


# ============================================================================ chart studio
def page_chart_studio(settings: Settings):
    st.header("📈 Chart Studio")
    s = _need_data()
    if not s:
        return
    df, cols = s.view(), ["(none)"] + list(s.view().columns)
    c1, c2, c3, c4 = st.columns(4)
    ctype = c1.selectbox("Chart type", viz.CHART_TYPES)
    x = c2.selectbox("X (category / date / numeric)", cols)
    y = c3.selectbox("Y (measure)", cols)
    color = c4.selectbox("Color / group", cols)
    d1, d2, d3, d4 = st.columns(4)
    size = d1.selectbox("Bubble size", cols)
    agg = d2.selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count", "nunique"])
    freq = d3.selectbox("Date grouping", ["auto", "day", "week", "month", "quarter", "year"])
    top_n = d4.number_input("Top N categories (0 = default)", 0, 200, 0)
    title = st.text_input("Chart title (optional)")
    n = lambda v: None if v == "(none)" else v
    try:
        res = viz.build_chart(df, ctype, n(x), n(y), n(color), n(size), agg, int(top_n) or None, None if freq == "auto" else freq, title or None)
    except AnalystError as e:
        st.warning(str(e))
        return
    st.info(f"**Why this chart:** {res.reason}")
    for note in res.notes:
        st.caption("ℹ️ " + note)
    left, right = st.columns([3, 2])
    with left:
        _fig(res.figure, "studio_chart")
    with right:
        st.write("**Data behind the chart**")
        data = res.data.copy()
        if st.checkbox("Conditional formatting (highlight max/min)", value=True):
            num = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            try:
                st.dataframe(state.arrow_safe(data.head(200)).style.highlight_max(subset=num, color="#bbf7d0").highlight_min(subset=num, color="#fecaca")
                             if num else state.arrow_safe(data.head(200)))
            except Exception:  # noqa: BLE001
                show_df(data.head(200))
        else:
            show_df(data.head(200))
    a, b = st.columns(2)
    if a.button("💾 Save chart to report/exports"):
        s.charts.append((res.title or "Chart", res.figure))
        st.success(f"Saved. {len(s.charts)} chart(s) saved.")
    b.download_button("⬇ Download chart (HTML)", viz.fig_to_html(res.figure, res.title or "chart"), file_name="chart.html", mime="text/html")


# ============================================================================ AI chat
def _render_entry(e: dict, i: int):
    with st.chat_message(e["role"]):
        st.markdown(e["text"])
        for n in e.get("notices", []):
            st.caption("ℹ️ " + n)
        for j, (title, df) in enumerate(e.get("tables", [])):
            with st.expander(f"📋 {title} ({len(df):,} rows)", expanded=j == 0 and len(df) <= 30):
                show_df(df.head(200))
        for j, (title, fig) in enumerate(e.get("figures", [])):
            _fig(fig, f"chat_{i}_{j}")
        if e.get("trace"):
            with st.expander(f"🔧 Tools used ({len(e['trace'])}) · mode: {e.get('mode')}"):
                st.json(e["trace"])


def page_chat(settings: Settings, use_ai: bool):
    st.header("🤖 AI Analyst Chat")
    s = _need_data()
    if not s:
        return
    st.caption("Ask in English, Hindi or Hinglish. Answers come from real computations (pandas/SciPy/scikit-learn) run by controlled tools - not from model guesses. "
               + ("**AI mode:** the model sees column names and tool outputs (aggregates / limited rows)." if use_ai and settings.ai_available else "**Offline mode:** rule-based analyst, no data leaves your machine."))
    if s.pending:
        st.warning(f"⏳ Waiting for your confirmation: **{s.pending['description']}**")
        a, b = st.columns(2)
        if a.button("✅ Confirm and apply", key="chat_confirm"):
            r = cleaning.run_operation(s, s.pending["operation"], s.pending["params"], confirmed=True)
            s.chat.append({"role": "assistant", "text": ("✅ " if r["status"] == "applied" else "❌ ") + r["message"]})
            st.rerun()
        if b.button("✖ Cancel", key="chat_cancel"):
            s.pending = None
            s.chat.append({"role": "assistant", "text": "Cancelled - nothing was changed."})
            st.rerun()
    for i, e in enumerate(s.chat):
        _render_entry(e, i)
    if not s.chat:
        st.write("Try one of these:")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex_{i}"):
                st.session_state["queued_question"] = ex
                st.rerun()
    q = st.chat_input("Ask a question about your data ...") or st.session_state.pop("queued_question", None)
    if q:
        s.chat.append({"role": "user", "text": q})
        with st.spinner("Analyzing ..."):
            try:
                r = Agent(s, settings, use_llm=use_ai).ask(q)
                s.chat.append({"role": "assistant", "text": r.text, "tables": r.tables, "figures": r.figures, "trace": r.trace, "mode": r.mode, "notices": r.notices})
            except Exception as e:  # noqa: BLE001
                s.chat.append({"role": "assistant", "text": f"Sorry, something went wrong: {e}"})
        st.rerun()
    if s.chat and st.button("Clear chat"):
        s.chat.clear()
        s.llm_history.clear()
        st.rerun()


# ============================================================================ report
def page_report(settings: Settings):
    st.header("📝 Insights & Report")
    s = _need_data()
    if not s:
        return
    for title, blocks in reporting.build_sections(s):
        st.subheader(title)
        for b in blocks:
            if isinstance(b, pd.DataFrame):
                show_df(b.head(30), hide_index=True)
            elif title.startswith(("5", "7")):
                st.markdown(f"- {b}")
            else:
                st.markdown(str(b))


# ============================================================================ export
def page_export(settings: Settings):
    st.header("📦 Export Center")
    s = _need_data()
    if not s:
        return
    safe = st.checkbox("Protect CSV against spreadsheet formula injection (prefix text starting with = + - @ with ')", value=True)
    labels = {"cleaned_csv": "Cleaned CSV (all working rows)", "filtered_csv": "Filtered CSV (current view)", "excel": "Excel workbook (multiple sheets)",
              "data_dictionary": "Data dictionary (CSV)", "statistics": "Statistical summary (CSV)", "report_md": "Report (Markdown)",
              "report_html": "Report (HTML with charts)", "report_pdf": "Report (PDF)", "charts_html": "Saved charts (interactive HTML)", "cleaning_log": "Cleaning log (CSV)"}
    cols = st.columns(2)
    for i, (k, lab) in enumerate(labels.items()):
        if cols[i % 2].button("Prepare: " + lab, key=f"exp_{k}"):
            try:
                with st.spinner("Building file ..."):
                    reporting.make_export(s, k, safe)
                st.success(f"{lab} is ready below.")
            except AnalystError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Export failed: {e}")
    st.subheader("Ready to download")
    if not s.exports:
        st.caption("Nothing prepared yet.")
    for name, (data, mime) in s.exports.items():
        st.download_button(f"⬇ {name} ({len(data) / 1024:.1f} KB)", data, file_name=name, mime=mime, key=f"dl_{name}")


PAGES = {
    "📁 Upload & Overview": page_upload, "🔎 Data Preview": page_preview, "🩺 Data Quality": page_quality, "🧹 Cleaning Workspace": page_cleaning,
    "🎯 Filter Builder": page_filters, "📈 Chart Studio": page_chart_studio, "📝 Insights & Report": page_report, "📦 Export Center": page_export,
}
