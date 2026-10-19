"""Controlled analytical tools exposed to the agent (LLM or rule-based).

The model can only call these functions with structured arguments - there is NO execution of
model-generated Python or shell code. Every tool returns a ToolResult with real computed values.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import analytics as A
from . import cleaning, reporting, viz
from .filtering import OPERATORS, describe_conditions
from .profiling import column_kind, find_duplicates, find_missing, profile_column, profile_dataset, schema_summary
from .util import AnalystError, fmt_num, resolve_column, resolve_columns, to_jsonable

log = logging.getLogger("analyst.tools")


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None
    figure: Any = None
    title: str = ""
    needs_confirmation: bool = False


@dataclass
class ToolDef:
    name: str
    description: str
    schema: dict
    fn: Callable


TOOLS: dict[str, ToolDef] = {}
_STR = {"type": "string"}
_STRS = {"type": "array", "items": {"type": "string"}}
_COND = {"type": "array", "description": "Filter conditions combined with AND (or OR via 'combine').",
         "items": {"type": "object", "properties": {"column": _STR, "op": {"type": "string", "enum": OPERATORS},
                                                    "value": {"description": "Value, or list for in/between"}}, "required": ["column", "op"]}}


def tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None):
    def deco(fn):
        TOOLS[name] = ToolDef(name, description, {"type": "object", "properties": properties or {}, "required": required or []}, fn)
        return fn
    return deco


def _try_chart(df, **kw):
    try:
        r = viz.build_chart(df, **kw)
        return r.figure, r
    except AnalystError as e:
        return None, str(e)


def _chart_note(r) -> str:
    return "" if not isinstance(r, str) else f" (Chart not drawn: {r})"


# ------------------------------------------------------------------------------ tools
@tool("inspect_dataset", "Overview of the dataset in view: size, column types, data-quality score and top issues. Call this first for broad or unclear questions.")
def inspect_dataset(s, a):
    df = s.view()
    prof = s.cached("profile", lambda: profile_dataset(df))
    cols = pd.DataFrame([{"column": p["name"], "type": p["kind"], "dtype": p["dtype"], "missing_pct": p["missing_pct"], "unique": p["unique"]}
                         for p in prof["columns_profile"]])
    order = {"high": 0, "warning": 1, "info": 2}
    issues = sorted(prof["issues"], key=lambda i: order.get(i["severity"], 3))
    q = prof["quality"]
    msg = (f"Dataset '{s.name}' (rows in view: {prof['rows']:,}, columns: {prof['columns']}, memory {prof['memory_mb']} MB). "
           f"Column types: {prof['kinds']}. Missing cells: {prof['missing_pct']}%. Duplicate rows: {prof['duplicate_rows']}. "
           f"Data-quality score: {q['score']}/100. " + (f"Active filters: {describe_conditions(s.filters, s.filter_combine)}. " if s.filters else "")
           + (f"Top issues: " + "; ".join(f"[{i['severity']}] {i['column'] or 'dataset'}: {i['issue']}" for i in issues[:6]) if issues else "No data-quality issues detected."))
    return ToolResult(True, msg, {"rows": prof["rows"], "columns": schema_summary(df, prof), "quality": q, "issue_count": len(issues),
                                  "top_issues": [{k: i[k] for k in ("severity", "column", "issue")} for i in issues[:10]]}, cols, title="Columns")


@tool("get_column_profile", "Detailed profile of one column (type, missing, unique, top values, range).", {"column": _STR}, ["column"])
def get_column_profile(s, a):
    df = s.view()
    c = resolve_column(df, a.get("column"))
    p = profile_column(df[c])
    top = pd.DataFrame(p.get("top_values", []))
    msg = f"Column '{c}': type {p['kind']} ({p['dtype']}), {p['count']:,} non-null, {p['missing']} missing ({p['missing_pct']}%), {p['unique']} unique."
    if p["kind"] == "numeric":
        msg += f" min {fmt_num(p['min'])}, max {fmt_num(p['max'])}, mean {fmt_num(p['mean'])}, median {fmt_num(p['median'])}."
    return ToolResult(True, msg, p, top if not top.empty else None, title=f"Top values of {c}")


@tool("calculate_statistics", "Descriptive statistics (count, sum, mean, median, mode, std, quartiles, skewness, 95% CI, frequency tables). Only suitable columns are computed.",
      {"columns": {**_STRS, "description": "Optional; default all columns"}})
def calculate_statistics(s, a):
    df = s.view()
    st = A.calculate_statistics(df, a.get("columns"))
    tab = A.stats_table(st)
    lines = [f"{c}: n={v['count']:,}, mean={fmt_num(v['mean'])}, median={fmt_num(v['median'])}, std={fmt_num(v['std'])}, min={fmt_num(v['min'])}, max={fmt_num(v['max'])}, {v['distribution']}"
             for c, v in st["numeric"].items() if v.get("count")]
    lines += [f"{c}: {v['unique']} categories, most frequent '{v['mode']}' ({v['frequency'][0]['pct']}%)" for c, v in st["categorical"].items() if v["frequency"]]
    if st["skipped"]:
        lines.append("Skipped (not meaningful): " + "; ".join(st["skipped"]))
    return ToolResult(True, "\n".join(lines) or "No suitable columns.", st, tab if not tab.empty else None, title="Numeric statistics")


@tool("find_missing_values", "Missing-value counts and percentages per column.")
def find_missing_values(s, a):
    df = s.view()
    t = find_missing(df)
    bad = t[t["missing"] > 0]
    rows_any = int(df.isna().any(axis=1).sum())
    msg = (f"{len(bad)} of {df.shape[1]} columns have missing values; {rows_any:,} of {len(df):,} rows have at least one. "
           + "; ".join(f"{r.column}: {r.missing} ({r.missing_pct}%)" for r in bad.head(8).itertuples()) if len(bad) else "No missing values found.")
    return ToolResult(True, msg, {"columns_with_missing": bad.to_dict("records"), "rows_with_any_missing": rows_any}, t, title="Missing values by column")


@tool("find_duplicates", "Count and show duplicate rows (optionally by a subset of columns).", {"subset": _STRS})
def find_dups(s, a):
    df = s.view()
    sub = resolve_columns(df, a.get("subset")) or None
    n, sample = find_duplicates(df, sub)
    return ToolResult(True, f"{n} duplicate row(s) ({100 * n / max(len(df), 1):.1f}%) by {sub or 'all columns'}; the table shows rows involved in duplication.",
                      {"duplicates": n, "subset": sub}, sample if n else None, title="Duplicate rows")


@tool("clean_dataset",
      "Apply a cleaning operation to the WORKING COPY (original is never changed; all steps are logged and undoable). Operations: "
      + ", ".join(cleaning.OPERATIONS) + ". Destructive/risky operations (removing rows/columns, outlier handling, lossy conversions) are NOT applied "
      "until the user confirms in the UI - set confirm=true ONLY if the user has explicitly approved that exact operation.",
      {"operation": {"type": "string", "enum": list(cleaning.OPERATIONS)},
       "params": {"type": "object", "description": "Operation parameters, e.g. fill_missing: {columns:[...], method:'median'}; remove_duplicates: {subset:[...]}; "
                                                   "convert_type: {column, target}; standardize_labels: {column, mapping}; filter_rows: {conditions:[...]}"},
       "confirm": {"type": "boolean"}}, ["operation"])
def clean_dataset(s, a):
    r = cleaning.run_operation(s, a["operation"], a.get("params") or {}, confirmed=bool(a.get("confirm")))
    if r["status"] == "error":
        return ToolResult(False, r["message"])
    res = r["result"]
    prev = cleaning.diff_preview(s.history[-1] if r["status"] == "applied" and s.history else s.working, res.df)
    tab = prev["changed_cells_sample"] if not prev["changed_cells_sample"].empty else (prev["removed_rows_sample"] if not prev["removed_rows_sample"].empty else None)
    msg = r["message"] + f" Rows: {res.rows_before:,} -> {res.rows_after:,}; cells changed: {res.cells_changed:,}."
    if res.warnings:
        msg += " Warnings: " + " ".join(res.warnings)
    return ToolResult(True, msg, {"status": r["status"], "rows_before": res.rows_before, "rows_after": res.rows_after, "cells_changed": res.cells_changed},
                      tab, title="Sample of changes", needs_confirmation=r["status"] == "needs_confirmation")


@tool("filter_dataset",
      "Set, add to, or clear the ACTIVE FILTER. All later analyses/charts run on the filtered rows. The underlying data is unchanged. Text equality is case-insensitive.",
      {"conditions": _COND, "mode": {"type": "string", "enum": ["replace", "add", "clear"]}, "combine": {"type": "string", "enum": ["and", "or"]}})
def filter_dataset(s, a):
    mode = a.get("mode", "replace")
    if mode == "clear":
        s.filters = []
        s.version += 1
        return ToolResult(True, f"Filters cleared. Showing all {len(s.working):,} rows.", {"rows": len(s.working)}, s.working.head(50), title="Data preview")
    conds = list(a.get("conditions") or [])
    if not conds:
        raise AnalystError("Provide at least one condition.")
    combine = a.get("combine", s.filter_combine if mode == "add" else "and")
    for c in conds:
        c["column"] = resolve_column(s.working, c["column"])
    new = (s.filters + conds) if mode == "add" else conds
    out = s.set_filters(new, combine)
    warn = " Warning: no rows match - check spelling of the values." if len(out) == 0 else ""
    return ToolResult(True, f"Filter active: {describe_conditions(new, combine)}. {len(out):,} of {len(s.working):,} rows match.{warn}",
                      {"rows_matching": len(out), "rows_total": len(s.working), "filters": new}, out.head(50), title="Filtered data (first 50 rows)")


def _norm_metrics(m):
    if isinstance(m, list):
        return {str(i.get("column", "*")): i.get("agg", "sum") for i in m if isinstance(i, dict)}
    if isinstance(m, str):
        return {m: "sum"}
    return m or {"*": "count"}


@tool("group_and_aggregate", "Group by one or more columns (a date column can be grouped by freq: day/week/month/quarter/year) and aggregate metrics (sum, mean, median, min, max, count, nunique, std). Use for rankings, totals per category, top/bottom N.",
      {"by": _STRS, "metrics": {"type": "object", "description": "{column: agg or [aggs]}; use '*': 'count' for row counts"}, "freq": _STR, "sort_by": _STR,
       "ascending": {"type": "boolean"}, "top_n": {"type": "integer"}, "pct_of_total": {"type": "boolean"}}, ["by"])
def group_and_aggregate(s, a):
    df = s.view()
    out, info = A.group_and_aggregate(df, a["by"], _norm_metrics(a.get("metrics")), a.get("freq"), a.get("sort_by"), a.get("ascending"), a.get("top_n"), bool(a.get("pct_of_total")))
    s.last_analysis = {"tool": "group_and_aggregate", "args": a}
    msg = f"Grouped {info['rows_used']:,} rows by {info['by']} into {info['groups']} group(s); metrics: {info['metrics']}."
    if info["rows_excluded_missing_keys"]:
        msg += f" {info['rows_excluded_missing_keys']} rows with missing group keys were excluded."
    if len(out):
        top = out.iloc[0]
        msg += f" Top: {top[info['by'][0]]} = {fmt_num(top[info['metrics'][0]])}."
    return ToolResult(True, msg, {**info, "result": out}, out, title="Aggregation result")


@tool("perform_correlation_analysis", "Correlations between numeric columns (matrix + strongest pairs with p-values), or - when 'target' is given - association of every other column with that target (numeric, categorical or yes/no).",
      {"columns": _STRS, "method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]}, "target": _STR})
def perform_correlation_analysis(s, a):
    df = s.view()
    if a.get("target"):
        r = A.target_associations(df, a["target"])
        tab = r["table"]
        msg = (f"Associations with '{r['target']}' (n={len(df):,} rows in view): " + "; ".join(f"{x.feature} ({x.measure}={x.association}, {x.strength})" for x in tab.head(5).itertuples())
               if len(tab) else "No suitable features to compare.") + " " + r["note"]
        return ToolResult(True, msg, {"target": r["target"], "table": tab}, tab, title=f"Association with {r['target']}")
    r = A.correlation_analysis(df, a.get("columns"), a.get("method", "pearson"))
    pairs = r["pairs"]
    fig, res = _try_chart(df, chart_type="correlation_heatmap")
    msg = "Strongest correlations: " + "; ".join(f"{x.column_a}~{x.column_b} r={x.correlation} ({x.strength}, n={x.n})" for x in pairs.head(5).itertuples()) + ". " + r["note"] + _chart_note(res)
    return ToolResult(True, msg, {"pairs": pairs, "method": r["method"]}, pairs, fig, "Strongest correlations")


@tool("analyze_time_series", "Trend analysis of a measure over time: period totals, month-over-month and year-over-year growth, rolling mean, linear-trend test, best/worst periods. Draws a line chart.",
      {"date_column": _STR, "value_column": {**_STR, "description": "Omit to count rows"}, "freq": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]},
       "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"]}, "last_n_periods": {"type": "integer"}}, ["date_column"])
def analyze_time_series(s, a):
    df = s.view()
    r = A.time_series_analysis(df, a["date_column"], a.get("value_column"), a.get("freq"), a.get("agg", "sum"), a.get("last_n_periods"))
    t = r["table"]
    fig, res = _try_chart(t, chart_type="line", x="period", y=r["value_label"], title=f"{r['value_label']} per period ({r['freq']})")
    s.last_analysis = {"tool": "analyze_time_series", "args": a}
    parts = [f"{r['periods']} {r['freq']}-periods analysed from {r['rows_used']:,} dated rows."]
    if "trend" in r:
        parts.append(f"Linear trend: {r['trend']['direction']} (p={r['trend']['p_value']:.3g}, R2={r['trend']['r_squared']:.2f}).")
    if "latest_vs_previous" in r and r["latest_vs_previous"]["pct_change"] is not None:
        lv = r["latest_vs_previous"]
        parts.append(f"Latest vs previous period: {fmt_num(lv['latest'])} vs {fmt_num(lv['previous'])} ({lv['pct_change']:+.1f}%).")
    parts.append(f"Best: {str(r['best_period']['period'])[:10]} ({fmt_num(r['best_period']['value'])}); worst: {str(r['worst_period']['period'])[:10]} ({fmt_num(r['worst_period']['value'])}).")
    if r["last_period_may_be_partial"]:
        parts.append("Caution: the last period may be incomplete, which can exaggerate a decline.")
    if r["missing_periods"]:
        parts.append(f"{r['missing_periods']} period(s) in range have no data.")
    return ToolResult(True, " ".join(parts) + _chart_note(res), {k: v for k, v in r.items() if k != "table"}, t, fig, "Time series")


@tool("detect_outliers", "Find unusual values in a numeric column (IQR or z-score) or across several columns (isolation_forest). Outliers are flagged, not assumed to be errors.",
      {"column": {**_STR, "description": "Column name (or list of names for isolation_forest)"}, "method": {"type": "string", "enum": ["iqr", "zscore", "isolation_forest"]}, "factor": {"type": "number"}})
def detect_outliers(s, a):
    df = s.view()
    r = A.detect_outliers(df, a.get("column") or [], a.get("method", "iqr"), a.get("factor"))
    where = r.get("column") or r.get("columns")
    bounds = f" Bounds: [{fmt_num(r['lower_bound'])}, {fmt_num(r['upper_bound'])}]." if "lower_bound" in r else ""
    return ToolResult(True, f"{r['n_outliers']} outlier(s) ({r['pct']}%) among {r['n_checked']:,} values of {where} using {r['method']}.{bounds} {r['note']}",
                      {k: v for k, v in r.items() if k != "rows"}, r["rows"], title="Outlier rows (top 20)")


@tool("calculate_kpis", "KPIs: record count plus aggregates of chosen columns; with a date column also latest-vs-previous period change.",
      {"metrics": {"type": "array", "items": {"type": "object", "properties": {"column": _STR, "agg": _STR}, "required": ["column"]}}, "date_column": _STR, "freq": _STR})
def calculate_kpis(s, a):
    df = s.view()
    t = A.calculate_kpis(df, a.get("metrics"), a.get("date_column"), a.get("freq"))
    msg = "; ".join(f"{r.kpi}: {fmt_num(r.value)}" + (f" (latest period {r.change_pct:+.1f}% vs previous)" if pd.notna(r.change_pct) else "") for r in t.itertuples())
    return ToolResult(True, msg, {"kpis": t}, t, title="KPIs")


@tool("generate_visualization", "Create an interactive chart. chart_type 'auto' picks the best chart from column types. Handles aggregation (agg), date grouping (freq), top_n. Types: " + ", ".join(viz.CHART_TYPES),
      {"chart_type": {"type": "string", "enum": viz.CHART_TYPES}, "x": _STR, "y": _STR, "color": _STR, "size": _STR,
       "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"]}, "top_n": {"type": "integer"}, "freq": _STR, "title": _STR, "intent": _STR})
def generate_visualization(s, a):
    df = s.view()
    r = viz.build_chart(df, a.get("chart_type", "auto"), a.get("x"), a.get("y"), a.get("color"), a.get("size"), a.get("agg", "sum"), a.get("top_n"),
                        a.get("freq"), a.get("title"), a.get("intent"))
    s.last_analysis = {"tool": "generate_visualization", "args": a}
    msg = f"Chart '{r.title}' ({r.chart_type}) built from {len(df):,} rows. Why: {r.reason} " + " ".join(r.notes)
    return ToolResult(True, msg, {"chart_type": r.chart_type, "title": r.title, "rows_in_chart": len(r.data)}, r.data, r.figure, r.title)


@tool("query_dataset", "Show rows: choose columns, structured filters, sort, and a row limit (max 500). Use for 'show me the records where ...'.",
      {"columns": _STRS, "filters": _COND, "sort_by": _STR, "ascending": {"type": "boolean"}, "limit": {"type": "integer"}})
def query_dataset(s, a):
    out, total = A.query_dataset(s.view(), a.get("columns"), a.get("filters"), a.get("sort_by"), a.get("ascending", True), min(int(a.get("limit", 50)), 500))
    return ToolResult(True, f"{total:,} matching row(s); showing {len(out)}.", {"matching_rows": total}, out, title="Query result")


@tool("export_results", "Prepare a downloadable file for the user. kind: " + ", ".join(reporting.EXPORT_KINDS),
      {"kind": {"type": "string", "enum": list(reporting.EXPORT_KINDS)}}, ["kind"])
def export_results(s, a):
    name, data, _ = reporting.make_export(s, a["kind"])
    return ToolResult(True, f"Export ready: {name} ({len(data) / 1024:.1f} KB). Use the download button in the Export Center.", {"file": name, "bytes": len(data)})


@tool("discover_insights", "Automatically compute the most important facts about the dataset (top categories, trends, correlations, quality problems). Use for 'what are the key insights' questions.")
def discover_insights(s, a):
    ins = reporting.generate_insights(s.view(), s.cached("profile", lambda: profile_dataset(s.view())))
    return ToolResult(True, "\n".join(f"- [{i['label']}] {i['text']}" for i in ins), {"insights": ins}, pd.DataFrame(ins), title="Key insights")


@tool("advanced_analysis",
      "Advanced analytics. kind: predictive_model (params: target, features?, task?), clustering (features?, k?), cohort (user_column, date_column, freq?), "
      "funnel (stage_column, user_column, stages[]), driver_analysis (date_column, value_column, dimensions[], freq?) = which categories drove the change between the last two periods.",
      {"kind": {"type": "string", "enum": ["predictive_model", "clustering", "cohort", "funnel", "driver_analysis"]}, "params": {"type": "object"}}, ["kind", "params"])
def advanced_analysis(s, a):
    df, k, p = s.view(), a["kind"], a.get("params") or {}
    if k == "predictive_model":
        r = A.train_model(df, p["target"], p.get("features"), p.get("task", "auto"))
        m = r["metrics"]
        msg = (f"{r['task'].title()} model for '{r['target']}' (train {r['n_train']:,} / test {r['n_test']:,} rows). Test metrics: " + ", ".join(f"{k2}={v:.3f}" for k2, v in m.items())
               + f". Top predictors: {', '.join(r['feature_importance']['feature'].head(3))}. Excluded: {r['excluded_features'] or 'none'}. {r['note']}")
        return ToolResult(True, msg, {k2: v for k2, v in r.items() if k2 != "feature_importance"}, r["feature_importance"], title="Permutation feature importance")
    if k == "clustering":
        r = A.cluster_data(df, p.get("features"), p.get("k"))
        return ToolResult(True, f"{r['k']} clusters found on {r['features']} (silhouette={r['silhouette']:.2f}). {r['note']}", {"k": r["k"], "silhouette": r["silhouette"]}, r["profile"], title="Cluster profile")
    if k == "cohort":
        r = A.cohort_retention(df, p["user_column"], p["date_column"], p.get("freq", "month"))
        return ToolResult(True, f"Cohort retention for {r['users']:,} users ({r['freq']} cohorts); values are % of cohort active in later periods.", {"users": r["users"]}, r["retention_pct"], title="Retention %")
    if k == "funnel":
        t = A.funnel_analysis(df, p["stage_column"], p["user_column"], p["stages"])
        return ToolResult(True, "Funnel: " + "; ".join(f"{r.stage}={r.users} users" for r in t.itertuples()) + ". (Stage order is not verified per user.)", {"funnel": t}, t, title="Funnel")
    if k == "driver_analysis":
        r = A.period_drivers(df, p["date_column"], p["value_column"], p.get("dimensions") or [], p.get("freq"))
        parts = [f"{p['value_column']} changed {fmt_num(r['total_change'])} from {r['previous_period']:%Y-%m-%d} ({fmt_num(r['total_previous'])}) to {r['current_period']:%Y-%m-%d} ({fmt_num(r['total_current'])})."]
        frames = []
        for dim, t in r["dimensions"].items():
            if len(t):
                top = t.iloc[0]
                parts.append(f"By {dim}: largest change from '{top['category']}' ({fmt_num(top['change'])}, {top['share_of_total_change_pct']}% of the total change; n={top['n_current']} current rows).")
                frames.append(t.assign(dimension=dim))
        parts.append(r["note"])
        return ToolResult(True, " ".join(parts), {k2: v for k2, v in r.items() if k2 != "dimensions"}, pd.concat(frames, ignore_index=True) if frames else None, title="Change drivers")
    raise AnalystError(f"Unknown advanced analysis '{k}'.")


# ------------------------------------------------------------------------------ execution + LLM formatting
def anthropic_tool_specs() -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in TOOLS.values()]


def execute_tool(session, name: str, args: dict | None, timeout_s: int = 90) -> ToolResult:
    """Run a tool with error handling and a timeout. Never raises."""
    t = TOOLS.get(name)
    if t is None:
        return ToolResult(False, f"Unknown tool '{name}'. Available: {list(TOOLS)}")
    args = dict(args or {})
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        res = ex.submit(t.fn, session, args).result(timeout=timeout_s)
        return res
    except cf.TimeoutError:
        return ToolResult(False, f"Tool '{name}' timed out after {timeout_s}s. Try a smaller question or filter the data first.")
    except AnalystError as e:
        return ToolResult(False, str(e))
    except KeyError as e:
        return ToolResult(False, f"Missing required argument {e} for tool '{name}'.")
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return ToolResult(False, f"Tool '{name}' failed unexpectedly: {type(e).__name__}: {e}")
    finally:
        ex.shutdown(wait=False)


def result_for_llm(res: ToolResult, max_rows: int = 20, max_chars: int = 9000) -> str:
    """Compact JSON for the model: message, key numbers, and a truncated table (never the full dataset)."""
    import json

    payload: dict[str, Any] = {"ok": res.ok, "message": res.message}
    if res.needs_confirmation:
        payload["status"] = "NOT APPLIED - waiting for the user to confirm in the UI. Tell the user."
    data = {k: v for k, v in res.data.items() if not isinstance(v, pd.DataFrame)}
    if data:
        payload["data"] = to_jsonable(data, 30)
    if res.table is not None:
        payload["table"] = {"total_rows": len(res.table), "columns": [str(c) for c in res.table.columns], "rows_shown": to_jsonable(res.table.head(max_rows), max_rows)}
    if res.figure is not None:
        payload["chart"] = "A chart was created and is displayed to the user."
    s = json.dumps(payload, ensure_ascii=False, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + '..."(truncated)'
