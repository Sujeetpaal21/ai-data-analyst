"""Dataset profiling, column type inference, data-quality scoring and issue detection."""
from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .util import fmt_num, is_datetime, is_numeric, is_text

_ID_TOKENS = {"id", "uuid", "guid", "key", "sku", "index"}
_NONNEG_NAME = re.compile(r"price|amount|qty|quantity|revenue|sales|cost|salary|units|income|count|age|profit_margin", re.I)
_DATE_GATE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}[\s-][A-Za-z]{3,9}[\s,-]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
)
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
_NOT_IDS = {"paid", "valid", "void", "android", "solid", "grid", "braid"}


def looks_like_id(name: Any) -> bool:
    n = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    if any(t in _ID_TOKENS for t in tokens):
        return True
    low = str(name).lower()
    return low.endswith("id") and len(low) > 2 and low not in _NOT_IDS


def _date_share(sample: pd.Series) -> float:
    sample = sample.astype(str)
    if sample.empty or sample.str.contains(_DATE_GATE).mean() < 0.8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _numeric_text_share(sample: pd.Series) -> float:
    cleaned = sample.astype(str).str.replace(r"[,\s$₹€£%]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def column_kind(s: pd.Series) -> str:
    """numeric | boolean | datetime | categorical | text | identifier | constant | empty"""
    nn = s.dropna()
    if nn.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if is_datetime(s):
        return "datetime"
    try:
        nun = nn.nunique()
    except TypeError:
        return "text"
    if nun == 1 and len(nn) > 1:
        return "constant"
    if is_numeric(s):
        if looks_like_id(s.name) and nun >= 0.9 * len(nn):
            return "identifier"
        if nun == 2 and set(pd.unique(nn)) <= {0, 1}:
            return "boolean"
        return "numeric"
    sample = nn.head(1000)
    if is_text(s):
        low = set(sample.astype(str).str.strip().str.lower().unique())
        if len(low) <= 2 and low <= _BOOL_WORDS:
            return "boolean"
        if _numeric_text_share(sample) >= 0.9 and not looks_like_id(s.name):
            return "numeric"
        if _date_share(sample) >= 0.8:
            return "datetime"
    ratio = nun / len(nn)
    if looks_like_id(s.name) and ratio >= 0.9:
        return "identifier"
    avg_len = sample.astype(str).str.len().mean()
    if avg_len > 50 and ratio > 0.5:
        return "text"
    if nun <= max(50, 0.05 * len(nn)):
        return "categorical"
    if ratio > 0.9:
        return "identifier"
    return "categorical"


def stored_as_text(s: pd.Series, kind: str) -> bool:
    return kind in {"numeric", "datetime"} and is_text(s)


def columns_of_kind(df: pd.DataFrame, *kinds: str) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) in kinds]


def profile_column(s: pd.Series) -> dict:
    n = len(s)
    miss = int(s.isna().sum())
    kind = column_kind(s)
    try:
        nun = int(s.nunique(dropna=True))
    except TypeError:
        nun = -1
    p: dict[str, Any] = {
        "name": s.name, "dtype": str(s.dtype), "kind": kind, "count": n - miss, "missing": miss,
        "missing_pct": round(100 * miss / n, 2) if n else 0.0, "unique": nun,
        "unique_pct": round(100 * nun / (n - miss), 2) if n - miss > 0 and nun >= 0 else 0.0,
        "stored_as_text": stored_as_text(s, kind),
    }
    nn = s.dropna()
    if nn.empty:
        return p
    if kind == "numeric":
        x = pd.to_numeric(s.astype(str).str.replace(r"[,\s$₹€£%]", "", regex=True), errors="coerce") if is_text(s) else s
        p.update(min=x.min(), max=x.max(), mean=x.mean(), median=x.median(), std=x.std())
    elif kind == "datetime":
        d = pd.to_datetime(s, errors="coerce", format="mixed") if not is_datetime(s) else s
        p.update(min=d.min(), max=d.max(), span_days=(d.max() - d.min()).days if d.notna().any() else None)
    elif kind in {"categorical", "boolean", "text", "identifier", "constant"}:
        vc = nn.astype(str).value_counts().head(5)
        p["top_values"] = [{"value": k, "count": int(v), "pct": round(100 * v / len(nn), 1)} for k, v in vc.items()]
        p["avg_length"] = round(float(nn.astype(str).str.len().mean()), 1)
    return p


def profile_dataset(df: pd.DataFrame) -> dict:
    cols = [profile_column(df[c]) for c in df.columns]
    dup = int(df.duplicated().sum()) if len(df) else 0
    total_cells = df.shape[0] * df.shape[1]
    miss_cells = int(df.isna().sum().sum())
    kinds: dict[str, int] = {}
    for c in cols:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    prof = {
        "rows": len(df), "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "duplicate_rows": dup, "duplicate_pct": round(100 * dup / len(df), 2) if len(df) else 0.0,
        "missing_cells": miss_cells, "missing_pct": round(100 * miss_cells / total_cells, 2) if total_cells else 0.0,
        "kinds": kinds, "columns_profile": cols,
        "high_cardinality": [c["name"] for c in cols if c["kind"] == "categorical" and c["unique"] > 50],
        "constant_columns": [c["name"] for c in cols if c["kind"] == "constant"],
    }
    issues = detect_issues(df, prof)
    prof["issues"] = issues
    prof["quality"] = quality_score(prof, issues)
    return prof


def quality_score(prof: dict, issues: list[dict]) -> dict:
    """Score 0-100 = 40% completeness + 20% uniqueness + 20% consistency + 20% validity."""
    ncols = max(prof["columns"], 1)
    completeness = 100 - prof["missing_pct"]
    uniqueness = 100 - prof["duplicate_pct"]
    cons_cols = {i["column"] for i in issues if i["code"] in {"whitespace", "label_variants"}}
    valid_cols = {i["column"] for i in issues if i["code"] in {"numeric_as_text", "date_as_text", "suspicious_values", "bad_dates"}}
    consistency = 100 * (1 - len(cons_cols) / ncols)
    validity = 100 * (1 - len(valid_cols) / ncols)
    score = 0.4 * completeness + 0.2 * uniqueness + 0.2 * consistency + 0.2 * validity
    return {
        "score": round(score, 1),
        "components": {"completeness": round(completeness, 1), "uniqueness": round(uniqueness, 1),
                       "consistency": round(consistency, 1), "validity": round(validity, 1)},
        "explanation": (
            "Score = 40% completeness (100 - % missing cells) + 20% uniqueness (100 - % duplicate rows) "
            "+ 20% consistency (share of columns without whitespace/label-variant problems) "
            "+ 20% validity (share of columns without type/date/suspicious-value problems)."
        ),
    }


def _issue(code, column, severity, detail, why, suggestion):
    return {"code": code, "column": column, "severity": severity, "issue": detail, "why_it_matters": why, "suggestion": suggestion}


def detect_issues(df: pd.DataFrame, prof: dict | None = None) -> list[dict]:
    """Detect data-quality issues. Unusual values are flagged, never assumed to be errors."""
    issues: list[dict] = []
    n = len(df)
    if n == 0:
        return [_issue("empty", None, "high", "The dataset has no rows.", "Nothing can be analysed.", "Check the source file.")]
    dup = int(df.duplicated().sum())
    if dup:
        issues.append(_issue("duplicates", None, "warning", f"{dup} duplicate rows ({100 * dup / n:.1f}%).",
                             "Duplicates can inflate totals and counts.", "Review them, then remove duplicates if they are not legitimate repeats."))
    profiles = {c["name"]: c for c in (prof or {"columns_profile": [profile_column(df[c]) for c in df.columns]})["columns_profile"]}
    for c in df.columns:
        s, p = df[c], profiles[c]
        kind = p["kind"]
        if p["missing"]:
            sev = "high" if p["missing_pct"] > 30 else "warning" if p["missing_pct"] > 5 else "info"
            issues.append(_issue("missing", c, sev, f"{p['missing']} missing values ({p['missing_pct']}%).",
                                 "Missing values can bias statistics and break models.",
                                 "Fill (median/mode/constant), drop rows, or leave as-is if missingness is meaningful."))
        if kind == "constant":
            issues.append(_issue("constant", c, "info", "Column has a single constant value.", "It carries no information.", "Consider dropping it."))
        if kind == "empty":
            issues.append(_issue("empty_column", c, "warning", "Column is completely empty.", "It carries no information.", "Consider dropping it."))
        if p["stored_as_text"]:
            code = "numeric_as_text" if kind == "numeric" else "date_as_text"
            issues.append(_issue(code, c, "warning", f"Looks {'numeric' if kind == 'numeric' else 'like dates'} but is stored as text.",
                                 "Text columns cannot be summed, averaged or sorted chronologically.", "Convert the data type."))
        if is_text(s) and kind in {"categorical", "text", "boolean"}:
            sv = s.dropna().astype(str)
            if (sv != sv.str.strip()).any() or sv.str.contains(r"\s{2,}").any():
                issues.append(_issue("whitespace", c, "warning", "Leading/trailing/double whitespace found.",
                                     "'Delhi' and 'Delhi ' are counted as different categories.", "Trim whitespace."))
            if kind == "categorical" and p["unique"] <= 500:
                key = sv.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                grp = sv.groupby(key).nunique()
                bad = grp[grp > 1]
                if len(bad):
                    ex = [sorted(sv[key == k].unique())[:3] for k in bad.index[:2]]
                    issues.append(_issue("label_variants", c, "warning", f"{len(bad)} label(s) appear in several spellings, e.g. {ex}.",
                                         "Inconsistent labels split one category into several.", "Standardize category labels."))
        if kind == "numeric":
            x = pd.to_numeric(s, errors="coerce") if is_text(s) else s
            nm = _NONNEG_NAME.search(str(c))
            if nm and (x < 0).any():
                issues.append(_issue("suspicious_values", c, "warning", f"{int((x < 0).sum())} negative values in '{c}'.",
                                     "Negative values may be errors or legitimate refunds/returns.", "Verify with the data owner before changing."))
            if re.search(r"(^|_| )age($|_| )", str(c).lower()) and ((x > 120).any() or (x < 0).any()):
                issues.append(_issue("suspicious_values", c, "high", "Impossible age values (<0 or >120).", "Likely entry errors.", "Verify and correct."))
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and len(x.dropna()) >= 10:
                out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
                if out:
                    issues.append(_issue("outliers", c, "info", f"{out} potential outliers by the 1.5xIQR rule ({100 * out / n:.1f}%).",
                                         "Outliers can distort means; but they may be genuine (e.g. big orders).", "Inspect before deciding to cap or remove."))
        if kind == "datetime" and is_text(s):
            d = pd.to_datetime(s.dropna(), errors="coerce", format="mixed")
            bad = int(d.isna().sum())
            if bad:
                issues.append(_issue("bad_dates", c, "warning", f"{bad} values cannot be parsed as dates.", "They become missing after conversion.", "Fix or drop them."))
        if kind == "identifier" and s.dropna().duplicated().any():
            issues.append(_issue("dup_ids", c, "warning", f"Identifier column has {int(s.dropna().duplicated().sum())} repeated values.",
                                 "IDs are usually unique; repeats may indicate duplicate or multi-row records.", "Check whether repeats are expected."))
        if kind == "categorical" and p["unique"] > 50:
            issues.append(_issue("high_cardinality", c, "info", f"{p['unique']} distinct categories.", "Hard to chart/model directly.", "Group rare categories or use Top-N."))
    # redundant (identical) columns
    cols = list(df.columns)
    if len(cols) <= 60:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                try:
                    if df[cols[i]].equals(df[cols[j]]):
                        issues.append(_issue("redundant", cols[j], "info", f"Identical to column '{cols[i]}'.", "Redundant columns add noise.", "Drop one."))
                except Exception:  # noqa: BLE001
                    pass
    return issues


def find_missing(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame({"column": m.index, "missing": m.values, "missing_pct": (100 * m / max(len(df), 1)).round(2).values,
                        "non_missing": (len(df) - m).values})
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def find_duplicates(df: pd.DataFrame, subset: list[str] | None = None, limit: int = 50) -> tuple[int, pd.DataFrame]:
    mask = df.duplicated(subset=subset or None, keep="first")
    allmask = df.duplicated(subset=subset or None, keep=False)
    sample = df[allmask]
    if subset:
        sample = sample.sort_values(subset)
    return int(mask.sum()), sample.head(limit)


def data_dictionary(df: pd.DataFrame, prof: dict | None = None) -> pd.DataFrame:
    prof = prof or profile_dataset(df)
    rows = []
    for p in prof["columns_profile"]:
        ex = df[p["name"]].dropna().head(3).astype(str).tolist()
        rows.append({"column": p["name"], "dtype": p["dtype"], "detected_type": p["kind"], "non_null": p["count"],
                     "missing": p["missing"], "missing_pct": p["missing_pct"], "unique": p["unique"], "examples": ", ".join(ex)})
    return pd.DataFrame(rows)


def schema_summary(df: pd.DataFrame, prof: dict | None = None, max_cols: int = 80) -> list[dict]:
    prof = prof or profile_dataset(df)
    return [{"name": p["name"], "type": p["kind"], "dtype": p["dtype"], "missing_pct": p["missing_pct"], "unique": p["unique"]}
            for p in prof["columns_profile"][:max_cols]]
