"""Cleaning engine. Every operation works on a copy, returns a CleanResult and is logged.

Operations that remove rows/columns or may lose/alter values are flagged ``risky`` and need
explicit confirmation before they are applied to the working copy.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .filtering import apply_conditions, describe_conditions
from .profiling import column_kind
from .util import AnalystError, ensure_datetime, is_numeric, is_text, resolve_column, resolve_columns


@dataclass
class CleanResult:
    df: pd.DataFrame
    description: str
    risky: bool = False
    warnings: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cells_changed: int = 0


def _count_changes(before: pd.DataFrame, after: pd.DataFrame) -> int:
    if list(before.columns) != list(after.columns) or not before.index.is_unique:
        return 0
    common = before.index.intersection(after.index)
    total = 0
    for c in before.columns:
        try:
            a, b = after.loc[common, c].astype(object), before.loc[common, c].astype(object)
            total += int((~((a == b) | (a.isna() & b.isna()))).sum())
        except Exception:  # noqa: BLE001
            continue
    return total


def _finish(before, after, desc, risky=False, warns=None) -> CleanResult:
    return CleanResult(after, desc, risky, warns or [], len(before), len(after), _count_changes(before, after))


def diff_preview(before: pd.DataFrame, after: pd.DataFrame, max_cells: int = 15) -> dict:
    """Summarise what an operation changed: shapes, columns, removed rows and sample changed cells."""
    out: dict[str, Any] = {"shape_before": before.shape, "shape_after": after.shape,
                           "columns_added": [c for c in after.columns if c not in before.columns],
                           "columns_removed": [c for c in before.columns if c not in after.columns],
                           "rows_removed": len(before) - len(after)}
    rows = []
    if before.index.is_unique:
        common = before.index.intersection(after.index)
        for c in [c for c in before.columns if c in after.columns]:
            try:
                a, b = after.loc[common, c].astype(object), before.loc[common, c].astype(object)
                neq = ~((a == b) | (a.isna() & b.isna()))
            except Exception:  # noqa: BLE001
                continue
            for idx in neq[neq].index[: max_cells - len(rows)]:
                rows.append({"row": idx, "column": c, "before": b.loc[idx], "after": a.loc[idx]})
            if len(rows) >= max_cells:
                break
    out["changed_cells_sample"] = pd.DataFrame(rows, columns=["row", "column", "before", "after"])
    removed = before.index.difference(after.index) if before.index.is_unique else []
    out["removed_rows_sample"] = before.loc[removed].head(max_cells) if len(removed) else before.iloc[0:0]
    return out


def _map_str(s: pd.Series, fn: Callable[[str], str]) -> pd.Series:
    """Apply fn to string cells only; non-strings and missing values are left untouched."""
    return s.map(lambda v: fn(v) if isinstance(v, str) else v)


# ----------------------------------------------------------------------------- operations
def remove_duplicates(df, subset=None, keep="first"):
    subset = resolve_columns(df, subset) or None
    out = df.drop_duplicates(subset=subset, keep=keep)
    n = len(df) - len(out)
    return _finish(df, out, f"Removed {n} duplicate row(s) (keep={keep}, subset={subset or 'all columns'}).", risky=n > 0)


def drop_missing_rows(df, columns=None, how="any"):
    cols = resolve_columns(df, columns) or None
    out = df.dropna(subset=cols, how=how)
    n = len(df) - len(out)
    return _finish(df, out, f"Dropped {n} row(s) with missing values in {cols or 'any column'} (how={how}).", risky=n > 0)


def fill_missing(df, columns=None, method="median", value=None):
    method = str(method).lower()
    out = df.copy()
    cols = resolve_columns(df, columns)
    auto = not cols
    if auto:
        cols = [c for c in df.columns if df[c].isna().any()]
        if method in {"mean", "median"}:
            cols = [c for c in cols if is_numeric(df[c])]
    if not cols:
        raise AnalystError("No columns with missing values match this operation.")
    filled, notes = {}, []
    for c in cols:
        n_na = int(out[c].isna().sum())
        if n_na == 0:
            continue
        s = out[c]
        if method in {"mean", "median"}:
            if not is_numeric(s):
                if auto:
                    continue
                raise AnalystError(f"Cannot use {method} on non-numeric column '{c}'. Use mode or a constant value.")
            fv = s.mean() if method == "mean" else s.median()
            out[c] = s.fillna(fv)
        elif method == "mode":
            m = s.mode(dropna=True)
            if m.empty:
                notes.append(f"'{c}' has no mode; skipped.")
                continue
            fv = m.iloc[0]
            out[c] = s.fillna(fv)
        elif method in {"value", "constant"}:
            if value is None:
                raise AnalystError("Provide a value to fill with.")
            fv = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0] if is_numeric(s) else value
            if is_numeric(s) and pd.isna(fv):
                raise AnalystError(f"'{value}' is not numeric but column '{c}' is.")
            out[c] = s.fillna(fv)
        elif method in {"ffill", "bfill"}:
            out[c] = s.ffill() if method == "ffill" else s.bfill()
            fv = method
        else:
            raise AnalystError("Method must be one of: mean, median, mode, value, ffill, bfill.")
        filled[c] = (n_na, fv)
    if not filled:
        raise AnalystError("Nothing to fill: the selected columns have no missing values.")
    parts = [f"{c}: {n} filled with {fv if isinstance(fv, str) else round(float(fv), 4) if isinstance(fv, (int, float, np.number)) else fv}" for c, (n, fv) in filled.items()]
    return _finish(df, out, f"Filled missing values using {method} -> " + "; ".join(parts), warns=notes)


def trim_text(df, columns=None, case="none"):
    cols = resolve_columns(df, columns) or [c for c in df.columns if is_text(df[c])]
    out = df.copy()
    caser = {"lower": str.lower, "upper": str.upper, "title": str.title, "none": lambda x: x}.get(str(case).lower())
    if caser is None:
        raise AnalystError("case must be none, lower, upper or title.")
    for c in cols:
        s = out[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        out[c] = _map_str(s, lambda v: caser(re.sub(r"\s+", " ", v).strip()))
    return _finish(df, out, f"Trimmed/normalized whitespace (case={case}) in {len(cols)} text column(s): {cols[:10]}.")


def standardize_labels(df, column, mapping=None):
    c = resolve_column(df, column)
    out = df.copy()
    s = out[c]
    if mapping:
        out[c] = _map_str(s, lambda v: mapping.get(v, mapping.get(v.strip(), v)))
        desc = f"Applied label mapping in '{c}': {mapping}"
    else:
        sv = s.dropna().astype(str)
        key = sv.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        canon = {}
        for k, grp in sv.groupby(key):
            vc = grp.str.strip().value_counts()
            canon[k] = vc.index[0]
        out[c] = s.map(lambda v: canon[re.sub(r"\s+", " ", v.strip().lower())] if isinstance(v, str) else v)
        desc = f"Standardized label variants in '{c}' to the most frequent spelling."
    return _finish(df, out, desc)


_TARGETS = {"numeric", "integer", "text", "category", "boolean", "datetime"}


def convert_type(df, column, target="numeric", date_format=None, dayfirst=False):
    c = resolve_column(df, column)
    target = str(target).lower()
    if target not in _TARGETS:
        raise AnalystError(f"target must be one of {sorted(_TARGETS)}")
    out, s = df.copy(), df[c]
    if target in {"numeric", "integer"}:
        cleaned = _map_str(s, lambda v: re.sub(r"[,\s$₹€£%]", "", v)) if is_text(s) else s
        new = pd.to_numeric(cleaned, errors="coerce")
        if target == "integer":
            if (new.dropna() % 1 != 0).any():
                raise AnalystError("Some values have decimals; use 'numeric' instead of 'integer'.")
            new = new.astype("Int64")
    elif target == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new = pd.to_datetime(s, errors="coerce", format=date_format or "mixed", dayfirst=bool(dayfirst))
    elif target == "boolean":
        mp = {"true": True, "yes": True, "y": True, "1": True, "t": True, "false": False, "no": False, "n": False, "0": False, "f": False}
        new = s.map(lambda v: mp.get(str(v).strip().lower()) if pd.notna(v) else None).astype("boolean")
    elif target == "category":
        new = s.astype("category")
    else:
        new = s.map(lambda v: v if pd.isna(v) else str(v))
    lost = int((new.isna() & s.notna()).sum())
    out[c] = new
    warns = [f"{lost} value(s) could not be converted and became missing."] if lost else []
    return _finish(df, out, f"Converted '{c}' to {target}." + (f" {lost} value(s) became missing." if lost else ""), risky=lost > 0, warns=warns)


def parse_dates(df, column, date_format=None, dayfirst=False):
    return convert_type(df, column, "datetime", date_format, dayfirst)


def rename_columns(df, mapping):
    if not mapping:
        raise AnalystError("Provide a mapping of old name -> new name.")
    m = {resolve_column(df, k): v for k, v in mapping.items()}
    new_names = [m.get(c, c) for c in df.columns]
    if len(set(new_names)) != len(new_names):
        raise AnalystError("Renaming would create duplicate column names.")
    return _finish(df, df.rename(columns=m), f"Renamed columns: {m}")


def reorder_columns(df, order):
    first = resolve_columns(df, order)
    rest = [c for c in df.columns if c not in first]
    return _finish(df, df[first + rest], f"Moved {first} to the front.")


def drop_columns(df, columns):
    cols = resolve_columns(df, columns)
    if not cols:
        raise AnalystError("Select at least one column.")
    if len(cols) >= df.shape[1]:
        raise AnalystError("Cannot drop all columns.")
    return _finish(df, df.drop(columns=cols), f"Dropped column(s): {cols}.", risky=True)


def filter_rows(df, conditions, combine="and"):
    out = apply_conditions(df, conditions, combine)
    return _finish(df, out, f"Kept {len(out)} of {len(df)} rows where {describe_conditions(conditions, combine)}.", risky=len(out) < len(df))


def handle_outliers(df, column, method="iqr", action="cap", factor=None):
    c = resolve_column(df, column)
    if not is_numeric(df[c]):
        raise AnalystError(f"'{c}' is not numeric.")
    s = df[c]
    if method == "zscore":
        f = float(factor or 3.0)
        mu, sd = s.mean(), s.std()
        lo, hi = mu - f * sd, mu + f * sd
    else:
        f = float(factor or 1.5)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo, hi = q1 - f * (q3 - q1), q3 + f * (q3 - q1)
    mask = (s < lo) | (s > hi)
    n = int(mask.sum())
    if n == 0:
        raise AnalystError(f"No outliers found in '{c}' with {method} (factor {f}).")
    out = df.copy()
    if action == "cap":
        out[c] = s.clip(lower=lo, upper=hi)
    elif action == "remove":
        out = out[~mask]
    elif action == "set_missing":
        out.loc[mask, c] = np.nan
    else:
        raise AnalystError("action must be cap, remove or set_missing.")
    return _finish(df, out, f"{action}: {n} outlier(s) in '{c}' outside [{lo:.4g}, {hi:.4g}] ({method}, factor {f}). Outliers are not necessarily errors.", risky=True)


def custom_rule(df, column, conditions, value, combine="and"):
    """User-defined rule: where <conditions> then set <column> = <value>."""
    c = resolve_column(df, column)
    mask = apply_conditions(df, conditions, combine).index
    out = df.copy()
    v = value
    if is_numeric(df[c]):
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(v):
            raise AnalystError(f"'{value}' is not numeric but '{c}' is.")
    out.loc[mask, c] = v
    return _finish(df, out, f"Rule: where {describe_conditions(conditions, combine)} set '{c}' = {v} ({len(mask)} rows).", risky=True)


# ----------------------------------------------------------------------------- registry
@dataclass
class Op:
    fn: Callable
    label: str
    help: str
    params: list[dict]


def _p(name, kind, **kw):
    return {"name": name, "kind": kind, **kw}


OPERATIONS: dict[str, Op] = {
    "remove_duplicates": Op(remove_duplicates, "Remove duplicate rows", "Drops repeated rows (keeps first/last occurrence).",
                            [_p("subset", "columns", optional=True), _p("keep", "choice", choices=["first", "last"], default="first")]),
    "drop_missing_rows": Op(drop_missing_rows, "Drop rows with missing values", "Removes rows that have missing values.",
                            [_p("columns", "columns", optional=True), _p("how", "choice", choices=["any", "all"], default="any")]),
    "fill_missing": Op(fill_missing, "Fill missing values", "Fill with mean/median/mode/constant/ffill/bfill. Empty column list = all suitable columns.",
                       [_p("columns", "columns", optional=True), _p("method", "choice", choices=["median", "mean", "mode", "value", "ffill", "bfill"], default="median"),
                        _p("value", "text", optional=True)]),
    "trim_text": Op(trim_text, "Trim whitespace / normalize text", "Strips and collapses whitespace; optional case change.",
                    [_p("columns", "columns", optional=True), _p("case", "choice", choices=["none", "lower", "upper", "title"], default="none")]),
    "standardize_labels": Op(standardize_labels, "Standardize category labels", "Merge spelling/case variants into the most frequent label, or apply your mapping.",
                             [_p("column", "column"), _p("mapping", "mapping", optional=True)]),
    "convert_type": Op(convert_type, "Convert data type", "Convert a column to numeric/integer/text/category/boolean/datetime.",
                       [_p("column", "column"), _p("target", "choice", choices=sorted(_TARGETS), default="numeric"),
                        _p("date_format", "text", optional=True), _p("dayfirst", "bool", default=False)]),
    "parse_dates": Op(parse_dates, "Parse date/time column", "Parse text into datetimes.",
                      [_p("column", "column"), _p("date_format", "text", optional=True), _p("dayfirst", "bool", default=False)]),
    "rename_columns": Op(rename_columns, "Rename columns", "Rename columns (old=new per line).", [_p("mapping", "mapping")]),
    "reorder_columns": Op(reorder_columns, "Reorder columns", "Move selected columns to the front.", [_p("order", "columns")]),
    "drop_columns": Op(drop_columns, "Remove columns", "Deletes columns from the working copy.", [_p("columns", "columns")]),
    "filter_rows": Op(filter_rows, "Keep only matching rows (permanent)", "Removes rows that do not match the condition.", [_p("conditions", "conditions")]),
    "handle_outliers": Op(handle_outliers, "Handle outliers", "Cap, remove or blank out outliers (IQR or z-score).",
                          [_p("column", "column"), _p("method", "choice", choices=["iqr", "zscore"], default="iqr"),
                           _p("action", "choice", choices=["cap", "remove", "set_missing"], default="cap"), _p("factor", "number", optional=True)]),
    "custom_rule": Op(custom_rule, "Custom rule: where <condition> set <column> = <value>", "User-defined value replacement rule.",
                      [_p("column", "column"), _p("conditions", "conditions"), _p("value", "text")]),
}


def run_operation(session, op_name: str, params: dict | None = None, confirmed: bool = False, dry_run: bool = False) -> dict:
    """Run a cleaning operation on the session's working copy.

    Returns {"status": applied|preview|needs_confirmation|error, "result": CleanResult, "message": str}.
    """
    params = dict(params or {})
    op = OPERATIONS.get(op_name)
    if op is None:
        return {"status": "error", "message": f"Unknown operation '{op_name}'. Available: {list(OPERATIONS)}"}
    try:
        res = op.fn(session.working, **params)
    except AnalystError as e:
        return {"status": "error", "message": str(e)}
    except TypeError as e:
        return {"status": "error", "message": f"Invalid parameters for {op_name}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"{op_name} failed: {e}"}
    if dry_run:
        return {"status": "preview", "result": res, "message": res.description}
    if res.risky and not confirmed:
        session.pending = {"operation": op_name, "params": params, "description": res.description}
        return {"status": "needs_confirmation", "result": res,
                "message": f"NOT APPLIED - this operation is potentially destructive and needs user confirmation: {res.description}"}
    session.commit(res.df, res.description, op_name, params, rows_before=res.rows_before, cells_changed=res.cells_changed)
    return {"status": "applied", "result": res, "message": res.description}
