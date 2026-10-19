"""Row filtering with a small, safe, structured condition language (no eval)."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .util import AnalystError, ensure_datetime, is_datetime, is_numeric, resolve_column

OPERATORS = [
    "==", "!=", ">", ">=", "<", "<=", "in", "not in",
    "contains", "not contains", "startswith", "endswith", "between", "is null", "not null",
]
_ALIASES = {"=": "==", "eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
            "isnull": "is null", "notnull": "not null", "is not null": "not null"}


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        raise AnalystError(f"'{v}' is not a number.")


def _clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def condition_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
    col = resolve_column(df, cond.get("column"))
    op = str(cond.get("op", "==")).strip().lower()
    op = _ALIASES.get(op, op)
    if op not in OPERATORS:
        raise AnalystError(f"Unknown operator '{op}'. Use one of: {OPERATORS}")
    s = df[col]
    val = cond.get("value")
    if op == "is null":
        return s.isna()
    if op == "not null":
        return s.notna()

    listy = op in {"in", "not in", "between"}
    if isinstance(val, (list, tuple)):
        vals = list(val)
    elif listy:
        vals = [v.strip() for v in str(val).split(",")]
    else:
        vals = [val]
    if op == "between" and len(vals) != 2:
        raise AnalystError("'between' needs exactly two values [low, high].")
    order_op = op in {">", ">=", "<", "<=", "between"}

    if is_numeric(s):
        m = _apply(s, op, [_num(x) for x in vals])
    elif is_datetime(s):
        m = _apply(s, op, [pd.Timestamp(x) for x in vals])
    elif order_op:  # text column compared with < > : try numbers, then dates
        try:
            v = [_num(x) for x in vals]
            m = _apply(pd.to_numeric(s, errors="coerce"), op, v)
        except AnalystError:
            m = _apply(ensure_datetime(s), op, [pd.Timestamp(x) for x in vals])
    else:
        st = s.astype(object).map(lambda x: _clean_text(x) if pd.notna(x) else None)
        v = [_clean_text(x) for x in vals]
        if op in {"contains", "not contains"}:
            m = st.map(lambda x: isinstance(x, str) and v[0] in x)
            m = ~m if op == "not contains" else m
        elif op == "startswith":
            m = st.map(lambda x: isinstance(x, str) and x.startswith(v[0]))
        elif op == "endswith":
            m = st.map(lambda x: isinstance(x, str) and x.endswith(v[0]))
        else:
            m = _apply(st, op, v)
    return m.fillna(False).astype(bool)


def _apply(s: pd.Series, op: str, v: list) -> pd.Series:
    if op == "==":
        return s == v[0]
    if op == "!=":
        return (s != v[0]) & s.notna()
    if op == ">":
        return s > v[0]
    if op == ">=":
        return s >= v[0]
    if op == "<":
        return s < v[0]
    if op == "<=":
        return s <= v[0]
    if op == "between":
        return (s >= v[0]) & (s <= v[1])
    if op == "in":
        return s.isin(v)
    if op == "not in":
        return ~s.isin(v) & s.notna()
    raise AnalystError(f"Operator '{op}' is not valid for this column type.")


def apply_conditions(df: pd.DataFrame, conditions: list[dict] | None, combine: str = "and") -> pd.DataFrame:
    """Return rows matching the conditions (text equality is case/whitespace-insensitive)."""
    if not conditions:
        return df
    masks = [condition_mask(df, c) for c in conditions]
    mask = masks[0]
    for m in masks[1:]:
        mask = (mask | m) if combine.lower() == "or" else (mask & m)
    return df[mask]


def describe_conditions(conditions: list[dict], combine: str = "and") -> str:
    if not conditions:
        return "no filters"
    parts = [f"{c.get('column')} {c.get('op', '==')} {c.get('value', '')}".strip() for c in conditions]
    return f" {combine.upper()} ".join(parts)
