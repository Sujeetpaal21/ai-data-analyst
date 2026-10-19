"""Small shared helpers: errors, dtype predicates, column resolution, JSON conversion."""
from __future__ import annotations

import datetime as _dt
import difflib
import math
import re
from typing import Any

import numpy as np
import pandas as pd


class AnalystError(Exception):
    """Expected, user-presentable error."""


class ColumnNotFoundError(AnalystError):
    pass


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_text(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve_column(df: pd.DataFrame, name: Any) -> str:
    """Map a (possibly slightly different) column name to a real column or raise."""
    cols = list(df.columns)
    if name in cols:
        return name
    low = {str(c).strip().lower(): c for c in cols}
    if str(name).strip().lower() in low:
        return low[str(name).strip().lower()]
    n = {norm(c): c for c in cols}
    if norm(name) in n:
        return n[norm(name)]
    m = difflib.get_close_matches(norm(name), list(n), n=1, cutoff=0.85)
    if m:
        return n[m[0]]
    raise ColumnNotFoundError(f"Column '{name}' not found. Available columns: {cols[:60]}")


def resolve_columns(df: pd.DataFrame, names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    return [resolve_column(df, n) for n in names]


def ensure_datetime(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    if is_datetime(s):
        return s
    return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)


def to_jsonable(obj: Any, max_items: int = 200) -> Any:
    """Convert numpy/pandas objects to JSON-safe python values (NaN -> None)."""
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Period, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_items) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, max_items) for v in list(obj)[:max_items]]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.head(max_items).to_dict(orient="records"), max_items)
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.head(max_items).to_dict(), max_items)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), max_items)
    return str(obj)


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f):
        return "n/a"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:,.{digits}f}"
