"""Analytical computations. All numbers come from pandas/NumPy/SciPy/scikit-learn - never from an LLM."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .profiling import column_kind, looks_like_id
from .util import AnalystError, ensure_datetime, is_datetime, is_numeric, is_text, resolve_column, resolve_columns

ALLOWED_AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std"}
FREQ_ALIASES = {"day": "D", "daily": "D", "d": "D", "week": "W", "weekly": "W", "w": "W", "month": "M", "monthly": "M",
                "m": "M", "quarter": "Q", "quarterly": "Q", "q": "Q", "year": "Y", "yearly": "Y", "annual": "Y", "y": "Y", "a": "Y"}
PERIODS_PER_YEAR = {"D": 365, "W": 52, "M": 12, "Q": 4, "Y": 1}


def norm_freq(freq: str | None) -> str | None:
    if not freq:
        return None
    f = FREQ_ALIASES.get(str(freq).strip().lower())
    if f is None:
        raise AnalystError(f"Unknown frequency '{freq}'. Use day, week, month, quarter or year.")
    return f


def auto_freq(dates: pd.Series) -> str:
    d = dates.dropna()
    if d.empty:
        return "M"
    span = (d.max() - d.min()).days
    return "D" if span <= 60 else "W" if span <= 180 else "M" if span <= 365 * 4 else "Q" if span <= 365 * 12 else "Y"


def period_start(dates: pd.Series, freq: str) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dates.dt.to_period(freq).dt.to_timestamp()


def numeric_series(s: pd.Series) -> pd.Series:
    """Numeric view of a column (handles numbers stored as text such as '1,200')."""
    if is_numeric(s):
        return s
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[,\s$₹€£%]", "", regex=True), errors="coerce")


def analysable_numeric_cols(df: pd.DataFrame, include_boolean: bool = True) -> list[str]:
    kinds = {"numeric", "boolean"} if include_boolean else {"numeric"}
    return [c for c in df.columns if column_kind(df[c]) in kinds and (is_numeric(df[c]) or column_kind(df[c]) == "numeric")]


# ------------------------------------------------------------------------------ statistics
def numeric_stats(s: pd.Series) -> dict:
    x = numeric_series(s).dropna().astype(float)
    n = len(x)
    if n == 0:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": n, "sum": x.sum(), "mean": x.mean(), "median": x.median(),
        "mode": (x.mode().iloc[0] if not x.mode().empty else None), "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
        "variance": x.var() if n > 1 else None, "std": x.std() if n > 1 else None,
        "q1": x.quantile(0.25), "q3": x.quantile(0.75), "p5": x.quantile(0.05), "p95": x.quantile(0.95),
        "skewness": float(stats.skew(x)) if n > 2 and x.std() > 0 else None,
        "kurtosis": float(stats.kurtosis(x)) if n > 3 and x.std() > 0 else None,
    }
    if n > 2 and x.std() > 0:
        lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
        out["mean_ci95"] = [float(lo), float(hi)]
        samp = x.sample(5000, random_state=0) if n > 5000 else x
        p = float(stats.shapiro(samp).pvalue)
        out["normality_p"] = p
        out["looks_normal"] = bool(p > 0.05)
    sk = out.get("skewness")
    out["distribution"] = "n/a" if sk is None else "right-skewed" if sk > 0.5 else "left-skewed" if sk < -0.5 else "roughly symmetric"
    return out


def categorical_stats(s: pd.Series, top: int = 20) -> dict:
    nn = s.dropna()
    vc = nn.astype(str).value_counts()
    return {"count": len(nn), "unique": int(vc.size), "mode": vc.index[0] if len(vc) else None,
            "frequency": [{"value": k, "count": int(v), "pct": round(100 * v / len(nn), 2)} for k, v in vc.head(top).items()]}


def calculate_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    cols = resolve_columns(df, columns) or list(df.columns)
    res: dict[str, Any] = {"numeric": {}, "categorical": {}, "datetime": {}, "skipped": {}}
    for c in cols:
        k = column_kind(df[c])
        if k == "numeric":
            res["numeric"][c] = numeric_stats(df[c])
        elif k == "boolean":
            res["categorical"][c] = categorical_stats(df[c])
        elif k == "categorical":
            res["categorical"][c] = categorical_stats(df[c])
        elif k == "datetime":
            d = ensure_datetime(df[c]).dropna()
            res["datetime"][c] = {"count": len(d), "min": d.min(), "max": d.max(), "span_days": (d.max() - d.min()).days if len(d) else None}
        else:
            res["skipped"][c] = f"{k} column: numeric/categorical statistics are not meaningful"
    return res


def stats_table(st: dict) -> pd.DataFrame:
    rows = []
    for c, s in st["numeric"].items():
        rows.append({"column": c, **{k: v for k, v in s.items() if k in {"count", "mean", "median", "std", "min", "q1", "q3", "max", "skewness", "distribution"}}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ grouping
def group_and_aggregate(df: pd.DataFrame, by: list[str], metrics: dict[str, Any] | None = None, freq: str | None = None,
                        sort_by: str | None = None, ascending: bool | None = None, top_n: int | None = None,
                        pct_of_total: bool = False) -> tuple[pd.DataFrame, dict]:
    by = resolve_columns(df, by)
    if not by:
        raise AnalystError("Provide at least one column to group by.")
    metrics = metrics or {"*": "count"}
    work = df.copy()
    freq = norm_freq(freq)
    time_key = False
    for c in by:
        if is_datetime(work[c]) or (freq and column_kind(work[c]) == "datetime"):
            d = ensure_datetime(work[c])
            f = freq or auto_freq(d)
            work[c] = period_start(d, f)
            time_key, freq = True, f
    named: dict[str, pd.NamedAgg] = {}
    need_count = False
    for col, aggs in metrics.items():
        for agg in ([aggs] if isinstance(aggs, str) else list(aggs)):
            agg = str(agg).lower()
            if agg not in ALLOWED_AGGS:
                raise AnalystError(f"Aggregation '{agg}' not allowed. Use one of {sorted(ALLOWED_AGGS)}.")
            if col == "*":
                if agg != "count":
                    raise AnalystError("'*' only supports count.")
                need_count = True
                continue
            c = resolve_column(work, col)
            if agg in {"sum", "mean", "median", "std", "min", "max"} and not is_numeric(work[c]):
                if agg in {"min", "max"} and is_datetime(work[c]):
                    pass
                else:
                    num = numeric_series(work[c])
                    if num.notna().sum() == 0:
                        raise AnalystError(f"Cannot compute {agg} of non-numeric column '{c}'.")
                    work[c] = num
            named[f"{agg}_{c}"] = pd.NamedAgg(column=c, aggfunc=agg)
    keys_na = int(work[by].isna().any(axis=1).sum())
    g = work.groupby(by, dropna=True, observed=True)
    out = g.agg(**named) if named else None
    if need_count or out is None:
        cnt = g.size().rename("row_count")
        out = cnt.to_frame() if out is None else out.join(cnt)
    out = out.reset_index()
    metric_cols = [c for c in out.columns if c not in by]
    if pct_of_total:
        for m in [m for m in metric_cols if m.startswith("sum_") or m == "row_count"]:
            tot = out[m].sum()
            out[f"pct_{m}"] = (100 * out[m] / tot).round(2) if tot else np.nan
    if time_key and not sort_by:
        out = out.sort_values(by[0])
    else:
        sb = metric_cols[0]
        if sort_by:
            try:
                sb = resolve_column(out, sort_by)
            except AnalystError:
                pass
        out = out.sort_values(sb, ascending=bool(ascending) if ascending is not None else False)
    if top_n:
        out = out.head(int(top_n))
    info = {"rows_used": len(df) - keys_na, "rows_excluded_missing_keys": keys_na, "metrics": metric_cols, "by": by, "freq": freq,
            "groups": len(out)}
    return out.reset_index(drop=True), info


# ------------------------------------------------------------------------------ correlation / association
def _strength(r: float) -> str:
    a = abs(r)
    return "very weak" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.8 else "very strong"


def correlation_analysis(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson", top_n: int = 10) -> dict:
    cols = resolve_columns(df, columns) or analysable_numeric_cols(df, include_boolean=False)
    cols = [c for c in cols if numeric_series(df[c]).nunique() > 1 and not (column_kind(df[c]) == "identifier")][:40]
    if len(cols) < 2:
        raise AnalystError("Need at least two numeric (non-identifier, non-constant) columns for correlation.")
    num = pd.DataFrame({c: numeric_series(df[c]) for c in cols})
    method = method.lower() if method.lower() in {"pearson", "spearman", "kendall"} else "pearson"
    corr = num.corr(method=method)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if not np.isnan(corr.loc[a, b]):
                pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: -abs(t[2]))
    rows = []
    for a, b, r in pairs[:top_n]:
        sub = num[[a, b]].dropna()
        p = None
        if len(sub) > 2:
            fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
            p = float(fn(sub[a], sub[b])[1])
        rows.append({"column_a": a, "column_b": b, "correlation": round(r, 4), "strength": _strength(r),
                     "p_value": p, "n": len(sub), "significant_5pct": (p is not None and p < 0.05)})
    return {"matrix": corr.round(4), "pairs": pd.DataFrame(rows), "method": method,
            "note": "Correlation measures association, not causation."}


def _eta(cat: pd.Series, num: pd.Series) -> float:
    d = pd.DataFrame({"c": cat, "x": num}).dropna()
    if len(d) < 3 or d["c"].nunique() < 2:
        return np.nan
    tot = ((d["x"] - d["x"].mean()) ** 2).sum()
    if tot == 0:
        return np.nan
    gm = d.groupby("c")["x"].agg(["count", "mean"])
    between = (gm["count"] * (gm["mean"] - d["x"].mean()) ** 2).sum()
    return float(np.sqrt(between / tot))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return np.nan
    ct = pd.crosstab(d["a"], d["b"])
    if min(ct.shape) < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (len(d) * (min(ct.shape) - 1))))


def target_associations(df: pd.DataFrame, target: str, top_n: int = 15) -> dict:
    t = resolve_column(df, target)
    tk = column_kind(df[t])
    t_num = tk == "numeric"
    if tk in {"identifier", "constant", "empty", "text"}:
        raise AnalystError(f"'{t}' ({tk}) is not suitable as a target.")
    if tk == "categorical" and df[t].nunique() > 20:
        raise AnalystError(f"Target '{t}' has too many categories ({df[t].nunique()}).")
    rows = []
    for c in df.columns:
        if c == t:
            continue
        k = column_kind(df[c])
        if k not in {"numeric", "categorical", "boolean"}:
            continue
        if k == "categorical" and df[c].nunique() > 30:
            continue
        n = int(df[[c, t]].dropna().shape[0])
        if t_num and k == "numeric":
            r = df[[c, t]].apply(numeric_series).corr(method="spearman").iloc[0, 1]
            m, val, direction = "spearman_r", r, ("positive" if r > 0 else "negative") if pd.notna(r) else ""
        elif t_num:
            m, val, direction = "eta (correlation ratio)", _eta(df[c], numeric_series(df[t])), ""
        elif k == "numeric":
            m, val, direction = "eta (correlation ratio)", _eta(df[t], numeric_series(df[c])), ""
        else:
            m, val, direction = "cramers_v", _cramers_v(df[c], df[t]), ""
        if val is not None and not np.isnan(val):
            rows.append({"feature": c, "measure": m, "association": round(float(val), 4), "strength": _strength(val),
                         "direction": direction, "n": n})
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.reindex(tab["association"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
    return {"target": t, "table": tab, "note": "Associations are not proof of causation; confounders are not controlled."}


# ------------------------------------------------------------------------------ time series
def time_series_analysis(df: pd.DataFrame, date_col: str, value_col: str | None = None, freq: str | None = None,
                         agg: str = "sum", last_n_periods: int | None = None) -> dict:
    dc = resolve_column(df, date_col)
    dates = ensure_datetime(df[dc])
    if dates.notna().sum() < 2:
        raise AnalystError(f"'{dc}' does not contain enough valid dates.")
    f = norm_freq(freq) or auto_freq(dates)
    d = pd.DataFrame({"period": period_start(dates, f)})
    if value_col and value_col != "*":
        vc = resolve_column(df, value_col)
        d["v"] = numeric_series(df[vc]).values
        agg = agg if agg in ALLOWED_AGGS else "sum"
        s = d.dropna(subset=["period"]).groupby("period")["v"].agg(agg)
        label = f"{agg}_{vc}"
    else:
        s = d.dropna(subset=["period"]).groupby("period").size()
        label, vc = "row_count", None
    s = s.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq=pd.tseries.frequencies.to_offset(
        {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS"}[f]))
    missing_periods = int(len(full.difference(s.index)))
    t = s.rename(label).to_frame()
    t["change"] = t[label].diff()
    t["pct_change"] = (t[label].pct_change() * 100).replace([np.inf, -np.inf], np.nan)
    ppy = PERIODS_PER_YEAR[f]
    t["yoy_pct"] = (t[label].pct_change(ppy) * 100).replace([np.inf, -np.inf], np.nan) if ppy > 1 and len(t) > ppy else np.nan
    t["rolling_mean_3"] = t[label].rolling(3, min_periods=1).mean()
    t = t.reset_index().rename(columns={"index": "period"})
    out: dict[str, Any] = {"freq": f, "value_label": label, "periods": len(t), "missing_periods": missing_periods}
    if len(t) >= 3:
        lr = stats.linregress(np.arange(len(t)), t[label].astype(float))
        sig = lr.pvalue < 0.05
        out["trend"] = {"slope_per_period": float(lr.slope), "r_squared": float(lr.rvalue ** 2), "p_value": float(lr.pvalue),
                        "direction": ("increasing" if lr.slope > 0 else "decreasing") if sig else "no statistically significant linear trend"}
    out["best_period"] = {"period": t.loc[t[label].idxmax(), "period"], "value": float(t[label].max())}
    out["worst_period"] = {"period": t.loc[t[label].idxmin(), "period"], "value": float(t[label].min())}
    if len(t) >= 2:
        out["latest_vs_previous"] = {"latest_period": t["period"].iloc[-1], "latest": float(t[label].iloc[-1]),
                                     "previous": float(t[label].iloc[-2]), "pct_change": None if t[label].iloc[-2] == 0 else float(100 * (t[label].iloc[-1] / t[label].iloc[-2] - 1))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        end = pd.Period(t["period"].iloc[-1], freq=f).end_time
    out["last_period_may_be_partial"] = bool(dates.max() < end.normalize())
    if last_n_periods:
        t = t.tail(int(last_n_periods)).reset_index(drop=True)
    out["table"] = t
    out["rows_used"] = int(dates.notna().sum())
    return out


def period_drivers(df: pd.DataFrame, date_col: str, value_col: str, dimensions: list[str], freq: str | None = None, agg: str = "sum", top: int = 5) -> dict:
    """Which categories contributed most to the change between the previous and latest period."""
    dc, vc = resolve_column(df, date_col), resolve_column(df, value_col)
    dates = ensure_datetime(df[dc])
    f = norm_freq(freq) or auto_freq(dates)
    per = period_start(dates, f)
    periods = sorted(per.dropna().unique())
    if len(periods) < 2:
        raise AnalystError("Need at least two time periods to compare.")
    cur, prev = periods[-1], periods[-2]
    val = numeric_series(df[vc])
    res = {"current_period": pd.Timestamp(cur), "previous_period": pd.Timestamp(prev), "dimensions": {}}
    tot_c, tot_p = val[per == cur].agg(agg), val[per == prev].agg(agg)
    res.update(total_current=float(tot_c), total_previous=float(tot_p), total_change=float(tot_c - tot_p))
    for dim in resolve_columns(df, dimensions):
        if df[dim].nunique() > 200:
            continue
        a = val[per == cur].groupby(df.loc[per == cur, dim]).agg(agg).rename("current")
        b = val[per == prev].groupby(df.loc[per == prev, dim]).agg(agg).rename("previous")
        t = pd.concat([b, a], axis=1).fillna(0)
        t["change"] = t["current"] - t["previous"]
        t["share_of_total_change_pct"] = (100 * t["change"] / (tot_c - tot_p)).round(1) if tot_c != tot_p else np.nan
        t["n_current"] = df.loc[per == cur, dim].value_counts().reindex(t.index).fillna(0).astype(int)
        t = t.sort_values("change", ascending=(tot_c < tot_p))
        res["dimensions"][dim] = t.head(top).reset_index().rename(columns={dim: "category", "index": "category"})
    res["note"] = ("Contributions describe where the change occurred in the data; they do not prove why it happened. "
                   "Check for partial periods, seasonality and external factors.")
    return res


# ------------------------------------------------------------------------------ outliers
def detect_outliers(df: pd.DataFrame, column: str | list[str], method: str = "iqr", factor: float | None = None, limit: int = 20) -> dict:
    method = method.lower()
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        cols = resolve_columns(df, column) or analysable_numeric_cols(df, include_boolean=False)
        X = pd.DataFrame({c: numeric_series(df[c]) for c in cols}).dropna()
        if len(X) < 20 or X.shape[1] < 1:
            raise AnalystError("Not enough complete numeric rows for IsolationForest (need >= 20).")
        pred = IsolationForest(contamination="auto", random_state=42).fit_predict(X)
        idx = X.index[pred == -1]
        return {"method": method, "columns": cols, "n_checked": len(X), "n_outliers": len(idx), "pct": round(100 * len(idx) / len(X), 2),
                "rows": df.loc[idx].head(limit), "note": "Anomalies are unusual, not necessarily wrong. Review before acting."}
    c = resolve_column(df, column if isinstance(column, str) else column[0])
    x = numeric_series(df[c])
    if x.dropna().empty:
        raise AnalystError(f"'{c}' has no numeric values.")
    if method == "zscore":
        f = float(factor or 3.0)
        lo, hi = x.mean() - f * x.std(), x.mean() + f * x.std()
    else:
        f = float(factor or 1.5)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        lo, hi = q1 - f * (q3 - q1), q3 + f * (q3 - q1)
    mask = (x < lo) | (x > hi)
    return {"method": method, "column": c, "factor": f, "lower_bound": float(lo), "upper_bound": float(hi), "n_checked": int(x.notna().sum()),
            "n_outliers": int(mask.sum()), "pct": round(100 * mask.sum() / max(x.notna().sum(), 1), 2),
            "rows": df[mask].sort_values(c, ascending=False).head(limit),
            "note": "Outliers are statistically unusual values; they may be genuine (e.g. large orders) and are not automatically errors."}


# ------------------------------------------------------------------------------ KPIs
def calculate_kpis(df: pd.DataFrame, metrics: list[dict] | None = None, date_col: str | None = None, freq: str | None = None) -> pd.DataFrame:
    rows = [{"kpi": "Records", "value": float(len(df)), "latest_period": None, "previous_period_value": None, "change_pct": None}]
    metrics = metrics or [{"column": c, "agg": "sum"} for c in analysable_numeric_cols(df, include_boolean=False)[:4]]
    dates = ensure_datetime(df[resolve_column(df, date_col)]) if date_col else None
    f = (norm_freq(freq) or auto_freq(dates)) if dates is not None else None
    per = period_start(dates, f) if dates is not None else None
    for m in metrics:
        c, agg = resolve_column(df, m["column"]), str(m.get("agg", "sum")).lower()
        if agg not in ALLOWED_AGGS:
            raise AnalystError(f"Aggregation '{agg}' not allowed.")
        v = numeric_series(df[c]) if agg != "nunique" and agg != "count" else df[c]
        row = {"kpi": f"{agg} of {c}", "value": float(v.agg(agg)), "latest_period": None, "previous_period_value": None, "change_pct": None}
        if per is not None:
            ps = sorted(per.dropna().unique())
            if len(ps) >= 2:
                cur, prev = v[per == ps[-1]].agg(agg), v[per == ps[-2]].agg(agg)
                row.update(latest_period=str(pd.Timestamp(ps[-1]).date()), previous_period_value=float(prev),
                           change_pct=None if prev == 0 else float(100 * (cur / prev - 1)))
                row["latest_period_value"] = float(cur)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ query
def query_dataset(df: pd.DataFrame, columns: list[str] | None = None, filters: list[dict] | None = None, sort_by: str | None = None,
                  ascending: bool = True, limit: int = 50) -> tuple[pd.DataFrame, int]:
    from .filtering import apply_conditions

    out = apply_conditions(df, filters or [])
    total = len(out)
    if sort_by:
        out = out.sort_values(resolve_column(out, sort_by), ascending=ascending)
    cols = resolve_columns(out, columns)
    if cols:
        out = out[cols]
    return out.head(int(limit)), total


# ------------------------------------------------------------------------------ ML
def _feature_frame(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = pd.DataFrame(index=df.index)
    num, cat = [], []
    for c in features:
        k = column_kind(df[c])
        if k == "numeric":
            X[c] = numeric_series(df[c]).astype(float)
            num.append(c)
        elif k in {"categorical", "boolean"}:
            X[c] = df[c].astype(object).where(df[c].notna(), np.nan)
            X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
            cat.append(c)
    return X, num, cat


def train_model(df: pd.DataFrame, target: str, features: list[str] | None = None, task: str = "auto", test_size: float = 0.25,
                seed: int = 42) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                                 r2_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    t = resolve_column(df, target)
    data = df[df[t].notna()]
    dropped_target_na = len(df) - len(data)
    if len(data) > 100_000:
        data = data.sample(100_000, random_state=seed)
    if len(data) < 30:
        raise AnalystError(f"Only {len(data)} rows with a target value; need at least 30 for a meaningful train/test split.")
    tk = column_kind(data[t])
    if task == "auto":
        task = "regression" if tk == "numeric" and data[t].nunique() > 10 else "classification"
    cand = resolve_columns(data, features) or [c for c in data.columns if c != t]
    excluded: dict[str, str] = {}
    use = []
    y_num = numeric_series(data[t]) if task == "regression" else None
    for c in cand:
        k = column_kind(data[c])
        if c == t:
            continue
        if k in {"identifier", "constant", "empty", "text", "datetime"}:
            excluded[c] = f"{k} column"
            continue
        if data[c].astype(str).equals(data[t].astype(str)):
            excluded[c] = "identical to target (leakage)"
            continue
        if task == "regression" and k == "numeric":
            r = numeric_series(data[c]).corr(y_num)
            if pd.notna(r) and abs(r) > 0.98:
                excluded[c] = f"|corr|={abs(r):.3f} with target (possible leakage)"
                continue
        if task == "classification" and k in {"categorical", "boolean"} and data[c].nunique() > 1:
            v = _cramers_v(data[c], data[t])
            if pd.notna(v) and v > 0.98:
                excluded[c] = f"Cramer's V={v:.3f} with target (possible leakage)"
                continue
        use.append(c)
    if not use:
        raise AnalystError("No usable feature columns remain after excluding identifiers/leaky columns.")
    X, num, cat = _feature_frame(data, use)
    y = y_num if task == "regression" else data[t].astype(str)
    if task == "classification":
        vc = y.value_counts()
        if len(vc) < 2:
            raise AnalystError("Target has a single class.")
        if len(vc) > 20:
            raise AnalystError("Target has more than 20 classes; classification is not appropriate.")
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20))]), cat),
    ])
    est = (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2) if task == "regression"
           else RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2))
    pipe = Pipeline([("pre", pre), ("model", est)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    if task == "regression":
        base = Pipeline([("pre", pre), ("model", DummyRegressor())]).fit(Xtr, ytr).predict(Xte)
        metrics = {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred), "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                   "baseline_mae_(predict_mean)": mean_absolute_error(yte, base)}
    else:
        base = Pipeline([("pre", pre), ("model", DummyClassifier(strategy="most_frequent"))]).fit(Xtr, ytr).predict(Xte)
        metrics = {"accuracy": accuracy_score(yte, pred), "balanced_accuracy": balanced_accuracy_score(yte, pred),
                   "f1_weighted": f1_score(yte, pred, average="weighted"), "baseline_accuracy_(majority_class)": accuracy_score(yte, base)}
        if y.nunique() == 2:
            try:
                proba = pipe.predict_proba(Xte)[:, 1]
                metrics["roc_auc"] = roc_auc_score(yte, proba)
            except Exception:  # noqa: BLE001
                pass
    Xs, ys = (Xte.head(2000), yte.loc[Xte.head(2000).index])
    pi = permutation_importance(pipe, Xs, ys, n_repeats=5, random_state=seed, n_jobs=1)
    imp = pd.DataFrame({"feature": use if len(use) == len(Xs.columns) else list(Xs.columns), "importance": pi.importances_mean})
    imp = imp.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
    return {"task": task, "target": t, "n_train": len(Xtr), "n_test": len(Xte), "features_used": list(X.columns), "excluded_features": excluded,
            "rows_without_target_excluded": dropped_target_na, "metrics": {k: float(v) for k, v in metrics.items()}, "feature_importance": imp,
            "note": ("Predictions are estimates from a random 75/25 split, not facts. Preprocessing was fit on training data only. "
                     "Feature importance shows predictive usefulness, not causation.")}


def cluster_data(df: pd.DataFrame, features: list[str] | None = None, k: int | None = None, seed: int = 42) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    cols = resolve_columns(df, features) or analysable_numeric_cols(df, include_boolean=False)
    cols = [c for c in cols if column_kind(df[c]) == "numeric"][:15]
    if len(cols) < 2:
        raise AnalystError("Clustering needs at least two numeric columns.")
    X = pd.DataFrame({c: numeric_series(df[c]) for c in cols})
    X = X.loc[X.notna().mean(axis=1) > 0.5]
    if len(X) < 20:
        raise AnalystError("Not enough rows (need >= 20) for clustering.")
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    best, scores = None, {}
    ks = [k] if k else range(2, min(7, len(X) // 5 + 1) or 3)
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        samp = np.random.RandomState(seed).choice(len(Z), min(5000, len(Z)), replace=False)
        sc = float(silhouette_score(Z[samp], km.labels_[samp])) if kk > 1 else np.nan
        scores[kk] = sc
        if best is None or sc > best[1]:
            best = (km, sc, kk)
    km, sc, kk = best
    prof = X.assign(cluster=km.labels_).groupby("cluster").agg(["mean"])
    prof.columns = [c for c, _ in prof.columns]
    prof.insert(0, "size", np.bincount(km.labels_))
    prof.insert(1, "pct", (100 * prof["size"] / len(X)).round(1))
    return {"k": kk, "silhouette": sc, "silhouette_by_k": scores, "features": cols, "profile": prof.reset_index().round(3),
            "assignments": pd.Series(km.labels_, index=X.index, name="cluster"),
            "note": "Clusters are exploratory groupings of similar rows; interpret them with domain knowledge. Silhouette near 0 means weak separation."}


def cohort_retention(df: pd.DataFrame, user_col: str, date_col: str, freq: str = "M") -> dict:
    u, dc, f = resolve_column(df, user_col), resolve_column(df, date_col), norm_freq(freq) or "M"
    d = pd.DataFrame({"u": df[u], "p": ensure_datetime(df[dc])}).dropna()
    if d.empty:
        raise AnalystError("No valid user/date rows.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per = d["p"].dt.to_period(f)
    d["ord"] = np.asarray(per.array.asi8)
    d["cohort"] = d.groupby("u")["ord"].transform("min")
    d["age"] = d["ord"] - d["cohort"]
    cnt = d.groupby(["cohort", "age"])["u"].nunique().unstack(fill_value=0)
    sizes = cnt[0]
    ret = cnt.div(sizes, axis=0).mul(100).round(1)
    first = pd.Period(ordinal=int(ret.index[0]), freq=f)
    ret.index = [str(pd.Period(ordinal=int(i), freq=f)) for i in ret.index]
    ret.columns = [f"+{c}" for c in ret.columns]
    ret.insert(0, "cohort_size", sizes.values)
    return {"retention_pct": ret.reset_index().rename(columns={"index": "cohort"}), "freq": f, "users": int(d["u"].nunique())}


def funnel_analysis(df: pd.DataFrame, stage_col: str, user_col: str, stages: list[str]) -> pd.DataFrame:
    sc, uc = resolve_column(df, stage_col), resolve_column(df, user_col)
    rows, first = [], None
    low = df[sc].astype(str).str.strip().str.lower()
    prev = None
    for st in stages:
        n = int(df.loc[low == str(st).strip().lower(), uc].nunique())
        first = n if first is None else first
        rows.append({"stage": st, "users": n, "pct_of_first_stage": round(100 * n / first, 1) if first else np.nan,
                     "pct_of_previous_stage": round(100 * n / prev, 1) if prev else np.nan})
        prev = n
    return pd.DataFrame(rows)
