"""Per-user dataset session: original (immutable), working copy, filters, undo, change log."""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .filtering import apply_conditions
from .util import AnalystError


@dataclass
class DatasetSession:
    name: str
    original: pd.DataFrame
    working: pd.DataFrame
    meta: dict = field(default_factory=dict)
    max_history: int = 15
    filters: list[dict] = field(default_factory=list)
    filter_combine: str = "and"
    log: list[dict] = field(default_factory=list)
    history: list[pd.DataFrame] = field(default_factory=list)
    pending: dict | None = None
    last_analysis: dict | None = None
    exports: dict[str, tuple[bytes, str]] = field(default_factory=dict)  # name -> (bytes, mime)
    charts: list[tuple[str, Any]] = field(default_factory=list)  # (title, plotly figure)
    chat: list[dict] = field(default_factory=list)  # UI transcript
    llm_history: list[dict] = field(default_factory=list)  # text-only memory for the model
    notices: list[str] = field(default_factory=list)
    version: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_df(cls, name: str, df: pd.DataFrame, meta: dict | None = None, max_history: int = 15):
        return cls(name=name, original=df.copy(), working=df.copy(), meta=meta or {}, max_history=max_history)

    # ------------------------------------------------------------ views
    def _sig(self) -> str:
        return json.dumps([self.version, self.filters, self.filter_combine], default=str, sort_keys=True)

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        k = (self._sig(), key)
        if k not in self._cache:
            if len(self._cache) > 60:
                self._cache.clear()
            self._cache[k] = fn()
        return self._cache[k]

    def view(self) -> pd.DataFrame:
        """Working copy with the active filters applied (what analyses run on)."""
        if not self.filters:
            return self.working
        def _make():
            try:
                return apply_conditions(self.working, self.filters, self.filter_combine)
            except AnalystError as e:
                self.notices.append(f"Active filters were cleared because they no longer apply: {e}")
                self.filters = []
                return self.working
        return self.cached("__view__", _make)

    def set_filters(self, conditions: list[dict], combine: str = "and") -> pd.DataFrame:
        apply_conditions(self.working, conditions, combine)  # validate first (raises AnalystError)
        self.filters, self.filter_combine = list(conditions), combine
        return self.view()

    # ------------------------------------------------------------ mutation
    def commit(self, new_df: pd.DataFrame, description: str, operation: str, params: dict,
               rows_before: int | None = None, cells_changed: int = 0) -> None:
        self.history.append(self.working)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.log.append({
            "step": len(self.log) + 1,
            "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "params": json.dumps(params, default=str, ensure_ascii=False)[:300],
            "description": description,
            "rows_before": len(self.working) if rows_before is None else rows_before,
            "rows_after": len(new_df),
            "cells_changed": cells_changed,
        })
        self.working = new_df
        self.version += 1
        self.pending = None

    def undo(self) -> bool:
        if not self.history:
            return False
        self.working = self.history.pop()
        if self.log:
            self.log.pop()
        self.version += 1
        return True

    def reset(self) -> None:
        self.working = self.original.copy()
        self.history.clear()
        self.log.clear()
        self.log.append({"step": 1, "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "operation": "reset", "params": "{}", "description": "Working copy reset to the original upload.",
                         "rows_before": len(self.working), "rows_after": len(self.working), "cells_changed": 0})
        self.filters = []
        self.pending = None
        self.version += 1

    def log_df(self) -> pd.DataFrame:
        cols = ["step", "time", "operation", "description", "rows_before", "rows_after", "cells_changed", "params"]
        return pd.DataFrame(self.log, columns=cols)
