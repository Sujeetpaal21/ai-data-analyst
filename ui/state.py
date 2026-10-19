"""Streamlit session-state helpers. st.session_state is private to each browser session,
so datasets, chat history and exports are isolated between users."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analyst.config import Settings
from analyst.ingestion import LoadedFile
from analyst.session import DatasetSession

MAX_DATASETS = 8
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def init() -> None:
    st.session_state.setdefault("sessions", {})  # name -> DatasetSession
    st.session_state.setdefault("active", None)
    st.session_state.setdefault("seen_uploads", set())
    st.session_state.setdefault("file_info", {})  # dataset name -> info dict
    st.session_state.setdefault("queued_question", None)


def sessions() -> dict[str, DatasetSession]:
    return st.session_state["sessions"]


def current() -> DatasetSession | None:
    name = st.session_state.get("active")
    return sessions().get(name) if name else None


def add_loaded(lf: LoadedFile, settings: Settings) -> list[str]:
    added = []
    for key, df in lf.datasets.items():
        name = lf.name if len(lf.datasets) == 1 else f"{lf.name} :: {key}"
        if len(sessions()) >= MAX_DATASETS and name not in sessions():
            st.warning(f"Maximum of {MAX_DATASETS} datasets per session reached; remove one first.")
            break
        sessions()[name] = DatasetSession.from_df(name, df, lf.meta, settings.max_history_states)
        st.session_state["file_info"][name] = {"file": lf.name, "format": lf.fmt, "size_kb": round(lf.size_bytes / 1024, 1),
                                               "sheets": list(lf.datasets), "warnings": lf.warnings}
        added.append(name)
    if added:
        st.session_state["active"] = added[0]
    return added


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame displayable by st.dataframe (mixed-type object columns -> text)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            nn = out[c].dropna()
            if len(nn) and nn.map(type).nunique() > 1:
                out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v))
    return out


def show_df(df: pd.DataFrame, **kw) -> None:
    st.dataframe(arrow_safe(df), **kw)
