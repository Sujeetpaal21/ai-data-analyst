import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyst.session import DatasetSession  # noqa: E402


def make_df(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_id": range(n), "order_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n), "product": rng.choice(["Laptop", "Phone", "Mouse"], n),
        "qty": rng.integers(1, 9, n), "price": rng.normal(100, 20, n).round(2)})
    df["revenue"] = (df.qty * df.price).round(2)
    df["churn"] = (rng.random(n) < 0.15 + 0.35 * (df.city == "Delhi")).astype(int)
    return df


def make_session(df=None, name="test") -> DatasetSession:
    return DatasetSession.from_df(name, make_df() if df is None else df)


def excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k, index=False)
    return buf.getvalue()


class FakeBlock(SimpleNamespace):
    pass


def text_block(t):
    return FakeBlock(type="text", text=t)


def tool_block(id_, name, inp):
    return FakeBlock(type="tool_use", id=id_, name=name, input=inp)


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic (no network)."""

    def __init__(self, script, fail=False):
        self.script, self.calls, self.fail = list(script), [], fail
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("network down")
        content = self.script.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop)
