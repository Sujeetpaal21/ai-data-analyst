"""UI smoke test with a stand-in `streamlit` module (real Streamlit is not required).

It executes every page function (including 'button clicked' paths) to catch NameErrors, bad attribute
access and wrong argument names. It does NOT verify visual layout - run the real app for that.
Skipped automatically when real Streamlit is installed (use `streamlit run app.py` to test it instead)."""
import sys
import types
import unittest
from pathlib import Path

try:
    import streamlit  # noqa: F401
    REAL = True
except ImportError:
    REAL = False


class Rerun(Exception):
    pass


class Ctx:
    def __init__(self, st): self.st = st
    def __enter__(self): return self.st
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(self.st, n)


def make_fake(click: bool):
    m = types.ModuleType("streamlit")
    m.session_state = {}
    ctx = Ctx(m)
    m.sidebar = ctx
    def first(opts, index=0, **k): opts = list(opts); return opts[index] if opts else None
    f = {
        "selectbox": lambda label, options, index=0, **k: first(options, index or 0), "radio": lambda label, options, index=0, **k: first(options, index or 0),
        "multiselect": lambda label, options=(), default=None, **k: default or [], "button": lambda *a, **k: click and not k.get("disabled"),
        "checkbox": lambda label, value=False, **k: True if click else value, "text_input": lambda *a, **k: "", "text_area": lambda *a, **k: "",
        "number_input": lambda label, min_value=0, max_value=None, value=None, *a, **k: (value if value is not None else (a[0] if a else min_value)),
        "file_uploader": lambda *a, **k: [], "chat_input": lambda *a, **k: None, "columns": lambda spec, **k: [ctx] * (spec if isinstance(spec, int) else len(spec)),
        "tabs": lambda names: [ctx] * len(names), "expander": lambda *a, **k: ctx, "spinner": lambda *a, **k: ctx, "chat_message": lambda *a, **k: ctx,
        "download_button": lambda *a, **k: False,
    }
    for k, v in f.items():
        setattr(m, k, v)
    def rerun(): raise Rerun()
    m.rerun = rerun
    m.__getattr__ = lambda name: (lambda *a, **k: None)
    return m


@unittest.skipIf(REAL, "real Streamlit installed")
class UISmoke(unittest.TestCase):
    def _run(self, click):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        for k in [k for k in sys.modules if k == "ui" or k.startswith("ui.")]:
            del sys.modules[k]
        sys.modules["streamlit"] = make_fake(click)
        from analyst.config import Settings
        from ui import state, views
        st = sys.modules["streamlit"]
        settings = Settings()
        state.init()
        p = state.SAMPLE_DIR / "sales_sample.csv"
        from analyst.ingestion import load_file
        state.add_loaded(load_file(p.name, p.read_bytes()), settings)
        s = state.current()
        s.chat.append({"role": "assistant", "text": "hi", "tables": [("t", s.working.head(3))], "figures": [], "trace": [{"tool": "x"}], "mode": "offline"})
        for name, fn in list(views.PAGES.items()) + [("chat", lambda se: views.page_chat(se, False))]:
            try:
                fn(settings)
            except Rerun:
                pass
        # chat question path
        st.chat_input = lambda *a, **k: "Top 3 products by revenue"
        try:
            views.page_chat(settings, False)
        except Rerun:
            pass
        if not click:
            self.assertTrue(any(e["role"] == "user" and "Top 3" in e["text"] for e in s.chat))

    def test_pages_render_default(self):
        self._run(False)

    def test_pages_with_buttons_clicked(self):
        self._run(True)


if __name__ == "__main__":
    unittest.main()
