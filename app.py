"""AI Data Analyst Agent - Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

from analyst.config import load_settings  # noqa: E402
from ui import state, views  # noqa: E402

settings = load_settings()
state.init()

st.markdown("<style>.block-container{padding-top:1.5rem}[data-testid=stMetricValue]{font-size:1.4rem}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 AI Data Analyst")
    names = list(state.sessions())
    if names:
        cur = st.session_state["active"] if st.session_state["active"] in names else names[0]
        st.session_state["active"] = st.selectbox("Active dataset / sheet", names, index=names.index(cur))
        if st.button("Remove this dataset"):
            state.sessions().pop(st.session_state["active"], None)
            st.session_state["active"] = None
            st.rerun()
    pages = list(views.PAGES)
    pages.insert(5, "🤖 AI Analyst Chat")
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.divider()
    use_ai = st.checkbox("Use external AI model", value=settings.ai_available, disabled=not settings.ai_available,
                         help="Requires ANTHROPIC_API_KEY in .env. When off, the built-in offline analyst is used.")
    if settings.ai_available:
        st.caption(f"Model: `{settings.model}`")
    else:
        st.caption("No API key found → offline rule-based analyst (add a key in `.env` for free-form questions).")
    with st.expander("🔒 Privacy"):
        st.caption("Your file stays in this app's memory for this browser session only. In AI mode the provider receives column names/types, your questions "
                   "and the *outputs of analysis tools* (aggregates and up to ~20 rows per result)" + (", plus 5 sample rows (SEND_SAMPLE_ROWS=true)" if settings.send_sample_rows else "")
                   + ". Model-written code is never executed. Turn the AI checkbox off to keep everything local.")
    st.caption("Theme: ⋮ menu → Settings → Theme (light/dark).")

if page == "🤖 AI Analyst Chat":
    views.page_chat(settings, use_ai)
else:
    views.PAGES[page](settings)
