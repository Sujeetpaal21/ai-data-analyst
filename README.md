# 📊 AI Data Analyst Agent
**🔴 Live demo:** https://ai-data-analyst-ixmjzln5f5w45p3s95pvdg.streamlit.app/
Upload a CSV / Excel / JSON / Parquet / TSV file and get an automatic analysis, a cleaning workspace, interactive charts, and a
**tool-using AI analyst** you can talk to in **English, Hindi or Hinglish**. Every number comes from real computation
(pandas / SciPy / scikit-learn) - the language model never does arithmetic from memory and never executes code.

* Works **without any API key** (built-in offline rule-based analyst). Add an Anthropic key for free-form questions.
* Original data is never modified; cleaning happens on a working copy with preview, confirmation, change log, undo and reset.

## Features
| Area | What you get |
|---|---|
| Ingestion | CSV, TSV, TXT, XLSX/XLS (all sheets), JSON (nested / lines), Parquet; encoding + delimiter detection; malformed rows are *reported*, not silently dropped |
| Overview & quality | types (numeric/categorical/date/boolean/ID/text), missing %, duplicates, memory, high-cardinality & constant columns, issue detector (whitespace, label variants, numbers/dates stored as text, suspicious values, outliers, redundant columns) and an explained 0-100 quality score |
| Statistics | descriptive stats, quartiles, skewness, 95% CI, normality test, frequency tables, correlations with p-values, target associations (Pearson/Spearman/eta/Cramér's V), group-bys, cross-tabs |
| Cleaning | duplicates, missing values (mean/median/mode/constant/ffill/bfill), trim/case, label standardization, type conversion, date parsing, rename/reorder/drop, row filters, outlier handling, custom rules; before/after preview, confirmation for risky steps, log, undo, reset |
| Charts | auto-recommended chart with explanation; bar, grouped/stacked bar, line, area, histogram, box, scatter, bubble, pie/donut, heatmap, correlation heatmap, KPI, table with conditional formatting; HTML download |
| AI agent | 17 controlled tools (inspect, profile, statistics, missing, duplicates, clean, filter, group/aggregate, correlation, time-series, outliers, KPIs, visualization, query, export, insights, advanced analysis incl. regression/classification, clustering, cohort, funnel, "why did it change" driver analysis); bounded tool loop, timeouts, conversation memory |
| Export | cleaned CSV, filtered CSV, multi-sheet Excel, data dictionary, statistics, report (Markdown / HTML with charts / PDF), saved charts (HTML), cleaning log |

## Project layout
```
app.py                 Streamlit entry point
ui/                    state.py (per-session state) · views.py (pages)
analyst/               UI-independent core
  ingestion.py  profiling.py  cleaning.py  filtering.py  analytics.py  viz.py
  tools.py (tool registry)   agent.py (LLM loop + offline router)   reporting.py   session.py   config.py
sample_data/           sales_sample.csv (deliberately messy), multi_sheet_sample.xlsx, generate_sample.py
tests/                 88 automated tests
run_app.ps1 / run_app.bat / run_tests.ps1   Windows launchers
```

## Run locally on Windows (VS Code)
1. **Check Python (3.10+)** - in the VS Code terminal (PowerShell):
   ```powershell
   py --version        # or: python --version
   ```
   If missing: install from python.org and tick **Add python.exe to PATH**, then reopen VS Code.
2. **Open the project folder** in VS Code (`File → Open Folder…`, pick the unzipped folder).
3. **Start (recommended - one command, handles venv + install):**
   ```powershell
   .\run_app.ps1
   ```
   If PowerShell says *"running scripts is disabled"*, use either:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_app.ps1
   ```
   or just double-click **run_app.bat**. (Alternative permanent fix: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)
4. The browser opens at **http://localhost:8501** (open it manually if not). Click **Load sample sales dataset** to try it immediately.

### Manual setup (if you prefer)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
(Calling the venv's `python.exe` directly means you never need to "activate" it.)

macOS/Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run app.py`

## Configure the AI (optional)
1. Get an API key from https://console.anthropic.com/.
2. `Copy-Item .env.example .env`, then edit `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   AI_MODEL=claude-sonnet-5-5
   ```
3. Restart the app. The sidebar shows the model and a **Use external AI model** checkbox. Without a key the offline analyst is used automatically.
   The model name is only a setting (`AI_MODEL`); provider code is isolated in `analyst/agent.py` (`make_client`).

**Privacy:** in AI mode the provider receives column names/types, your questions and the *outputs of analysis tools* (aggregates and ≤ ~20 rows per
result). Raw sample rows are sent only if you set `SEND_SAMPLE_ROWS=true`. Untick the AI checkbox to keep everything local.

## Example questions
`Top 10 products by revenue` · `Monthly sales trend` · `Pichhle 6 months ka sales trend dikhao` · `Sirf Delhi ke customers dikhao` ·
`Missing values ko median se fill karo` · `What factors are associated with Churned?` · `Why did Revenue decrease?` · `Find outliers` ·
`Is data ka complete report banao` · follow-up: `Now show only the top five`, `Is chart ko region-wise bana do`.

## Run the tests
```powershell
.\run_tests.ps1                                   # pytest
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .   # without pytest
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `running scripts is disabled on this system` | `powershell -ExecutionPolicy Bypass -File .\run_app.ps1` or double-click `run_app.bat` |
| `py`/`python` not recognized | Install Python 3.10+ with "Add to PATH"; reopen terminal |
| `pip install` fails on `pyarrow`/`scikit-learn` | Use 64-bit Python 3.10-3.12; upgrade pip; delete `.venv` and rerun |
| Port 8501 busy | `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` |
| "Plotly is not installed" | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| AI answers say "offline" | `.env` missing/blank key, or `AI_ENABLED=false`; check the sidebar message |
| Reading `.xls` fails | needs `xlrd` (in requirements) |
| File too large | raise `MAX_UPLOAD_MB` in `.env` **and** `maxUploadSize` in `.streamlit/config.toml` |

## Push to GitHub
```powershell
git init
git add .
git commit -m "AI Data Analyst Agent"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
`.env` is git-ignored - **never commit your API key**.

## Deploy
**Streamlit Community Cloud (easiest):** push to GitHub → https://share.streamlit.io → *New app* → pick repo, branch `main`, main file `app.py` →
*Advanced settings → Secrets*, paste:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
AI_MODEL = "claude-sonnet-5-5"
```
(Top-level secrets are exposed as environment variables, which `analyst/config.py` reads.) Public apps let anyone use your key/credit - keep the app
private/invite-only or leave the key out to run offline.
**Docker / any VM:** `docker build -t ai-analyst . && docker run -p 8501:8501 --env-file .env ai-analyst`.

Multi-user notes: data lives in each browser session's memory (isolated), nothing is written to disk, uploads are size/type-limited and tools have
timeouts. For public deployments add authentication in front (reverse proxy / platform access control).

## Security design
No `exec`/`eval`/shell on model output - the model can only call whitelisted tools with structured arguments · dataset text is treated as untrusted data
(system prompt + not sent as instructions) · API key only from environment · CSV/Excel exports neutralise formula injection · risky edits need confirmation.

## Known limitations
* Tested here with Python 3.12 / pandas 3.0 / scikit-learn 1.8; the Streamlit UI, Plotly figures and the live Anthropic API could **not** be executed in the build sandbox (no internet) - see "Verification status" below.
* In-memory processing: very large files (>~1M rows) may be slow; undo keeps up to 15 copies of the working frame.
* Offline router is keyword-based; ambiguous phrasing triggers a clarification instead of a guess. Free-form questions need the AI mode.
* Regression/classification use a random 75/25 split (no time-based split); predictions are estimates. Cohort/funnel assume sensible event data.
* PDF reports omit charts and cannot render non-Latin scripts (use the HTML report).
* Timeouts stop waiting for a slow tool but cannot kill its thread.

## Verification status
Verified by automated tests in the build sandbox (88 tests, 2 skipped): ingestion, cleaning/undo/confirmation, profiling, statistics, grouping, correlation,
time-series, outliers, KPIs, ML (with leakage guards), cohort/funnel, tool execution/timeouts/errors, offline routing (EN/Hinglish/Hindi), the LLM tool loop
(with a scripted fake client: limits, API failure fallback, no-key mode), CSV/Excel/Markdown/HTML/PDF exports, session isolation, and every UI page against a stand-in `streamlit`.
**Not verified in the sandbox:** real Streamlit rendering, Plotly figure construction (`test_chart_render_when_plotly_available` runs automatically once Plotly is installed), real Anthropic API calls, `.xls`/Parquet reading, Windows scripts. Run `.\run_tests.ps1` after setup to confirm on your machine.
