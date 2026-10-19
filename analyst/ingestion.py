"""File ingestion: CSV/TSV/TXT, Excel (all sheets), JSON (nested / lines), Parquet.

Nothing is silently dropped: malformed rows, skipped sheets, renamed columns and
encoding fallbacks are all reported in ``LoadedFile.warnings``.
"""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from .util import AnalystError

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}


class IngestionError(AnalystError):
    pass


@dataclass
class LoadedFile:
    name: str
    size_bytes: int
    fmt: str
    datasets: dict[str, pd.DataFrame]
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def load_file(name: str, data: bytes, max_mb: int = 200, max_rows: int = 2_000_000) -> LoadedFile:
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext or name}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not data:
        raise IngestionError("The file is empty (0 bytes).")
    if len(data) > max_mb * 1024 * 1024:
        raise IngestionError(f"File is {len(data) / 1e6:.1f} MB; the limit is {max_mb} MB.")

    warnings: list[str] = []
    meta: dict = {}
    if ext in {".csv", ".tsv", ".txt"}:
        datasets = {"data": _read_delimited(data, ext, warnings, meta)}
        fmt = {"csv": "CSV", "tsv": "TSV", "txt": "Text"}[ext.strip(".")]
    elif ext in {".xlsx", ".xls"}:
        datasets = _read_excel(data, warnings)
        fmt = "Excel"
    elif ext == ".json":
        datasets = _read_json(data, warnings)
        fmt = "JSON"
    else:
        datasets = {"data": _read_parquet(data)}
        fmt = "Parquet"

    prepared: dict[str, pd.DataFrame] = {}
    for key, df in datasets.items():
        df = _prepare(df, key, warnings, max_rows)
        if df.shape[1] == 0:
            warnings.append(f"Dataset '{key}' has no usable columns and was skipped.")
            continue
        prepared[key] = df
    if not prepared:
        raise IngestionError("No data could be read from the file (empty or unreadable).")
    return LoadedFile(name, len(data), fmt, prepared, warnings, meta)


# ---------------------------------------------------------------- readers
def _decode_check(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    try:
        data.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        return "latin-1"


def _read_delimited(data: bytes, ext: str, warnings: list[str], meta: dict) -> pd.DataFrame:
    enc = _decode_check(data)
    if enc not in {"utf-8-sig", "utf-8"}:
        warnings.append(f"File is not valid UTF-8; decoded as {enc}. Check special characters.")
    sample = data[:50_000].decode(enc, errors="replace")
    delim = "\t" if ext == ".tsv" else None
    if delim is None:
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
    meta.update(encoding=enc, delimiter=delim)
    kwargs = dict(sep=delim, encoding=enc)
    try:
        return pd.read_csv(io.BytesIO(data), **kwargs)
    except pd.errors.EmptyDataError:
        raise IngestionError("The file contains no data.")
    except (pd.errors.ParserError, ValueError):
        # Malformed rows: keep rows that match the header width, but REPORT every skipped row.
        text = data.decode(enc, errors="replace")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            raise IngestionError("The file contains no data.")
        kept, skipped = [], []
        for i, row in enumerate(reader, start=2):
            if not row:
                continue
            (kept if len(row) == len(header) else skipped).append(row if len(row) == len(header) else (i, row))
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=delim)
        w.writerow(header)
        w.writerows(kept)
        buf.seek(0)
        try:
            df = pd.read_csv(buf, sep=delim)
        except Exception as e:  # noqa: BLE001
            raise IngestionError(f"Could not parse the file as delimited text: {e}")
        if skipped:
            warnings.append(f"{len(skipped)} malformed row(s) with the wrong number of fields were skipped "
                            f"(first at line {skipped[0][0]}: {str(skipped[0][1])[:100]}).")
        return df


def _read_excel(data: bytes, warnings: list[str]) -> dict[str, pd.DataFrame]:
    try:
        xl = pd.ExcelFile(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Could not read the Excel file (is it corrupted or password-protected?): {e}")
    out: dict[str, pd.DataFrame] = {}
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Sheet '{sheet}' could not be read and was skipped: {e}")
            continue
        if df.shape[1] == 0:
            warnings.append(f"Sheet '{sheet}' is empty and was skipped.")
            continue
        out[str(sheet)] = df
    return out


def _read_json(data: bytes, warnings: list[str]) -> dict[str, pd.DataFrame]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            return {"data": pd.read_json(io.StringIO(text), lines=True)}
        except Exception as e:  # noqa: BLE001
            raise IngestionError(f"Invalid JSON: {e}")
    if isinstance(obj, list):
        if obj and not isinstance(obj[0], dict):
            return {"data": pd.DataFrame({"value": obj})}
        return {"data": pd.json_normalize(obj)}
    if isinstance(obj, dict):
        lists = {k: v for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)}
        if lists:
            return {str(k): pd.json_normalize(v) for k, v in lists.items()}
        return {"data": pd.json_normalize([obj])}
    raise IngestionError("JSON must contain an object or a list of records.")


def _read_parquet(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data))
    except ImportError:
        raise IngestionError("Reading Parquet requires 'pyarrow' (pip install pyarrow).")
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Could not read Parquet file: {e}")


# ---------------------------------------------------------------- post-processing
def _prepare(df: pd.DataFrame, key: str, warnings: list[str], max_rows: int) -> pd.DataFrame:
    if len(df) > max_rows:
        raise IngestionError(
            f"Dataset '{key}' has {len(df):,} rows; the limit is {max_rows:,}. Raise MAX_ROWS or split the file."
        )
    new_cols: list[str] = []
    seen: dict[str, int] = {}
    changed = 0
    for i, c in enumerate(df.columns):
        orig = str(c)
        name = orig.strip() or f"unnamed_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        if name != orig:
            changed += 1
        new_cols.append(name)
    if changed:
        warnings.append(f"[{key}] {changed} column name(s) were trimmed, made unique or given a name.")
    df = df.copy()
    df.columns = new_cols
    if any(c.lower().startswith("unnamed") for c in new_cols):
        warnings.append(f"[{key}] Some columns have no header (named 'unnamed_*'); check the first row of the file.")
    for c in df.columns:  # nested lists/dicts -> JSON text so duplicates/unique counts work
        if df[c].dtype == object:
            head = df[c].dropna().head(50)
            if any(isinstance(v, (list, dict)) for v in head):
                df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                warnings.append(f"[{key}] Column '{c}' contained nested data; stored as JSON text.")
    return df.reset_index(drop=True)
