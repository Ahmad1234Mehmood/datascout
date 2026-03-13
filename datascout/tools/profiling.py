"""Profiling tools. The LLM never sees raw rows, only these summaries."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pt

from ..models.findings import ColumnProfile

ID_NAME = re.compile(r"(^|_)(id|uuid|guid|idx|index)$|^id_", re.I)
DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


class DataLoadError(ValueError):
    """Raised when the uploaded file cannot be read as a table."""


def _detect_delimiter(p: Path) -> str:
    """Pick the delimiter from the header line (some public CSVs use ';', e.g. UCI bank-full.csv)."""
    try:
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
    except OSError:
        return ","
    counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def load_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(p, sep=_detect_delimiter(p))
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            raise DataLoadError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx")
    except DataLoadError:
        raise
    except Exception as e:  # malformed file
        raise DataLoadError(f"Could not parse '{p.name}': {e}") from e
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("The file contains no data")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise DataLoadError("Duplicate column names are not supported")
    df, notes = normalize_text_columns(df)
    df.attrs["load_notes"] = notes
    return df


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pt.is_string_dtype(s)


def normalize_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Blank strings become NaN; all-numeric text columns become numeric; every change is reported."""
    df = df.copy()
    notes: list[str] = []
    for col in df.columns:
        s = df[col]
        if not _is_text(s):
            continue
        blank = s.notna() & (s.astype(str).str.strip() == "")
        if blank.any():
            s = s.mask(blank)
            notes.append(f"'{col}': {int(blank.sum())} blank text value(s) treated as missing")
        nonnull = s.dropna()
        if len(nonnull) and not ID_NAME.search(col):
            txt = nonnull.astype(str).str.strip()
            has_leading_zero = txt.str.match(r"^0\d").any()
            num = pd.to_numeric(txt, errors="coerce")
            if num.notna().all() and not has_leading_zero:
                s = pd.to_numeric(s.astype(str).str.strip().where(s.notna()), errors="coerce")
                notes.append(f"'{col}': text column converted to numeric")
        df[col] = s
    return df, notes


def detect_semantic_type(s: pd.Series) -> str:
    n = s.dropna()
    if len(n) == 0:
        return "empty"
    if n.nunique() <= 1:
        return "constant"
    if pt.is_bool_dtype(s):
        return "boolean"
    if pt.is_datetime64_any_dtype(s):
        return "datetime"
    name_is_id = bool(ID_NAME.search(str(s.name)))
    if pt.is_numeric_dtype(s):
        if pt.is_integer_dtype(s) and name_is_id:
            return "id"
        return "numeric"
    # string / object columns
    if name_is_id:
        return "id"
    sample = n.astype(str).head(50)
    if sample.map(lambda v: bool(DATE_LIKE.match(v))).mean() >= 0.9:
        return "datetime"
    avg_len = n.astype(str).str.len().mean()
    if avg_len > 30 or (n.nunique() / len(n) > 0.5 and n.nunique() > 50):
        return "text"
    return "categorical"


def is_numeric_feature(s: pd.Series) -> bool:
    return pt.is_numeric_dtype(s) and not pt.is_bool_dtype(s)


def infer_task_type(df: pd.DataFrame, target: str) -> str:
    """'classification' or 'regression', from the target column alone."""
    y = df[target].dropna()
    if not pt.is_numeric_dtype(y) or pt.is_bool_dtype(y):
        return "classification"
    if y.nunique() <= 20 and np.allclose(y, np.round(y)):
        return "classification"
    return "regression"


def _py(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    n = len(df)
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        sem = detect_semantic_type(s)
        miss = int(s.isna().sum())
        uniq = int(s.nunique(dropna=True))
        p = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            semantic_type=sem,
            missing_count=miss,
            missing_pct=round(100 * miss / n, 4),
            unique_count=uniq,
            unique_pct=round(100 * uniq / max(n - miss, 1), 4),
        )
        if is_numeric_feature(s) and sem != "empty":
            x = s.dropna().astype(float)
            p.mean, p.std = _py(x.mean()), _py(x.std())
            p.min, p.max = _py(x.min()), _py(x.max())
            p.skew = _py(x.skew()) if len(x) > 2 else None
        if sem in ("categorical", "boolean", "constant"):
            p.top_values = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        profiles.append(p)
    return profiles


def compute_stats(df: pd.DataFrame, column: str) -> dict:
    """Detailed statistics for one column (used in before/after comparisons)."""
    s = df[column]
    out: dict = {"rows": int(len(s)), "missing_count": int(s.isna().sum()), "unique_count": int(s.nunique())}
    if is_numeric_feature(s) and s.notna().any():
        x = s.dropna().astype(float)
        out.update(mean=_py(x.mean()), std=_py(x.std()), min=_py(x.min()), max=_py(x.max()), median=_py(x.median()))
    return out
