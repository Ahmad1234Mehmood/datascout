"""Load a Hugging Face dataset by name into a DataFrame (optional dependency: `pip install datasets`)."""
from __future__ import annotations

import re

import pandas as pd

from .profiling import DataLoadError, normalize_text_columns

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$")
MAX_ROWS = 50_000


def load_hf_dataset(name: str, split: str = "train", config: str | None = None, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    name, split = name.strip(), split.strip()
    if not NAME_RE.match(name):
        raise DataLoadError(f"'{name}' is not a valid Hugging Face dataset name (example: scikit-learn/churn-prediction)")
    if not re.match(r"^[A-Za-z0-9_]+$", split):
        raise DataLoadError(f"invalid split '{split}' (example: train)")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataLoadError("Hugging Face support needs: pip install datasets") from e
    try:
        ds = load_dataset(name, config, split=f"{split}[:{int(max_rows)}]")
        df = ds.to_pandas()
    except Exception as e:
        raise DataLoadError(f"Could not load Hugging Face dataset '{name}' (split '{split}'): {e}") from e

    notes = [f"loaded from Hugging Face: {name} (split '{split}', at most {max_rows} rows)"]
    nested = [c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict, tuple))).any()]
    if nested:
        df = df.drop(columns=nested)
        notes.append(f"dropped nested column(s) that are not tabular: {nested}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("The dataset contains no tabular data")
    df.columns = [str(c).strip() for c in df.columns]
    df, more = normalize_text_columns(df)
    df.attrs["load_notes"] = notes + more
    return df
