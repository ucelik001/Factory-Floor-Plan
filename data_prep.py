from __future__ import annotations

import io

import numpy as np
import pandas as pd
from config import EventLogConfig


def load_event_log(file_bytes: bytes, file_name: str, sheet_name=0) -> pd.DataFrame:
    """CSV or XLSX raw text (dtype=str) is read as-is. Column meanings
    are not yet known — this is resolved by prepare_dataframe() with the cfg."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="utf-8", engine="python")
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {file_name!r} (only .csv / .xlsx)")


def prepare_dataframe(raw: pd.DataFrame, cfg: EventLogConfig) -> pd.DataFrame:
    """Validates required columns, sets types and keeps only operation rows.

    Rows that do not carry an operation (e.g. Set Order / Update Order / Request Order Complete rows,
    which have no work center, station or start time) are dropped here automatically."""
    df = raw.copy()

    required = [cfg.case_id_col, cfg.activity_code_col] + list(cfg.hier_code_cols)
    required += [cfg.start_timestamp_col, cfg.end_timestamp_col] if cfg.has_interval else [cfg.timestamp_col]
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}\nAvailable columns: {list(df.columns)}")

    ts_cols = [c for c in (cfg.timestamp_col, cfg.start_timestamp_col, cfg.end_timestamp_col)
               if c and c in df.columns]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    key_cols = [cfg.case_id_col, cfg.activity_code_col, *cfg.hier_code_cols, cfg.ts_col]
    n0 = len(df)
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    df.attrs["dropped_rows"] = n0 - len(df)

    str_cols = list(cfg.hier_code_cols) + [cfg.activity_code_col]
    str_cols += [lvl.name_col for lvl in cfg.hierarchy_levels if lvl.name_col]
    if cfg.activity_name_col:
        str_cols.append(cfg.activity_name_col)
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in list(cfg.hier_code_cols) + [cfg.activity_code_col]:
        df[col] = df[col].astype("category")
    return df


def safe_sort(frame: pd.DataFrame, cols) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]
    if len(frame) == 0:
        return frame.reset_index(drop=True)
    keys = [frame[c].to_numpy() for c in reversed(cols)]
    order = np.lexsort(keys)
    return frame.iloc[order].reset_index(drop=True)


def format_duration(seconds):
    seconds = abs(seconds)
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.0f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"
