"""Model evaluation utilities — IO helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from joblib import dump

def project_root() -> Path:
    """
    Returns the data_processing/ directory.
    utils/ sits at data_processing/04_model_evaluation/utils/, so parents[2] resolves correctly regardless of the
    working directory.
    """
    return Path(__file__).resolve().parents[2]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write obj as indented JSON. NaN should be replaced with None before calling."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def upsert_summary_row(
    csv_path: Path,
    row: Dict[str, Any],
    key_cols: Iterable[str] = ("split", "regime", "model", "seed"),
) -> None:
    """
    Write row to the summary CSV, replacing any existing row with the same key.
    This makes evaluate.py idempotent - rerunning a configuration overwrites the previous result rather than
    appending a duplicate row.
    """
    ensure_dir(csv_path.parent)
    df_new = pd.DataFrame([row])

    if not csv_path.exists():
        df_new.to_csv(csv_path, index=False)
        return

    df_old = pd.read_csv(csv_path)
    key_cols = list(key_cols)

    for k in key_cols:
        if k not in df_new.columns:
            raise ValueError(f"Key column '{k}' missing from new row.")
        if k not in df_old.columns:
            df_out = pd.concat([df_old, df_new], ignore_index=True)
            df_out.to_csv(csv_path, index=False)
            return

    mask = pd.Series(True, index=df_old.index)
    for k in key_cols:
        mask &= df_old[k] == df_new.loc[0, k]

    df_old = df_old.loc[~mask]
    all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
    df_out = pd.concat(
        [df_old.reindex(columns=all_cols), df_new.reindex(columns=all_cols)],
        ignore_index=True,
    )
    df_out.to_csv(csv_path, index=False)

def serialized_size_mb(obj: Any) -> float:
    """Measure the serialised size of a model or vectoriser in MB using joblib."""
    buf = io.BytesIO()
    dump(obj, buf)
    return float(len(buf.getvalue())) / (1024.0 * 1024.0)
