"""Cached parquet loader for the CurbAI Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CURBAI_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CURBAI_ROOT / "data"
SCORED_PATH = DATA_DIR / "sf_scored.parquet"


@lru_cache(maxsize=1)
def load_sf_scored() -> pd.DataFrame:
    """Load the precomputed SF scored grid (1 row per H3 cell)."""
    if not SCORED_PATH.exists():
        raise FileNotFoundError(
            f"{SCORED_PATH} not found. Run:\n"
            "  python scripts/bootstrap_data.py\n"
            "  python scripts/fetch_osm.py\n"
            "  python scripts/build_sf.py"
        )
    df = pd.read_parquet(SCORED_PATH)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns used for scoring / similarity."""
    score_cols = {"score_av", "score_delivery", "score_eats"}
    meta_cols = {
        "h3_index",
        "center_lat",
        "center_lon",
        "top_category",
        "deep_city",
        "deep_country",
        "overture",
    }
    out = []
    for c in df.columns:
        if c in score_cols or c in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out
