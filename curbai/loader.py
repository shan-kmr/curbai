"""Cached parquet / pickle loaders for the CurbAI Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CURBAI_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CURBAI_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"

SCORED_PATH = DATA_DIR / "sf_scored.parquet"
POIS_PATH = DATA_DIR / "pois_sf.parquet"
POIS_RAW_PATH = DATA_RAW / "pois_sf.parquet"
ROAD_GRAPH_PATH = DATA_DIR / "road_graph.pkl"


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


@lru_cache(maxsize=1)
def load_pois_sf() -> pd.DataFrame:
    """Load the slim POI parquet (lat, lon, category, name).

    Prefers the deploy-friendly copy at `data/pois_sf.parquet`; falls back to
    `data/raw/pois_sf.parquet` for local dev if build_sf.py hasn't been rerun.
    """
    path = POIS_PATH if POIS_PATH.exists() else POIS_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{POIS_PATH} not found and {POIS_RAW_PATH} missing. "
            "Run scripts/bootstrap_data.py then scripts/build_sf.py."
        )
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_road_graph():
    """Load the pickled OSM road graph (NetworkX undirected Graph)."""
    if not ROAD_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{ROAD_GRAPH_PATH} not found. Run scripts/build_sf.py to regenerate."
        )
    from curbai.catchment import load_pickled_graph

    return load_pickled_graph(ROAD_GRAPH_PATH)


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
