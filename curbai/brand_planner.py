"""
Brand Location Planner — interactive category-specific white-space analysis.

Given a business category (e.g. "coffee_shop", "gym", "pharmacy"), scores
every H3 cell for opportunity: high demand + low same-category supply.

Data: reads category_counts_sf.parquet (pre-aggregated by build_sf.py)
and joins against the base cell features for the demand side.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h3
import numpy as np
import pandas as pd


CURBAI_ROOT = Path(__file__).resolve().parents[1]
CAT_COUNTS_PATH = CURBAI_ROOT / "data" / "category_counts_sf.parquet"


@lru_cache(maxsize=1)
def load_category_counts() -> pd.DataFrame:
    """Load pre-aggregated category counts per H3 cell."""
    if not CAT_COUNTS_PATH.exists():
        raise FileNotFoundError(
            f"{CAT_COUNTS_PATH} not found. Run scripts/build_sf.py first."
        )
    return pd.read_parquet(CAT_COUNTS_PATH)


def list_categories(min_count: int = 10) -> list[str]:
    """Return categories with at least `min_count` total POIs, sorted by count descending."""
    df = load_category_counts()
    totals = df.groupby("category")["count"].sum().sort_values(ascending=False)
    return [c for c, n in totals.items() if n >= min_count]


def _kring_sum(values: dict[str, int], h3_ids: list[str], k: int = 2) -> dict[str, float]:
    """For each h3 in h3_ids, sum values across k-ring neighbors."""
    out = {}
    for h3_id in h3_ids:
        neighbors = h3.k_ring(h3_id, k)
        out[h3_id] = sum(values.get(nb, 0) for nb in neighbors)
    return out


def _norm_series(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi <= lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def compute_opportunity(
    category: str,
    cells_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute white-space opportunity score for `category` across all cells.

    Returns a copy of cells_df with added columns:
        cat_count_cell    — POIs of this category in the cell
        cat_count_kring   — POIs of this category in the k=2 ring
        demand_proxy      — composite demand signal
        opportunity       — final 0-1 opportunity score
        nearest_distance_m — approx meters to nearest same-category POI (placeholder)
    """
    cat_df = load_category_counts()
    cat_subset = cat_df[cat_df["category"] == category]

    # Build h3 -> count dict for selected category.
    cat_by_cell = dict(zip(cat_subset["h3_index"], cat_subset["count"]))

    result = cells_df.copy()
    h3_ids = result["h3_index"].tolist()

    # Per-cell count.
    result["cat_count_cell"] = result["h3_index"].map(
        lambda h: cat_by_cell.get(h, 0)
    ).astype(int)

    # K-ring neighborhood count.
    kring_counts = _kring_sum(cat_by_cell, h3_ids, k=2)
    result["cat_count_kring"] = result["h3_index"].map(kring_counts).fillna(0).astype(int)

    # Demand proxy: foot traffic × accessibility.
    poi = result["poi_count"].fillna(0)
    amenity = result["amenity_count"].fillna(0)
    transit = result["transit_stop_count"].fillna(0)
    demand = (
        _norm_series(poi) * 0.4
        + _norm_series(amenity) * 0.3
        + _norm_series(transit) * 0.3
    )
    result["demand_proxy"] = demand

    # Saturation: how much of this category is already here.
    max_kring = result["cat_count_kring"].max()
    if max_kring > 0:
        saturation = result["cat_count_kring"] / (max_kring + 1)
    else:
        saturation = pd.Series(0.0, index=result.index)

    # Opportunity = high demand × low saturation.
    result["opportunity"] = (demand * (1.0 - saturation)).clip(0.0, 1.0)

    # Re-normalize to use full 0-1 range.
    opp = result["opportunity"]
    opp_min, opp_max = opp.min(), opp.max()
    if opp_max > opp_min:
        result["opportunity"] = ((opp - opp_min) / (opp_max - opp_min)).clip(0.0, 1.0)

    return result


def category_display_name(cat: str) -> str:
    """Convert Overture category slug to a readable name."""
    return cat.replace("_", " ").replace("and ", "& ").title()
