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


# ---------------------------------------------------------------------------
# Nearest competitors (POI-level detail for the side panel)
# ---------------------------------------------------------------------------


_EARTH_R_M = 6_371_008.8
_COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _haversine_m(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance in meters from one point to an array."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_R_M * np.arcsin(np.sqrt(a))


def _bearing_to_cardinal(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Initial bearing from (lat1, lon1) to (lat2, lon2), bucketed to 8 compass points."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)
    y = np.sin(dlon_r) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    idx = int(((bearing + 22.5) % 360.0) // 45.0)
    return _COMPASS_8[idx]


def nearest_competitors(
    category: str,
    h3_id: str,
    pois_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    k: int = 5,
    max_m: float = 1500.0,
) -> list[dict]:
    """Return the k nearest POIs of `category` to the selected cell's center.

    Each dict has `name`, `distance_m` (int meters), `direction` (N/NE/.../NW).
    POIs with no name render as "(unnamed)".
    """
    cell = cells_df[cells_df["h3_index"] == h3_id]
    if len(cell) == 0:
        return []
    center_lat = float(cell["center_lat"].iloc[0])
    center_lon = float(cell["center_lon"].iloc[0])

    sub = pois_df[pois_df["category"] == category]
    sub = sub.dropna(subset=["lat", "lon"])
    if len(sub) == 0:
        return []

    lats = sub["lat"].to_numpy(dtype=np.float64)
    lons = sub["lon"].to_numpy(dtype=np.float64)
    dists = _haversine_m(center_lat, center_lon, lats, lons)

    mask = dists <= max_m
    if not mask.any():
        return []

    idx_sorted = np.argsort(dists[mask])[:k]
    sub_masked = sub.loc[mask]
    names = sub_masked["name"].to_numpy() if "name" in sub_masked.columns else np.array([""] * len(sub_masked))
    lat_masked = lats[mask]
    lon_masked = lons[mask]
    dist_masked = dists[mask]

    out: list[dict] = []
    for i in idx_sorted:
        raw_name = names[i]
        if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)) or str(raw_name).strip() == "" or str(raw_name) == "nan":
            display_name = "(unnamed)"
        else:
            display_name = str(raw_name)
        out.append(
            {
                "name": display_name,
                "distance_m": int(round(float(dist_masked[i]))),
                "direction": _bearing_to_cardinal(
                    center_lat, center_lon, float(lat_masked[i]), float(lon_masked[i])
                ),
            }
        )
    return out
