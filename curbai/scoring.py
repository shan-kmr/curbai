"""
Scoring functions for CurbIndex — geo data platform capabilities demo.

Three scores, each demonstrating a different commercial application of
block-level geospatial intelligence:

1. Site Intelligence — "where should a business open?"
2. Brand Location Planner — dynamic, category-specific (computed at runtime)
3. Neighborhood Character — "what is this block's functional identity?"

Each static score (1 and 3) returns a 0-1 value per cell and a per-component
breakdown dataframe.  Score 2 is computed dynamically by brand_planner.py.
"""

from __future__ import annotations

import pandas as pd


def _norm(series: pd.Series) -> pd.Series:
    """Robust min-max to [0, 1] using 5th/95th percentiles, clipped."""
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0.0, 1.0)


def _inv(series: pd.Series) -> pd.Series:
    return 1.0 - _norm(series)


# ---------------------------------------------------------------------------
# 1. Site Intelligence
# ---------------------------------------------------------------------------

SITE_WEIGHTS: dict[str, float] = {
    "foot_traffic": 0.25,
    "accessibility": 0.25,
    "commercial_vibrancy": 0.20,
    "demographic_density": 0.15,
    "retail_ecosystem": 0.15,
}


def site_intelligence_score(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Where should a business open? Composite readiness score."""
    poi = df["poi_count"].fillna(0)
    amenity = df["amenity_count"].fillna(0)
    transit = df["transit_stop_count"].fillna(0)
    intersections = df["intersection_count"].fillna(0)
    entropy = df["category_entropy"].fillna(0)
    nightlife = df["nightlife_count"].fillna(0)
    buildings = df["building_count"].fillna(0)
    shops = df["shop_count"].fillna(0)
    restaurants = df["restaurant_count"].fillna(0)

    foot_traffic = (_norm(poi) + _norm(amenity)) / 2.0
    accessibility = (_norm(transit) + _norm(intersections)) / 2.0
    commercial_vibrancy = (_norm(entropy) + _norm(nightlife)) / 2.0
    demographic_density = _norm(buildings)
    retail_ecosystem = (_norm(shops) + _norm(restaurants)) / 2.0

    components = pd.DataFrame({
        "foot_traffic": foot_traffic,
        "accessibility": accessibility,
        "commercial_vibrancy": commercial_vibrancy,
        "demographic_density": demographic_density,
        "retail_ecosystem": retail_ecosystem,
    })
    score = sum(SITE_WEIGHTS[c] * components[c] for c in SITE_WEIGHTS)
    return score.clip(0, 1), components


# ---------------------------------------------------------------------------
# 3. Neighborhood Character Index
# ---------------------------------------------------------------------------

CHARACTER_WEIGHTS: dict[str, float] = {
    "amenity_walkability": 0.20,
    "green_access": 0.15,
    "safety_perception": 0.15,
    "evening_vibrancy": 0.15,
    "connectivity": 0.20,
    "mixed_use": 0.15,
}


def neighborhood_character_score(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """What is this block's livability and commercial character?"""
    amenity = df["amenity_count"].fillna(0)
    shops = df["shop_count"].fillna(0)
    greenery = df["greenery_count"].fillna(0)
    safety = df["safety_count"].fillna(0)
    nightlife = df["nightlife_count"].fillna(0)
    restaurants = df["restaurant_count"].fillna(0)
    transit = df["transit_stop_count"].fillna(0)
    intersections = df["intersection_count"].fillna(0)
    entropy = df["category_entropy"].fillna(0)

    amenity_walkability = (_norm(amenity) + _norm(shops)) / 2.0
    green_access = _norm(greenery)
    safety_perception = (_norm(safety) * 0.6 + _norm(nightlife) * 0.4)
    evening_vibrancy = (_norm(nightlife) + _norm(restaurants)) / 2.0
    connectivity = (_norm(transit) + _norm(intersections)) / 2.0
    mixed_use = _norm(entropy)

    components = pd.DataFrame({
        "amenity_walkability": amenity_walkability,
        "green_access": green_access,
        "safety_perception": safety_perception,
        "evening_vibrancy": evening_vibrancy,
        "connectivity": connectivity,
        "mixed_use": mixed_use,
    })
    score = sum(CHARACTER_WEIGHTS[c] * components[c] for c in CHARACTER_WEIGHTS)
    return score.clip(0, 1), components


# ---------------------------------------------------------------------------
# Unified helpers
# ---------------------------------------------------------------------------

SCORES = {
    "site": (site_intelligence_score, SITE_WEIGHTS, "Site Intelligence"),
    "character": (neighborhood_character_score, CHARACTER_WEIGHTS, "Neighborhood Character"),
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add score_site and score_character to df and return it."""
    df = df.copy()
    df["score_site"], _ = site_intelligence_score(df)
    df["score_character"], _ = neighborhood_character_score(df)
    return df
