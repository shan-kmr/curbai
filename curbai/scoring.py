"""
Three scoring functions over per-cell features. Each returns a 0-1 score per
row. Scores are weighted sums of min-max-normalized features passed through a
light rescaling — the point is that they are interpretable, tunable by editing
the `WEIGHTS` dict, and reproducible offline into `data/sf_scored.parquet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _norm(series: pd.Series) -> pd.Series:
    """Robust min-max to [0, 1] using 5th/95th percentiles, clipped."""
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    out = (series - lo) / (hi - lo)
    return out.clip(0.0, 1.0)


def _inv(series: pd.Series) -> pd.Series:
    return 1.0 - _norm(series)


@dataclass(frozen=True)
class ScoreBreakdown:
    """What a score is actually made of, per cell — for the click-to-explain panel."""

    name: str
    score: float
    components: dict[str, float]  # feature label -> normalized contribution in [0,1]
    weights: dict[str, float]
    description: str


# ---------------------------------------------------------------------------
# AV Rider Launch Readiness
# ---------------------------------------------------------------------------

AV_WEIGHTS: dict[str, float] = {
    "road_simplicity": 0.25,     # prefer simpler geometry = fewer intersections per km
    "pickup_curb_proxy": 0.20,   # building footprint density relative to road density
    "demand_density": 0.25,      # where riders actually are
    "transit_complement": 0.15,  # near transit but not saturated
    "ambient_population": 0.15,  # walk-up demand
}


def av_readiness_score(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return (score in [0,1], per-component contributions dataframe)."""
    # Road simplicity: an AV prefers fewer intersections per unit road length.
    # intersections / (edges+1) low = more "grid-like", high = more "spaghetti".
    intersection_density = df["intersection_count"].fillna(0)
    road_simplicity = _inv(intersection_density)

    # Pickup curb proxy: building count is a rough proxy for "there's a curb here
    # with addresses". Too many buildings = dense urban canyon. Balance around mid.
    curb = df["building_count"].fillna(0)
    curb_norm = _norm(curb)
    # Peak at the middle (neither empty lots nor skyscraper cluster).
    pickup_curb = 1.0 - 2.0 * (curb_norm - 0.5).abs()

    # Demand: POIs from Overture master grid + amenity count from OSM.
    poi_count = df["poi_count"].fillna(0)
    amenity_count = df["amenity_count"].fillna(0)
    demand_density = (_norm(poi_count) + _norm(amenity_count)) / 2.0

    # Transit complement: not zero stops (need demand), not saturated (don't compete).
    transit = df["transit_stop_count"].fillna(0)
    transit_norm = _norm(transit)
    transit_complement = 1.0 - 2.0 * (transit_norm - 0.5).abs()

    # Ambient population: use category entropy as a proxy for mixed-use density.
    entropy = df["category_entropy"].fillna(0)
    ambient = _norm(entropy)

    components = pd.DataFrame(
        {
            "road_simplicity": road_simplicity,
            "pickup_curb_proxy": pickup_curb,
            "demand_density": demand_density,
            "transit_complement": transit_complement,
            "ambient_population": ambient,
        }
    )
    score = sum(AV_WEIGHTS[c] * components[c] for c in AV_WEIGHTS)
    return score.clip(0, 1), components


# ---------------------------------------------------------------------------
# Autonomous Delivery Handoff
# ---------------------------------------------------------------------------

DELIVERY_WEIGHTS: dict[str, float] = {
    "curb_accessible": 0.25,    # can a robot physically pull up?
    "low_canyon": 0.15,         # line-of-sight to the door
    "pedestrian_density": 0.20, # walkable path from curb to building
    "safety_proxy": 0.15,       # nearby police/hospital/fire
    "recovery_zones": 0.15,     # benches/shelters for retry waits
    "daytime_light": 0.10,      # not a parking lot edge; POI evidence
}


def delivery_handoff_score(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    building_count = df["building_count"].fillna(0)
    curb = _norm(building_count)
    # Peak at mid-density — rural = no customer, downtown = nowhere to stop.
    curb_accessible = 1.0 - 2.0 * (curb - 0.5).abs()

    # Canyon proxy: very high building density = canyon. Use high as inverse.
    low_canyon = _inv(building_count)

    # Pedestrian density proxy = amenities + intersections (walkable neighborhood).
    amenity = df["amenity_count"].fillna(0)
    intersections = df["intersection_count"].fillna(0)
    pedestrian_density = (_norm(amenity) + _norm(intersections)) / 2.0

    # Safety: nearby police / hospital / fire station counts (already derived).
    safety = df["safety_count"].fillna(0)
    safety_proxy = _norm(safety)

    # Recovery zones: greenery (parks) + benches + shelters (from amenities tags).
    recovery = df["greenery_count"].fillna(0)
    recovery_zones = _norm(recovery)

    # Daytime light proxy = POI density (dense POIs mean "there's activity here").
    poi = df["poi_count"].fillna(0)
    daytime_light = _norm(poi)

    components = pd.DataFrame(
        {
            "curb_accessible": curb_accessible,
            "low_canyon": low_canyon,
            "pedestrian_density": pedestrian_density,
            "safety_proxy": safety_proxy,
            "recovery_zones": recovery_zones,
            "daytime_light": daytime_light,
        }
    )
    score = sum(DELIVERY_WEIGHTS[c] * components[c] for c in DELIVERY_WEIGHTS)
    return score.clip(0, 1), components


# ---------------------------------------------------------------------------
# Rides -> Eats Conversion Upside
# ---------------------------------------------------------------------------

EATS_WEIGHTS: dict[str, float] = {
    "restaurant_supply": 0.35,    # Eats partners within walking distance
    "drop_gravity": 0.20,         # ride drop-off density proxy
    "category_diversity": 0.20,   # more cuisines = more conversion options
    "evening_activity": 0.15,     # nightlife + bar proxy
    "underserved_home": 0.10,     # weak home-zone restaurant supply = order-in upside
}


def eats_upside_score(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    # Restaurant supply: restaurant_count_kring (500m neighborhood).
    rest_kring = df["restaurant_count_kring"].fillna(0)
    restaurant_supply = _norm(rest_kring)

    # Drop gravity: POI density is a proxy for "people get dropped off here".
    poi = df["poi_count"].fillna(0)
    drop_gravity = _norm(poi)

    # Category diversity: unique_categories + category_entropy blend.
    uniq = df["unique_categories"].fillna(0)
    entropy = df["category_entropy"].fillna(0)
    category_diversity = (_norm(uniq) + _norm(entropy)) / 2.0

    # Evening activity: bar/cafe-heavy cells (from amenities).
    evening = df["nightlife_count"].fillna(0)
    evening_activity = _norm(evening)

    # Underserved home proxy: low restaurant_count at the cell itself (not k-ring).
    rest_local = df["restaurant_count"].fillna(0)
    underserved_home = _inv(rest_local)

    components = pd.DataFrame(
        {
            "restaurant_supply": restaurant_supply,
            "drop_gravity": drop_gravity,
            "category_diversity": category_diversity,
            "evening_activity": evening_activity,
            "underserved_home": underserved_home,
        }
    )
    score = sum(EATS_WEIGHTS[c] * components[c] for c in EATS_WEIGHTS)
    return score.clip(0, 1), components


# ---------------------------------------------------------------------------
# Unified helpers
# ---------------------------------------------------------------------------

SCORES = {
    "av": (av_readiness_score, AV_WEIGHTS, "AV Rider Launch Readiness"),
    "delivery": (delivery_handoff_score, DELIVERY_WEIGHTS, "Autonomous Delivery Handoff"),
    "eats": (eats_upside_score, EATS_WEIGHTS, "Rides → Eats Conversion Upside"),
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add score_av, score_delivery, score_eats to df and return it."""
    df = df.copy()
    df["score_av"], _ = av_readiness_score(df)
    df["score_delivery"], _ = delivery_handoff_score(df)
    df["score_eats"], _ = eats_upside_score(df)
    return df
