"""
Temporal Intelligence — "when is this block alive?"

Derives a 4-bucket activity profile (morning / midday / evening / late night)
from the POI category mix already computed per H3 cell. This is an open-data
primitive: category composition tells you WHEN a block is busy. A first-party
version would measure actual device density per cell per hour; this module
infers it from WHAT is there.

Weights mirror the rationale baked into the plan:
    - morning     : transit + intersections + shops (commute + opening hours)
    - midday      : restaurants + shops + amenities (lunch + errands)
    - evening     : nightlife + restaurants + amenities (dinner + going out)
    - late_night  : nightlife + restaurants (bars, late dining)
"""

from __future__ import annotations

import pandas as pd


TEMPORAL_BUCKETS: list[str] = ["morning", "midday", "evening", "late_night"]


TEMPORAL_WEIGHTS: dict[str, dict[str, float]] = {
    "morning": {
        "transit_stop_count": 0.5,
        "intersection_count": 0.3,
        "shop_count": 0.2,
    },
    "midday": {
        "restaurant_count": 0.4,
        "shop_count": 0.3,
        "amenity_count": 0.3,
    },
    "evening": {
        "nightlife_count": 0.4,
        "restaurant_count": 0.3,
        "amenity_count": 0.3,
    },
    "late_night": {
        "nightlife_count": 0.7,
        "restaurant_count": 0.3,
    },
}


def _norm(series: pd.Series) -> pd.Series:
    """Robust min-max to [0, 1] using 5th/95th percentiles, clipped.

    Same shape as curbai.scoring._norm — duplicated to keep modules decoupled.
    """
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0.0, 1.0)


def compute_temporal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-column temporal activity profile for every row of `df`.

    Returns a DataFrame with columns:
        activity_morning, activity_midday, activity_evening, activity_late_night

    Each value is a 0-1 "expected activity" score for that time window.
    """
    normed: dict[str, pd.Series] = {}
    needed_cols = {col for bucket in TEMPORAL_WEIGHTS.values() for col in bucket}
    for col in needed_cols:
        src = df[col].fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)
        normed[col] = _norm(src)

    out = pd.DataFrame(index=df.index)
    for bucket, weights in TEMPORAL_WEIGHTS.items():
        acc = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            acc = acc + w * normed[col]
        out[f"activity_{bucket}"] = acc.clip(0.0, 1.0)
    return out


def temporal_columns() -> list[str]:
    return [f"activity_{b}" for b in TEMPORAL_BUCKETS]


def bucket_display_name(bucket: str) -> str:
    mapping = {
        "morning": "Morning (6–11)",
        "midday": "Midday (11–16)",
        "evening": "Evening (16–22)",
        "late_night": "Late night (22–2)",
    }
    return mapping.get(bucket, bucket.replace("_", " ").title())
