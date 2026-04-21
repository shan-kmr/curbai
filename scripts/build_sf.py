"""
Feature engineering + scoring precompute for CurbAI SF.

Reads (all from curbai/data/raw/):
    cells_sf.parquet        — 1,112 H3 res-9 cells with Overture POI aggregates
    pois_sf.parquet         — 51K individual Overture POIs (lat, lon, category)
    osm/sf_roads.gpkg       — drive graph (nodes = intersections)
    osm/sf_buildings.parquet — 158K building polygons (WKB)
    osm/sf_transit.parquet  — 7K transit stops (lat, lon, type)
    osm/sf_amenities.parquet — 15K restaurants, shops, safety, greenery, etc.

Writes:
    data/sf_scored.parquet   — cells × features × (score_av, score_delivery, score_eats)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely.wkb

CURBAI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURBAI_ROOT))

from curbai import scoring  # noqa: E402

DATA_RAW = CURBAI_ROOT / "data" / "raw"
DATA_OUT = CURBAI_ROOT / "data"
OUT = DATA_OUT / "sf_scored.parquet"

H3_RES = 9


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def latlon_to_h3(lat: float, lon: float) -> str:
    # h3 3.x API
    return h3.geo_to_h3(lat, lon, H3_RES)


def count_by_h3(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.Series:
    """Return a Series: h3_index -> count."""
    # Drop rows with missing coords
    df = df[[lat_col, lon_col]].dropna()
    df = df[(df[lat_col] > -90) & (df[lat_col] < 90) & (df[lon_col] > -180) & (df[lon_col] < 180)]
    h3_ids = [latlon_to_h3(lat, lon) for lat, lon in zip(df[lat_col].values, df[lon_col].values)]
    return pd.Series(Counter(h3_ids))


def load_cells() -> pd.DataFrame:
    path = DATA_RAW / "cells_sf.parquet"
    df = pd.read_parquet(path)
    log(f"cells_sf: {len(df):,} rows — columns: {list(df.columns)}")
    return df


def load_pois() -> pd.DataFrame:
    path = DATA_RAW / "pois_sf.parquet"
    df = pd.read_parquet(path)
    log(f"pois_sf: {len(df):,} rows — columns: {list(df.columns)}")
    return df


def load_road_nodes() -> pd.DataFrame:
    path = DATA_RAW / "osm" / "sf_roads.gpkg"
    nodes = gpd.read_file(path, layer="nodes")
    log(f"road nodes: {len(nodes):,}")
    # Geometry is Point in EPSG:4326 from OSMnx
    nodes = nodes.set_crs(epsg=4326, allow_override=True)
    out = pd.DataFrame(
        {
            "lat": nodes.geometry.y.values,
            "lon": nodes.geometry.x.values,
        }
    )
    return out


def load_buildings() -> pd.DataFrame:
    path = DATA_RAW / "osm" / "sf_buildings.parquet"
    raw = pd.read_parquet(path)
    log(f"buildings: {len(raw):,}")
    # Convert WKB to centroid lat/lon (approximate — ok for binning)
    lats = []
    lons = []
    for wkb_bytes in raw["geom_wkb"].values:
        try:
            geom = shapely.wkb.loads(bytes(wkb_bytes))
            c = geom.centroid
            lats.append(c.y)
            lons.append(c.x)
        except Exception:
            lats.append(np.nan)
            lons.append(np.nan)
    return pd.DataFrame({"lat": lats, "lon": lons})


def load_transit() -> pd.DataFrame:
    path = DATA_RAW / "osm" / "sf_transit.parquet"
    df = pd.read_parquet(path)
    log(f"transit: {len(df):,}")
    return df


def load_amenities() -> pd.DataFrame:
    path = DATA_RAW / "osm" / "sf_amenities.parquet"
    df = pd.read_parquet(path)
    log(f"amenities: {len(df):,}")
    return df


# ---- restaurant / nightlife / safety / greenery classification --------------

RESTAURANT_AMENITIES = {
    "restaurant",
    "cafe",
    "fast_food",
    "food_court",
    "ice_cream",
    "bar",
    "pub",
}
NIGHTLIFE_AMENITIES = {"bar", "pub", "cafe", "ice_cream"}
SAFETY_AMENITIES = {"police", "hospital", "clinic", "fire_station"}
GREENERY_LEISURE = {"park", "garden"}


def classify_amenities(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return dict of label -> h3_index-count Series."""
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["h3"] = [latlon_to_h3(a, b) for a, b in zip(df["lat"].values, df["lon"].values)]

    restaurants = df[df["amenity"].isin(RESTAURANT_AMENITIES)]
    nightlife = df[df["amenity"].isin(NIGHTLIFE_AMENITIES)]
    safety = df[df["amenity"].isin(SAFETY_AMENITIES)]
    greenery = df[df["leisure"].isin(GREENERY_LEISURE)]
    shops = df[df["shop"].astype(bool) & (df["shop"] != "") & (df["shop"] != "nan")]

    return {
        "amenity_count": df.groupby("h3").size(),
        "restaurant_count": restaurants.groupby("h3").size(),
        "nightlife_count": nightlife.groupby("h3").size(),
        "safety_count": safety.groupby("h3").size(),
        "greenery_count": greenery.groupby("h3").size(),
        "shop_count": shops.groupby("h3").size(),
    }


def kring_sum(values: pd.Series, k: int = 2) -> pd.Series:
    """For each h3 in the index of `values`, sum `values` across k-ring neighbors."""
    out = {}
    known = set(values.index)
    for h3_id in values.index:
        neighbors = h3.k_ring(h3_id, k)
        out[h3_id] = sum(values.get(nb, 0) for nb in neighbors if nb in known)
    return pd.Series(out)


# ---- main -------------------------------------------------------------------


def main() -> None:
    log("=== CurbAI build_sf ===")

    cells = load_cells()

    # Start from the master cell grid.
    cells = cells.rename(columns={"h3_index": "h3_index", "center_lat": "center_lat", "center_lon": "center_lon"})
    cells["h3_index"] = cells["h3_index"].astype(str)

    # --- road intersections per cell ----
    road_nodes = load_road_nodes()
    intersection_counts = count_by_h3(road_nodes)
    cells["intersection_count"] = cells["h3_index"].map(intersection_counts).fillna(0).astype(int)
    log(f"intersections -> cells with >0: {(cells['intersection_count'] > 0).sum():,}")

    # --- buildings per cell ----
    buildings = load_buildings()
    building_counts = count_by_h3(buildings)
    cells["building_count"] = cells["h3_index"].map(building_counts).fillna(0).astype(int)
    log(f"buildings -> cells with >0: {(cells['building_count'] > 0).sum():,}")

    # --- transit stops per cell ----
    transit = load_transit()
    transit_counts = count_by_h3(transit)
    cells["transit_stop_count"] = cells["h3_index"].map(transit_counts).fillna(0).astype(int)
    log(f"transit -> cells with >0: {(cells['transit_stop_count'] > 0).sum():,}")

    # --- amenity classifications ----
    amenities = load_amenities()
    amen = classify_amenities(amenities)
    for label, series in amen.items():
        cells[label] = cells["h3_index"].map(series).fillna(0).astype(int)
    log(
        f"amenities -> restaurants: {int(cells['restaurant_count'].sum())}  "
        f"nightlife: {int(cells['nightlife_count'].sum())}  "
        f"safety: {int(cells['safety_count'].sum())}  "
        f"greenery: {int(cells['greenery_count'].sum())}"
    )

    # --- k-ring neighborhood restaurant supply ----
    rest_by_h3 = pd.Series(
        cells["restaurant_count"].values, index=cells["h3_index"].values
    )
    cells["restaurant_count_kring"] = cells["h3_index"].map(kring_sum(rest_by_h3, k=2))
    log(f"restaurant_count_kring: mean {cells['restaurant_count_kring'].mean():.1f}")

    # --- per-category POI counts (for Brand Location Planner) ----
    pois_path = DATA_RAW / "pois_sf.parquet"
    if pois_path.exists():
        pois = pd.read_parquet(pois_path)
        pois = pois.dropna(subset=["lat", "lon", "category"])
        pois["h3_index"] = [latlon_to_h3(lat, lon) for lat, lon in zip(pois["lat"].values, pois["lon"].values)]
        cat_counts = pois.groupby(["h3_index", "category"]).size().reset_index(name="count")
        cat_out = DATA_OUT / "category_counts_sf.parquet"
        cat_counts.to_parquet(cat_out, index=False)
        n_cats = cat_counts["category"].nunique()
        log(f"category_counts: {len(cat_counts):,} rows, {n_cats} unique categories -> {cat_out}")
    else:
        log("WARN: pois_sf.parquet not found; skipping category_counts")

    # --- apply scoring functions ----
    scored = scoring.compute_all(cells)
    log(
        f"scores — site: mean {scored['score_site'].mean():.3f} max {scored['score_site'].max():.3f} | "
        f"character: mean {scored['score_character'].mean():.3f} max {scored['score_character'].max():.3f}"
    )

    # --- write ----
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT, index=False)
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) — {len(scored):,} rows, {len(scored.columns)} cols")
    log(f"columns: {list(scored.columns)}")


if __name__ == "__main__":
    main()
