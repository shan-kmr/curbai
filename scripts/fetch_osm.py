"""
Fetch OpenStreetMap data for the San Francisco bbox using OSMnx and write
GeoPackage / parquet files into curbai/data/raw/osm/. One-time run; ~1-2 min.

What we fetch:
  - Road network (driveable graph) -> sf_roads.gpkg
  - Buildings -> sf_buildings.parquet
  - Transit stops (bus + subway + train) -> sf_transit.parquet
  - Amenities relevant to scoring (restaurants, retail, benches, schools,
    police/hospital/fire for safety proxies) -> sf_amenities.parquet

All outputs are gitignored and re-creatable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

# SF bbox (west, south, east, north) — OSMnx 2.x signature
SF_BBOX = (-122.52, 37.71, -122.36, 37.83)

CURBAI_ROOT = Path(__file__).resolve().parents[1]
OSM_DIR = CURBAI_ROOT / "data" / "raw" / "osm"
OSM_DIR.mkdir(parents=True, exist_ok=True)

ROADS_OUT = OSM_DIR / "sf_roads.gpkg"
BUILDINGS_OUT = OSM_DIR / "sf_buildings.parquet"
TRANSIT_OUT = OSM_DIR / "sf_transit.parquet"
AMENITIES_OUT = OSM_DIR / "sf_amenities.parquet"


def log(msg: str) -> None:
    print(f"[osm] {msg}", flush=True)


def fetch_roads() -> None:
    if ROADS_OUT.exists():
        log(f"roads already cached: {ROADS_OUT}")
        return

    log("downloading drive road graph...")
    G = ox.graph_from_bbox(bbox=SF_BBOX, network_type="drive", simplify=True)
    log(f"  nodes: {len(G.nodes):,}  edges: {len(G.edges):,}")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Reset index so the multi-index columns become regular columns
    # (pyogrio can't write a MultiIndex).
    nodes_out = nodes_gdf.reset_index(drop=False)
    edges_out = edges_gdf.reset_index(drop=False)

    # Coerce any list-valued columns to strings for GeoPackage compatibility.
    for gdf in (nodes_out, edges_out):
        for col in gdf.columns:
            if gdf[col].dtype == object and gdf[col].apply(lambda v: isinstance(v, list)).any():
                gdf[col] = gdf[col].astype(str)

    nodes_out.to_file(ROADS_OUT, layer="nodes", driver="GPKG")
    edges_out.to_file(ROADS_OUT, layer="edges", driver="GPKG")
    log(f"  wrote {ROADS_OUT}")


def fetch_buildings() -> None:
    if BUILDINGS_OUT.exists():
        log(f"buildings already cached: {BUILDINGS_OUT}")
        return

    log("downloading buildings...")
    gdf = ox.features_from_bbox(bbox=SF_BBOX, tags={"building": True})
    log(f"  buildings: {len(gdf):,}")

    keep_cols = [c for c in ["building", "height", "building:levels", "name"] if c in gdf.columns]
    out = gdf[keep_cols + ["geometry"]].copy()

    # For storage as parquet, convert geometry to WKB.
    out = out.to_crs(epsg=4326)
    out["geom_wkb"] = out.geometry.to_wkb()
    out = pd.DataFrame(out.drop(columns="geometry"))
    out.to_parquet(BUILDINGS_OUT, index=False)
    log(f"  wrote {BUILDINGS_OUT}")


def fetch_transit() -> None:
    if TRANSIT_OUT.exists():
        log(f"transit already cached: {TRANSIT_OUT}")
        return

    log("downloading transit stops...")
    tags = {
        "public_transport": ["stop_position", "platform", "station", "stop_area"],
        "highway": "bus_stop",
        "railway": ["station", "tram_stop", "subway_entrance", "halt"],
    }
    gdf = ox.features_from_bbox(bbox=SF_BBOX, tags=tags)
    log(f"  raw transit features: {len(gdf):,}")

    # Point-only (we only need stop locations).
    gdf = gdf[gdf.geometry.geom_type.isin(["Point"])].to_crs(epsg=4326)
    out = pd.DataFrame(
        {
            "lat": gdf.geometry.y.values,
            "lon": gdf.geometry.x.values,
            "name": gdf.get("name", pd.Series(index=gdf.index, dtype=object)).astype(str).values,
            "transit_type": gdf.get("public_transport", pd.Series(index=gdf.index, dtype=object)).astype(str).values,
            "highway": gdf.get("highway", pd.Series(index=gdf.index, dtype=object)).astype(str).values,
            "railway": gdf.get("railway", pd.Series(index=gdf.index, dtype=object)).astype(str).values,
        }
    )
    out.to_parquet(TRANSIT_OUT, index=False)
    log(f"  wrote {TRANSIT_OUT}  (points: {len(out):,})")


def fetch_amenities() -> None:
    if AMENITIES_OUT.exists():
        log(f"amenities already cached: {AMENITIES_OUT}")
        return

    log("downloading amenities (restaurants, retail, safety, greenery)...")
    tags = {
        "amenity": [
            "restaurant",
            "cafe",
            "bar",
            "fast_food",
            "food_court",
            "ice_cream",
            "pub",
            "bicycle_parking",
            "bench",
            "shelter",
            "police",
            "hospital",
            "fire_station",
            "clinic",
            "school",
            "university",
        ],
        "shop": True,
        "leisure": ["park", "garden"],
    }
    gdf = ox.features_from_bbox(bbox=SF_BBOX, tags=tags)
    log(f"  raw amenity features: {len(gdf):,}")

    # Centroid of every feature as a point (handles polygons too).
    gdf_proj = gdf.to_crs(epsg=4326)

    lat = gdf_proj.geometry.centroid.y
    lon = gdf_proj.geometry.centroid.x

    def col_or_blank(name: str) -> pd.Series:
        if name in gdf.columns:
            return gdf[name].astype(str)
        return pd.Series("", index=gdf.index)

    out = pd.DataFrame(
        {
            "lat": lat.values,
            "lon": lon.values,
            "amenity": col_or_blank("amenity").values,
            "shop": col_or_blank("shop").values,
            "leisure": col_or_blank("leisure").values,
            "name": col_or_blank("name").values,
        }
    )
    # drop any rows with bad coords
    out = out.dropna(subset=["lat", "lon"])
    out.to_parquet(AMENITIES_OUT, index=False)
    log(f"  wrote {AMENITIES_OUT}  (rows: {len(out):,})")


if __name__ == "__main__":
    log("=== CurbAI OSM fetch (San Francisco) ===")
    try:
        fetch_roads()
        fetch_buildings()
        fetch_transit()
        fetch_amenities()
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        raise
    log("done.")
