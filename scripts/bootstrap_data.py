"""
One-time bootstrap: pull San Francisco data out of the sibling geofm-global
project, copy the needed subsets into curbai/data/raw/, and never touch the
sibling again after this runs. CurbAI is standalone from that point on.

What we take:
  - geofm-global/data/processed/cells.parquet  (11.7M rows, H3 res-9 + POI counts)
      -> curbai/data/raw/cells_sf.parquet  (SF bbox only, ~1-2k rows)
  - geofm-global/data/raw/overture/pois_global.parquet (3.1 GB Overture+FSQ dedup)
      -> curbai/data/raw/pois_sf.parquet  (SF bbox only)

Run once:
    python scripts/bootstrap_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

# ---- paths ----
CURBAI_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CURBAI_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SIBLING_ROOT = CURBAI_ROOT.parent / "geofm-global"
CELLS_SRC = SIBLING_ROOT / "data" / "processed" / "cells.parquet"
POIS_SRC = SIBLING_ROOT / "data" / "raw" / "overture" / "pois_global.parquet"

# San Francisco bbox (from geofm-global/configs/cities_deep.yaml)
SF_BBOX = {
    "min_lon": -122.52,
    "min_lat": 37.71,
    "max_lon": -122.36,
    "max_lat": 37.83,
}

CELLS_OUT = RAW_DIR / "cells_sf.parquet"
POIS_OUT = RAW_DIR / "pois_sf.parquet"


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def bootstrap_cells() -> None:
    """Filter geofm-global's master cell grid to the SF bbox and copy out."""
    if not CELLS_SRC.exists():
        log(f"ERROR: {CELLS_SRC} not found. Is geofm-global next to curbai?")
        sys.exit(1)

    log(f"reading {CELLS_SRC.name} ({CELLS_SRC.stat().st_size / 1e6:.0f} MB)...")

    con = duckdb.connect(":memory:")
    con.execute(
        f"""
        CREATE VIEW cells AS
        SELECT * FROM read_parquet('{CELLS_SRC.as_posix()}')
        """
    )

    total = con.execute("SELECT count(*) FROM cells").fetchone()[0]
    log(f"total global cells: {total:,}")

    # Filter by center lat/lon inside SF bbox.
    sf_rows = con.execute(
        f"""
        SELECT *
        FROM cells
        WHERE center_lon BETWEEN {SF_BBOX['min_lon']} AND {SF_BBOX['max_lon']}
          AND center_lat BETWEEN {SF_BBOX['min_lat']} AND {SF_BBOX['max_lat']}
        """
    ).df()

    log(f"SF cells: {len(sf_rows):,}")
    if len(sf_rows) == 0:
        log("ERROR: zero rows inside SF bbox. Aborting.")
        sys.exit(1)

    sf_rows.to_parquet(CELLS_OUT, index=False)
    log(f"wrote {CELLS_OUT} ({CELLS_OUT.stat().st_size / 1e6:.1f} MB)")


def bootstrap_pois() -> None:
    """Filter the big Overture POI parquet to SF bbox and copy out."""
    if not POIS_SRC.exists():
        log(f"WARN: {POIS_SRC} not found. Skipping (scoring will use cells.parquet aggregates only).")
        return

    log(f"reading {POIS_SRC.name} ({POIS_SRC.stat().st_size / 1e9:.1f} GB) via DuckDB...")

    # Peek at columns so we can select only what we need.
    schema = pq.ParquetFile(POIS_SRC).schema_arrow
    cols = [f.name for f in schema]
    log(f"POI columns available: {cols}")

    # We want h3_index (if present), lat/lon, category, and any subcategory.
    # Pick whichever lat/lon and category columns exist.
    lat_col = next((c for c in ["lat", "latitude", "y"] if c in cols), None)
    lon_col = next((c for c in ["lon", "longitude", "lng", "x"] if c in cols), None)
    cat_col = next((c for c in ["category", "primary_category", "top_category", "cat"] if c in cols), None)

    if lat_col is None or lon_col is None:
        log(f"ERROR: cannot find lat/lon in POI schema. cols={cols}")
        sys.exit(1)

    select_cols = [lat_col, lon_col]
    if cat_col:
        select_cols.append(cat_col)
    for maybe in ["name", "id", "h3_index", "h3", "country"]:
        if maybe in cols and maybe not in select_cols:
            select_cols.append(maybe)

    select_sql = ", ".join(f'"{c}"' for c in select_cols)

    con = duckdb.connect(":memory:")
    sf_pois = con.execute(
        f"""
        SELECT {select_sql}
        FROM read_parquet('{POIS_SRC.as_posix()}')
        WHERE "{lon_col}" BETWEEN {SF_BBOX['min_lon']} AND {SF_BBOX['max_lon']}
          AND "{lat_col}" BETWEEN {SF_BBOX['min_lat']} AND {SF_BBOX['max_lat']}
        """
    ).df()

    log(f"SF POIs: {len(sf_pois):,}")
    if len(sf_pois) == 0:
        log("WARN: zero POIs in SF bbox. Skipping file write.")
        return

    # Normalize column names for downstream code.
    rename = {}
    if lat_col != "lat":
        rename[lat_col] = "lat"
    if lon_col != "lon":
        rename[lon_col] = "lon"
    if cat_col and cat_col != "category":
        rename[cat_col] = "category"
    sf_pois = sf_pois.rename(columns=rename)

    sf_pois.to_parquet(POIS_OUT, index=False)
    log(f"wrote {POIS_OUT} ({POIS_OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    log("=== CurbAI bootstrap: one-time sibling-project copy ===")
    bootstrap_cells()
    bootstrap_pois()
    log("done. curbai/ is now independent of geofm-global/.")
