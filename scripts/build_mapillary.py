"""
Build the street-furniture / pavement-detections layer for the Hex Atlas
from Mapillary map features (computer-vision detections on street imagery).

Source: Mapillary vector tiles, map-feature POINTS layer (the non-sign
object/marking classes, plus generic object--traffic-sign--front/back blobs):
  https://tiles.mapillary.com/maps/vtp/mly_map_feature_point/2/{z}/{x}/{y}
    ?access_token=MLY|...
Fetched at z=14 (~2.4 km tiles). Each feature carries `value` (class string),
`id`, `first_seen_at` / `last_seen_at` (ms epoch), and a tile-local point
geometry. Decoded with mapbox-vector-tile; y-axis orientation was verified
empirically against the Graph API (decode yields origin bottom-left, i.e.
py counts up from the tile's south edge). Points in the tile buffer
(px/py outside [0, extent)) are duplicates of neighbouring tiles and are
dropped, so no cross-tile id dedupe is needed.

Scope: 5 US deep-city metros, generous bboxes (NYC, LA, Chicago, Houston,
SF) — ~2.4k tiles total. Extending coverage is just a bbox change below.

Each detection is bucketed into an Uber H3 res-9 cell and aggregated:

  mly_crosswalks       value contains 'crosswalk' (zebra marking + plain)
  mly_cones            'traffic-cone' (live-construction proxy)
  mly_streetlights     'street-light'
  mly_poles            'pole' (utility-pole + support pole)
  mly_benches          'bench'
  mly_hydrants         'fire-hydrant'
  mly_drainage         'catch-basin' + 'manhole'
  mly_bike_racks       'bike-rack' / 'bicycle-rack'
  mly_features_total   all point features (incl. classes not broken out)
  mly_first_seen       min(first_seen_at) over the cell's features
  mly_last_seen        max(last_seen_at) over the cell's features

No scores — raw counts only (int32). Writes data/mapillary_us_h3.parquet
keyed by h3_index (string) with a `city` slug column.

Token: env MAPILLARY_TOKEN, with a fallback to the local geofm-global
project config. Get one at https://www.mapillary.com/dashboard/developers.

Raw tiles are cached gzipped under data/raw/mapillary/tiles/ (gitignored),
per-city aggregates under data/raw/mapillary/agg_{city}.parquet, so reruns
and per-city runs are cheap and resumable.

Usage:
  python scripts/build_mapillary.py                 # all 5 cities + finalize
  python scripts/build_mapillary.py --city nyc sf   # subset (finalizes when
                                                    #   all 5 aggregates exist)
  python scripts/build_mapillary.py --verify-only   # re-print checks
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h3
import mapbox_vector_tile
import mercantile
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mapillary"
TILE_CACHE = RAW_DIR / "tiles"
OUT_PATH = ROOT / "data" / "mapillary_us_h3.parquet"

Z = 14
H3_RES = 9
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly_map_feature_point/2/{z}/{x}/{y}"
N_WORKERS = 12
TIMEOUT = 30
MAX_RETRIES = 5

# west, south, east, north
BBOXES: dict[str, tuple[float, float, float, float]] = {
    "nyc": (-74.20, 40.55, -73.65, 40.95),
    "la": (-118.55, 33.85, -118.10, 34.25),
    "chicago": (-87.85, 41.70, -87.50, 42.05),
    "houston": (-95.65, 29.55, -95.15, 30.05),
    "sf": (-122.55, 37.60, -122.30, 37.85),
}

COUNT_COLS = [
    "mly_crosswalks",
    "mly_cones",
    "mly_streetlights",
    "mly_poles",
    "mly_benches",
    "mly_hydrants",
    "mly_drainage",
    "mly_bike_racks",
    "mly_features_total",
]
N_GROUPS = len(COUNT_COLS)
TOTAL_IDX = N_GROUPS - 1


# --------------------------------------------------------------------------
# token
# --------------------------------------------------------------------------

def get_token() -> tuple[str, str]:
    """Return (token, source-description). Never print the token itself."""
    tok = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if tok.startswith("MLY|"):
        return tok, "env:MAPILLARY_TOKEN"
    # Fallback: Mapillary token from the local geofm-global project config.
    cfg = Path.home() / "Downloads" / "Final Semester" / "geofm-global" / "configs" / "curbside.yaml"
    if cfg.exists():
        m = re.search(r'access_token:\s*"(MLY\|[^"]+)"', cfg.read_text())
        if m:
            return m.group(1), f"yaml:{cfg}"
    sys.exit(
        "No Mapillary token found. Set MAPILLARY_TOKEN=MLY|... "
        "(create one at https://www.mapillary.com/dashboard/developers)"
    )


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

_CLASS_CACHE: dict[str, np.ndarray] = {}


def classify(value: str) -> np.ndarray:
    """Map a Mapillary class string to a 0/1 vector over COUNT_COLS."""
    flags = _CLASS_CACHE.get(value)
    if flags is None:
        flags = np.zeros(N_GROUPS, dtype=np.int64)
        if "crosswalk" in value:
            flags[0] = 1
        if "traffic-cone" in value:
            flags[1] = 1
        if "street-light" in value:
            flags[2] = 1
        if "pole" in value:  # utility-pole + support--pole
            flags[3] = 1
        if "bench" in value:
            flags[4] = 1
        if "fire-hydrant" in value:
            flags[5] = 1
        if "catch-basin" in value or "manhole" in value:
            flags[6] = 1
        if "bike-rack" in value or "bicycle-rack" in value:
            flags[7] = 1
        flags[TOTAL_IDX] = 1  # every point feature counts toward the total
        _CLASS_CACHE[value] = flags
    return flags


# --------------------------------------------------------------------------
# tile fetch (cached)
# --------------------------------------------------------------------------

def _cache_path(t: mercantile.Tile) -> Path:
    return TILE_CACHE / str(t.z) / str(t.x) / f"{t.y}.mvt.gz"


def fetch_tile(session: requests.Session, t: mercantile.Tile, token: str) -> bytes:
    """Return raw MVT bytes for a tile (b'' if empty), using the disk cache."""
    p = _cache_path(t)
    if p.exists():
        raw = p.read_bytes()
        return gzip.decompress(raw) if raw else b""

    url = TILE_URL.format(z=t.z, x=t.x, y=t.y)
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params={"access_token": token}, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            data = r.content
        elif r.status_code == 404:  # outside coverage -> empty
            data = b""
        elif r.status_code in (401, 403):
            sys.exit(f"Mapillary rejected the token (HTTP {r.status_code}). Aborting.")
        elif r.status_code == 429 or r.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"tile {t}: HTTP {r.status_code} after {MAX_RETRIES} tries")
            time.sleep(delay)
            delay *= 2
            continue
        else:
            raise RuntimeError(f"tile {t}: unexpected HTTP {r.status_code}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(data, 6) if data else b"")
        return data
    return b""  # unreachable


# --------------------------------------------------------------------------
# tile -> per-hex aggregation
# --------------------------------------------------------------------------

def process_tile(
    t: mercantile.Tile,
    data: bytes,
    bbox: tuple[float, float, float, float],
    counts: dict[str, np.ndarray],
    seen: dict[str, list],
) -> int:
    """Aggregate one tile's point features into the per-hex dicts."""
    if not data:
        return 0
    layer = mapbox_vector_tile.decode(data).get("point")
    if not layer:
        return 0
    extent = float(layer["extent"])
    n = 2.0 ** t.z
    w, s, e, nb = bbox
    kept = 0
    for f in layer["features"]:
        geom = f["geometry"]
        if geom["type"] == "Point":
            pts = [geom["coordinates"]]
        elif geom["type"] == "MultiPoint":
            pts = geom["coordinates"]
        else:
            continue
        props = f["properties"]
        value = props.get("value", "")
        flags = classify(value)
        # epoch-0 / negative timestamps are Mapillary "unknown" sentinels
        first = props.get("first_seen_at")
        if first is not None and first <= 0:
            first = None
        last = props.get("last_seen_at")
        if last is not None and last <= 0:
            last = None
        for px, py in pts:
            # buffer points duplicate neighbouring tiles; half-open interval
            # keeps each point in exactly one tile of the sweep
            if not (0 <= px < extent and 0 <= py < extent):
                continue
            lon = (t.x + px / extent) / n * 360.0 - 180.0
            ytile = t.y + (1.0 - py / extent)  # decode origin is bottom-left
            lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n))))
            if not (w <= lon < e and s <= lat < nb):
                continue
            hx = h3.geo_to_h3(lat, lon, H3_RES)
            vec = counts.get(hx)
            if vec is None:
                counts[hx] = flags.copy()
                seen[hx] = [first, last]
            else:
                vec += flags
                fl = seen[hx]
                if first is not None and (fl[0] is None or first < fl[0]):
                    fl[0] = first
                if last is not None and (fl[1] is None or last > fl[1]):
                    fl[1] = last
            kept += 1
    return kept


def build_city(city: str, token: str) -> pd.DataFrame:
    bbox = BBOXES[city]
    tiles = list(mercantile.tiles(*bbox, Z))
    print(f"[{city}] {len(tiles)} tiles at z={Z}", flush=True)

    counts: dict[str, np.ndarray] = {}
    seen: dict[str, list] = {}
    n_feats = 0
    t0 = time.time()
    with requests.Session() as session, ThreadPoolExecutor(N_WORKERS) as ex:
        futs = {ex.submit(fetch_tile, session, t, token): t for t in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            n_feats += process_tile(t, fut.result(), bbox, counts, seen)
            if i % 100 == 0 or i == len(tiles):
                print(
                    f"[{city}] {i}/{len(tiles)} tiles | {n_feats:,} features "
                    f"| {len(counts):,} hexes | {time.time()-t0:.0f}s",
                    flush=True,
                )

    hexes = sorted(counts)
    mat = np.vstack([counts[h] for h in hexes]) if hexes else np.zeros((0, N_GROUPS), np.int64)
    df = pd.DataFrame(mat, columns=COUNT_COLS)
    df.insert(0, "h3_index", hexes)
    df.insert(1, "city", city)
    df["mly_first_seen"] = pd.array([seen[h][0] for h in hexes], dtype="Int64")
    df["mly_last_seen"] = pd.array([seen[h][1] for h in hexes], dtype="Int64")
    return df


# --------------------------------------------------------------------------
# finalize + verify
# --------------------------------------------------------------------------

def finalize() -> pd.DataFrame | None:
    aggs = {c: RAW_DIR / f"agg_{c}.parquet" for c in BBOXES}
    missing = [c for c, p in aggs.items() if not p.exists()]
    if missing:
        print(f"Not finalizing yet — missing city aggregates: {missing}")
        return None
    df = pd.concat([pd.read_parquet(p) for p in aggs.values()], ignore_index=True)
    for c in COUNT_COLS:
        df[c] = df[c].astype("int32")
    for c in ("mly_first_seen", "mly_last_seen"):
        df[c] = pd.to_datetime(df[c], unit="ms")  # UTC ms epoch -> naive UTC ts
    df = df.sort_values(["city", "h3_index"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(df):,} rows, {OUT_PATH.stat().st_size/1e6:.1f} MB)")
    return df


def verify(df: pd.DataFrame) -> None:
    print("\n=== schema ===")
    print(df.dtypes.to_string())

    print("\n=== per-city hexes / features ===")
    per = df.groupby("city").agg(
        hexes=("h3_index", "size"), features=("mly_features_total", "sum")
    )
    print(per.to_string())
    print(f"TOTAL: {per['hexes'].sum():,} hexes, {per['features'].sum():,} features")

    print("\n=== class totals (all cities) ===")
    print(df[COUNT_COLS].sum().to_string())

    # Manhattan avenue spot-check: Times Square + Fifth Ave/34th
    print("\n=== Manhattan avenue hexes ===")
    for label, lat, lon in [
        ("Times Square 7th Ave & 45th", 40.7580, -73.9857),
        ("Fifth Ave & 34th (Empire State)", 40.7484, -73.9857),
    ]:
        hx = h3.geo_to_h3(lat, lon, H3_RES)
        row = df[df.h3_index == hx]
        if row.empty:
            print(f"  {label}: hex {hx} NOT PRESENT (unexpected)")
            continue
        r = row.iloc[0]
        core = int(r.mly_streetlights + r.mly_poles + r.mly_crosswalks)
        print(
            f"  {label}: {hx} lights={r.mly_streetlights} poles={r.mly_poles} "
            f"xwalks={r.mly_crosswalks} cones={r.mly_cones} total={r.mly_features_total} "
            f"-> lights+poles+xwalks={core} {'OK' if core > 0 else 'FAIL'}"
        )

    print("\n=== top-5 crosswalk hexes ===")
    top = df.nlargest(5, "mly_crosswalks")
    for _, r in top.iterrows():
        lat, lon = h3.h3_to_geo(r.h3_index)
        print(
            f"  {r.h3_index} ({r.city}) lat={lat:.5f} lon={lon:.5f} "
            f"crosswalks={r.mly_crosswalks} total={r.mly_features_total}"
        )

    print("\n=== cones per city (live-construction proxy) ===")
    cones = df.groupby("city")["mly_cones"].sum()
    print(cones.to_string())
    print("cones in every metro:", "OK" if (cones > 0).all() else "FAIL")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--city", nargs="+", choices=list(BBOXES), default=list(BBOXES))
    ap.add_argument("--verify-only", action="store_true", help="re-run checks on existing output")
    args = ap.parse_args()

    if args.verify_only:
        verify(pd.read_parquet(OUT_PATH))
        return

    token, source = get_token()
    print(f"Mapillary token loaded from {source}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for city in args.city:
        agg_path = RAW_DIR / f"agg_{city}.parquet"
        if agg_path.exists():
            print(f"[{city}] aggregate exists, skipping (delete {agg_path} to rebuild)")
            continue
        df = build_city(city, token)
        df.to_parquet(agg_path, index=False)
        print(f"[{city}] saved {agg_path.name}: {len(df):,} hexes, "
              f"{int(df.mly_features_total.sum()):,} features")

    out = finalize()
    if out is not None:
        verify(out)


if __name__ == "__main__":
    main()
