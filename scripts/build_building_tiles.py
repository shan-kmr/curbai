#!/usr/bin/env python3
"""Build vector tiles of NYC building footprints -> data/tiles/nyc_buildings.pmtiles.

Source: NYC Open Data "Building Footprints" (dataset 5zhs-2jue, ~1.08M rows),
fetched via the Socrata SoQL JSON API with keyset pagination on `objectid`
(50k rows/page) and a server-side borough filter (base_bbl prefix 1/3/4 =
Manhattan / Brooklyn / Queens, ~836k rows).

Verified live field names: the_geom, bin, base_bbl, height_roof (ft),
ground_elevation, construction_year, objectid.

Stages (each resumable / skippable if its artifact already exists):
  1. download  -> data/raw/nyc_bldg/pages/page_NNNNN.json.gz  (raw cache)
  2. convert   -> data/raw/nyc_bldg/nyc_bldg.ndjson  (one Feature per line,
                  properties {h: height_m rounded 0.1 clamped 2..450,
                              y: construction year 1650..2026 else omitted,
                              b: borough int 1=MN 3=BK 4=QN})
  3. tile      -> tippecanoe -o data/tiles/nyc_buildings.pmtiles
                  -l buildings -Z11 -z16 --drop-smallest-as-needed
                  --extend-zooms-if-still-dropping --simplification=4 --force
  4. verify    -> PMTiles v3 header + metadata sanity checks

Usage:
  .venv/bin/python scripts/build_building_tiles.py [--redownload] [--reconvert] [--retile]
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "nyc_bldg"
PAGES_DIR = RAW_DIR / "pages"
NDJSON_PATH = RAW_DIR / "nyc_bldg.ndjson"
STATS_PATH = RAW_DIR / "convert_stats.json"
DOWNLOAD_MARKER = RAW_DIR / "_DOWNLOAD_COMPLETE.json"
TILES_PATH = ROOT / "data" / "tiles" / "nyc_buildings.pmtiles"

API_URL = "https://data.cityofnewyork.us/resource/5zhs-2jue.json"
SELECT = "objectid,bin,base_bbl,height_roof,ground_elevation,construction_year,the_geom"
BOROUGH_WHERE = (
    "(starts_with(base_bbl,'1') OR starts_with(base_bbl,'3') "
    "OR starts_with(base_bbl,'4'))"
)
PAGE_SIZE = 50_000
HTTP_TIMEOUT_S = 900
HTTP_RETRIES = 4

BOROUGH_NAMES = {1: "Manhattan", 3: "Brooklyn", 4: "Queens"}
FT_TO_M = 0.3048
H_MIN, H_MAX = 2.0, 450.0
Y_MIN, Y_MAX = 1650, 2026


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- download

def _fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "janus-curbai-tiles/1.0 (building footprint tiler)",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def fetch_page(cursor: int) -> list[dict]:
    params = {
        "$select": SELECT,
        "$where": f"{BOROUGH_WHERE} AND objectid > {cursor}",
        "$order": "objectid",
        "$limit": str(PAGE_SIZE),
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            t0 = time.time()
            body = _fetch(url)
            rows = json.loads(body)
            log(
                f"  fetched {len(rows):>6} rows after objectid>{cursor} "
                f"({len(body) / 1e6:.1f} MB, {time.time() - t0:.1f}s)"
            )
            return rows
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            wait = 15 * attempt
            log(f"  fetch attempt {attempt}/{HTTP_RETRIES} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on page after objectid>{cursor}: {last_err}")


def _page_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def download(force: bool) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = RAW_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")

    if force:
        for p in PAGES_DIR.glob("page_*.json.gz"):
            p.unlink()
        DOWNLOAD_MARKER.unlink(missing_ok=True)

    if DOWNLOAD_MARKER.exists():
        info = json.loads(DOWNLOAD_MARKER.read_text())
        log(f"download: cache complete ({info['total_rows']} rows in {info['pages']} pages), skipping")
        return

    existing = sorted(PAGES_DIR.glob("page_*.json.gz"))
    cursor = 0
    page_no = 0
    total = 0
    if existing:
        # Resume: rows are globally ordered by objectid, so the cursor is the
        # last row of the last cached page.
        for p in existing[:-1]:
            page_no += 1
        last_rows = _page_rows(existing[-1])
        page_no += 1
        cursor = int(last_rows[-1]["objectid"])
        total = (len(existing) - 1) * PAGE_SIZE + len(last_rows)  # estimate for logging only
        log(f"download: resuming after {len(existing)} cached pages (cursor objectid={cursor})")

    while True:
        rows = fetch_page(cursor)
        if not rows:
            break
        page_no += 1
        total += len(rows)
        out = PAGES_DIR / f"page_{page_no:05d}.json.gz"
        tmp = out.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(rows, f, separators=(",", ":"))
        tmp.rename(out)
        cursor = int(rows[-1]["objectid"])
        log(f"  wrote {out.name} (cum ~{total} rows)")
        if len(rows) < PAGE_SIZE:
            break

    DOWNLOAD_MARKER.write_text(
        json.dumps({"total_rows": total, "pages": page_no, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    )
    log(f"download: complete — {total} rows in {page_no} pages")


# ----------------------------------------------------------------- convert

def _parse_height_m(raw: object, stats: Counter) -> float:
    try:
        h_ft = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        h_ft = math.nan
    if not math.isfinite(h_ft):
        stats["h_missing_defaulted"] += 1
        return H_MIN
    h_m = h_ft * FT_TO_M
    if h_m < H_MIN:
        stats["h_clamped_low"] += 1
        return H_MIN
    if h_m > H_MAX:
        stats["h_clamped_high"] += 1
        return H_MAX
    return h_m


def _parse_year(raw: object) -> int | None:
    try:
        y = int(float(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return y if Y_MIN <= y <= Y_MAX else None


def convert(force: bool) -> dict:
    if not force and NDJSON_PATH.exists() and STATS_PATH.exists():
        stats = json.loads(STATS_PATH.read_text())
        log(f"convert: {NDJSON_PATH.name} exists ({stats['kept_total']} features), skipping")
        return stats

    pages = sorted(PAGES_DIR.glob("page_*.json.gz"))
    if not pages:
        raise RuntimeError("no cached pages; run download first")

    kept_by_borough: Counter = Counter()
    drops: Counter = Counter()
    quality: Counter = Counter()
    heights: list[float] = []
    years_min, years_max, years_null = Y_MAX + 1, Y_MIN - 1, 0
    tallest: list[tuple[float, str]] = []  # min-heap of (h, bin)

    t0 = time.time()
    tmp = NDJSON_PATH.with_suffix(".ndjson.tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        for i, page in enumerate(pages, 1):
            for row in _page_rows(page):
                geom = row.get("the_geom")
                if not geom or not geom.get("coordinates"):
                    drops["no_geometry"] += 1
                    continue
                shp = shape(geom)
                if shp.is_empty or shp.area <= 0.0:
                    drops["empty_or_zero_area"] += 1
                    continue

                src = row.get("base_bbl") or row.get("bin") or ""
                b = int(src[0]) if src[:1].isdigit() else 0
                if b not in BOROUGH_NAMES:
                    drops["bad_borough"] += 1
                    continue

                h = round(_parse_height_m(row.get("height_roof"), quality), 1)
                y = _parse_year(row.get("construction_year"))

                props: dict = {"h": h, "b": b}
                if y is not None:
                    props["y"] = y
                    if y < years_min:
                        years_min = y
                    if y > years_max:
                        years_max = y
                else:
                    years_null += 1

                kept_by_borough[b] += 1
                heights.append(h)
                item = (h, row.get("bin") or "?")
                if len(tallest) < 8:
                    heapq.heappush(tallest, item)
                elif item > tallest[0]:
                    heapq.heapreplace(tallest, item)

                out.write(json.dumps(
                    {"type": "Feature", "properties": props, "geometry": geom},
                    separators=(",", ":"),
                ))
                out.write("\n")
            log(f"  page {i}/{len(pages)} done (kept so far {sum(kept_by_borough.values())})")
    tmp.rename(NDJSON_PATH)

    heights.sort()
    n = len(heights)
    stats = {
        "kept_total": n,
        "kept_by_borough": {BOROUGH_NAMES[k]: v for k, v in sorted(kept_by_borough.items())},
        "dropped": dict(drops),
        "height_quality": dict(quality),
        "height_m": {
            "min": heights[0],
            "median": round(statistics.median(heights), 1),
            "mean": round(sum(heights) / n, 1),
            "p99": heights[int(n * 0.99)],
            "max": heights[-1],
        },
        "tallest_bins": [
            {"h_m": h, "bin": b} for h, b in sorted(tallest, reverse=True)
        ],
        "year": {"min": years_min, "max": years_max, "null_or_out_of_range": years_null},
        "ndjson_bytes": NDJSON_PATH.stat().st_size,
        "convert_seconds": round(time.time() - t0, 1),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    log(f"convert: kept {n} features in {stats['convert_seconds']}s -> {NDJSON_PATH.name} "
        f"({stats['ndjson_bytes'] / 1e9:.2f} GB)")
    return stats


# -------------------------------------------------------------------- tile

def tile(force: bool) -> None:
    TILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and TILES_PATH.exists()
        and TILES_PATH.stat().st_mtime > NDJSON_PATH.stat().st_mtime
    ):
        log(f"tile: {TILES_PATH.name} is newer than ndjson, skipping (use --retile to force)")
        return

    cmd = [
        "tippecanoe",
        "-o", str(TILES_PATH),
        "-l", "buildings",
        "-Z11", "-z16",
        "--drop-smallest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--simplification=4",
        "--force",
        "-P",  # parallel read; input is newline-delimited
        str(NDJSON_PATH),
    ]
    log("tile: " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"tippecanoe exited with code {proc.returncode}")
    log(f"tile: done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ verify

def verify() -> None:
    size = TILES_PATH.stat().st_size
    with open(TILES_PATH, "rb") as f:
        hdr = f.read(127)
        if hdr[:7] != b"PMTiles":
            raise RuntimeError(f"bad magic bytes: {hdr[:7]!r}")
        version = hdr[7]
        (root_off, root_len, meta_off, meta_len, leaf_off, leaf_len,
         data_off, data_len, addressed, entries, contents) = struct.unpack_from("<11Q", hdr, 8)
        clustered, internal_comp, tile_comp, tile_type, min_z, max_z = hdr[96:102]
        min_lon, min_lat, max_lon, max_lat = (v / 1e7 for v in struct.unpack_from("<4i", hdr, 102))
        f.seek(meta_off)
        meta_raw = f.read(meta_len)
        if internal_comp == 2:
            meta_raw = gzip.decompress(meta_raw)
        meta = json.loads(meta_raw)

    tile_types = {1: "mvt", 2: "png", 3: "jpeg", 4: "webp", 5: "avif"}
    log("verify: PMTiles header OK")
    log(f"  version={version} tile_type={tile_types.get(tile_type, tile_type)} "
        f"compression={'gzip' if tile_comp == 2 else tile_comp} clustered={bool(clustered)}")
    log(f"  zooms {min_z}..{max_z}, addressed_tiles={addressed}, tile_entries={entries}, "
        f"tile_contents={contents}")
    log(f"  bounds lon {min_lon:.4f}..{max_lon:.4f} lat {min_lat:.4f}..{max_lat:.4f}")
    log(f"  size {size / 1e6:.1f} MB")

    layers = meta.get("vector_layers", [])
    log(f"  layers: {[l.get('id') for l in layers]}")
    for lyr in meta.get("tilestats", {}).get("layers", []):
        log(f"  tilestats: layer={lyr.get('layer')} count={lyr.get('count')} "
            f"geometry={lyr.get('geometry')}")

    assert version == 3, "expected PMTiles v3"
    assert tile_type == 1, "expected MVT tiles"
    assert min_z == 11 and max_z >= 16, f"unexpected zoom range {min_z}..{max_z}"
    assert -74.3 < min_lon < -73.6 and 40.4 < min_lat < 41.0, "bounds not NYC-ish"
    assert any(l.get("id") == "buildings" for l in layers), "missing 'buildings' layer"
    log("verify: all assertions passed")


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redownload", action="store_true", help="clear page cache and re-fetch")
    ap.add_argument("--reconvert", action="store_true", help="rebuild ndjson from cached pages")
    ap.add_argument("--retile", action="store_true", help="re-run tippecanoe even if tiles are fresh")
    args = ap.parse_args()

    t0 = time.time()
    download(force=args.redownload)
    stats = convert(force=args.reconvert or args.redownload)
    tile(force=args.retile or args.reconvert or args.redownload)
    verify()

    log("---- summary ----")
    log(f"kept {stats['kept_total']} footprints: {stats['kept_by_borough']}")
    log(f"dropped: {stats['dropped']}  height_quality: {stats['height_quality']}")
    log(f"height_m: {stats['height_m']}")
    log(f"year: {stats['year']}")
    log(f"tallest: {stats['tallest_bins'][:3]}")
    log(f"tiles: {TILES_PATH} ({TILES_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"total wall time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
