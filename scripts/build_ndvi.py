#!/usr/bin/env python
"""Build per-hex summer greenness (NDVI) layer for the continental US.

Source: Microsoft Planetary Computer STAC, collection `modis-13Q1-061`
(MODIS Terra Vegetation Indices, 16-day, 250 m), NDVI asset per granule.

Method:
  * Download the CONUS granules for two mid-summer 16-day composites
    (2023-07-04 and 2023-07-20) -> ~44 COGs cached under data/raw/ndvi/.
  * Load 4.34M res-9 hex centroids from data/us_r9/*.parquet.
  * Reproject centroids lon/lat -> MODIS sinusoidal once (chunked).
  * For each granule: read the NDVI band, map points to pixels via the
    inverse affine transform, mask fill (-3000) / invalid (< -2000),
    scale by 0.0001, and max-value composite across granules/dates.
  * Gap-fill: hexes whose pixel is fill in EVERY composite sit on the
    static MOD44W water mask (a known artifact over urban waterfronts,
    e.g. most of Manhattan south of Central Park). For those, take the
    mean of the nearest valid pixel ring (Chebyshev, <= 12 px ~ 2.8 km)
    per raster and max-composite as usual.
  * Write data/ndvi_us_h3.parquet (h3_index, ndvi_summer f32, ndvi_period).

Usage: .venv/bin/python scripts/build_ndvi.py
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import pystac_client
import planetary_computer
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ndvi"
OUT_PATH = ROOT / "data" / "ndvi_us_h3.parquet"
HEX_GLOB = str(ROOT / "data" / "us_r9" / "*.parquet")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
SEARCH_WINDOW = "2023-07-01/2023-08-15"
# 16-day composites that start (and mostly fall) inside July 2023.
# The PC collection mixes Terra (MOD13Q1) and Aqua (MYD13Q1) granules:
# 2023-07-04 / 2023-07-20 are Aqua composite starts, 2023-07-12 is Terra.
# Using all three gives platform redundancy for the max-value composite.
WANT_DATES = {"2023-07-04", "2023-07-12", "2023-07-20"}
PERIOD = "2023-07"
CONUS_BBOX = [-125.0, 24.0, -66.5, 50.0]

NDVI_ASSET = "250m_16_days_NDVI"
FILL_VALUE = -3000
VALID_MIN = -2000  # MODIS valid range is [-2000, 10000] before scaling
VALID_MAX = 10000
SCALE = 0.0001
CHUNK = 500_000
DOWNLOAD_WORKERS = 6
# Ocean-heavy tiles compress to <1 MB; anything smaller than this is junk.
MIN_COG_BYTES = 50_000
MAX_FAILED_GRANULES = 3
GAPFILL_RADIUS = 12  # pixels (~2.8 km) for water-mask artifact fill

SPOT_CHECKS = [
    ("Central Park NYC", 40.783, -73.965, "> 0.6"),
    ("Midtown NYC", 40.754, -73.984, "< 0.3"),
    ("Phoenix suburb", 33.45, -112.07, "low"),
    ("Vermont forest", 44.0, -72.7, "> 0.8"),
]


def fetch_items():
    """Query PC STAC for the CONUS MOD13Q1 granules of the target composites."""
    cat = pystac_client.Client.open(STAC_URL)
    search = cat.search(
        collections=[COLLECTION], datetime=SEARCH_WINDOW, bbox=CONUS_BBOX
    )
    items = [
        it
        for it in search.items()
        if it.properties["start_datetime"][:10] in WANT_DATES
    ]
    print(f"[stac] {len(items)} granules for composites {sorted(WANT_DATES)}")
    if not items:
        raise SystemExit("STAC search returned no granules — aborting.")
    return items


def _is_valid_cog(path):
    try:
        with rasterio.open(path) as src:
            src.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def _download_one(item):
    """Download one granule's NDVI COG (signed URL) unless already cached."""
    dest = RAW_DIR / f"{item.id}_NDVI.tif"
    if (dest.exists() and dest.stat().st_size > MIN_COG_BYTES
            and _is_valid_cog(dest)):
        return dest, "cached"
    tmp = dest.with_suffix(".tif.part")
    for attempt in range(4):
        href = planetary_computer.sign(item.assets[NDVI_ASSET].href)
        r = subprocess.run(
            ["curl", "-sSfL", "--retry", "3", "--max-time", "600",
             "-o", str(tmp), href],
            capture_output=True, text=True,
        )
        if (r.returncode == 0 and tmp.exists()
                and tmp.stat().st_size > MIN_COG_BYTES and _is_valid_cog(tmp)):
            tmp.rename(dest)
            return dest, "downloaded"
        tmp.unlink(missing_ok=True)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {item.id}")


def download_all(items):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths, failed = [], []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(_download_one, it): it.id for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                path, how = fut.result()
            except RuntimeError as err:
                failed.append(futs[fut])
                print(f"[dl {i:>2}/{len(items)}] WARNING {err}", flush=True)
                continue
            paths.append(path)
            print(f"[dl {i:>2}/{len(items)}] {path.name} ({how}, "
                  f"{path.stat().st_size/1e6:.1f} MB)", flush=True)
    if failed:
        print(f"[dl] {len(failed)} granules failed: {failed}", flush=True)
        if len(failed) > MAX_FAILED_GRANULES:
            raise SystemExit(
                "too many failed granules — coverage would suffer; aborting."
            )
    return sorted(paths)


def load_hexes():
    con = duckdb.connect()
    df = con.execute(
        f"SELECT h3_index, center_lat, center_lon FROM '{HEX_GLOB}'"
    ).df()
    print(f"[hex] loaded {len(df):,} hex centroids")
    return df


def project_points(lons, lats, dst_crs):
    """lon/lat -> MODIS sinusoidal, chunked to bound memory."""
    tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    n = len(lons)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        xs[s:e], ys[s:e] = tr.transform(lons[s:e], lats[s:e])
    return xs, ys


def sample_raster(path, xs, ys, best):
    """Max-composite NDVI values from one granule into `best` (float32)."""
    with rasterio.open(path) as src:
        band = src.read(1)
        inv = ~src.transform
        h, w = band.shape
        touched = 0
        for s in range(0, len(xs), CHUNK):
            e = min(s + CHUNK, len(xs))
            x, y = xs[s:e], ys[s:e]
            cols = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64)
            rows = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64)
            m = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            if not m.any():
                continue
            vals = band[rows[m], cols[m]]
            ok = (vals >= VALID_MIN) & (vals <= VALID_MAX) & (vals != FILL_VALUE)
            if not ok.any():
                continue
            idx = np.nonzero(m)[0][ok] + s
            ndvi = vals[ok].astype(np.float32) * SCALE
            best[idx] = np.fmax(best[idx], ndvi)
            touched += len(idx)
    return touched


def gapfill(paths, xs, ys, best):
    """Fill hexes that were fill-valued in every composite (MOD44W
    water-mask artifacts) from the nearest valid pixel ring per raster,
    max-composited across rasters. Mutates `best`; returns #filled."""
    nan_idx = np.nonzero(np.isnan(best))[0]
    if len(nan_idx) == 0:
        return 0
    fillvals = np.full(len(nan_idx), np.nan, dtype=np.float32)
    R = GAPFILL_RADIUS
    xn, yn = xs[nan_idx], ys[nan_idx]
    for p in paths:
        with rasterio.open(p) as src:
            band = src.read(1)
            inv = ~src.transform
            h, w = band.shape
        cols = np.floor(inv.a * xn + inv.b * yn + inv.c).astype(np.int64)
        rows = np.floor(inv.d * xn + inv.e * yn + inv.f).astype(np.int64)
        m = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        for j in np.nonzero(m)[0]:
            r0, c0 = rows[j], cols[j]
            rs, re = max(r0 - R, 0), min(r0 + R + 1, h)
            cs, ce = max(c0 - R, 0), min(c0 + R + 1, w)
            win = band[rs:re, cs:ce]
            valid = (win != FILL_VALUE) & (win >= VALID_MIN) & (win <= VALID_MAX)
            if not valid.any():
                continue
            cheb = np.maximum(
                np.abs(np.arange(rs, re)[:, None] - r0),
                np.abs(np.arange(cs, ce)[None, :] - c0),
            )
            dmin = cheb[valid].min()
            v = win[valid & (cheb == dmin)].mean() * SCALE
            fillvals[j] = np.fmax(fillvals[j], np.float32(v))
    best[nan_idx] = fillvals
    n_filled = int(np.isfinite(fillvals).sum())
    print(f"[gapfill] filled {n_filled:,} of {len(nan_idx):,} empty hexes "
          f"(rest are outside granule coverage or >{R} px from valid land)")
    return n_filled


def spot_check(df):
    import h3
    out = []
    lut = df.set_index("h3_index")["ndvi_summer"]
    for name, lat, lon, expect in SPOT_CHECKS:
        cell = h3.geo_to_h3(lat, lon, 9)
        note = ""
        if cell in lut.index:
            val = float(lut.loc[cell])
        else:
            # cell not in the (population-filtered) hex universe:
            # use the mean of the nearest ring of hexes that are present.
            val = float("nan")
            for k in range(1, 31):
                ring = [c for c in h3.hex_ring(cell, k) if c in lut.index]
                if ring:
                    val = float(lut.loc[ring].mean())
                    note = f" [nearest ring k={k}, n={len(ring)}]"
                    break
        out.append((name, cell, val, expect))
        print(f"[check] {name:<18} {cell}  ndvi={val:.3f}  "
              f"(expect {expect}){note}")
    return out


def main():
    t0 = time.time()
    items = fetch_items()
    paths = download_all(items)

    hexes = load_hexes()
    lons = hexes["center_lon"].to_numpy(dtype=np.float64)
    lats = hexes["center_lat"].to_numpy(dtype=np.float64)

    with rasterio.open(paths[0]) as src:
        sinu_crs = src.crs
    xs, ys = project_points(lons, lats, sinu_crs)
    print(f"[proj] reprojected {len(xs):,} centroids to MODIS sinusoidal")

    best = np.full(len(hexes), np.nan, dtype=np.float32)
    for i, p in enumerate(paths, 1):
        n = sample_raster(p, xs, ys, best)
        print(f"[sample {i:>2}/{len(paths)}] {p.name}: {n:,} valid samples")

    covered = ~np.isnan(best)
    print(f"[composite] hexes covered: {covered.sum():,} / {len(hexes):,}")

    gapfill(paths, xs, ys, best)
    covered = ~np.isnan(best)
    print(f"[final] hexes covered: {covered.sum():,} / {len(hexes):,}")

    out = pd.DataFrame(
        {
            "h3_index": hexes.loc[covered, "h3_index"].to_numpy(),
            "ndvi_summer": np.clip(best[covered], -0.2, 1.0),
            "ndvi_period": PERIOD,
        }
    )
    table = pa.Table.from_pandas(out, preserve_index=False).cast(
        pa.schema(
            [
                ("h3_index", pa.string()),
                ("ndvi_summer", pa.float32()),
                ("ndvi_period", pa.string()),
            ]
        )
    )
    pq.write_table(table, OUT_PATH, compression="zstd")
    print(f"[write] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    v = out["ndvi_summer"]
    print(f"[stats] mean={v.mean():.4f} median={v.median():.4f} "
          f"p5={v.quantile(0.05):.3f} p95={v.quantile(0.95):.3f} "
          f"min={v.min():.3f} max={v.max():.3f}")
    spot_check(out)
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
