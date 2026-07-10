#!/usr/bin/env python
"""Build per-hex terrain slope layer for NYC from USGS 3DEP elevation.

The geometric half of "sidewalk physics" (surface roughness comes from
IMU data elsewhere): how steep is the ground each hex sits on.

Source: USGS 3DEP 1/3 arc-second (~10 m) seamless DEM, staged GeoTIFFs
    https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/
    Tiles n41w074 + n41w075 cover every NYC hex (lon -74.26..-73.70,
    lat 40.49..40.92, Staten Island included) in one consistent product.

Why not the 1 m DEM (v1 decision, documented): TNM Access lists 18 x 1 m
tiles / 3.2 GB over just the Manhattan-Brooklyn-Queens core bbox, spread
across four different lidar projects (NY_CMPG_2013, NY_FEMAR2_Central_2018,
NJ_NW_..._2017, NJ_SdL5_2014) with mixed vintages — a patchwork, and heavy.
10 m slope UNDERESTIMATES absolute local grades (short steep pitches get
averaged) but ranks hexes correctly, which is what the atlas consumes.
v2 upgrade path: swap TILES/SLOPE_SRC to the 1 m project mosaic.

Method:
  * Download + cache the two tiles under data/raw/3dep/ (~670 MB).
  * Mosaic a window covering all hex centers (+300 m margin), nodata -> NaN.
  * Slope via np.gradient with per-row cell size in metres (the raster is
    geographic NAD83, so deg -> m uses the standard latitude series; a
    projected-CRS input would use the affine cell size directly).
    slope_pct = 100 * sqrt((dz/dx)^2 + (dz/dy)^2).
  * For each of the 15,368 res-9 hexes in data/nyc_base.parquet, collect
    pixels within ~90 m of the center and reduce:
    slope_pct_med, slope_pct_p95, elev_m_med, elev_range_m.
  * Write data/slope_nyc_h3.parquet (+ slope_src provenance column).

Verified behaviour / caveats (2026-07 build; tile vintages n41w074
2024-09-26, n41w075 2022-11-28):
  * The nyc_base grid extends into North Jersey, so the citywide
    steepest hexes are the Hudson Palisades west-bank cliffs and Snake
    Hill (Secaucus) — real terrain, transect-checked, not water-edge
    artifacts. Steepest NYC-borough hexes: Highbridge Pk, Fort
    Washington, Fort George/Inwood, Grymes/Ward Hill (SI).
  * Bridges are hydro-flattened in the DTM (GW / Brooklyn / Verrazzano
    midspans read 0% slope) — no bridge spikes in med or p95.
  * Open ocean is nodata: 2 hexes off the Rockaways drop out (15,366 of
    15,368 covered). Rivers/harbor are flattened valid pixels (~0 m),
    so water hexes read ~0% slope.

Usage: .venv/bin/python scripts/build_slope.py
"""

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.merge import merge as rio_merge

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "3dep"
HEX_PATH = ROOT / "data" / "nyc_base.parquet"
OUT_PATH = ROOT / "data" / "slope_nyc_h3.parquet"

TILE_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
    "current/{tile}/USGS_13_{tile}.tif"
)
SLOPE_SRC = "3dep_10m"  # '3dep_1m' when/if v2 swaps in the 1 m mosaic

SAMPLE_RADIUS_M = 90.0  # pixels within this distance of the hex center
MARGIN_DEG = 0.003      # mosaic margin beyond the hex bbox (~300 m)
MIN_VALID_PX = 8        # fewer valid slope pixels than this -> hex dropped
NODATA_FALLBACK = -999999.0

SPOT_CHECKS = [  # (name, lat, lon, expectation)
    ("Washington Hts / Ft George", 40.852, -73.937, "among steepest, med > 4%"),
    ("Midtown Manhattan", 40.754, -73.984, "flat, med < 2%"),
    ("Brooklyn Heights bluff", 40.696, -73.996, "elevated vs DUMBO"),
    ("DUMBO waterfront", 40.7033, -73.9894, "low elevation"),
]


def tiles_for(lat_min, lat_max, lon_min, lon_max):
    """1x1-degree staged tiles (named by NW corner) covering the bbox."""
    tiles = set()
    for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
            tiles.add(f"n{lat + 1:02d}w{-lon:03d}")
    return sorted(tiles)


def _is_valid_tif(path):
    try:
        with rasterio.open(path) as src:
            src.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def download_tiles(tiles):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile in tiles:
        dest = RAW_DIR / f"USGS_13_{tile}.tif"
        if dest.exists() and _is_valid_tif(dest):
            print(f"[dl] {dest.name} (cached, {dest.stat().st_size/1e6:.0f} MB)")
            paths.append(dest)
            continue
        tmp = dest.with_suffix(".tif.part")
        url = TILE_URL.format(tile=tile)
        for attempt in range(4):
            r = subprocess.run(
                ["curl", "-sSfL", "--retry", "3", "--max-time", "1800",
                 "-o", str(tmp), url],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and tmp.exists() and _is_valid_tif(tmp):
                tmp.rename(dest)
                break
            tmp.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
        else:
            raise SystemExit(f"failed to download {url}")
        print(f"[dl] {dest.name} (downloaded, {dest.stat().st_size/1e6:.0f} MB)")
        paths.append(dest)
    return paths


def build_mosaic(paths, bounds):
    """Merge tiles into one float32 elevation array; nodata -> NaN."""
    srcs = [rasterio.open(p) for p in paths]
    crs = srcs[0].crs
    for s in srcs[1:]:
        if s.crs != crs:
            raise SystemExit(f"mixed CRS across tiles: {crs} vs {s.crs}")
    nodata = srcs[0].nodata if srcs[0].nodata is not None else NODATA_FALLBACK
    arr, transform = rio_merge(srcs, bounds=bounds, nodata=nodata)
    for s in srcs:
        s.close()
    elev = arr[0].astype(np.float32, copy=False)
    elev[elev == np.float32(nodata)] = np.nan
    print(f"[mosaic] {elev.shape[0]} x {elev.shape[1]} px, crs={crs}, "
          f"res=({transform.a:.7f}, {-transform.e:.7f}), "
          f"nodata px: {np.isnan(elev).sum():,}")
    return elev, transform, crs


def metres_per_degree(lat_rad):
    """Length of one degree of lat/lon (m) at given latitude (standard series)."""
    m_lat = (111132.92 - 559.82 * np.cos(2 * lat_rad)
             + 1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad))
    m_lon = (111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
             + 0.118 * np.cos(5 * lat_rad))
    return m_lat, m_lon


def slope_percent(elev, transform, crs):
    """Slope in percent from np.gradient, with correct metre cell sizes."""
    dz_drow, dz_dcol = np.gradient(elev)  # per-pixel differences (float32)
    if crs.is_geographic:
        res_x_deg, res_y_deg = transform.a, -transform.e
        row_lat = transform.f + transform.e * (np.arange(elev.shape[0]) + 0.5)
        m_lat, m_lon = metres_per_degree(np.deg2rad(row_lat))
        dy_m = (res_y_deg * m_lat).astype(np.float32)[:, None]  # per row
        dx_m = (res_x_deg * m_lon).astype(np.float32)[:, None]
    else:  # projected: affine units are metres
        dy_m = np.float32(abs(transform.e))
        dx_m = np.float32(abs(transform.a))
    dz_drow /= dy_m
    dz_dcol /= dx_m
    slope = 100.0 * np.sqrt(dz_drow ** 2 + dz_dcol ** 2)
    v = slope[np.isfinite(slope)]
    print(f"[slope] valid px {v.size:,}; med={np.median(v):.2f}% "
          f"p95={np.percentile(v, 95):.2f}% max={v.max():.1f}%")
    return slope


def sample_hexes(hexes, elev, slope, transform, crs):
    """Per-hex stats over pixels within SAMPLE_RADIUS_M of the center."""
    lons = hexes["center_lon"].to_numpy(np.float64)
    lats = hexes["center_lat"].to_numpy(np.float64)
    inv = ~transform
    cols = np.floor(inv.a * lons + inv.b * lats + inv.c).astype(np.int64)
    rows = np.floor(inv.d * lons + inv.e * lats + inv.f).astype(np.int64)
    if crs.is_geographic:
        m_lat, m_lon = metres_per_degree(np.deg2rad(lats))
        r_row = np.ceil(SAMPLE_RADIUS_M / (-transform.e * m_lat)).astype(int)
        r_col = np.ceil(SAMPLE_RADIUS_M / (transform.a * m_lon)).astype(int)
    else:
        r_row = np.full(len(lats), math.ceil(SAMPLE_RADIUS_M / abs(transform.e)))
        r_col = np.full(len(lats), math.ceil(SAMPLE_RADIUS_M / abs(transform.a)))

    H, W = slope.shape
    out = np.full((len(hexes), 4), np.nan, dtype=np.float32)
    for i in range(len(hexes)):
        r0 = max(rows[i] - r_row[i], 0)
        r1 = min(rows[i] + r_row[i] + 1, H)
        c0 = max(cols[i] - r_col[i], 0)
        c1 = min(cols[i] + r_col[i] + 1, W)
        if r1 <= r0 or c1 <= c0:
            continue
        sw = slope[r0:r1, c0:c1]
        sv = sw[np.isfinite(sw)]
        if sv.size < MIN_VALID_PX:
            continue
        ew = elev[r0:r1, c0:c1]
        ev = ew[np.isfinite(ew)]
        s_med, s_p95 = np.percentile(sv, [50.0, 95.0])
        out[i] = (s_med, s_p95, np.median(ev), ev.max() - ev.min())
    covered = np.isfinite(out[:, 0])
    print(f"[sample] hexes covered: {covered.sum():,} / {len(hexes):,} "
          f"(dropped: {(~covered).sum()})")
    df = pd.DataFrame(
        {
            "h3_index": hexes["h3_index"].to_numpy(),
            "slope_pct_med": out[:, 0],
            "slope_pct_p95": out[:, 1],
            "elev_m_med": out[:, 2],
            "elev_range_m": out[:, 3],
            "slope_src": SLOPE_SRC,
        }
    )[covered]
    return df.reset_index(drop=True)


def verify(df):
    import h3

    lut = df.set_index("h3_index")
    med = lut["slope_pct_med"]
    print(f"[stats] slope_pct_med: mean={med.mean():.2f} med={med.median():.2f} "
          f"p90={med.quantile(0.9):.2f} max={med.max():.2f} | "
          f"elev_m_med: min={lut.elev_m_med.min():.1f} "
          f"max={lut.elev_m_med.max():.1f}")

    checks = {}
    for name, lat, lon, expect in SPOT_CHECKS:
        cell = h3.geo_to_h3(lat, lon, 9)
        if cell not in lut.index:
            print(f"[check] {name:<28} {cell}  NOT IN LAYER (expect {expect})")
            continue
        row = lut.loc[cell]
        pctl = 100.0 * (med < row.slope_pct_med).mean()
        checks[name] = row
        print(f"[check] {name:<28} {cell}  slope_med={row.slope_pct_med:.2f}% "
              f"(pctl {pctl:.1f}) p95={row.slope_pct_p95:.2f}% "
              f"elev={row.elev_m_med:.1f} m range={row.elev_range_m:.1f} m "
              f"(expect {expect})")
    bh = checks.get("Brooklyn Heights bluff")
    du = checks.get("DUMBO waterfront")
    if bh is not None and du is not None:
        print(f"[check] Brooklyn Hts elev {bh.elev_m_med:.1f} m vs DUMBO "
              f"{du.elev_m_med:.1f} m -> delta {bh.elev_m_med - du.elev_m_med:+.1f} m")

    print("[top5] steepest hexes by slope_pct_med:")
    for h, row in lut.nlargest(5, "slope_pct_med").iterrows():
        lat, lon = h3.h3_to_geo(h)
        print(f"        {h} ({lat:.4f}, {lon:.4f}) med={row.slope_pct_med:.2f}% "
              f"p95={row.slope_pct_p95:.2f}% elev={row.elev_m_med:.1f} m "
              f"range={row.elev_range_m:.1f} m")


def main():
    t0 = time.time()
    hexes = pd.read_parquet(HEX_PATH, columns=["h3_index", "center_lat",
                                               "center_lon"])
    print(f"[hex] {len(hexes):,} res-9 hexes from {HEX_PATH.name}")
    lat_min, lat_max = hexes.center_lat.min(), hexes.center_lat.max()
    lon_min, lon_max = hexes.center_lon.min(), hexes.center_lon.max()
    print(f"[hex] extent lon {lon_min:.4f}..{lon_max:.4f} "
          f"lat {lat_min:.4f}..{lat_max:.4f}")

    tiles = tiles_for(lat_min, lat_max, lon_min, lon_max)
    print(f"[tiles] {tiles}")
    paths = download_tiles(tiles)

    bounds = (lon_min - MARGIN_DEG, lat_min - MARGIN_DEG,
              lon_max + MARGIN_DEG, lat_max + MARGIN_DEG)
    elev, transform, crs = build_mosaic(paths, bounds)
    slope = slope_percent(elev, transform, crs)

    df = sample_hexes(hexes, elev, slope, transform, crs)

    table = pa.Table.from_pandas(df, preserve_index=False).cast(
        pa.schema(
            [
                ("h3_index", pa.string()),
                ("slope_pct_med", pa.float32()),
                ("slope_pct_p95", pa.float32()),
                ("elev_m_med", pa.float32()),
                ("elev_range_m", pa.float32()),
                ("slope_src", pa.string()),
            ]
        )
    )
    pq.write_table(table, OUT_PATH, compression="zstd")
    print(f"[write] {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, "
          f"{len(df):,} rows)")

    verify(df)
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
