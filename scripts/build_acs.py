"""
Build the US census (ACS 5-year 2022) layer for the Janus Hex Atlas.

For every US res-9 hex in the geofm universe, attach the American Community
Survey values of the census TRACT its centroid falls in.

Output  data/acs_us_h3.parquet   (one row per US res-9 hex)
  h3_index, tract_geoid, acs_population, acs_median_income, acs_median_age,
  acs_median_rent, acs_median_home_value, acs_pct_no_vehicle

Data sources (both keyless, downloaded over HTTPS and cached under data/):

  * Tract polygons -> Census cartographic boundary file cb_2022_us_tract_500k.
  * ACS values     -> Census 2022 ACS 5-yr *table-based Summary File* (.dat).
                      The public ACS API now requires an API key (every keyless
                      request 302-redirects to missing_key.html). The Summary
                      File is the keyless bulk equivalent and carries the
                      identical estimates the API serves. In it a tract row is
                      GEO_ID = '1400000US<11-digit GEOID>' (summary level 140),
                      and the API variable B19013_001E maps to column B19013_E001.

Method: res-9 hex centroids -> gpd.sjoin(predicate='within') -> tract GEOID
-> left-join the ACS table by GEOID. Sentinel negatives (-666666666, ...) -> NaN.
"""

from __future__ import annotations

import subprocess
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
FE = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed").as_posix()
CELLS = f"{FE}/cells.parquet"
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data"
OUT = OUTDIR / "acs_us_h3.parquet"

# Cheap bbox pre-filter: lower-48 + AK + HI (same convention as build_us.py).
BBOX = (
    "((center_lat BETWEEN 24 AND 49.5 AND center_lon BETWEEN -125 AND -66.5) "
    "OR (center_lat BETWEEN 51 AND 72 AND center_lon BETWEEN -170 AND -129) "
    "OR (center_lat BETWEEN 18 AND 23 AND center_lon BETWEEN -161 AND -154))"
)

# Tract cartographic boundary (national 2022, 1:500k generalized).
TRACT_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_tract_500k.zip"
TRACT_ZIP = OUTDIR / "cb_2022_us_tract_500k.zip"
TRACT_DIR = OUTDIR / "cb_2022_us_tract_500k"

# ACS 2022 5-yr table-based Summary File (keyless bulk .dat downloads).
SF_BASE = ("https://www2.census.gov/programs-surveys/acs/summary_file/2022/"
           "table-based-SF/data/5YRData")
SF_DIR = OUTDIR / "acs_sf_2022"
TRACT_PREFIX = "1400000US"  # summary level 140 = census tract

# One entry per Summary File table -> {output_col_or_helper: SF_column}.
SF_SPEC: dict[str, dict[str, str]] = {
    "b01003": {"acs_population": "B01003_E001"},          # total population
    "b19013": {"acs_median_income": "B19013_E001"},       # median household income
    "b01002": {"acs_median_age": "B01002_E001"},          # median age
    "b25064": {"acs_median_rent": "B25064_E001"},         # median gross rent
    "b25077": {"acs_median_home_value": "B25077_E001"},   # median home value
    "b08201": {"_hh_total": "B08201_E001",                # total households
               "_hh_no_vehicle": "B08201_E002"},          # households w/ no vehicle
}

OUTPUT_COLS = [
    "h3_index", "tract_geoid", "acs_population", "acs_median_income",
    "acs_median_age", "acs_median_rent", "acs_median_home_value",
    "acs_pct_no_vehicle",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path) -> None:
    """curl a URL to dest (follows redirects, fails loudly). Skips if cached."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    r = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "900", url, "-o", str(dest)]
    )
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"download failed ({r.returncode}): {url}")
    print(f"[download] -> {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


def sentinel_to_nan(s: pd.Series) -> pd.Series:
    """Numeric-coerce and null out ACS sentinel negatives (-666666666, etc.).
    All seven variables are non-negative in reality, so any value < 0 is a jam
    value ('median in open interval', 'N/A', 'not available') -> NaN."""
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x < 0)


# --------------------------------------------------------------------------- #
# 1. Hex universe
# --------------------------------------------------------------------------- #
def load_hexes() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"SELECT h3_index, center_lat, center_lon "
        f"FROM read_parquet('{CELLS}') WHERE {BBOX}"
    ).df()
    con.close()
    print(f"[hexes] {len(df):,} US res-9 hexes in bbox")
    return df


# --------------------------------------------------------------------------- #
# 2. Tract polygons
# --------------------------------------------------------------------------- #
def load_tracts() -> gpd.GeoDataFrame:
    download(TRACT_URL, TRACT_ZIP)
    if not TRACT_DIR.exists():
        with zipfile.ZipFile(TRACT_ZIP) as z:
            z.extractall(TRACT_DIR)
    shp = next(TRACT_DIR.glob("*.shp"))
    gdf = gpd.read_file(shp)[["GEOID", "geometry"]]
    # CB files are NAD83 (EPSG:4269); reproject to WGS84 to match hex centroids.
    if gdf.crs is None:
        gdf = gdf.set_crs(4269)
    gdf = gdf.to_crs(4326)
    print(f"[tracts] {len(gdf):,} tract polygons (crs -> {gdf.crs.to_epsg()})")
    return gdf


# --------------------------------------------------------------------------- #
# 3. Spatial join: hex centroid -> tract GEOID  (chunked to cap memory)
# --------------------------------------------------------------------------- #
def assign_tracts(hexes: pd.DataFrame, tracts: gpd.GeoDataFrame,
                  chunk: int = 1_000_000) -> pd.DataFrame:
    parts = []
    n = len(hexes)
    for i in range(0, n, chunk):
        sub = hexes.iloc[i:i + chunk]
        pts = gpd.GeoDataFrame(
            {"h3_index": sub["h3_index"].to_numpy()},
            geometry=gpd.points_from_xy(sub["center_lon"], sub["center_lat"]),
            crs="EPSG:4326",
        )
        j = gpd.sjoin(pts, tracts, predicate="within", how="left")
        # A centroid on a shared edge could match >1 tract; keep one.
        j = j.drop_duplicates("h3_index")
        parts.append(j[["h3_index", "GEOID"]])
        print(f"[sjoin] {min(i + chunk, n):,}/{n:,}")
    out = (pd.concat(parts, ignore_index=True)
           .rename(columns={"GEOID": "tract_geoid"}))
    matched = out["tract_geoid"].notna().sum()
    print(f"[sjoin] matched {matched:,}/{n:,} ({100*matched/n:.2f}%)")
    return out


# --------------------------------------------------------------------------- #
# 4. ACS table (tract level) from the Summary File
# --------------------------------------------------------------------------- #
def load_acs() -> pd.DataFrame:
    acs: pd.DataFrame | None = None
    for table, cols in SF_SPEC.items():
        dest = SF_DIR / f"acsdt5y2022-{table}.dat"
        download(f"{SF_BASE}/acsdt5y2022-{table}.dat", dest)
        usecols = ["GEO_ID"] + list(cols.values())
        raw = pd.read_csv(dest, sep="|", usecols=usecols, dtype=str,
                          na_values=[""], keep_default_na=False)
        raw = raw[raw["GEO_ID"].str.startswith(TRACT_PREFIX)].copy()
        raw["tract_geoid"] = raw["GEO_ID"].str[len(TRACT_PREFIX):]
        frame = pd.DataFrame({"tract_geoid": raw["tract_geoid"].to_numpy()})
        for out_col, sf_col in cols.items():
            frame[out_col] = sentinel_to_nan(raw[sf_col]).to_numpy()
        acs = frame if acs is None else acs.merge(frame, on="tract_geoid", how="outer")
        print(f"[acs] {table}: {len(frame):,} tracts, cols {list(cols)}")

    hh = acs["_hh_total"]
    acs["acs_pct_no_vehicle"] = np.where(
        hh > 0, 100.0 * acs["_hh_no_vehicle"] / hh, np.nan)
    acs = acs.drop(columns=["_hh_total", "_hh_no_vehicle"])
    print(f"[acs] assembled {len(acs):,} tracts x {acs.shape[1]-1} vars")
    return acs


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(final: pd.DataFrame, hexes: pd.DataFrame, acs: pd.DataFrame) -> None:
    import h3

    n = len(final)
    matched = final["tract_geoid"].notna().sum()
    print("\n================= VERIFY =================")
    print(f"hexes total            : {n:,}")
    print(f"hexes matched to tract : {matched:,} ({100*matched/n:.2f}%)")

    # Population summed over DISTINCT matched tracts (NOT per-hex) ~ 330M US total.
    matched_tracts = final["tract_geoid"].dropna().unique()
    distinct = acs[acs["tract_geoid"].isin(matched_tracts)]
    pop = distinct["acs_population"].sum()
    print(f"distinct matched tracts: {len(matched_tracts):,}")
    print(f"pop over distinct tracts: {pop:,.0f}  (expect ~330M national)")

    # Non-null coverage of each ACS variable among matched hexes.
    for c in OUTPUT_COLS[2:]:
        nn = final[c].notna().sum()
        print(f"  {c:<24}: {100*nn/n:5.1f}% non-null  "
              f"(min {np.nanmin(final[c].values):.1f}, "
              f"median {np.nanmedian(final[c].values):.1f}, "
              f"max {np.nanmax(final[c].values):.1f})")

    # Spot-check a Manhattan hex ~ (40.75, -73.99).
    th = h3.geo_to_h3(40.75, -73.99, 9)
    row = final[final["h3_index"] == th]
    if len(row):
        r = row.iloc[0]
        tag = "exact target hex"
    else:
        d = ((hexes["center_lat"] - 40.75) ** 2
             + (hexes["center_lon"] + 73.99) ** 2)
        nb = hexes.loc[d.idxmin(), "h3_index"]
        r = final[final["h3_index"] == nb].iloc[0]
        tag = "nearest hex in universe"
    print(f"\nManhattan spot-check ({tag}): h3={r['h3_index']}")
    print(f"  tract_geoid={r['tract_geoid']}  "
          f"median_income={r['acs_median_income']}  "
          f"population={r['acs_population']}  "
          f"pct_no_vehicle={r['acs_pct_no_vehicle']:.1f}")
    print("=========================================\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    hexes = load_hexes()
    tracts = load_tracts()
    assigned = assign_tracts(hexes, tracts)
    hexes = hexes.merge(assigned, on="h3_index", how="left")

    acs = load_acs()
    final = hexes.merge(acs, on="tract_geoid", how="left")[OUTPUT_COLS]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT, index=False)
    mb = OUT.stat().st_size / 1e6
    print(f"[write] {len(final):,} rows x {final.shape[1]} cols | "
          f"{mb:.1f} MB -> {OUT}")

    verify(final, hexes, acs)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
