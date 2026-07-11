#!/usr/bin/env python
"""Build per-hex (H3 res-9) workplace-jobs layer from Census LEHD LODES8 WAC.

Output: data/lodes_us_h3.parquet
  h3_index (string), lodes_jobs, lodes_jobs_retail, lodes_jobs_food,
  lodes_jobs_health, lodes_jobs_edu (all int32), lodes_year (int32,
  per-hex max of source years).

Method
------
1. For all 50 states + DC, download LODES8 WAC S000 JT00 (all jobs, all
   private+public) per-state CSVs. Preferred year 2022; states whose
   latest vintage is older fall back down the year ladder (MI -> 2021,
   AK -> 2016 per LODES8.3 coverage).
2. Block coordinates: Census Centers of Population 2020. Block-level
   files do NOT exist for the 2020 vintage (only county/tract/blkgrp
   directories exist on www2.census.gov), so we use the block-group
   files CenPop2020_Mean_BG{SS}.txt and join on the 12-digit
   block-group prefix of w_geocode. Zero-population BGs are present in
   the files with valid (geographic-fallback) coordinates, so
   employment-only BGs are covered.
3. blocks -> BG sums -> h3.geo_to_h3(lat, lon, 9) -> national groupby.

Downloads are cached under data/raw/lodes/ (gitignored).
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lodes"
OUT = ROOT / "data" / "lodes_us_h3.parquet"
RAW.mkdir(parents=True, exist_ok=True)

H3_RES = 9
PREFERRED_YEAR = 2022
YEAR_LADDER = list(range(PREFERRED_YEAR, 2015, -1))  # 2022..2016 (AK ends 2016)

WAC_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/{st}_wac_S000_JT00_{year}.csv.gz"
CENPOP_BG_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG{fips}.txt"

WAC_COLS = ["w_geocode", "C000", "CNS07", "CNS18", "CNS16", "CNS15"]
RENAME = {
    "C000": "lodes_jobs",
    "CNS07": "lodes_jobs_retail",
    "CNS18": "lodes_jobs_food",
    "CNS16": "lodes_jobs_health",
    "CNS15": "lodes_jobs_edu",
}
COUNT_COLS = list(RENAME.values())

# 50 states + DC: postal abbrev -> 2-digit FIPS
STATES = {
    "al": "01", "ak": "02", "az": "04", "ar": "05", "ca": "06", "co": "08",
    "ct": "09", "de": "10", "dc": "11", "fl": "12", "ga": "13", "hi": "15",
    "id": "16", "il": "17", "in": "18", "ia": "19", "ks": "20", "ky": "21",
    "la": "22", "me": "23", "md": "24", "ma": "25", "mi": "26", "mn": "27",
    "ms": "28", "mo": "29", "mt": "30", "ne": "31", "nv": "32", "nh": "33",
    "nj": "34", "nm": "35", "ny": "36", "nc": "37", "nd": "38", "oh": "39",
    "ok": "40", "or": "41", "pa": "42", "ri": "44", "sc": "45", "sd": "46",
    "tn": "47", "tx": "48", "ut": "49", "vt": "50", "va": "51", "wa": "53",
    "wv": "54", "wi": "55", "wy": "56",
}


def curl(url: str, dest: Path) -> str:
    """Download url -> dest atomically. Returns 'ok' | 'notfound' | 'error'.

    404 must be distinguished from transient failures: the WAC year
    ladder may only step down on a real 404 — stepping down on a flaky
    transfer would silently pin a state to an older vintage.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(
        ["curl", "-sS", "--retry", "5", "--retry-delay", "5",
         "--connect-timeout", "30", "--max-time", "900",
         "--speed-limit", "1024", "--speed-time", "60",  # abort stalled transfers
         "-w", "%{http_code}", "-o", str(tmp), url],
        capture_output=True, text=True,
    )
    code = r.stdout.strip()
    if r.returncode == 0 and code == "200":
        tmp.rename(dest)
        return "ok"
    tmp.unlink(missing_ok=True)
    return "notfound" if code == "404" else "error"


def curl_retry(url: str, dest: Path, attempts: int = 3) -> str:
    """curl() with extra whole-transfer retries on transient errors."""
    for i in range(attempts):
        status = curl(url, dest)
        if status != "error":
            return status
        time.sleep(10 * (i + 1))
    return "error"


def fetch_state(st: str, fips: str):
    """Ensure WAC (walking the year ladder) + CenPop BG file for one state.

    Returns (st, year_used, wac_path, bg_path).
    """
    wac_path, year_used = None, None
    for year in YEAR_LADDER:
        p = RAW / f"{st}_wac_S000_JT00_{year}.csv.gz"
        if p.exists():
            wac_path, year_used = p, year
            break
        status = curl_retry(WAC_URL.format(st=st, year=year), p)
        if status == "ok":
            wac_path, year_used = p, year
            break
        if status == "error":
            raise RuntimeError(f"{st}: WAC {year} download failed (transient)")
        # 404 -> vintage doesn't exist for this state; step down the ladder.
    if wac_path is None:
        raise RuntimeError(f"{st}: no WAC file found for any year {YEAR_LADDER}")

    bg_path = RAW / f"CenPop2020_Mean_BG{fips}.txt"
    if not bg_path.exists() and curl_retry(CENPOP_BG_URL.format(fips=fips), bg_path) != "ok":
        raise RuntimeError(f"{st}: CenPop BG file download failed")
    return st, year_used, wac_path, bg_path


def process_state(st, year, wac_path, bg_path):
    """One state's WAC blocks -> BG coords -> res-9 hex sums."""
    wac = pd.read_csv(wac_path, usecols=WAC_COLS, dtype={"w_geocode": str})
    total_jobs = int(wac["C000"].sum())

    # All blocks in a BG share the BG's centroid -> same hex, so summing
    # to the 12-digit block-group prefix first is lossless.
    wac["bg12"] = wac["w_geocode"].str[:12]
    bgsum = wac.groupby("bg12", as_index=False)[list(RENAME)].sum()

    bg = pd.read_csv(bg_path, dtype=str, encoding="utf-8-sig")
    bg["bg12"] = bg["STATEFP"] + bg["COUNTYFP"] + bg["TRACTCE"] + bg["BLKGRPCE"]
    bg["lat"] = bg["LATITUDE"].str.lstrip("+").astype(float)
    bg["lon"] = bg["LONGITUDE"].str.lstrip("+").astype(float)

    m = bgsum.merge(bg[["bg12", "lat", "lon"]], on="bg12", how="left")
    lost = m[m["lat"].isna()]
    lost_jobs = int(lost["C000"].sum())
    m = m.dropna(subset=["lat", "lon"])

    m["h3_index"] = [h3.geo_to_h3(la, lo, H3_RES) for la, lo in zip(m["lat"], m["lon"])]
    hex_df = m.groupby("h3_index", as_index=False)[list(RENAME)].sum()
    hex_df = hex_df.rename(columns=RENAME)
    hex_df["lodes_year"] = year
    return hex_df, total_jobs, lost_jobs, len(lost)


def main():
    print(f"Fetching WAC + CenPop for {len(STATES)} states/DC ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        fetched = list(ex.map(lambda kv: fetch_state(*kv), STATES.items()))

    fallback = {st: yr for st, yr, *_ in fetched if yr != PREFERRED_YEAR}
    if fallback:
        print(f"Year fallbacks (not {PREFERRED_YEAR}): {fallback}")
    else:
        print(f"All states at {PREFERRED_YEAR}.")

    parts, nat_jobs_src, nat_lost_jobs, nat_lost_bgs = [], 0, 0, 0
    for st, year, wac_path, bg_path in fetched:
        hex_df, total, lost_jobs, lost_bgs = process_state(st, year, wac_path, bg_path)
        parts.append(hex_df)
        nat_jobs_src += total
        nat_lost_jobs += lost_jobs
        nat_lost_bgs += lost_bgs
        print(f"  {st} {year}: {total:>11,} jobs -> {len(hex_df):>7,} hexes"
              + (f"  [DROPPED {lost_jobs:,} jobs / {lost_bgs} unmatched BGs]" if lost_bgs else ""),
              flush=True)

    us = pd.concat(parts, ignore_index=True)
    # Hexes can straddle state lines (BG centroids of two states in one cell).
    us = us.groupby("h3_index", as_index=False).agg(
        {**{c: "sum" for c in COUNT_COLS}, "lodes_year": "max"}
    )
    for c in COUNT_COLS + ["lodes_year"]:
        us[c] = us[c].astype(np.int32)
    us["h3_index"] = us["h3_index"].astype("string")
    us = us.sort_values("h3_index", ignore_index=True)

    us.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

    # ---- verification -------------------------------------------------
    print(f"\nNational lodes_jobs sum : {us['lodes_jobs'].sum():,}")
    print(f"  (source WAC job sum   : {nat_jobs_src:,}; dropped {nat_lost_jobs:,} jobs"
          f" in {nat_lost_bgs} unmatched BGs)")
    print(f"# hexes                 : {len(us):,}")
    print(f"Schema                  : {dict(us.dtypes.astype(str))}")

    mid = h3.geo_to_h3(40.754, -73.984, H3_RES)
    row = us[us["h3_index"] == mid]
    mj = int(row["lodes_jobs"].iloc[0]) if len(row) else 0
    print(f"\nMidtown hex {mid}: {mj:,} jobs "
          f"(rank {int((us['lodes_jobs'] > mj).sum()) + 1} nationally)")

    print("\nTop-5 job hexes:")
    for _, r in us.nlargest(5, "lodes_jobs").iterrows():
        lat, lon = h3.h3_to_geo(r["h3_index"])
        print(f"  {r['h3_index']}  {r['lodes_jobs']:>9,} jobs  ({lat:.4f}, {lon:.4f})")


if __name__ == "__main__":
    sys.exit(main())
