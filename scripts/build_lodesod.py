#!/usr/bin/env python
"""Build per-hex (H3 res-9) "who works here" layer from Census LEHD LODES8 OD.

For every workplace hex: the profile of where its workers LIVE.

Output: data/lodesod_us_h3.parquet
  h3_index (string),
  lodesod_workers            int32   sum of S000 (main + aux OD parts)
  lodesod_home_income        float32 worker-weighted MEDIAN of home-tract
                                     ACS median household incomes
  lodesod_home_novehicle_pct float32 worker-weighted MEAN of home-tract
                                     ACS pct-no-vehicle
  lodesod_pct_far            float32 % of workers whose home-BG centroid is
                                     > 25 km (haversine) from the workplace-BG
                                     centroid
  lodesod_top_origin_county  string  FIPS of most common home county
  lodesod_top_origin_share   float32 that county's share of workers (0-100)
  lodesod_year               int32   per-hex max of source years

Method
------
1. For all 50 states + DC, download LODES8 OD JT00 (all jobs) per-state
   CSVs: {st}_od_main (home+work in-state) + {st}_od_aux (work in-state,
   home out-of-state). Year per state matches the WAC build (2022; MI 2021,
   AK 2016) — taken from the cached WAC filename, falling down the year
   ladder only on a real 404 (flagged loudly).
2. Geocodes -> coordinates via Census Centers of Population 2020
   block-group files (CenPop2020_Mean_BG{SS}.txt, already cached by the
   WAC build): both w_geocode and h_geocode are collapsed to their
   12-digit block-group prefix, which is lossless for hex assignment and
   for BG-centroid distances.
3. Per state (duckdb over the gz CSVs, aggregated to BG pairs FIRST so raw
   block rows never accumulate): emit three small parquets under
   data/raw/lodes/od_agg/ —
     flows  (w_hex, h_tract11, S000)   for income/no-vehicle profiles
     county (w_hex, h_county5, S000)   for top origin county
     hexagg (w_hex, workers, far_workers, home_unmatched, year)
   A state whose *_meta.json exists is skipped on re-runs.
4. Home-tract ACS profile from data/acs_us_h3.parquet: tract_geoid ->
   (acs_median_income, acs_pct_no_vehicle); values are tract-level
   constants so one row per tract (max() = the constant, skips NULLs).
   GEOGRAPHY FALLBACK: LODES8/CenPop2020 use 2020-tabulation Connecticut
   counties (09001-09015) while the ACS layer uses 2022 planning regions
   (09110-09190). Home tracts that miss the direct join get ACS values via
   their CenPop BG centroids' res-9 hexes looked up in acs_us_h3.parquet
   (k-ring<=1 rescue), population-weighted mean per tract.
5. Weighted median (documented method): LOWER WEIGHTED MEDIAN — sort a
   hex's home tracts by income, cumulative-sum the S000 weights, take the
   smallest income whose cumulative weight >= 50% of the hex's total
   matched weight. Tracts with NULL income are excluded from the median
   (and from the no-vehicle mean) but still count as workers.
6. lodesod_pct_far denominator = workers whose home BG matched a CenPop
   centroid (unmatched homes are excluded; count reported).

Downloads are cached under data/raw/lodes/ (gitignored).
"""
import glob
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lodes"
AGG = RAW / "od_agg"
ACS = ROOT / "data" / "acs_us_h3.parquet"
WAC_OUT = ROOT / "data" / "lodes_us_h3.parquet"
OUT = ROOT / "data" / "lodesod_us_h3.parquet"
RAW.mkdir(parents=True, exist_ok=True)
AGG.mkdir(parents=True, exist_ok=True)

H3_RES = 9
FAR_KM = 25.0
PREFERRED_YEAR = 2022
YEAR_LADDER = list(range(PREFERRED_YEAR, 2015, -1))  # 2022..2016 (AK ends 2016)

OD_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{year}.csv.gz"
CENPOP_BG_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG{fips}.txt"

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


def wac_year(st: str):
    """Year the WAC build used for this state (from the cached filename)."""
    hits = sorted(RAW.glob(f"{st}_wac_S000_JT00_*.csv.gz"))
    if not hits:
        return None
    return int(hits[-1].stem.split("_")[-1].split(".")[0])


def fetch_state(st: str, fips: str):
    """Ensure OD main+aux (matching the WAC year) + CenPop BG for one state.

    Returns (st, year_used, main_path, aux_path, year_mismatch).
    """
    want = wac_year(st)
    ladder = ([want] + [y for y in YEAR_LADDER if y < want]) if want else YEAR_LADDER

    main_path = aux_path = year_used = None
    for year in ladder:
        mp = RAW / f"{st}_od_main_JT00_{year}.csv.gz"
        if not mp.exists():
            status = curl_retry(OD_URL.format(st=st, part="main", year=year), mp)
            if status == "error":
                raise RuntimeError(f"{st}: OD main {year} download failed (transient)")
            if status == "notfound":
                continue  # real 404 -> step down the ladder
        ap = RAW / f"{st}_od_aux_JT00_{year}.csv.gz"
        if not ap.exists():
            status = curl_retry(OD_URL.format(st=st, part="aux", year=year), ap)
            if status == "error":
                raise RuntimeError(f"{st}: OD aux {year} download failed (transient)")
            if status == "notfound":
                # main+aux are published together; a lone main is unusable
                raise RuntimeError(f"{st}: OD aux {year} missing while main exists")
        main_path, aux_path, year_used = mp, ap, year
        break
    if main_path is None:
        raise RuntimeError(f"{st}: no OD files found for any year {ladder}")

    bg_path = RAW / f"CenPop2020_Mean_BG{fips}.txt"
    if not bg_path.exists() and curl_retry(CENPOP_BG_URL.format(fips=fips), bg_path) != "ok":
        raise RuntimeError(f"{st}: CenPop BG file download failed")

    mismatch = bool(want and year_used != want)
    if mismatch:
        print(f"  !! {st}: OD year {year_used} != WAC year {want}", flush=True)
    return st, year_used, main_path, aux_path, mismatch


def load_bg_table() -> pd.DataFrame:
    """All 51 CenPop BG files -> national (bg12, lat, lon, pop, h3_index)."""
    parts = []
    for fips in STATES.values():
        p = RAW / f"CenPop2020_Mean_BG{fips}.txt"
        bg = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
        bg["bg12"] = bg["STATEFP"] + bg["COUNTYFP"] + bg["TRACTCE"] + bg["BLKGRPCE"]
        bg["lat"] = bg["LATITUDE"].str.lstrip("+").astype(float)
        bg["lon"] = bg["LONGITUDE"].str.lstrip("+").astype(float)
        bg["pop"] = bg["POPULATION"].astype(int)
        parts.append(bg[["bg12", "lat", "lon", "pop"]])
    us = pd.concat(parts, ignore_index=True)
    us["h3_index"] = [h3.geo_to_h3(la, lo, H3_RES) for la, lo in zip(us["lat"], us["lon"])]
    return us


HAVERSINE_KM = (
    "2*6371.0088*asin(sqrt("
    "pow(sin(radians(hb.lat - o.wlat)/2),2)"
    " + cos(radians(o.wlat))*cos(radians(hb.lat))"
    "  *pow(sin(radians(hb.lon - o.wlon)/2),2)))"
)


def process_state(con, st: str, year: int, main_path: Path, aux_path: Path) -> dict:
    """One state's OD blocks -> BG pairs -> per-hex aggregates on disk."""
    meta_p = AGG / f"{st}_meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE od AS
        SELECT substr(w_geocode, 1, 12) AS w_bg12,
               substr(h_geocode, 1, 12) AS h_bg12,
               sum(S000)::BIGINT        AS s
        FROM read_csv_auto([?, ?],
                           types={'w_geocode':'VARCHAR','h_geocode':'VARCHAR'})
        GROUP BY 1, 2
        """,
        [str(main_path), str(aux_path)],
    )
    src_total = con.execute("SELECT coalesce(sum(s),0) FROM od").fetchone()[0]

    # Workplace side must geocode; rows whose w_bg12 has no CenPop centroid
    # are dropped (counted). Home side stays LEFT-joined for the far metric.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE odw AS
        SELECT o.w_bg12, o.h_bg12, o.s, b.h3_index AS w_hex,
               b.lat AS wlat, b.lon AS wlon
        FROM od o JOIN bg b ON o.w_bg12 = b.bg12
        """
    )
    matched = con.execute("SELECT coalesce(sum(s),0) FROM odw").fetchone()[0]

    def copy_atomic(select_sql: str, dest: Path):
        tmp = dest.with_suffix(".tmp.parquet")
        con.execute(f"COPY ({select_sql}) TO '{tmp}' (FORMAT PARQUET)")
        os.replace(tmp, dest)

    copy_atomic(
        "SELECT w_hex, substr(h_bg12,1,11) AS h_tract, sum(s) AS s "
        "FROM odw GROUP BY 1,2",
        AGG / f"{st}_flows.parquet",
    )
    copy_atomic(
        "SELECT w_hex, substr(h_bg12,1,5) AS cty, sum(s) AS s "
        "FROM odw GROUP BY 1,2",
        AGG / f"{st}_county.parquet",
    )
    copy_atomic(
        f"""
        SELECT w_hex,
               sum(s)                                                    AS workers,
               sum(CASE WHEN hb.bg12 IS NOT NULL
                         AND {HAVERSINE_KM} > {FAR_KM} THEN s ELSE 0 END) AS far_workers,
               sum(CASE WHEN hb.bg12 IS NULL THEN s ELSE 0 END)          AS home_unmatched,
               {year}                                                    AS yr
        FROM odw o LEFT JOIN bg hb ON o.h_bg12 = hb.bg12
        GROUP BY 1
        """,
        AGG / f"{st}_hexagg.parquet",
    )
    con.execute("DROP TABLE IF EXISTS od")
    con.execute("DROP TABLE IF EXISTS odw")

    meta = {"st": st, "year": year, "src_total": int(src_total),
            "matched": int(matched), "dropped_w": int(src_total - matched)}
    meta_p.write_text(json.dumps(meta))
    return meta


def build_tract_profile(con, bg_df: pd.DataFrame):
    """home-tract11 -> (income, novehicle), direct ACS join + geo fallback.

    Returns (profile_df, stats_dict).
    """
    direct = con.execute(
        f"""
        SELECT tract_geoid AS h_tract,
               max(acs_median_income)  AS income,
               max(acs_pct_no_vehicle) AS nv
        FROM read_parquet('{ACS}')
        WHERE tract_geoid IS NOT NULL
        GROUP BY 1
        """
    ).df()

    needed = con.execute(
        f"SELECT DISTINCT h_tract FROM read_parquet('{AGG}/*_flows.parquet')"
    ).df()["h_tract"]
    missing = sorted(set(needed) - set(direct["h_tract"]))

    # Geo fallback: unmatched home tracts (renumbered CT geographies etc.)
    # get ACS values via their CenPop BG centroids' res-9 hexes.
    fb_rows, rescued_bgs = [], 0
    if missing:
        bgs = bg_df[bg_df["bg12"].str[:11].isin(set(missing))].copy()
        hex_pool = set(bgs["h3_index"])
        rings = {h: h3.k_ring(h, 1) for h in set(bgs["h3_index"])}
        for neigh in rings.values():
            hex_pool.update(neigh)
        hex_lu = con.execute(
            f"""
            SELECT h3_index,
                   max(acs_median_income)  AS income,
                   max(acs_pct_no_vehicle) AS nv
            FROM read_parquet('{ACS}')
            WHERE h3_index IN (SELECT unnest(?))
            GROUP BY 1
            """,
            [list(hex_pool)],
        ).df().set_index("h3_index")

        def hex_val(hx):
            if hx in hex_lu.index:
                r = hex_lu.loc[hx]
                return r["income"], r["nv"]
            vals = [hex_lu.loc[n] for n in rings[hx] if n in hex_lu.index]
            if not vals:
                return np.nan, np.nan
            v = pd.DataFrame(vals)
            return v["income"].mean(), v["nv"].mean()  # NaN-skipping

        vals = [hex_val(hx) for hx in bgs["h3_index"]]
        bgs["income"] = [v[0] for v in vals]
        bgs["nv"] = [v[1] for v in vals]
        bgs["tract"] = bgs["bg12"].str[:11]
        rescued_bgs = int(bgs["income"].notna().sum())

        def wavg(g, col):
            v = g[g[col].notna()]
            if v.empty:
                return np.nan
            w = v["pop"].to_numpy(dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(v))
            return float(np.average(v[col], weights=w))

        for tract, g in bgs.groupby("tract"):
            fb_rows.append({"h_tract": tract,
                            "income": wavg(g, "income"), "nv": wavg(g, "nv")})

    fb = pd.DataFrame(fb_rows, columns=["h_tract", "income", "nv"])
    prof = pd.concat([direct, fb], ignore_index=True)
    rescued = int(fb["income"].notna().sum()) if len(fb) else 0
    stats = {"needed": int(len(needed)), "direct": int(len(set(needed) & set(direct["h_tract"]))),
             "missing": len(missing), "rescued": rescued, "rescued_bgs": rescued_bgs}
    return prof, stats


def main():
    print(f"Fetching OD main+aux + CenPop for {len(STATES)} states/DC ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        fetched = list(ex.map(lambda kv: fetch_state(*kv), STATES.items()))

    fallback = {st: yr for st, yr, *_ in fetched if yr != PREFERRED_YEAR}
    print(f"Year fallbacks (not {PREFERRED_YEAR}): {fallback or 'none'}")
    mismatches = [st for st, _, _, _, mm in fetched if mm]
    if mismatches:
        print(f"!! OD/WAC year mismatches: {mismatches}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    bg_df = load_bg_table()
    con.register("bg", bg_df)
    print(f"CenPop BG centroids loaded: {len(bg_df):,}", flush=True)

    nat_src, nat_dropped = 0, 0
    for st, year, main_path, aux_path, _ in fetched:
        t0 = time.time()
        meta = process_state(con, st, year, main_path, aux_path)
        nat_src += meta["src_total"]
        nat_dropped += meta["dropped_w"]
        print(f"  {st} {meta['year']}: {meta['src_total']:>11,} workers"
              f"  ({time.time()-t0:4.1f}s)"
              + (f"  [DROPPED {meta['dropped_w']:,} @ unmatched workplace BGs]"
                 if meta["dropped_w"] else ""),
              flush=True)

    print("\nBuilding home-tract ACS profile (direct + geo fallback) ...", flush=True)
    prof, tstats = build_tract_profile(con, bg_df)
    con.register("tract_prof", prof)
    print(f"  home tracts needed {tstats['needed']:,}: direct {tstats['direct']:,}, "
          f"missing {tstats['missing']:,}, rescued via geo fallback {tstats['rescued']:,} "
          f"(from {tstats['rescued_bgs']:,} BG-hex hits)", flush=True)

    print("Final national assembly ...", flush=True)
    us = con.execute(
        f"""
        WITH hexagg AS (
            SELECT w_hex, sum(workers) AS workers, sum(far_workers) AS far,
                   sum(home_unmatched) AS unm, max(yr) AS yr
            FROM read_parquet('{AGG}/*_hexagg.parquet') GROUP BY 1
        ),
        flows AS (
            SELECT w_hex, h_tract, sum(s) AS s
            FROM read_parquet('{AGG}/*_flows.parquet') GROUP BY 1, 2
        ),
        joined AS (
            SELECT f.w_hex, f.s, t.income, t.nv
            FROM flows f JOIN tract_prof t USING (h_tract)
        ),
        med AS (  -- lower weighted median of home-tract incomes
            SELECT w_hex, min(income) AS home_income FROM (
                SELECT w_hex, income, s,
                       sum(s) OVER (PARTITION BY w_hex ORDER BY income
                                    ROWS UNBOUNDED PRECEDING) AS cum,
                       sum(s) OVER (PARTITION BY w_hex) AS tot
                FROM joined WHERE income IS NOT NULL
            ) WHERE cum >= tot * 0.5 GROUP BY 1
        ),
        nvm AS (
            SELECT w_hex, sum(s * nv) / sum(s) AS novehicle
            FROM joined WHERE nv IS NOT NULL GROUP BY 1
        ),
        cty AS (
            SELECT w_hex, arg_max(cty, s) AS top_cty,
                   max(s) * 100.0 / sum(s) AS top_share
            FROM (SELECT w_hex, cty, sum(s) AS s
                  FROM read_parquet('{AGG}/*_county.parquet') GROUP BY 1, 2)
            GROUP BY 1
        )
        SELECT h.w_hex                                   AS h3_index,
               h.workers                                 AS lodesod_workers,
               m.home_income                             AS lodesod_home_income,
               n.novehicle                               AS lodesod_home_novehicle_pct,
               h.far * 100.0 / nullif(h.workers - h.unm, 0) AS lodesod_pct_far,
               c.top_cty                                 AS lodesod_top_origin_county,
               c.top_share                               AS lodesod_top_origin_share,
               h.yr                                      AS lodesod_year,
               h.unm                                     AS _home_unmatched
        FROM hexagg h
        LEFT JOIN med m ON h.w_hex = m.w_hex
        LEFT JOIN nvm n ON h.w_hex = n.w_hex
        LEFT JOIN cty c ON h.w_hex = c.w_hex
        ORDER BY h3_index
        """
    ).df()

    nat_unmatched_homes = int(us["_home_unmatched"].sum())
    us = us.drop(columns=["_home_unmatched"])

    us["h3_index"] = us["h3_index"].astype("string")
    us["lodesod_workers"] = us["lodesod_workers"].astype(np.int32)
    us["lodesod_year"] = us["lodesod_year"].astype(np.int32)
    us["lodesod_top_origin_county"] = us["lodesod_top_origin_county"].astype("string")
    for c in ["lodesod_home_income", "lodesod_home_novehicle_pct",
              "lodesod_pct_far", "lodesod_top_origin_share"]:
        us[c] = us[c].astype(np.float32)

    us.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

    # ---- verification -------------------------------------------------
    tot = int(us["lodesod_workers"].sum())
    print(f"\n# hexes                  : {len(us):,}")
    print(f"National worker sum      : {tot:,}")
    print(f"  (source OD S000 sum    : {nat_src:,}; dropped {nat_dropped:,} at "
          f"unmatched workplace BGs; {nat_unmatched_homes:,} workers with "
          f"unmatched home BG kept but excluded from pct_far denominator)")
    if WAC_OUT.exists():
        wac_tot = int(pd.read_parquet(WAC_OUT, columns=["lodes_jobs"])["lodes_jobs"].sum())
        print(f"WAC lodes_jobs sum       : {wac_tot:,}  (OD delta "
              f"{100.0*(tot-wac_tot)/wac_tot:+.2f}%)")
    print(f"Schema                   : {dict(us.dtypes.astype(str))}")

    def show(label, hx):
        row = us[us["h3_index"] == hx]
        if row.empty:
            # Workplace hexes are sparse (BG-centroid snapped); fall back to
            # the biggest-worker hex within k-ring 3 of the requested cell.
            near = us[us["h3_index"].isin(h3.k_ring(hx, 3))]
            if near.empty:
                print(f"  {label:34s} {hx}: (not present)")
                return
            row = near.nlargest(1, "lodesod_workers")
            hx = f"{row.iloc[0]['h3_index']} (nearby)"
        r = row.iloc[0]
        print(f"  {label:34s} {hx}: {int(r['lodesod_workers']):>9,} workers | "
              f"home income ${r['lodesod_home_income']:>9,.0f} | "
              f"no-vehicle {r['lodesod_home_novehicle_pct']:5.2f}% | "
              f"far>25km {r['lodesod_pct_far']:5.1f}% | "
              f"top county {r['lodesod_top_origin_county']} "
              f"({r['lodesod_top_origin_share']:.1f}%) | {int(r['lodesod_year'])}")

    print("\nSpot checks:")
    show("Chicago Loop", "892664c1a9bffff")
    show("Apple Park, Cupertino (suburb HQ)", h3.geo_to_h3(37.3349, -122.0090, H3_RES))
    show("Sand Hill Rd, Menlo Park (wealthy)", h3.geo_to_h3(37.4209, -122.2129, H3_RES))
    show("Vernon, CA (industrial)", h3.geo_to_h3(34.0039, -118.2160, H3_RES))
    show("Elk Grove Village, IL (industrial)", h3.geo_to_h3(42.0084, -87.9773, H3_RES))

    print("\nTop-5 worker hexes:")
    for _, r in us.nlargest(5, "lodesod_workers").iterrows():
        lat, lon = h3.h3_to_geo(r["h3_index"])
        print(f"  {r['h3_index']}  {int(r['lodesod_workers']):>9,} workers  "
              f"income ${r['lodesod_home_income']:,.0f}  far {r['lodesod_pct_far']:.1f}%  "
              f"({lat:.4f}, {lon:.4f})")


if __name__ == "__main__":
    sys.exit(main())
