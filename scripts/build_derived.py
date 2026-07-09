"""
Cross-layer derived features for the Hex Atlas — rates with real units, not
scores. Each metric names its numerator, denominator, and caveat.

  pop_calibrated            census tract totals (ACS, true) allocated to hexes
                            by Kontur weights (spatial pattern, ~5x inflated in
                            absolute terms — used only as a within-tract share)
  drv_visits_per_resident   GPS visits / calibrated residents ("draw factor" —
                            destinations pull >> 1, dormitories ~ <1)
  drv_fatal_per_100k_aadt   annualized FARS fatal crashes per 100k daily
                            vehicles on the busiest covered road (proxy rate:
                            AADT-max, not per-VMT — we don't carry segment
                            length per hex)
  drv_jobs_per_resident     LODES workplace jobs / calibrated residents
                            (daytime pull)                    [when LODES lands]
  nri_eal_alloc             tract Expected Annual Loss $ allocated to hexes by
                            calibrated-population share       [when NRI lands]
  drv_eal_per_capita        allocated EAL $ / calibrated resident / year
                                                              [when NRI lands]

Also exports the US subset of worldmove O-D flows (inflow/outflow/diversity)
as data/worldmove_us_h3.parquet for the card's Flows section.

Writes: data/us_derived_h3.parquet  (re-run after new side layers land)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FE = (Path.home() / "Downloads/Final Semester/geofm-global/data/processed").as_posix()

R9_GLOB = str(DATA / "us_r9" / "*.parquet")
ACS = DATA / "acs_us_h3.parquet"
FARS = DATA / "fars_us_h3.parquet"
HPMS = DATA / "hpms_us_h3.parquet"
LODES = DATA / "lodes_us_h3.parquet"
NRI = DATA / "nri_us_h3.parquet"
GDELT = DATA / "gdelt_us_h3.parquet"
EAGLEI = DATA / "eaglei_us_h3.parquet"
NDVI = DATA / "ndvi_us_h3.parquet"
OUT = DATA / "us_derived_h3.parquet"
WM_OUT = DATA / "worldmove_us_h3.parquet"

# Guards: rates over tiny denominators are noise, not signal.
MIN_POP = 25        # residents needed before a per-resident rate is shown
MIN_AADT = 1000     # veh/day needed before a crash-per-traffic rate is shown
FARS_YEARS = 3      # 2022–2024 window -> annualize


def main() -> None:
    con = duckdb.connect()

    # base: every US res-9 hex with its kontur weight + movement counts
    df = con.execute(f"""
        SELECT h3_index, kontur_population, wt_visit_count
        FROM read_parquet('{R9_GLOB}')
    """).df()

    # --- calibrated population: ACS tract truth x kontur within-tract share ---
    acs = pd.read_parquet(ACS, columns=["h3_index", "tract_geoid", "acs_population"])
    df = df.merge(acs, on="h3_index", how="left")
    kon = df.kontur_population.fillna(0)
    tract_kon = kon.groupby(df.tract_geoid).transform("sum")
    share = np.where(tract_kon > 0, kon / tract_kon, np.nan)
    df["pop_calibrated"] = df.acs_population * share

    # --- visits per resident ---
    pop = df.pop_calibrated
    df["drv_visits_per_resident"] = np.where(
        pop >= MIN_POP, df.wt_visit_count.fillna(0) / pop, np.nan)

    # --- fatal crashes per 100k daily vehicles (annualized, proxy rate) ---
    if FARS.exists() and HPMS.exists():
        fars = pd.read_parquet(FARS, columns=["h3_index", "fars_crashes"])
        hpms = pd.read_parquet(HPMS, columns=["h3_index", "hpms_aadt_max"])
        df = df.merge(fars, on="h3_index", how="left").merge(hpms, on="h3_index", how="left")
        aadt = df.hpms_aadt_max
        df["drv_fatal_per_100k_aadt"] = np.where(
            aadt >= MIN_AADT,
            (df.fars_crashes.fillna(0) / FARS_YEARS) / aadt * 100_000,
            np.nan)
        df = df.drop(columns=["fars_crashes", "hpms_aadt_max"])

    # --- jobs per resident (daytime pull) ---
    if LODES.exists():
        lodes = pd.read_parquet(LODES, columns=["h3_index", "lodes_jobs"])
        df = df.merge(lodes, on="h3_index", how="left")
        df["drv_jobs_per_resident"] = np.where(
            pop >= MIN_POP, df.lodes_jobs.fillna(0) / pop, np.nan)
        df = df.drop(columns=["lodes_jobs"])

    # --- NRI dollars: allocate tract EAL to hexes by calibrated-pop share ---
    if NRI.exists():
        nri = pd.read_parquet(NRI, columns=["h3_index", "nri_eal_total"])
        df = df.merge(nri, on="h3_index", how="left")
        tract_pop = df.pop_calibrated.groupby(df.tract_geoid).transform("sum")
        pshare = np.where(tract_pop > 0, df.pop_calibrated / tract_pop, np.nan)
        df["nri_eal_alloc"] = df.nri_eal_total * pshare
        df["drv_eal_per_capita"] = np.where(
            pop >= MIN_POP, df.nri_eal_alloc / pop, np.nan)
        df = df.drop(columns=["nri_eal_total"])

    keep = ["h3_index", "pop_calibrated", "drv_visits_per_resident"] + [
        c for c in ("drv_fatal_per_100k_aadt", "drv_jobs_per_resident",
                    "nri_eal_alloc", "drv_eal_per_capita") if c in df.columns]
    out = df[keep]
    out.to_parquet(OUT, index=False)

    print(f"[derived] {len(out):,} hexes -> {OUT.name} ({OUT.stat().st_size/1e6:.0f} MB)")
    print(f"[derived] pop_calibrated national sum: {out.pop_calibrated.sum()/1e6:.1f} M "
          f"(ACS distinct-tract truth ~330.7 M)")
    for c in keep[2:]:
        v = out[c].dropna()
        print(f"[derived] {c:26s} n={len(v):>9,}  median={v.median():,.2f}  p95={v.quantile(.95):,.2f}")
    if "nri_eal_alloc" in out.columns:
        print(f"[derived] nri_eal_alloc national sum: ${out.nri_eal_alloc.sum()/1e9:.1f} B/yr")

    # --- res-5 rollups for the national overview (precomputed so the app
    #     never pays parent-computation at load; parent-map keeps hexes that
    #     sit outside the cells universe, e.g. rural interstates) ---
    import h3 as h3lib
    rolls: pd.DataFrame | None = None

    def parent_roll(path: Path, aggs: dict[str, tuple[str, str]]) -> pd.DataFrame:
        cols = ["h3_index"] + [src for src, _ in aggs.values()]
        t = pd.read_parquet(path, columns=cols)
        t["res5"] = t.h3_index.map(lambda h: h3lib.h3_to_parent(h, 5))
        out = t.groupby("res5").agg(**{
            dst: (src, how) for dst, (src, how) in aggs.items()}).reset_index()
        return out.rename(columns={"res5": "h3_index"})

    if FARS.exists():
        rolls = parent_roll(FARS, {"fars_crashes": ("fars_crashes", "sum"),
                                   "fars_killed": ("fars_killed", "sum")})
    if HPMS.exists():
        r = parent_roll(HPMS, {"hpms_aadt_max": ("hpms_aadt_max", "max")})
        rolls = r if rolls is None else rolls.merge(r, on="h3_index", how="outer")
    if LODES.exists():
        r = parent_roll(LODES, {"lodes_jobs": ("lodes_jobs", "sum")})
        rolls = r if rolls is None else rolls.merge(r, on="h3_index", how="outer")
    if GDELT.exists():
        r = parent_roll(GDELT, {"gdelt_events": ("gdelt_events", "sum")})
        rolls = r if rolls is None else rolls.merge(r, on="h3_index", how="outer")
    if NDVI.exists():
        r = con.execute(f"""
            SELECT b.res5 AS h3_index, avg(t.ndvi_summer) AS ndvi_summer
            FROM read_parquet('{NDVI.as_posix()}') t
            JOIN read_parquet('{R9_GLOB}') b USING (h3_index) GROUP BY b.res5""").df()
        rolls = r if rolls is None else rolls.merge(r, on="h3_index", how="outer")
    if EAGLEI.exists():
        r = con.execute(f"""
            SELECT b.res5 AS h3_index, max(t.eaglei_hrs_dark_per_cust_yr) AS eaglei_hrs_dark_per_cust_yr
            FROM read_parquet('{EAGLEI.as_posix()}') t
            JOIN read_parquet('{R9_GLOB}') b USING (h3_index) GROUP BY b.res5""").df()
        rolls = r if rolls is None else rolls.merge(r, on="h3_index", how="outer")
    if "nri_eal_alloc" in out.columns:
        e = con.execute(f"""
            SELECT b.res5 AS h3_index, sum(d.nri_eal_alloc) AS nri_eal_alloc
            FROM out d JOIN read_parquet('{R9_GLOB}') b USING (h3_index)
            GROUP BY b.res5""").df()
        rolls = e if rolls is None else rolls.merge(e, on="h3_index", how="outer")
    if rolls is not None:
        rp = DATA / "us_r5_rollups.parquet"
        rolls.to_parquet(rp, index=False)
        print(f"[rollups] {len(rolls):,} res-5 cells x {rolls.shape[1]-1} metrics -> {rp.name}")

    # --- worldmove O-D flows, US subset for the card ---
    wm_src = f"{FE}/features/worldmove_features.parquet"
    wm = con.execute(f"""
        SELECT w.* FROM read_parquet('{wm_src}') w
        JOIN read_parquet('{R9_GLOB}') b USING (h3_index)
    """).df()
    wm.to_parquet(WM_OUT, index=False)
    print(f"[flows] worldmove US subset: {len(wm):,} hexes -> {WM_OUT.name}")


if __name__ == "__main__":
    main()
