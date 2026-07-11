#!/usr/bin/env python
"""
Build the US geocoded-news-events layer for the Hex Atlas from GDELT 1.0.

Source: GDELT 1.0 daily event files (tab-separated, no header, 58 columns):
  http://data.gdeltproject.org/events/{YYYYMMDD}.export.CSV.zip
Column layout per the GDELT 1.0 EVENT codebook — 0-based indices used here:
  1  SQLDATE                28 EventRootCode         30 GoldsteinScale
  34 AvgTone                49 ActionGeo_Type        50 ActionGeo_FullName
  51 ActionGeo_CountryCode  52 ActionGeo_ADM1Code    53 ActionGeo_Lat
  54 ActionGeo_Long

Window: the most recent ~180 daily files available. We probe backward from
yesterday for the newest published file, then fetch that day and the 179
before it (4 parallel curl workers). Individual missing days are skipped and
counted, never fatal.

Filter: ActionGeo_CountryCode == 'US' AND ActionGeo_Type in (3, 4) — city- /
landmark-level geocoding only. Types 1-2 (country / state centroids) are
garbage at hex resolution and are dropped. Lat/lon must parse, be in range,
and not be (0, 0).

HONESTY NOTE — read before using this layer:
  GDELT geocodes events to CITY / LANDMARK CENTROIDS (GNIS/GNS feature
  points), not street addresses. At H3 res 9 (~0.1 km^2) every event coded to
  a city lands on the single hex containing that city's centroid. This layer
  is therefore a news-attention signal per PLACE (city / landmark), NOT a
  per-street signal. Join and label it accordingly (place-level covariate,
  or spread over the city polygon); never read it as "news happened on this
  exact block".

Aggregation per res-9 hex over the window — counts with real units, one
explainable tone mean, no text blobs, no composite scores:
  gdelt_events     total geocoded events (all CAMEO root codes)
  gdelt_protest    events with EventRootCode '14' (protest)
  gdelt_conflict   EventRootCode in ('18','19','20') (assault / fight /
                   unconventional mass violence)
  gdelt_coerce     EventRootCode '17' (coercion: seizures, curfews, ...)
  gdelt_aid        EventRootCode '07' (provide aid)
  gdelt_tone_mean  mean AvgTone across the hex's events; negative = negative
                   coverage (typical range roughly -10..+10)
  gdelt_days       distinct days with >= 1 event, out of the window
  gdelt_window     'YYYY-MM-DD..YYYY-MM-DD' fetch window (same on every row)

Day identity for gdelt_days is the FILE date (the day GDELT ingested the
article = publication day), which is the news-attention clock and is bounded
by the window. SQLDATE (the coded event date) carries a historical-reference
tail that would let day counts exceed the window, so it is not used for this.

Output: data/gdelt_us_h3.parquet keyed h3_index (string); counts int32.
Cache:  data/raw/gdelt/{YYYYMMDD}.export.CSV.zip (gitignored via data/raw/).

Usage:
  python scripts/build_gdelt.py                     # 180 newest available days
  python scripts/build_gdelt.py --days 3 --out /tmp/smoke.parquet
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd

RES = 9
WINDOW_DAYS = 180
WORKERS = 4
PROBE_BACK = 30
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gdelt"
OUT = ROOT / "data" / "gdelt_us_h3.parquet"
URL_TMPL = "http://data.gdeltproject.org/events/{d}.export.CSV.zip"

# 0-based column indices in the 58-column GDELT 1.0 daily export.
USECOLS = [1, 28, 34, 49, 50, 51, 53, 54]
COLNAMES = {
    1: "sqldate",   # SQLDATE (loaded for layout sanity; day identity = file date)
    28: "root",     # EventRootCode (two-char string, e.g. '07', '14')
    34: "tone",     # AvgTone
    49: "gtype",    # ActionGeo_Type: 3=US city, 4=world city/landmark
    50: "gname",    # ActionGeo_FullName (verification printing only — never written)
    51: "gcc",      # ActionGeo_CountryCode (FIPS 10-4; 'US')
    53: "lat",      # ActionGeo_Lat
    54: "lon",      # ActionGeo_Long
}

COUNT_COLS = ["gdelt_events", "gdelt_protest", "gdelt_conflict",
              "gdelt_coerce", "gdelt_aid", "gdelt_days"]


# ---------------------------------------------------------------- fetch --------
def day_url(d: date) -> str:
    return URL_TMPL.format(d=d.strftime("%Y%m%d"))


def day_path(d: date) -> Path:
    return RAW / f"{d.strftime('%Y%m%d')}.export.CSV.zip"


def probe_newest(start: date, back: int = PROBE_BACK) -> date:
    """Newest date whose daily file exists, probing backward from `start`."""
    for i in range(back):
        d = start - timedelta(days=i)
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "15", "-I", day_url(d)],
            capture_output=True, text=True)
        if r.stdout.strip() == "200":
            return d
    sys.exit(f"FATAL: no GDELT daily file found in the last {back} days — "
             "is data.gdeltproject.org reachable?")


def _download_one(d: date) -> tuple[date, bool]:
    dest = day_path(d)
    if dest.exists():
        if dest.stat().st_size > 1024 and dest.open("rb").read(2) == b"PK":
            return d, True          # cached
        dest.unlink()               # corrupt stub — refetch
    tmp = dest.with_suffix(".part")
    r = subprocess.run(
        ["curl", "-fsS", "--connect-timeout", "15", "--max-time", "300",
         "--retry", "2", "--retry-delay", "3", "-o", str(tmp), day_url(d)],
        capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.open("rb").read(2) != b"PK":
        tmp.unlink(missing_ok=True)
        return d, False             # 404 (day never published) or bad payload
    os.replace(tmp, dest)
    return d, True


def download(dates: list[date]) -> tuple[list[date], list[date]]:
    ok, missing = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (d, got) in enumerate(ex.map(_download_one, dates), 1):
            (ok if got else missing).append(d)
            if i % 30 == 0 or i == len(dates):
                print(f"  fetch {i}/{len(dates)} (missing so far: {len(missing)})",
                      flush=True)
    return sorted(ok), sorted(missing)


# ---------------------------------------------------------------- parse --------
def parse_day(args: tuple[str, str]) -> dict:
    """One daily zip -> per-hex partial aggregate. Runs in a worker process."""
    path_str, day_iso = args
    try:
        df = pd.read_csv(
            path_str, sep="\t", header=None, usecols=USECOLS, dtype=str,
            quoting=csv.QUOTE_NONE, na_filter=False, on_bad_lines="skip",
            encoding="latin-1", compression="zip")
    except Exception as e:  # truncated zip / layout drift — skip the day
        return {"day": day_iso, "error": f"{type(e).__name__}: {e}"}
    df = df.rename(columns=COLNAMES)
    raw_rows = len(df)

    # City/landmark-level US events only.
    df = df[(df["gcc"] == "US") & df["gtype"].isin(("3", "4"))]

    # Coordinates: parse once per unique (lat, lon) string pair — GDELT reuses
    # the same centroid coords thousands of times per file.
    pairs = df[["lat", "lon"]].drop_duplicates().copy()
    la = pd.to_numeric(pairs["lat"], errors="coerce")
    lo = pd.to_numeric(pairs["lon"], errors="coerce")
    valid = (la.between(-90, 90) & lo.between(-180, 180)
             & ~((la == 0.0) & (lo == 0.0)))          # between() is False for NaN
    pairs = pairs.loc[valid]
    pairs["h3_index"] = [h3.geo_to_h3(a, b, RES)
                         for a, b in zip(la[valid].to_numpy(), lo[valid].to_numpy())]
    df = df.merge(pairs, on=["lat", "lon"], how="inner")

    root = df["root"]
    part = pd.DataFrame({
        "h3_index": df["h3_index"],
        "protest":  (root == "14"),
        "conflict": root.isin(("18", "19", "20")),
        "coerce":   (root == "17"),
        "aid":      (root == "07"),
        "tone":     pd.to_numeric(df["tone"], errors="coerce"),
    })
    g = part.groupby("h3_index", sort=False)
    out = g.agg(events=("protest", "size"),
                protest=("protest", "sum"),
                conflict=("conflict", "sum"),
                coerce=("coerce", "sum"),
                aid=("aid", "sum"),
                tone_sum=("tone", "sum"),
                tone_n=("tone", "count")).reset_index()
    out["day"] = day_iso
    names = df.drop_duplicates("h3_index").set_index("h3_index")["gname"].to_dict()
    return {"day": day_iso, "part": out, "names": names,
            "raw": raw_rows, "kept": len(df)}


# ---------------------------------------------------------------- verify -------
def _haversine_km(lat, lon, lats, lons):
    p1, p2 = np.radians(lat), np.radians(lats)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _pile_hex(final: pd.DataFrame, coords: np.ndarray, lat: float, lon: float,
              km: float = 5.0):
    """Highest-events hex within `km` of a point (city pile-up hex)."""
    dist = _haversine_km(lat, lon, coords[:, 0], coords[:, 1])
    near = final.loc[dist <= km]
    return None if near.empty else near.loc[near["gdelt_events"].idxmax()]


def verify(final: pd.DataFrame, names: dict, window: str,
           n_fetched: int, n_missing: int) -> None:
    coords = np.array([h3.h3_to_geo(hh) for hh in final["h3_index"]])
    total = int(final["gdelt_events"].sum())
    print("\n================ VERIFY ================")
    print(f"window            : {window}")
    print(f"days fetched      : {n_fetched}   missing: {n_missing}")
    print(f"hexes             : {len(final):,}")
    print(f"total events      : {total:,}  (~{total / max(n_fetched, 1):,.0f}/day)")
    print(f"gdelt_days max    : {int(final['gdelt_days'].max())} (must be <= {n_fetched})")
    print(f"tone_mean range   : {final['gdelt_tone_mean'].min():.2f} .. "
          f"{final['gdelt_tone_mean'].max():.2f}")

    print("\ntop-10 hexes by events (h3_to_geo -> should be big-city centroids):")
    top = final.head(10)
    for i, (_, r) in enumerate(top.iterrows()):
        la, lo = h3.h3_to_geo(r["h3_index"])
        nm = names.get(r["h3_index"], "?")
        print(f"  {i + 1:>2}. {r['h3_index']}  ({la:8.4f},{lo:10.4f})  "
              f"events={int(r['gdelt_events']):>7,}  protest={int(r['gdelt_protest']):>6,}  "
              f"tone={r['gdelt_tone_mean']:6.2f}  days={int(r['gdelt_days'])}  | {nm}")

    nat_share = final["gdelt_protest"].sum() / total
    print(f"\nprotest share, national: {nat_share:.3%}")
    for label, la, lo in [("Washington DC (38.9072,-77.0369)", 38.9072, -77.0369),
                          ("Naperville IL suburb (41.7508,-88.1535)", 41.7508, -88.1535),
                          ("Plano TX suburb (33.0198,-96.6989)", 33.0198, -96.6989)]:
        r = _pile_hex(final, coords, la, lo)
        if r is None:
            print(f"  {label}: no hex within 5 km")
            continue
        share = r["gdelt_protest"] / r["gdelt_events"]
        print(f"  {label}: hex {r['h3_index']}  events={int(r['gdelt_events']):,}  "
              f"protest={int(r['gdelt_protest']):,}  share={share:.3%}  "
              f"| {names.get(r['h3_index'], '?')}")
    print("========================================")


# ---------------------------------------------------------------- main ---------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--days", type=int, default=WINDOW_DAYS,
                    help=f"window length in days (default {WINDOW_DAYS})")
    ap.add_argument("--out", type=Path, default=OUT,
                    help=f"output parquet (default {OUT})")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    newest = probe_newest(date.today() - timedelta(days=1))
    dates = [newest - timedelta(days=i) for i in range(args.days - 1, -1, -1)]
    window = f"{dates[0].isoformat()}..{dates[-1].isoformat()}"
    print(f"newest available file: {newest}  ->  window {window} ({len(dates)} days)")

    print(f"downloading to {RAW} ({WORKERS} workers) ...", flush=True)
    ok_dates, missing = download(dates)
    print(f"fetched {len(ok_dates)} days; missing {len(missing)}"
          + (f" -> {[d.isoformat() for d in missing]}" if missing else ""))

    jobs = [(str(day_path(d)), d.isoformat()) for d in ok_dates]
    parts, names, failed = [], {}, []
    raw_total = kept_total = 0
    print(f"parsing {len(jobs)} files ({WORKERS} workers) ...", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(parse_day, jobs, chunksize=4), 1):
            if "error" in res:
                failed.append(res["day"])
                print(f"  PARSE FAIL {res['day']}: {res['error']}", flush=True)
            else:
                parts.append(res["part"])
                raw_total += res["raw"]
                kept_total += res["kept"]
                for k, v in res["names"].items():
                    names.setdefault(k, v)
            if i % 30 == 0 or i == len(jobs):
                print(f"  parse {i}/{len(jobs)}", flush=True)
    if failed:
        missing = sorted(missing + [date.fromisoformat(d) for d in failed])
    if not parts:
        sys.exit("FATAL: no days parsed")
    n_fetched = len(parts)
    print(f"rows: {raw_total:,} global -> {kept_total:,} US city/landmark-level "
          f"({kept_total / max(raw_total, 1):.1%})")

    allp = pd.concat(parts, ignore_index=True)
    final = (allp.groupby("h3_index", sort=False)
             .agg(gdelt_events=("events", "sum"),
                  gdelt_protest=("protest", "sum"),
                  gdelt_conflict=("conflict", "sum"),
                  gdelt_coerce=("coerce", "sum"),
                  gdelt_aid=("aid", "sum"),
                  tone_sum=("tone_sum", "sum"),
                  tone_n=("tone_n", "sum"),
                  gdelt_days=("day", "nunique"))
             .reset_index())
    final["gdelt_tone_mean"] = np.where(final["tone_n"] > 0,
                                        final["tone_sum"] / final["tone_n"], np.nan)
    for c in COUNT_COLS:
        final[c] = final[c].astype(np.int32)
    final["gdelt_window"] = window
    final = (final[["h3_index", "gdelt_events", "gdelt_protest", "gdelt_conflict",
                    "gdelt_coerce", "gdelt_aid", "gdelt_tone_mean", "gdelt_days",
                    "gdelt_window"]]
             .sort_values(["gdelt_events", "h3_index"], ascending=[False, True])
             .reset_index(drop=True))

    final.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  ({len(final):,} hexes, "
          f"{args.out.stat().st_size / 1e6:.1f} MB)")
    print(final.dtypes.to_string())

    verify(final, names, window, n_fetched, len(missing))


if __name__ == "__main__":
    main()
