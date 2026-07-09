"""
Build the US-wide fatal-crash layer for the Hex Atlas from NHTSA FARS.

Source: NHTSA FARS annual national CSV files
  https://static.nhtsa.gov/nhtsa/downloads/FARS/{YEAR}/National/FARS{YEAR}NationalCSV.zip
Each zip carries accident.csv (one row per fatal crash / ST_CASE) with, among
others: ST_CASE, LATITUDE, LONGITUD, FATALS, YEAR, MONTH, DAY_WEEK, HOUR.

We fetch the N most recent *available* years (auto-discovered by probing the
CDN; FARS lags the calendar by ~1.5y so the newest year is resolved at run
time), keep rows with real coordinates, bucket each crash into an Uber H3
res-9 cell, and aggregate per cell:

  fars_crashes         count of fatal crashes
  fars_killed          sum(FATALS)
  fars_peak_hour       modal HOUR 0..23  (99=unknown dropped; -1 if none)
  fars_peak_dow        modal DAY_WEEK 1..7  (FARS: 1=Sun..7=Sat; -1 if none)
  fars_peak_dow_label  weekday abbrev for the code above (Sun..Sat)
  fars_hour_hist       JSON list[24] of counts by hour
  fars_years_covered   distinct YEARs contributing to the cell
  fars_year_min/max    span of contributing years
  center_lat/lon       h3_to_geo of the cell (for rendering / joins)

No scores — raw counts only. Writes data/fars_us_h3.parquet keyed by h3_index.

Usage:
  python scripts/build_fars.py                 # 3 newest available years
  python scripts/build_fars.py 2022 2021 2020  # explicit years (adjacent-year
                                                 # fallback if one 404s)
  python scripts/build_fars.py --years 4       # N newest available years
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import h3
import numpy as np
import pandas as pd

RES = 9
N_YEARS = 3
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "fars_us_h3.parquet"
URL_TMPL = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{y}/National/FARS{y}NationalCSV.zip"

# FARS DAY_WEEK coding is 1=Sunday .. 7=Saturday (9=unknown, dropped).
DOW_LABEL = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}


# ---------------------------------------------------------------- fetch --------
def year_url(year: int) -> str:
    return URL_TMPL.format(y=year)


def head_ok(url: str) -> bool:
    """True iff the CDN serves a real zip (HTTP 200 + zip content-type)."""
    try:
        r = subprocess.run(
            ["curl", "-sIL", "-o", "/dev/null", "-w", "%{http_code} %{content_type}", url],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    out = r.stdout.strip().lower()
    return out.startswith("200") and "zip" in out


def resolve_recent(n: int, start: int) -> list[int]:
    """The n newest available years, probing downward from `start`."""
    years: list[int] = []
    y = start
    while len(years) < n and y >= 2010:
        if head_ok(year_url(y)):
            years.append(y)
        y -= 1
    return years


def resolve_one(year: int) -> int | None:
    """Nearest available year to `year` (exact, then +/-1, +/-2)."""
    for cand in (year, year - 1, year + 1, year - 2, year + 2):
        if head_ok(year_url(cand)):
            return cand
    return None


def download(year: int) -> Path:
    """Fetch FARS{year}NationalCSV.zip into data/raw (cached if already sound)."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"FARS{year}NationalCSV.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[fetch] {year}: cached ({dest.stat().st_size:,} B)")
        return dest
    url = year_url(year)
    print(f"[fetch] {year}: downloading {url}")
    subprocess.run(
        ["curl", "-sSL", "--fail", "--retry", "3", "-o", str(dest), url],
        check=True, timeout=600,
    )
    print(f"[fetch] {year}: {dest.stat().st_size:,} B")
    return dest


# ---------------------------------------------------------------- load ---------
def load_accident(zip_path: Path, year: int) -> pd.DataFrame:
    """Extract accident.csv, keep valid-coordinate rows, tag each with its H3 cell."""
    with zipfile.ZipFile(zip_path) as z:
        member = next(
            n for n in z.namelist() if n.lower().rsplit("/", 1)[-1] == "accident.csv"
        )
        raw = z.read(member)

    df = None
    for enc in ("utf-8", "latin-1"):  # some annual files ship latin-1
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise RuntimeError(f"could not decode accident.csv in {zip_path.name}")

    df.columns = [c.upper() for c in df.columns]

    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUD"], errors="coerce")
    # Real US extent. This range alone rejects every FARS sentinel:
    #   lat 77.7777 / 88.8888 / 99.9999  (all > 72)
    #   lon 777.7777 / 888.8888 / 999.9999 and any positive lon (all > -60)
    keep = lat.between(17, 72) & lon.between(-180, -60)

    out = pd.DataFrame({
        "lat": lat[keep].to_numpy(),
        "lon": lon[keep].to_numpy(),
        "FATALS": pd.to_numeric(df.loc[keep, "FATALS"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "HOUR": pd.to_numeric(df.loc[keep, "HOUR"], errors="coerce").fillna(99).astype(int).to_numpy(),
        "DAY_WEEK": pd.to_numeric(df.loc[keep, "DAY_WEEK"], errors="coerce").fillna(9).astype(int).to_numpy(),
        "YEAR": year,  # authoritative from the source year, not the row
    })
    out["h3_index"] = [h3.geo_to_h3(la, lo, RES) for la, lo in zip(out.lat, out.lon)]
    print(f"[load] {year}: {len(df):,} rows -> {len(out):,} valid-coord crashes "
          f"({len(df) - len(out):,} dropped)")
    return out


# ---------------------------------------------------------------- aggregate ----
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("h3_index", sort=True)
    agg = g.agg(
        fars_crashes=("FATALS", "size"),
        fars_killed=("FATALS", "sum"),
        fars_years_covered=("YEAR", "nunique"),
        fars_year_min=("YEAR", "min"),
        fars_year_max=("YEAR", "max"),
    )

    # Hour histogram over valid hours only (drop 99=unknown).
    hv = df[df.HOUR.between(0, 23)]
    hour_ct = (pd.crosstab(hv.h3_index, hv.HOUR)
                 .reindex(index=agg.index, columns=range(24), fill_value=0))
    hour_arr = hour_ct.to_numpy()
    hour_sum = hour_arr.sum(axis=1)
    agg["fars_peak_hour"] = np.where(hour_sum > 0, hour_arr.argmax(axis=1), -1).astype(int)
    agg["fars_hour_hist"] = [json.dumps(r.tolist()) for r in hour_arr]

    # Day-of-week mode over valid codes 1..7 (drop 9=unknown).
    dv = df[df.DAY_WEEK.between(1, 7)]
    dow_ct = (pd.crosstab(dv.h3_index, dv.DAY_WEEK)
                .reindex(index=agg.index, columns=range(1, 8), fill_value=0))
    dow_arr = dow_ct.to_numpy()
    dow_sum = dow_arr.sum(axis=1)
    peak_dow = np.where(dow_sum > 0, dow_arr.argmax(axis=1) + 1, -1).astype(int)
    agg["fars_peak_dow"] = peak_dow
    agg["fars_peak_dow_label"] = [DOW_LABEL.get(int(c), "") for c in peak_dow]

    centers = [h3.h3_to_geo(h) for h in agg.index]
    agg["center_lat"] = [c[0] for c in centers]
    agg["center_lon"] = [c[1] for c in centers]

    agg = agg.reset_index()
    agg["h3_index"] = agg["h3_index"].astype("string")
    for c in ("fars_crashes", "fars_killed", "fars_years_covered",
              "fars_year_min", "fars_year_max"):
        agg[c] = agg[c].astype(int)

    return agg[[
        "h3_index", "fars_crashes", "fars_killed",
        "fars_peak_hour", "fars_peak_dow", "fars_peak_dow_label",
        "fars_hour_hist", "fars_years_covered", "fars_year_min", "fars_year_max",
        "center_lat", "center_lon",
    ]]


# ---------------------------------------------------------------- main ---------
def choose_years() -> list[int]:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--years" in sys.argv:
        n = int(sys.argv[sys.argv.index("--years") + 1])
        return resolve_recent(n, date.today().year)
    if args:  # explicit years, each snapped to the nearest available
        years: list[int] = []
        for a in args:
            got = resolve_one(int(a))
            if got is None:
                print(f"[warn] no FARS zip near {a}; skipping")
            elif got != int(a):
                print(f"[warn] {a} unavailable; using adjacent {got}")
                years.append(got)
            else:
                years.append(got)
        return sorted(set(years), reverse=True)
    return resolve_recent(N_YEARS, date.today().year)


def main() -> None:
    years = choose_years()
    if not years:
        raise SystemExit("no FARS years resolved")
    print(f"[plan] years = {years}")

    frames = [load_accident(download(y), y) for y in years]
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] total valid-coord crashes: {len(df):,}")

    agg = aggregate(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)

    # ---- verify ---------------------------------------------------------------
    worst = agg.loc[agg.fars_crashes.idxmax()]
    wlat, wlon = h3.h3_to_geo(worst.h3_index)
    mb = OUT.stat().st_size / 1e6
    print("\n[verify] ------------------------------------------------------------")
    print(f"[verify] years covered : {min(years)}-{max(years)}  ({len(years)} years)")
    print(f"[verify] hexes         : {len(agg):,}")
    print(f"[verify] total crashes : {int(agg.fars_crashes.sum()):,}")
    print(f"[verify] total killed  : {int(agg.fars_killed.sum()):,}")
    print(f"[verify] worst hex     : {worst.h3_index}  "
          f"crashes={int(worst.fars_crashes)}  killed={int(worst.fars_killed)}")
    print(f"[verify]   -> h3_to_geo : ({wlat:.5f}, {wlon:.5f})")
    print(f"[verify] columns       : {list(agg.columns)}")
    print(f"[verify] wrote {OUT.name}: {len(agg):,} rows, {agg.shape[1]} cols, {mb:.1f} MB")


if __name__ == "__main__":
    main()
