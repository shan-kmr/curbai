"""
Build the US-wide power-reliability layer for the Hex Atlas from ORNL EAGLE-I.

Source: "The Environment for Analysis of Geo-Located Energy Information's
Recorded Electricity Outages" (ORNL EAGLE-I historical release), figshare
mirror, article 24237376 (v4 covers 2014-2025):
  https://doi.org/10.6084/m9.figshare.24237376
  https://figshare.com/articles/dataset/_/24237376
File list is discovered live via the figshare API (FALLBACK_FILES if the API
is down). We use the 3 most recent year files (currently 2023, 2024, 2025)
plus MCC.csv (modeled county customers) for normalization.

Per-year CSVs are 15-minute county outage snapshots. Schema drifts by year:
  2023: fips_code,county,state,sum,run_start_time
  2024: fips_code,county,state,customers_out,run_start_time,total_customers
  2025: fips_code,county,state,customers_out,run_start_time
so the outage column is sniffed per file (customers_out > sum > max).

KEY ASSUMPTION (documented per task spec): EAGLE-I records a row only when a
county has a NONZERO outage at that 15-min snapshot; absent county-intervals
mean 0 customers out. Aggregates therefore treat missing intervals as zero:
  eaglei_cust_out_mean  = sum(customers_out) / total_15min_intervals_in_window
  eaglei_outage_hours_yr= nonzero intervals x 0.25h / years_in_window
where total_15min_intervals_in_window is the CALENDAR interval count between
the earliest and latest run_start_time seen across the 3 files (inclusive).
The script counts zero-valued rows and prints them to validate the assumption.
Duplicate county-timestamp rows (rare feed artifacts) are collapsed with MAX.

Normalization: MCC.csv ships County_FIPS (unpadded) + Customers -- ORNL's
modeled count of electricity customers per county (single 2022-era snapshot,
not per-year). eaglei_pct_out_mean = cust_out_mean / customers x 100.

Hex mapping: county FIPS = tract_geoid[:5] from data/acs_us_h3.parquet
(h3_index -> tract_geoid), plain key join, no spatial work. County values are
REPEATED on every hex in the county -- same pattern as the NRI tract fields.

Output data/eaglei_us_h3.parquet, one row per hex:
  h3_index               res-9 cell (from the ACS hex map)
  county_fips            5-digit county FIPS (null if hex has no tract)
  eaglei_cust_out_mean   mean customers out (missing intervals = 0)
  eaglei_pct_out_mean    eaglei_cust_out_mean / county customers x 100
  eaglei_hrs_dark_per_cust_yr  SAIDI-like HEADLINE: expected hours without
                         power per customer per year (pct_out_mean x 8766).
                         Size-normalized — use this, not outage_hours_yr.
  eaglei_outage_hours_yr est. hours/year with ANY customers out. Degenerate
                         for large counties (saturates at ~8766: somewhere in
                         Wayne/LA County is always dark) — kept for small-county
                         event-frequency reading only
  eaglei_max_out         max customers out in the window (worst event)
  eaglei_years           'YYYY-YYYY' window label

Usage:
  python scripts/build_eaglei.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eaglei"
ACS = ROOT / "data" / "acs_us_h3.parquet"
OUT = ROOT / "data" / "eaglei_us_h3.parquet"

N_YEARS = 3  # most recent year files to use
FIGSHARE_API = "https://api.figshare.com/v2/articles/24237376"
DOI = "10.6084/m9.figshare.24237376"

# Fallback if the figshare API is unreachable: name -> (download_url, bytes).
# Sizes are from API v4 (2026-07); cache check tolerates ±1% when using these.
FALLBACK_FILES = {
    "eaglei_outages_2023.csv": ("https://ndownloader.figshare.com/files/44574907", 1_199_800_000),
    "eaglei_outages_2024.csv": ("https://ndownloader.figshare.com/files/53581661", 1_444_800_000),
    "eaglei_outages_2025.csv": ("https://ndownloader.figshare.com/files/62164877", 1_402_300_000),
    "MCC.csv": ("https://ndownloader.figshare.com/files/42547708", 43_000),
}

# outage-count column, in order of preference (schema drifts across years)
OUTAGE_COLS = ["customers_out", "sum", "max"]


# ---------------------------------------------------------------- fetch --------
def figshare_files() -> tuple[dict[str, tuple[str, int]], bool]:
    """name -> (download_url, size_bytes) from the figshare API; exact=False on fallback."""
    try:
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", "60", FIGSHARE_API],
            capture_output=True, text=True, timeout=90, check=True,
        )
        meta = json.loads(r.stdout)
        files = {f["name"]: (f["download_url"], int(f["size"])) for f in meta["files"]}
        print(f"[fetch] figshare API: '{meta['title']}' doi={meta['doi']} ({len(files)} files)")
        return files, True
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
        print(f"[warn] figshare API failed ({e}); using hardcoded fallback URLs")
        return dict(FALLBACK_FILES), False


def pick_year_files(files: dict[str, tuple[str, int]]) -> list[tuple[int, str]]:
    """The N_YEARS most recent (year, name) outage files present in the article."""
    years = []
    for name in files:
        if name.startswith("eaglei_outages_") and name.endswith(".csv"):
            y = name[len("eaglei_outages_"):-len(".csv")]
            if y.isdigit():
                years.append((int(y), name))
    if len(years) < N_YEARS:
        raise SystemExit(f"only {len(years)} eaglei_outages_YYYY.csv files found: {years}")
    return sorted(years)[-N_YEARS:]


def download(name: str, url: str, size: int, exact: bool) -> Path:
    """Fetch one file into data/raw/eaglei (cached when the size checks out)."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / name
    if dest.exists():
        have = dest.stat().st_size
        ok = have == size if exact else abs(have - size) <= 0.01 * size
        if ok:
            print(f"[fetch] cached: {name} ({have:,} B)")
            return dest
        print(f"[fetch] {name}: cached size {have:,} != expected {size:,}, resuming")
    print(f"[fetch] downloading {name} from {url}")
    subprocess.run(
        ["curl", "-sSL", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True, timeout=7200,
    )
    print(f"[fetch] {name}: {dest.stat().st_size:,} B")
    return dest


def sniff_outage_col(path: Path) -> str:
    """Which outage-count column this year's file uses."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    cols = {c.strip().lower() for c in header}
    need = {"fips_code", "run_start_time"}
    if not need <= cols:
        raise SystemExit(f"{path.name}: missing {need - cols} (header: {header})")
    for c in OUTAGE_COLS:
        if c in cols:
            return c
    raise SystemExit(f"{path.name}: no outage column among {OUTAGE_COLS} (header: {header})")


# ---------------------------------------------------------------- aggregate ----
def aggregate(con: duckdb.DuckDBPyConnection, year_paths: list[tuple[int, Path]]) -> dict:
    """Stream the year CSVs through duckdb into per-county aggregates.

    Returns window stats. Materializes: raw union -> dedup (county,ts) -> agg.
    """
    selects = []
    for year, path in year_paths:
        col = sniff_outage_col(path)
        print(f"[load] {path.name}: outage column = '{col}'")
        selects.append(f"""
            SELECT lpad(CAST(fips_code AS VARCHAR), 5, '0') AS fips,
                   county, state,
                   CAST("{col}" AS DOUBLE) AS cust,
                   run_start_time AS ts
            FROM read_csv('{path}', header=true,
                          types={{'fips_code':'VARCHAR','run_start_time':'TIMESTAMP'}})
        """)
    union = "\nUNION ALL\n".join(selects)

    # collapse any duplicate county-timestamp rows (MAX: idempotent for true dups)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dedup AS
        SELECT fips, ts, max(cust) AS cust, count(*) AS n_src,
               any_value(county) AS county, any_value(state) AS state
        FROM ({union})
        WHERE fips IS NOT NULL AND ts IS NOT NULL AND cust IS NOT NULL
        GROUP BY fips, ts
    """)
    raw_rows, n_rows, n_zero, ts_min, ts_max = con.execute("""
        SELECT sum(n_src), count(*), count(*) FILTER (WHERE cust = 0), min(ts), max(ts)
        FROM dedup
    """).fetchone()

    # calendar 15-min intervals in [ts_min, ts_max], missing intervals count as 0
    total_intervals = int((ts_max - ts_min).total_seconds() // 900) + 1
    years_span = total_intervals * 900 / (86400 * 365.25)
    print(f"[agg ] rows: {raw_rows:,} raw -> {n_rows:,} after county-ts dedup "
          f"({raw_rows - n_rows:,} dup rows collapsed)")
    print(f"[agg ] zero-valued rows: {n_zero:,} of {n_rows:,} "
          f"({100 * n_zero / n_rows:.3f}%) -- dataset is (near-)nonzero-only, "
          f"missing intervals treated as 0")
    print(f"[agg ] window: {ts_min} .. {ts_max}  "
          f"= {total_intervals:,} calendar 15-min intervals ({years_span:.3f} yr)")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE county_agg AS
        SELECT fips,
               any_value(county) AS county,
               any_value(state)  AS state,
               sum(cust) / {total_intervals}                            AS cust_out_mean,
               count(*) FILTER (WHERE cust > 0) * 0.25 / {years_span}   AS outage_hours_yr,
               max(cust)                                                AS max_out
        FROM dedup
        GROUP BY fips
    """)
    n_counties = con.execute("SELECT count(*) FROM county_agg").fetchone()[0]
    print(f"[agg ] {n_counties:,} counties with >=1 outage row in window")
    return {"ts_min": ts_min, "ts_max": ts_max, "total_intervals": total_intervals,
            "years_span": years_span, "n_counties": n_counties}


def load_mcc(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """MCC.csv -> temp table mcc(fips, customers); returns county count."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mcc AS
        SELECT lpad(CAST(County_FIPS AS VARCHAR), 5, '0') AS fips,
               CAST(Customers AS DOUBLE) AS customers
        FROM read_csv('{path}', header=true, types={{'County_FIPS':'VARCHAR'}})
        WHERE try_cast(County_FIPS AS BIGINT) IS NOT NULL   -- drops 'Grand Total'
          AND try_cast(Customers AS DOUBLE) > 0
    """)
    n, total = con.execute("SELECT count(*), sum(customers) FROM mcc").fetchone()
    print(f"[mcc ] {n:,} counties, {total / 1e6:.1f}M modeled customers")
    return n


# ---------------------------------------------------------------- build --------
def build(con: duckdb.DuckDBPyConnection, years_label: str) -> None:
    """Join county aggregates (+MCC pct) onto the hex->tract map; write parquet."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE county_final AS
        SELECT c.fips, c.county, c.state,
               c.cust_out_mean                                    AS eaglei_cust_out_mean,
               CASE WHEN m.customers > 0
                    THEN c.cust_out_mean / m.customers * 100 END  AS eaglei_pct_out_mean,
               -- SAIDI-like headline: expected hours WITHOUT power per customer
               -- per year (mean share of customers out x 8766 h). Size-normalized,
               -- unlike outage_hours_yr which saturates for any big county
               -- (somewhere is always dark in Wayne/LA County).
               CASE WHEN m.customers > 0
                    THEN c.cust_out_mean / m.customers * 8766 END AS eaglei_hrs_dark_per_cust_yr,
               c.outage_hours_yr                                  AS eaglei_outage_hours_yr,
               c.max_out                                          AS eaglei_max_out
        FROM county_agg c LEFT JOIN mcc m USING (fips)
    """)
    con.execute(f"""
        COPY (
          SELECT a.h3_index,
                 substr(a.tract_geoid, 1, 5)     AS county_fips,
                 c.eaglei_cust_out_mean,
                 c.eaglei_pct_out_mean,
                 c.eaglei_hrs_dark_per_cust_yr,
                 c.eaglei_outage_hours_yr,
                 c.eaglei_max_out,
                 CASE WHEN c.fips IS NOT NULL
                      THEN '{years_label}' END   AS eaglei_years
          FROM read_parquet('{ACS}') a
          LEFT JOIN county_final c ON substr(a.tract_geoid, 1, 5) = c.fips
          ORDER BY a.h3_index
        ) TO '{OUT}' (FORMAT PARQUET)
    """)


# ---------------------------------------------------------------- verify -------
def county_row(con: duckdb.DuckDBPyConnection, fips: str):
    return con.execute("""
        SELECT county, state, eaglei_hrs_dark_per_cust_yr, eaglei_pct_out_mean,
               eaglei_cust_out_mean, eaglei_max_out
        FROM county_final WHERE fips = ?
    """, [fips]).fetchone()


def verify(con: duckdb.DuckDBPyConnection, url: str, stats: dict, years_label: str) -> None:
    total, matched, n_cty, n_pct = con.execute(f"""
        SELECT count(*), count(eaglei_outage_hours_yr),
               count(DISTINCT county_fips) FILTER (WHERE eaglei_outage_hours_yr IS NOT NULL),
               count(eaglei_pct_out_mean)
        FROM read_parquet('{OUT}')
    """).fetchone()

    print("\n[verify] ------------------------------------------------------------")
    print(f"[verify] source        : {url} (doi:{DOI})")
    print(f"[verify] window        : {years_label}  "
          f"({stats['ts_min']} .. {stats['ts_max']}, {stats['years_span']:.3f} yr)")
    print(f"[verify] hexes         : {total:,} total, {matched:,} with EAGLE-I "
          f"({100 * matched / total:.2f}%), {n_pct:,} with pct_out_mean; "
          f"{n_cty:,} counties on hexes")

    print(f"[verify] top-5 counties by hrs_dark_per_cust_yr (SAIDI-like; "
          f"expect hurricane/ice-storm country, min 10k customers):")
    for r in con.execute("""
        SELECT f.county, f.state, f.eaglei_hrs_dark_per_cust_yr, f.eaglei_pct_out_mean, f.eaglei_max_out
        FROM county_final f JOIN mcc m USING (fips)
        WHERE m.customers >= 10000
        ORDER BY f.eaglei_hrs_dark_per_cust_yr DESC LIMIT 5
    """).fetchall():
        pct = f"{r[3]:.2f}%" if r[3] is not None else "n/a"
        print(f"[verify]   {r[0]}, {r[1]:<15s} {r[2]:8.1f} h dark/cust/yr  "
              f"mean {pct} out  max {r[4]:,.0f}")

    ny = county_row(con, "36061")   # New York County, NY (reliable urban)
    tb = county_row(con, "22109")   # Terrebonne Parish, LA (hurricane country)
    for label, r in [("New York County NY (36061)", ny), ("Terrebonne Parish LA (22109)", tb)]:
        if r is None:
            print(f"[verify] {label}: NOT IN DATA")
            continue
        pct = f"{r[3]:.3f}%" if r[3] is not None else "n/a"
        print(f"[verify] {label}: {r[2]:.1f} h dark/cust/yr, mean {pct} out, "
              f"max {r[5]:,.0f}")
    ok = (ny is not None and tb is not None
          and ny[2] is not None and tb[2] is not None and tb[2] > 2 * ny[2])
    print(f"[verify] parish >> urban check (hrs dark/cust): "
          f"{'PASS' if ok else 'FAIL'} (Terrebonne vs New York County)")

    import pyarrow.parquet as pq
    schema = pq.ParquetFile(OUT).schema_arrow
    print("[verify] schema        : " + ", ".join(f"{f.name}:{f.type}" for f in schema))
    print(f"[verify] wrote {OUT.name}: {total:,} rows, {len(schema)} cols, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    if not ok:
        raise SystemExit("spot check FAILED -- inspect output before shipping")


# ---------------------------------------------------------------- main ---------
def main() -> None:
    if not ACS.exists():
        raise SystemExit(f"missing hex->tract map: {ACS} (run build_acs.py first)")

    files, exact = figshare_files()
    year_names = pick_year_files(files)
    years_label = f"{year_names[0][0]}-{year_names[-1][0]}"
    print(f"[plan] years: {[y for y, _ in year_names]} -> eaglei_years='{years_label}'")

    year_paths = [(y, download(n, *files[n], exact)) for y, n in year_names]
    if "MCC.csv" in files:
        mcc_path = download("MCC.csv", *files["MCC.csv"], exact)
    else:
        mcc_path = None
        print("[warn] MCC.csv absent from article -- eaglei_pct_out_mean will be null")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='20GB'")
    con.execute(f"SET temp_directory='{RAW / 'duckdb_tmp'}'")

    stats = aggregate(con, year_paths)
    if mcc_path is not None:
        load_mcc(con, mcc_path)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE mcc (fips VARCHAR, customers DOUBLE)")

    build(con, years_label)
    verify(con, "https://figshare.com/articles/dataset/_/24237376", stats, years_label)


if __name__ == "__main__":
    main()
