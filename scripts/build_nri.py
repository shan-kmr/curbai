"""
Build the US-wide FEMA National Risk Index layer for the Hex Atlas.

Source: FEMA National Risk Index (NRI), census-tract bulk table
  https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
  current zip (v1.20, Dec 2025):
  https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_CensusTracts.zip
The legacy hazards.fema.gov/nri/Content/StaticDocuments/... path now 301s to
a generic fema.gov page, so resolve_url() probes known candidates and, if
they all fail, scrapes the OpenFEMA data-set page for the current link.

NRI values are per 2020-vintage census tract. data/acs_us_h3.parquet already
maps every US res-9 hex to its tract (h3_index -> tract_geoid), so this is a
plain key join -- no spatial work. IMPORTANT: tract-level values are REPEATED
on every hex in the tract; a downstream job allocates to hexes (do NOT divide
by hex count here).

Per hex:
  h3_index          res-9 cell (from the ACS hex map)
  tract_geoid       11-digit tract FIPS (join key, == NRI TRACTFIPS)
  nri_risk_score    RISK_SCORE  composite national percentile 0..100
  nri_risk_rating   RISK_RATNG  "Very Low" .. "Very High" (string)
  nri_eal_total     EAL_VALT    expected annual loss, all consequences ($/yr)
  nri_eal_building  EAL_VALB    expected annual loss, buildings ($/yr)
  nri_sovi          SOVI_SCORE  social vulnerability score
  nri_resilience    RESL_SCORE  community resilience score
  nri_top_hazards   JSON top-3 hazards by per-hazard total EAL, e.g.
                    [["Hurricane", 1234567.0], ["Inland Flooding", ...]]
                    zero/NaN hazards excluded; "[]" if nothing positive.
                    Names come from the zip's NRI_HazardInfo.csv; note v1.20
                    renamed "Riverine Flooding" (RFLD) -> "Inland Flooding"
                    (IFLD).
  nri_tract_pop     POPULATION  tract population (for later per-capita math)

Usage:
  python scripts/build_nri.py
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd

RES = 9
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ACS = ROOT / "data" / "acs_us_h3.parquet"
OUT = ROOT / "data" / "nri_us_h3.parquet"

CANDIDATE_URLS = [
    # current OpenFEMA location (version dir bumps on re-release)
    "https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_CensusTracts.zip",
    # legacy location (301s to a generic page as of mid-2026, kept as a probe)
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/"
    "NRI_Table_CensusTracts/NRI_Table_CensusTracts.zip",
]
DATASET_PAGE = "https://www.fema.gov/about/openfema/data-sets/national-risk-index-data"

# NRI hazard-type prefixes -> human-readable names. Fallback only: the zip
# ships NRI_HazardInfo.csv with the authoritative Prefix->Hazard map for its
# version, and load_nri() prefers that. Covers both the pre-v1.20 RFLD
# "Riverine Flooding" and its v1.20 (Dec 2025) replacement IFLD
# "Inland Flooding".
HAZARDS_FALLBACK = {
    "AVLN": "Avalanche",
    "CFLD": "Coastal Flooding",
    "CWAV": "Cold Wave",
    "DRGT": "Drought",
    "ERQK": "Earthquake",
    "HAIL": "Hail",
    "HWAV": "Heat Wave",
    "HRCN": "Hurricane",
    "IFLD": "Inland Flooding",
    "ISTM": "Ice Storm",
    "LNDS": "Landslide",
    "LTNG": "Lightning",
    "RFLD": "Riverine Flooding",
    "SWND": "Strong Wind",
    "TRND": "Tornado",
    "TSUN": "Tsunami",
    "VLCN": "Volcanic Activity",
    "WFIR": "Wildfire",
    "WNTW": "Winter Weather",
}

# NRI column -> output column (core scalar fields).
CORE = {
    "TRACTFIPS": "tract_geoid",
    "RISK_SCORE": "nri_risk_score",
    "RISK_RATNG": "nri_risk_rating",
    "EAL_VALT": "nri_eal_total",
    "EAL_VALB": "nri_eal_building",
    "SOVI_SCORE": "nri_sovi",
    "RESL_SCORE": "nri_resilience",
    "POPULATION": "nri_tract_pop",  # optional -- skipped with a warning if absent
}


# ---------------------------------------------------------------- fetch --------
def head_ok(url: str) -> bool:
    """True iff the URL serves a real zip (HTTP 200 + zip content-type)."""
    try:
        r = subprocess.run(
            ["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code} %{content_type}", url],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    out = r.stdout.strip().lower()
    return out.startswith("200") and "zip" in out


def resolve_url() -> str:
    """Working NRI_Table_CensusTracts.zip URL: known candidates, then page scrape."""
    for url in CANDIDATE_URLS:
        if head_ok(url):
            print(f"[fetch] url ok: {url}")
            return url
        print(f"[fetch] url dead: {url}")
    print(f"[fetch] scraping {DATASET_PAGE}")
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "120", DATASET_PAGE],
        capture_output=True, text=True, timeout=180, check=True,
    )
    hits = re.findall(r'https?://[^"\'\s]*NRI_Table_CensusTracts\.zip', r.stdout)
    for url in dict.fromkeys(hits):
        if head_ok(url):
            print(f"[fetch] url ok (scraped): {url}")
            return url
    raise SystemExit("no working NRI_Table_CensusTracts.zip URL found")


def download() -> tuple[Path, str]:
    """Fetch the tract-table zip into data/raw (cached if already sound)."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "NRI_Table_CensusTracts.zip"
    if dest.exists() and dest.stat().st_size > 50_000_000 and zipfile.is_zipfile(dest):
        print(f"[fetch] cached: {dest.name} ({dest.stat().st_size:,} B)")
        return dest, "(cached) " + CANDIDATE_URLS[0]
    url = resolve_url()
    print(f"[fetch] downloading {url}")
    subprocess.run(
        ["curl", "-sSL", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True, timeout=3600,
    )
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a valid zip")
    print(f"[fetch] {dest.name}: {dest.stat().st_size:,} B")
    return dest, url


# ---------------------------------------------------------------- load ---------
def pick_member(z: zipfile.ZipFile) -> str:
    """The tract-table CSV inside the zip (skip data dictionaries etc.)."""
    csvs = [i for i in z.infolist() if i.filename.lower().endswith(".csv")]
    if not csvs:
        raise SystemExit("no CSV member in NRI zip")
    named = [i for i in csvs
             if "censustracts" in i.filename.lower().replace("_", "")
             and "dictionary" not in i.filename.lower()]
    pool = named or csvs
    return max(pool, key=lambda i: i.file_size).filename


def hazard_names(z: zipfile.ZipFile) -> dict[str, str]:
    """Prefix -> hazard name, preferring the zip's own NRI_HazardInfo.csv."""
    names = dict(HAZARDS_FALLBACK)
    info = next((n for n in z.namelist()
                 if n.lower().rsplit("/", 1)[-1] == "nri_hazardinfo.csv"), None)
    if info:
        hi = pd.read_csv(io.BytesIO(z.read(info)), encoding="utf-8-sig")
        hi.columns = [c.strip().upper() for c in hi.columns]
        if {"PREFIX", "HAZARD"} <= set(hi.columns):
            names.update(dict(zip(hi["PREFIX"].str.strip(), hi["HAZARD"].str.strip())))
            print(f"[load] hazard names from {info}: {len(hi)} hazards")
    return names


def load_nri(zip_path: Path) -> pd.DataFrame:
    """Read the tract table, keep/rename the atlas fields, build top-hazards JSON."""
    with zipfile.ZipFile(zip_path) as z:
        HAZARDS = hazard_names(z)
        member = pick_member(z)
        print(f"[load] csv member: {member}")
        with z.open(member) as f:
            header = io.TextIOWrapper(f, encoding="utf-8-sig").readline()
        cols = [c.strip().strip('"') for c in header.rstrip("\r\n").split(",")]
        by_upper = {c.upper(): c for c in cols}

        missing_core = [c for c in CORE if c not in by_upper and c != "POPULATION"]
        if missing_core:
            raise SystemExit(f"NRI header missing expected columns: {missing_core}")
        if "POPULATION" not in by_upper:
            print("[warn] POPULATION column absent -- nri_tract_pop will be skipped")

        haz_cols = {code: by_upper[f"{code}_EALT"]
                    for code in HAZARDS if f"{code}_EALT" in by_upper}
        absent = sorted(set(HAZARDS) - set(haz_cols))
        if absent:
            print(f"[warn] hazard EALT columns absent, skipped: {absent}")

        usecols = [by_upper[c] for c in CORE if c in by_upper] + list(haz_cols.values())
        df = pd.read_csv(
            z.open(member), usecols=usecols, encoding="utf-8-sig",
            dtype={by_upper["TRACTFIPS"]: "string"}, low_memory=False,
        )
    df = df.rename(columns={v: k for k, v in by_upper.items()})  # -> canonical UPPER
    print(f"[load] {len(df):,} tracts x {df.shape[1]} cols")

    out = pd.DataFrame({"tract_geoid": df["TRACTFIPS"].str.strip().str.zfill(11)})
    for src, dst in CORE.items():
        if src in ("TRACTFIPS",) or src not in df.columns:
            continue
        out[dst] = (df[src].astype("string") if dst == "nri_risk_rating"
                    else pd.to_numeric(df[src], errors="coerce"))

    # top-3 hazards by per-hazard total EAL (exclude zero/NaN), as JSON
    codes = list(haz_cols)
    vals = (df[[f"{c}_EALT" for c in codes]]
            .apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
    masked = np.where(np.isfinite(vals) & (vals > 0), vals, -np.inf)
    top_idx = np.argsort(-masked, axis=1)[:, :3]
    names = [HAZARDS[c] for c in codes]
    out["nri_top_hazards"] = [
        json.dumps([[names[j], round(float(masked[i, j]), 2)]
                    for j in top_idx[i] if masked[i, j] > 0])
        for i in range(len(df))
    ]

    dups = out["tract_geoid"].duplicated().sum()
    if dups:
        print(f"[warn] {dups} duplicate TRACTFIPS rows dropped (kept first)")
        out = out.drop_duplicates("tract_geoid", keep="first")
    return out


# ---------------------------------------------------------------- build --------
ORDERED = [
    "h3_index", "tract_geoid",
    "nri_risk_score", "nri_risk_rating",
    "nri_eal_total", "nri_eal_building",
    "nri_sovi", "nri_resilience",
    "nri_top_hazards", "nri_tract_pop",
]


def build(con: duckdb.DuckDBPyConnection, nri: pd.DataFrame) -> None:
    """Left-join tract-level NRI onto the hex->tract map and write the parquet."""
    con.register("nri", nri)
    nri_sel = ",\n           ".join(f"n.{c}" for c in ORDERED[2:] if c in nri.columns)
    con.execute(f"""
        COPY (
          SELECT a.h3_index, a.tract_geoid,
                 {nri_sel}
          FROM read_parquet('{ACS}') a
          LEFT JOIN nri n ON a.tract_geoid = n.tract_geoid
          ORDER BY a.h3_index
        ) TO '{OUT}' (FORMAT PARQUET)
    """)


# ---------------------------------------------------------------- verify -------
def spot_check(con: duckdb.DuckDBPyConnection, label: str,
               lat: float, lon: float, want: str) -> bool:
    """Nearest populated hex to (lat,lon); True iff `want` is in its top hazards."""
    center = h3.geo_to_h3(lat, lon, RES)
    cells = [center] + sorted(h3.k_ring(center, 3) - {center})
    inlist = ",".join(f"'{c}'" for c in cells)
    row = con.execute(f"""
        SELECT h3_index, tract_geoid, nri_risk_score, nri_risk_rating,
               nri_eal_total, nri_top_hazards
        FROM read_parquet('{OUT}')
        WHERE h3_index IN ({inlist}) AND nri_top_hazards IS NOT NULL
        ORDER BY (h3_index = '{center}') DESC, h3_index
        LIMIT 1
    """).fetchone()
    if row is None:
        print(f"[verify] {label}: FAIL -- no matched hex within k=3 of {center}")
        return False
    hx, tract, score, rating, eal, top = row
    hazards = json.loads(top)
    ok = any(name == want for name, _ in hazards)
    print(f"[verify] {label}: hex={hx}{' (ring)' if hx != center else ''}  tract={tract}")
    print(f"[verify]   risk={score:.1f} ({rating})  EAL=${eal:,.0f}/yr")
    print(f"[verify]   top hazards: {top}")
    print(f"[verify]   contains '{want}': {'PASS' if ok else 'FAIL'}")
    return ok


def verify(con: duckdb.DuckDBPyConnection, nri: pd.DataFrame, url: str) -> None:
    total, matched = con.execute(f"""
        SELECT count(*), count(nri_top_hazards) FROM read_parquet('{OUT}')
    """).fetchone()

    eal_out, tracts_out, pop_out = con.execute(f"""
        SELECT sum(nri_eal_total), count(*), sum(nri_tract_pop) FROM (
          SELECT DISTINCT tract_geoid, nri_eal_total, nri_tract_pop
          FROM read_parquet('{OUT}') WHERE nri_top_hazards IS NOT NULL
        )
    """).fetchone()
    eal_src = float(nri["nri_eal_total"].sum())

    print("\n[verify] ------------------------------------------------------------")
    print(f"[verify] source url    : {url}")
    print(f"[verify] hexes         : {total:,} total, {matched:,} with NRI "
          f"({100 * matched / total:.2f}%)")
    print(f"[verify] tracts        : {tracts_out:,} distinct matched in output, "
          f"{len(nri):,} in NRI table")
    print(f"[verify] national EAL  : ${eal_out / 1e9:.2f}B/yr over distinct matched tracts "
          f"(${eal_src / 1e9:.2f}B/yr over full NRI table)")
    if pop_out:
        print(f"[verify] population    : {pop_out / 1e6:.1f}M over distinct matched tracts")

    unmatched = con.execute(f"""
        SELECT substr(tract_geoid, 1, 2) AS st, count(*) AS hexes,
               count(DISTINCT tract_geoid) AS tracts
        FROM read_parquet('{OUT}') WHERE nri_top_hazards IS NULL
        GROUP BY 1 ORDER BY hexes DESC LIMIT 5
    """).fetchall()
    if unmatched:
        print(f"[verify] unmatched by state FIPS (top {len(unmatched)}): "
              + ", ".join(f"{st}: {hx:,} hexes/{tr:,} tracts" for st, hx, tr in unmatched))

    ok_mia = spot_check(con, "Miami (25.77,-80.19)", 25.77, -80.19, "Hurricane")
    ok_sf = spot_check(con, "Bay Area (37.77,-122.42)", 37.77, -122.42, "Earthquake")

    import pyarrow.parquet as pq
    schema = pq.ParquetFile(OUT).schema_arrow
    print(f"[verify] schema        : "
          + ", ".join(f"{f.name}:{f.type}" for f in schema))
    print(f"[verify] wrote {OUT.name}: {total:,} rows, {len(schema)} cols, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    if not (ok_mia and ok_sf):
        raise SystemExit("spot checks FAILED -- inspect output before shipping")


# ---------------------------------------------------------------- main ---------
def main() -> None:
    if not ACS.exists():
        raise SystemExit(f"missing hex->tract map: {ACS} (run build_acs.py first)")
    zip_path, url = download()
    nri = load_nri(zip_path)

    con = duckdb.connect()
    build(con, nri)
    verify(con, nri, url)


if __name__ == "__main__":
    main()
