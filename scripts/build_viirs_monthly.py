#!/usr/bin/env python
"""
Build the MONTHLY night-lights pulse layer for the Janus Hex Atlas.

Output (long format):
    data/viirs_monthly_us_h3.parquet
        h3_index        string   res-9 H3 cell
        month           string   'YYYY-MM'
        viirs_radiance  float32  monthly average radiance (nW / cm^2 / sr)

Hex universe: continental-US res-9 hexes from data/us_r9/*.parquet with
nightlight_2021 > 1 OR kontur_population > 100 (~3.3M hexes).

Sources are probed IN ORDER; the first one with working credentials wins.
None of them is anonymous as of 2026-07 (probed live), so this script needs
ONE of the following before it will write any output:

  1. EOG monthly VNL v10 (Colorado School of Mines)
     - Register (free): https://eogdata.mines.edu/products/register/
     - export EOG_USER=... EOG_PASSWORD=...
       (optional EOG_CLIENT_ID / EOG_CLIENT_SECRET if EOG rotates the
        public client documented at
        https://eogdata.mines.edu/products/register/#automated_access ;
        the historical `eogdata_oidc` client now returns invalid_client)
  2. NASA Black Marble VNP46A3 via LAADS DAAC
     - Earthdata Login: https://urs.earthdata.nasa.gov  ->  Profile ->
       "Generate Token"
     - export EARTHDATA_TOKEN=...   (or put machine urs.earthdata.nasa.gov
       login/password in ~/.netrc -- a token is preferred)
     - needs `h5py` in the venv (pip install h5py) when this branch runs
  3. Microsoft Planetary Computer STAC
     - anonymous, but PC carries NO monthly nightlights collection
       (135 collections checked 2026-07; closest is `hrea`, an ANNUAL
       electricity-access product). Probe kept in case one appears.

EOG branch needs `rasterio` in the venv (pip install rasterio).

If every source is blocked the script prints the auth report above,
writes NOTHING, and exits with status 2. It never fabricates data.
"""

from __future__ import annotations

import calendar
import datetime as dt
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "data" / "us_r9"
RAW = ROOT / "data" / "raw" / "viirs"
OUT = ROOT / "data" / "viirs_monthly_us_h3.parquet"

N_MONTHS = 24            # target window
FALLBACK_MONTHS = 12     # if output exceeds SIZE_CAP_MB
SIZE_CAP_MB = 150
# continental US clip (excludes AK / HI / PR)
CONUS = dict(lon_min=-125.0, lon_max=-66.5, lat_min=24.3, lat_max=49.6)
KEEP_RASTERS = os.environ.get("VIIRS_KEEP_RASTERS", "1") != "0"

EOG_BASE = "https://eogdata.mines.edu/nighttime_light/monthly/v10"
EOG_TOKEN_URL = "https://eogauth.mines.edu/realms/eog/protocol/openid-connect/token"
LAADS_API = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details"
LAADS_ARCHIVE = "https://ladsweb.modaps.eosdis.nasa.gov/archive"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

VERIFY_SPOTS = [
    ("Vegas Strip, NV", 36.117, -115.172),
    ("Williston Basin oil field, ND", 48.146, -103.618),
    ("Ocean City, MD (beach town)", 38.3837, -75.0675),
]


# --------------------------------------------------------------------------
# hex universe
# --------------------------------------------------------------------------

def load_hexes() -> pd.DataFrame:
    import duckdb

    q = f"""
        SELECT h3_index, center_lat, center_lon
        FROM '{SHARDS}/*.parquet'
        WHERE (nightlight_2021 > 1 OR kontur_population > 100)
          AND center_lon BETWEEN {CONUS['lon_min']} AND {CONUS['lon_max']}
          AND center_lat BETWEEN {CONUS['lat_min']} AND {CONUS['lat_max']}
    """
    df = duckdb.connect().execute(q).df()
    print(f"[hexes] universe: {len(df):,} res-9 hexes "
          f"(nightlight_2021>1 OR kontur_population>100, CONUS clip)")
    return df


# --------------------------------------------------------------------------
# source 1: EOG monthly VNL v10
# --------------------------------------------------------------------------

def eog_token() -> tuple[str | None, str]:
    user, pw = os.environ.get("EOG_USER"), os.environ.get("EOG_PASSWORD")
    if not (user and pw):
        return None, ("no EOG credentials: every eogdata.mines.edu path 302s to "
                      "eogauth.mines.edu (Keycloak OIDC). Register free at "
                      "https://eogdata.mines.edu/products/register/ then "
                      "export EOG_USER / EOG_PASSWORD.")
    payload = {
        "grant_type": "password",
        "client_id": os.environ.get("EOG_CLIENT_ID", "eogdata_oidc"),
        "username": user,
        "password": pw,
    }
    secret = os.environ.get("EOG_CLIENT_SECRET",
                            "2677ad81-521b-4869-8480-6d05b9e57d48")
    if secret:
        payload["client_secret"] = secret
    r = requests.post(EOG_TOKEN_URL, data=payload, timeout=60)
    if r.status_code != 200:
        return None, (f"EOG token endpoint refused ({r.status_code}: "
                      f"{r.text[:120]}). If the error is invalid_client, EOG "
                      "rotated the public client -- copy the current "
                      "client_id/client_secret from their download-automation "
                      "docs into EOG_CLIENT_ID / EOG_CLIENT_SECRET.")
    return r.json()["access_token"], "ok"


def _eog_ls(url: str, tok: str) -> list[str]:
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    return re.findall(r'href="([^"?/][^"]*)"', r.text)


def probe_eog() -> tuple[dict | None, str]:
    """Return ({month: file_url}, 'ok') for the last N_MONTHS, or (None, why)."""
    tok, why = eog_token()
    if tok is None:
        return None, why
    months: dict[str, str] = {}
    try:
        years = sorted((h.strip("/") for h in _eog_ls(f"{EOG_BASE}/", tok)
                        if re.fullmatch(r"\d{4}/?", h)), reverse=True)
        for y in years:
            for ym in sorted((h.strip("/") for h in _eog_ls(f"{EOG_BASE}/{y}/", tok)
                              if re.fullmatch(r"\d{6}/?", h)), reverse=True):
                d = f"{EOG_BASE}/{y}/{ym}/vcmcfg/"
                files = _eog_ls(d, tok)
                # prefer the masked average-radiance product, fall back to raw avg
                pick = ([f for f in files if f.endswith((".avg_rade9h.masked.tif",
                                                         ".avg_rade9h.masked.tif.gz"))]
                        or [f for f in files if ".avg_rade9h.tif" in f])
                if pick:
                    months[f"{ym[:4]}-{ym[4:]}"] = d + pick[0]
                if len(months) >= N_MONTHS:
                    break
            if len(months) >= N_MONTHS:
                break
    except Exception as e:  # noqa: BLE001
        return None, f"EOG listing failed after auth: {e}"
    if not months:
        return None, "EOG auth worked but no monthly vcmcfg files found"
    return months, "ok"


def sample_eog(months: dict[str, str], tok: str, hexes: pd.DataFrame) -> pd.DataFrame:
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        sys.exit("EOG branch needs rasterio:  .venv/bin/pip install rasterio")

    RAW.mkdir(parents=True, exist_ok=True)
    lon = hexes["center_lon"].to_numpy()
    lat = hexes["center_lat"].to_numpy()
    frames = []
    for month, url in sorted(months.items()):
        local = RAW / url.rsplit("/", 1)[1]
        if not local.exists():
            print(f"[eog] downloading {month}: {local.name}")
            with requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                              stream=True, timeout=1800) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=1 << 20)
        path = str(local)
        if path.endswith(".gz"):  # rasterio reads gzip via /vsigzip/
            path = "/vsigzip/" + path
        with rasterio.open(path) as src:
            win = from_bounds(CONUS["lon_min"], CONUS["lat_min"],
                              CONUS["lon_max"], CONUS["lat_max"], src.transform)
            arr = src.read(1, window=win)
            wt = src.window_transform(win)
            col = np.floor((lon - wt.c) / wt.a).astype(np.int64)
            row = np.floor((lat - wt.f) / wt.e).astype(np.int64)
            ok = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
            vals = np.full(len(hexes), np.nan, np.float32)
            vals[ok] = arr[row[ok], col[ok]].astype(np.float32)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        vals[vals < 0] = 0.0  # EOG background noise can dip slightly negative
        frames.append(pd.DataFrame({"h3_index": hexes["h3_index"],
                                    "month": month, "viirs_radiance": vals}))
        if not KEEP_RASTERS:
            local.unlink(missing_ok=True)
        print(f"[eog] {month}: sampled, mean={np.nanmean(vals):.3f}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# source 2: NASA Black Marble VNP46A3 via LAADS
# --------------------------------------------------------------------------

def earthdata_token() -> tuple[str | None, str]:
    tok = os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_BEARER")
    if tok:
        return tok, "ok"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text():
        return None, ("~/.netrc has urs.earthdata.nasa.gov but LAADS archive "
                      "downloads want a bearer token; generate one at "
                      "https://urs.earthdata.nasa.gov (Profile -> Generate "
                      "Token) and export EARTHDATA_TOKEN.")
    return None, ("no Earthdata credentials: LAADS file downloads AND deep "
                  "listings under allData/5000/VNP46A3 303-redirect to "
                  "urs.earthdata.nasa.gov OAuth. Create an Earthdata Login, "
                  "generate a token, export EARTHDATA_TOKEN.")


def probe_laads() -> tuple[dict | None, str, str | None]:
    """Return ({month: [file_urls for CONUS tiles]}, 'ok', token) or (None, why, None)."""
    tok, why = earthdata_token()
    if tok is None:
        return None, why, None
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

    def ls(path: str):
        r = requests.get(f"{LAADS_API}/{path}", headers=hdr, timeout=120)
        if r.status_code != 200 or "urs.earthdata.nasa.gov" in r.url:
            raise RuntimeError(f"listing {path} -> {r.status_code} ({r.url})")
        return json.loads(r.text).get("content", [])

    # CONUS tile ids on the Black Marble 10-degree grid
    hs = range(int((CONUS["lon_min"] + 180) // 10), int((CONUS["lon_max"] + 180) // 10) + 1)
    vs = range(int((90 - CONUS["lat_max"]) // 10), int((90 - CONUS["lat_min"]) // 10) + 1)
    tiles = {f"h{h:02d}v{v:02d}" for h in hs for v in vs}

    months: dict[str, list[str]] = {}
    try:
        years = sorted((c["name"] for c in ls("allData/5000/VNP46A3")
                        if c["name"].isdigit()), reverse=True)
        for y in years:
            for doy in sorted((c["name"] for c in ls(f"allData/5000/VNP46A3/{y}")
                               if c["name"].isdigit()), key=int, reverse=True):
                date = dt.date(int(y), 1, 1) + dt.timedelta(days=int(doy) - 1)
                key = f"{date:%Y-%m}"
                fs = [c["name"] for c in ls(f"allData/5000/VNP46A3/{y}/{doy}")
                      if c["name"].endswith(".h5")
                      and any(t in c["name"] for t in tiles)]
                if fs:
                    months[key] = [f"{LAADS_ARCHIVE}/allData/5000/VNP46A3/{y}/{doy}/{f}"
                                   for f in fs]
                if len(months) >= N_MONTHS:
                    break
            if len(months) >= N_MONTHS:
                break
    except Exception as e:  # noqa: BLE001
        return None, f"LAADS listing failed with token: {e}", None
    if not months:
        return None, "LAADS token worked but no VNP46A3 CONUS tiles found", None
    return months, "ok", tok


def sample_laads(months: dict[str, list[str]], tok: str,
                 hexes: pd.DataFrame) -> pd.DataFrame:
    try:
        import h5py
    except ImportError:
        sys.exit("LAADS branch needs h5py:  .venv/bin/pip install h5py")

    RAW.mkdir(parents=True, exist_ok=True)
    DS = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/NearNadir_Composite_Snow_Free"
    lon = hexes["center_lon"].to_numpy()
    lat = hexes["center_lat"].to_numpy()
    res = 10.0 / 2400.0  # 15 arc-second
    frames = []
    for month, urls in sorted(months.items()):
        vals = np.full(len(hexes), np.nan, np.float32)
        for url in urls:
            local = RAW / url.rsplit("/", 1)[1]
            if not local.exists():
                with requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                                  stream=True, timeout=1800) as r:
                    r.raise_for_status()
                    with open(local, "wb") as f:
                        shutil.copyfileobj(r.raw, f, length=1 << 20)
            m = re.search(r"\.h(\d{2})v(\d{2})\.", local.name)
            h, v = int(m.group(1)), int(m.group(2))
            lon0, lat0 = -180.0 + 10.0 * h, 90.0 - 10.0 * v
            sel = ((lon >= lon0) & (lon < lon0 + 10) &
                   (lat <= lat0) & (lat > lat0 - 10))
            if not sel.any():
                continue
            with h5py.File(local, "r") as f:
                d = f[DS]
                arr = d[...].astype(np.float32)
                fill = d.attrs.get("_FillValue", [65535])[0]
                scale = float(d.attrs.get("scale_factor", [0.1])[0])
                off = float(d.attrs.get("add_offset", [0.0])[0])
            arr[arr == fill] = np.nan
            arr = arr * scale + off
            row = np.floor((lat0 - lat[sel]) / res).astype(np.int64).clip(0, 2399)
            col = np.floor((lon[sel] - lon0) / res).astype(np.int64).clip(0, 2399)
            vals[sel] = arr[row, col]
            if not KEEP_RASTERS:
                local.unlink(missing_ok=True)
        frames.append(pd.DataFrame({"h3_index": hexes["h3_index"],
                                    "month": month, "viirs_radiance": vals}))
        print(f"[laads] {month}: sampled, mean={np.nanmean(vals):.3f}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# source 3: Microsoft Planetary Computer
# --------------------------------------------------------------------------

def probe_pc() -> tuple[None, str]:
    try:
        r = requests.get(f"{PC_STAC}/collections", timeout=60)
        r.raise_for_status()
        cols = r.json()["collections"]
    except Exception as e:  # noqa: BLE001
        return None, f"PC STAC unreachable: {e}"
    hits = [c["id"] for c in cols
            if any(k in (c["id"] + " " + (c.get("title") or "")).lower()
                   for k in ("viirs", "nighttime", "night-time", "black marble",
                             "vnp46", "dnb"))]
    if hits:
        return None, (f"PC now lists possible nightlights collections {hits} -- "
                      "add a sampling branch for them (not implemented because "
                      "none existed when this script was written).")
    return None, (f"PC has no monthly nightlights collection ({len(cols)} "
                  "collections checked; no VIIRS DNB / Black Marble / "
                  "nighttime-lights entry; `hrea` is annual electricity access, "
                  "not monthly radiance). Nothing to authenticate -- the data "
                  "simply is not on PC.")


# --------------------------------------------------------------------------
# output + verification
# --------------------------------------------------------------------------

def write_and_verify(long_df: pd.DataFrame, hexes: pd.DataFrame, source: str):
    import h3

    long_df = long_df.dropna(subset=["viirs_radiance"])
    long_df["viirs_radiance"] = long_df["viirs_radiance"].astype(np.float32)
    long_df = long_df.sort_values(["h3_index", "month"], ignore_index=True)

    def _write(df: pd.DataFrame):
        df.to_parquet(OUT, index=False, compression="zstd")
        return OUT.stat().st_size / 1e6

    mb = _write(long_df)
    months = sorted(long_df["month"].unique())
    if mb > SIZE_CAP_MB and len(months) > FALLBACK_MONTHS:
        keep = months[-FALLBACK_MONTHS:]
        print(f"[size] {mb:.0f} MB > {SIZE_CAP_MB} MB cap -> keeping last "
              f"{FALLBACK_MONTHS} months")
        long_df = long_df[long_df["month"].isin(keep)].reset_index(drop=True)
        mb = _write(long_df)
        months = keep

    print(f"\n[out] {OUT}  ({mb:.1f} MB, {len(long_df):,} rows, "
          f"{long_df['h3_index'].nunique():,} hexes x {len(months)} months, "
          f"source={source})")

    # wide summary
    nat = long_df.groupby("month")["viirs_radiance"].agg(["count", "mean", "median"])
    print("\n[wide summary] national per-month stats:")
    print(nat.round(3).to_string())

    # temporal-smoothness / artifact check
    ratio = (nat["mean"] / nat["mean"].shift(1)).dropna()
    bad = ratio[(ratio > 3) | (ratio < 1 / 3)]
    if len(bad):
        print("\n[flag] adjacent-month national-mean jumps >3x (likely "
              "stray-light / snow artifact months, EOG marks winter "
              "high-latitude composites low-quality):")
        print(bad.round(2).to_string())
    else:
        print("\n[ok] national mean is temporally smooth (no >3x adjacent-month jumps)")

    # spot checks
    for name, la, lo in VERIFY_SPOTS:
        hx = h3.geo_to_h3(la, lo, 9)
        s = long_df.loc[long_df["h3_index"] == hx,
                        ["month", "viirs_radiance"]].set_index("month")
        if s.empty:  # nearest in-universe neighbour
            ring = h3.k_ring(hx, 3)
            cand = long_df[long_df["h3_index"].isin(ring)]
            if cand.empty:
                print(f"\n[spot] {name}: no hex in universe near ({la},{lo})")
                continue
            hx = cand["h3_index"].iloc[0]
            s = cand[cand["h3_index"] == hx][["month", "viirs_radiance"]].set_index("month")
        v = s["viirs_radiance"]
        print(f"\n[spot] {name} ({hx}): mean={v.mean():.2f} "
              f"cv={(v.std() / v.mean() if v.mean() else np.nan):.2f}")
        print(v.round(2).to_string())


# --------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("VIIRS monthly pulse builder -- probing sources in order")
    print("=" * 72)

    blockers: list[tuple[str, str]] = []

    eog_months, eog_why = probe_eog()
    if eog_months:
        print(f"[probe] EOG monthly v10: OK ({len(eog_months)} months)")
        hexes = load_hexes()
        tok, _ = eog_token()
        write_and_verify(sample_eog(eog_months, tok, hexes), hexes, "EOG VNL v10 vcmcfg")
        return 0
    blockers.append(("EOG monthly VNL v10", eog_why))
    print(f"[probe] EOG monthly v10: BLOCKED -- {eog_why}\n")

    laads_months, laads_why, laads_tok = probe_laads()
    if laads_months:
        print(f"[probe] LAADS VNP46A3: OK ({len(laads_months)} months)")
        hexes = load_hexes()
        write_and_verify(sample_laads(laads_months, laads_tok, hexes), hexes,
                         "NASA Black Marble VNP46A3 (NearNadir_Composite_Snow_Free)")
        return 0
    blockers.append(("NASA LAADS VNP46A3", laads_why))
    print(f"[probe] LAADS VNP46A3: BLOCKED -- {laads_why}\n")

    _, pc_why = probe_pc()
    blockers.append(("MS Planetary Computer", pc_why))
    print(f"[probe] Planetary Computer: BLOCKED -- {pc_why}\n")

    print("=" * 72)
    print("ALL SOURCES BLOCKED -- no parquet written (nothing fabricated).")
    print("Auth needed, per source:")
    for name, why in blockers:
        print(f"\n  * {name}:\n      {why}")
    print("\nRe-run this script after exporting EOG_USER/EOG_PASSWORD or "
          "EARTHDATA_TOKEN.")
    print("=" * 72)
    return 2


if __name__ == "__main__":
    sys.exit(main())
