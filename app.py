"""
The Hex Atlas — click any hex, see everything open data + the geofm model know
about that cell, raw (no scores).

Two scopes:
  · United States — national res-5 overview (32k cells), click to drill into
    res-9 (174 m). Safety = NHTSA FARS fatal crashes; census/traffic sections
    appear automatically when their parquets exist.
  · Ten deep cities (US + India) — res-9 direct. Crashes are NYC-only
    (open Vision Zero); the base card is global.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curbai import ui  # noqa: E402

st.set_page_config(page_title="Janus — The Hex Atlas", page_icon="🔬", layout="wide")
ui.inject()

DATADIR = Path(__file__).resolve().parents[1] / "data"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
US = "New York · tiled"

# label -> (slug, lat, lon, zoom)
CITIES = {
    "New York": ("nyc", 40.739, -74.001, 10.2),
    "Los Angeles": ("la", 34.048, -118.363, 9.5),
    "Chicago": ("chicago", 41.83, -87.72, 9.8),
    "Houston": ("houston", 29.80, -95.42, 9.3),
    "San Francisco": ("sf", 37.763, -122.44, 11.3),
    "Delhi": ("delhi", 28.60, 77.14, 10.0),
    "Mumbai": ("mumbai", 19.11, 72.88, 10.6),
    "Bengaluru": ("bangalore", 12.98, 77.61, 10.4),
    "Hyderabad": ("hyderabad", 17.42, 78.46, 10.3),
    "Chennai": ("chennai", 13.06, 80.23, 10.6),
}

# Base metrics that shade the map (global). Crash layers are prepended
# where crash data exists (Vision Zero for NYC, FARS for the US).
LAYERS = {
    "Population": ("kontur_population", "{:,.0f} residents"),
    "POIs / places": ("poi_count", "{:,.0f} POIs"),
    "Buildings": ("building_count", "{:,.0f} buildings"),
    "Movement · visits": ("wt_visit_count", "{:,.0f} visits"),
    "Roads": ("road_count", "{:,.0f} road segments"),
    "Night-lights": ("nightlight_2021", "{:.0f} brightness"),
}

US_R5 = DATADIR / "us_r5.parquet"
US_R9_GLOB = str(DATADIR / "us_r9" / "*.parquet")
FARS = DATADIR / "fars_us_h3.parquet"
ACS = DATADIR / "acs_us_h3.parquet"
HPMS = DATADIR / "hpms_us_h3.parquet"
NRI = DATADIR / "nri_us_h3.parquet"
LODES = DATADIR / "lodes_us_h3.parquet"
LODESOD = DATADIR / "lodesod_us_h3.parquet"
GDELT = DATADIR / "gdelt_us_h3.parquet"
EAGLEI = DATADIR / "eaglei_us_h3.parquet"
NDVI = DATADIR / "ndvi_us_h3.parquet"
MLY = DATADIR / "mapillary_us_h3.parquet"
OSMPED = DATADIR / "osmped_us_h3.parquet"
WZDX = DATADIR / "wzdx_us_h3.parquet"
SLOPE = DATADIR / "slope_nyc_h3.parquet"
SWP = DATADIR / "sidewalkphys_nyc_h3.parquet"
SWD = DATADIR / "nycsw_width_h3.parquet"
OSMW = DATADIR / "osmwidth_us_h3.parquet"
DERIVED = DATADIR / "us_derived_h3.parquet"
WM = DATADIR / "worldmove_us_h3.parquet"

DATASET_REPO = "skay97/curbai-data"


@st.cache_resource(show_spinner="First boot — pulling the data layers from the hub…")
def ensure_data() -> None:
    """The Space ships code only (1 GB cap); layers live in a private HF
    dataset. No-op when data is already on disk (local dev / warm container).
    Needs HF_TOKEN as a Space secret to read the private repo."""
    if US_R5.exists():
        return
    from huggingface_hub import snapshot_download
    snapshot_download(DATASET_REPO, repo_type="dataset",
                      local_dir=DATADIR.parent,
                      allow_patterns=["data/*.parquet", "data/us_r9/*.parquet",
                                      "data/us_lod/*.parquet", "data/us_boundary.wkt"],
                      token=os.environ.get("HF_TOKEN"))


ensure_data()


# ---------- loaders ----------

@st.cache_data(show_spinner="Loading city grid…")
def load(slug: str) -> pd.DataFrame:
    df = pd.read_parquet(DATADIR / f"{slug}_base.parquet")
    if slug == "nyc" and (DATADIR / "nyc_crashes_h3.parquet").exists():
        crash = pd.read_parquet(DATADIR / "nyc_crashes_h3.parquet")
        cols = [c for c in crash.columns if c not in ("center_lat", "center_lon")]
        df = df.merge(crash[cols], on="h3_index", how="left")
    return df


ROLLUPS = DATADIR / "us_r5_rollups.parquet"


@st.cache_data(show_spinner="Loading the United States…")
def load_us_overview() -> pd.DataFrame:
    df = pd.read_parquet(US_R5)
    if ROLLUPS.exists():
        # precomputed by scripts/build_derived.py — no parent math at load
        df = df.merge(pd.read_parquet(ROLLUPS), on="h3_index", how="left")
    elif FARS.exists():
        fars = pd.read_parquet(FARS, columns=["h3_index", "fars_crashes", "fars_killed"])
        fars["res5"] = fars.h3_index.map(lambda h: h3.h3_to_parent(h, 5))
        g = fars.groupby("res5")[["fars_crashes", "fars_killed"]].sum().reset_index()
        df = df.merge(g, left_on="h3_index", right_on="res5", how="left").drop(columns=["res5"])
    return df


@st.cache_data(show_spinner="Opening this cell at 174 m…")
def load_us_children(r5: str) -> pd.DataFrame:
    con = duckdb.connect()
    kids = con.execute(
        f"SELECT * FROM read_parquet('{US_R9_GLOB}') WHERE res5 = ?", [r5]
    ).df()
    con.register("kids", kids[["h3_index"]])
    for side in (FARS, ACS, HPMS, NRI, LODES, LODESOD, GDELT, EAGLEI, NDVI,
                 MLY, OSMPED, WZDX, SLOPE, SWP, SWD, OSMW, DERIVED, WM):
        if not side.exists():
            continue
        s = con.execute(
            f"SELECT t.* FROM read_parquet('{side.as_posix()}') t "
            "JOIN kids USING (h3_index)"
        ).df()
        # first layer to bring a column wins (centers, tract_geoid, ...)
        s = s.drop(columns=[c for c in s.columns if c != "h3_index" and c in kids.columns])
        kids = kids.merge(s, on="h3_index", how="left")
    return kids


def apply_layer(df: pd.DataFrame, col: str, valfmt: str, elev_max: float = 320.0) -> pd.DataFrame:
    v = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    top = float(v.max()) or 1.0
    df["_color"] = (np.log1p(v) / (np.log1p(top) or 1)).apply(ui.count_color)
    df["_elev"] = (v / top * elev_max).astype(float)
    df["_val_str"] = v.apply(lambda x: valfmt.format(x))
    return df


# ---------- small formatters ----------

def fmt_hour(h) -> str:
    if h is None or h < 0:
        return "—"
    h = int(h)
    return f"{(h % 12) or 12}{'am' if h < 12 else 'pm'}"


def _g(row, col):
    if col not in row.index:
        return None
    v = row.get(col)
    # pd.isna catches numpy float32/float64 NaN too — isinstance(float) doesn't
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def _fmt(v, f="{:,.0f}") -> str:
    return "—" if v is None else f.format(v)


def _dist(km) -> str:
    return "—" if km is None else (f"{km*1000:.0f} m" if km < 1 else f"{km:.1f} km")


def _cat(s) -> str:
    if not s:
        return "—"
    s = str(s).strip("[]'\" ")
    return " · ".join(p.strip() for p in s.split(">")[-2:]) if ">" in s else s


def _amenities(row) -> str:
    have = [n for n, c in [("hospital", "has_hospital"), ("school", "has_school"),
                           ("park", "has_park"), ("pharmacy", "has_pharmacy"),
                           ("worship", "has_worship")] if _g(row, c)]
    return ", ".join(have) if have else "—"


def _hist(values) -> str:
    mx = max(values) or 1
    bars = "".join(f'<div class="b" style="height:{max(2, int(v/mx*100))}%"></div>' for v in values)
    return (f'<div class="jx-hist">{bars}</div>'
            '<div class="jx-hticks"><span>12a</span><span>6a</span><span>12p</span><span>6p</span><span>11p</span></div>')


# ---------- card sections ----------

def base_html(row, top: str = "", census: str = "", traffic: str = "",
              flows: str = "", street: str = "") -> str:
    """The shared raw-data card. Optional fragments slot in at fixed points:
    top (safety/rates), census (after Who's here), traffic (after Roads),
    flows + street (after Movement)."""
    pop = _g(row, "kontur_population") or _g(row, "population")
    nl = _g(row, "nightlight_2021")
    bc, fl, mh = _g(row, "building_count"), _g(row, "avg_floors"), _g(row, "max_height")
    poi, cat, ntr = _g(row, "poi_count"), _cat(_g(row, "top_category")), _g(row, "count_transit")
    dh, dp, dt = _g(row, "dist_hospital_km"), _g(row, "dist_park_km"), _g(row, "dist_transit_km")
    rc, rpri, rres = _g(row, "road_count"), _g(row, "road_primary"), _g(row, "road_residential")
    vis, pkh, nf, rog = (_g(row, "wt_visit_count"), _g(row, "wt_peak_hour"),
                         _g(row, "wt_night_fraction"), _g(row, "wt_radius_of_gyration_km"))
    temp, precip = _g(row, "annual_mean_temp"), _g(row, "annual_precipitation")
    txt = _g(row, "llmgeovec_text")

    # calibrated population (census tract truth x kontur weights) when derived
    cal = _g(row, "pop_calibrated")
    if cal is not None:
        pop_rows = (f'<div class="jx-row"><span class="k">Residents</span>'
                    f'<span class="v"><b>{_fmt(cal)}</b> census-calibrated</span></div>'
                    f'<div class="jx-row"><span class="k">Modeled index</span>'
                    f'<span class="v">{_fmt(pop)}</span></div>')
        who_src = "ACS × Kontur"
    else:
        pop_rows = (f'<div class="jx-row"><span class="k">Population</span>'
                    f'<span class="v"><b>{_fmt(pop)}</b> modeled residents</span></div>')
        who_src = "Kontur"

    return f"""
    <div class="jx-card">
      <div class="jx-cid">R9 · {row.h3_index[-12:]} · 0.105 km²</div>
      {top}
      <div class="jx-lab jx-sec">◆ Who's here — {who_src}</div>
      {pop_rows}
      <div class="jx-row"><span class="k">Night-lights</span><span class="v">{_fmt(nl, '{:.0f}')}</span></div>
      {census}
      <div class="jx-lab jx-sec">◆ Built form — Overture</div>
      <div class="jx-row"><span class="k">Buildings</span><span class="v"><b>{_fmt(bc)}</b> · avg {_fmt(fl, '{:.0f}')} fl</span></div>
      <div class="jx-row"><span class="k">Tallest</span><span class="v">{_fmt(mh, '{:.0f}')} m</span></div>
      <div class="jx-lab jx-sec">◆ Places — Overture / FSQ</div>
      <div class="jx-row"><span class="k">POIs</span><span class="v"><b>{_fmt(poi)}</b> · {_fmt(ntr, '{:.0f}')} transit</span></div>
      <div class="jx-row"><span class="k">Character</span><span class="v">{cat}</span></div>
      <div class="jx-row"><span class="k">On / near</span><span class="v">{_amenities(row)}</span></div>
      <div class="jx-lab jx-sec">◆ Access — nearest</div>
      <div class="jx-row"><span class="k">Hospital · Park · Transit</span><span class="v">{_dist(dh)} · {_dist(dp)} · {_dist(dt)}</span></div>
      <div class="jx-lab jx-sec">◆ Roads — OpenStreetMap</div>
      <div class="jx-row"><span class="k">Segments</span><span class="v"><b>{_fmt(rc)}</b> · {_fmt(rpri)} primary · {_fmt(rres)} resid.</span></div>
      {traffic}
      <div class="jx-lab jx-sec">◆ Movement — trajectory data</div>
      <div class="jx-row"><span class="k">Visits</span><span class="v"><b>{_fmt(vis)}</b> · peak {fmt_hour(pkh) if pkh is not None else '—'}</span></div>
      <div class="jx-row"><span class="k">Night · radius</span><span class="v">{_fmt(nf*100 if nf is not None else None, '{:.0f}')}% · {_fmt(rog, '{:.1f}')} km</span></div>
      {flows}
      {street}
      <div class="jx-lab jx-sec">◆ Climate — WorldClim{' / MODIS' if _g(row, 'ndvi_summer') is not None else ''}</div>
      <div class="jx-row"><span class="k">Temp · rain</span><span class="v">{_fmt(temp, '{:.0f}')}°C · {_fmt(precip)} mm/yr</span></div>
      {f'<div class="jx-row"><span class="k">Greenness (summer NDVI)</span><span class="v"><b>{_g(row, "ndvi_summer"):.2f}</b></span></div>' if _g(row, 'ndvi_summer') is not None else ''}
      <div class="jx-lab jx-sec">◆ The model reads</div>
      <div class="jx-txt">{(str(txt)[:210] + '…') if txt else '—'}</div>
    </div>
    """


def nyc_safety_html(row) -> str:
    if _g(row, "crashes") is None or _g(row, "hour_hist") is None:
        return ""
    fac = "".join(f'<div class="jx-fac"><span>{k.title()}</span><span class="c">{v}</span></div>'
                  for k, v in json.loads(row.top_factors)) or '<div class="jx-fac"><span class="c">— none —</span></div>'
    return f"""
      <div class="jx-lab" style="margin-top:6px">◆ Safety — NYPD Vision Zero</div>
      <div class="jx-big">{int(row.crashes):,}<small> collisions</small></div>
      <div class="jx-row"><span class="k">Killed / injured</span><span class="v"><b>{int(row.killed)}</b> · {int(row.injured):,}</span></div>
      <div class="jx-row"><span class="k">Pedestrian</span><span class="v">{int(row.ped_inj)} inj · {int(row.ped_kill)} killed</span></div>
      <div class="jx-row"><span class="k">Peak</span><span class="v"><b>{DOW[int(row.peak_dow)] if row.peak_dow >= 0 else '—'} {fmt_hour(row.peak_hour)}</b></span></div>
      {_hist(json.loads(row.hour_hist))}
      <div class="jx-lab jx-sec" style="margin-top:6px">Top contributing factors</div>{fac}"""


def fars_html(row) -> str:
    """FARS fatal-crash section — US drill only. A quiet row when clean."""
    if "fars_crashes" not in row.index:
        return ""
    c = _g(row, "fars_crashes")
    if c is None:
        return ('<div class="jx-lab" style="margin-top:6px">◆ Safety — NHTSA FARS 2022–2024</div>'
                '<div class="jx-row"><span class="k">Fatal crashes</span><span class="v">none recorded</span></div>')
    y0, y1 = _g(row, "fars_year_min"), _g(row, "fars_year_max")
    yrs = f"{int(y0)}–{int(y1)}" if y0 and y1 and y0 != y1 else (f"{int(y0)}" if y0 else "2022–2024")
    dow = _g(row, "fars_peak_dow_label") or "—"
    hist = _hist(json.loads(row.fars_hour_hist)) if _g(row, "fars_hour_hist") else ""
    return f"""
      <div class="jx-lab" style="margin-top:6px">◆ Safety — NHTSA FARS {yrs}</div>
      <div class="jx-big">{int(c):,}<small> fatal crash{'es' if c != 1 else ''}</small></div>
      <div class="jx-row"><span class="k">Killed</span><span class="v"><b>{int(row.fars_killed)}</b></span></div>
      <div class="jx-row"><span class="k">Peak</span><span class="v"><b>{dow} {fmt_hour(_g(row, 'fars_peak_hour'))}</b></span></div>
      {hist}"""


def acs_html(row) -> str:
    if _g(row, "acs_median_income") is None and _g(row, "acs_population") is None:
        return ""
    geoid = _g(row, "tract_geoid")
    return f"""
      <div class="jx-lab jx-sec">◆ Census — ACS 5-yr · tract {geoid or '—'}</div>
      <div class="jx-row"><span class="k">Tract population</span><span class="v"><b>{_fmt(_g(row, 'acs_population'))}</b> census</span></div>
      <div class="jx-row"><span class="k">Median income</span><span class="v"><b>{_fmt(_g(row, 'acs_median_income'), '${:,.0f}')}</b></span></div>
      <div class="jx-row"><span class="k">Median age</span><span class="v">{_fmt(_g(row, 'acs_median_age'), '{:.0f}')}</span></div>
      <div class="jx-row"><span class="k">Rent · home value</span><span class="v">{_fmt(_g(row, 'acs_median_rent'), '${:,.0f}')} · {_fmt(_g(row, 'acs_median_home_value'), '${:,.0f}')}</span></div>
      <div class="jx-row"><span class="k">No-vehicle households</span><span class="v">{_fmt(_g(row, 'acs_pct_no_vehicle'), '{:.0f}%')}</span></div>"""


def hpms_html(row) -> str:
    a = _g(row, "hpms_aadt_max")
    if a is None:
        return ""
    trucks = _g(row, "hpms_pct_truck")
    truck_row = (f'<div class="jx-row"><span class="k">Trucks</span><span class="v">{_fmt(trucks, "{:.0f}%")}</span></div>'
                 if trucks is not None else "")
    src = {"hpms2023": "FHWA HPMS 2023", "hpms2022": "FHWA HPMS 2022 · backbone"}.get(
        str(_g(row, "hpms_source")), "FHWA HPMS")
    return f"""
      <div class="jx-lab jx-sec">◆ Traffic — {src}</div>
      <div class="jx-row"><span class="k">AADT (busiest)</span><span class="v"><b>{_fmt(a)}</b> veh/day</span></div>
      <div class="jx-row"><span class="k">Covered segments</span><span class="v">{_fmt(_g(row, 'hpms_seg_count'))}</span></div>
      {truck_row}"""


def rates_html(row) -> str:
    """Cross-layer rates — real units, named denominators. No scores."""
    rows = []
    r = _g(row, "drv_fatal_per_100k_aadt")
    if r is not None:
        rows.append(f'<div class="jx-row"><span class="k">Fatal / traffic</span>'
                    f'<span class="v"><b>{r:.2f}</b> /yr per 100k veh·day</span></div>')
    v = _g(row, "drv_visits_per_resident")
    if v is not None:
        rows.append(f'<div class="jx-row"><span class="k">Visits / resident</span>'
                    f'<span class="v"><b>{v:,.1f}×</b></span></div>')
    j = _g(row, "drv_jobs_per_resident")
    if j is not None:
        rows.append(f'<div class="jx-row"><span class="k">Jobs / resident</span>'
                    f'<span class="v"><b>{j:,.1f}×</b> daytime pull</span></div>')
    e = _g(row, "drv_eal_per_capita")
    if e is not None:
        rows.append(f'<div class="jx-row"><span class="k">Hazard loss / person</span>'
                    f'<span class="v"><b>${e:,.0f}</b> /yr expected</span></div>')
    if not rows:
        return ""
    return f'<div class="jx-lab jx-sec">◆ Rates — cross-layer</div>{"".join(rows)}'


def nri_html(row) -> str:
    """FEMA National Risk Index — tract-level hazard economics."""
    eal = _g(row, "nri_eal_total")
    if eal is None:
        return ""
    top = ""
    th = _g(row, "nri_top_hazards")
    if th:
        try:
            parts = " · ".join(f"{name} ${v/1e6:.1f}M" if v >= 1e6 else f"{name} ${v/1e3:.0f}k"
                               for name, v in json.loads(th)[:3])
            top = f'<div class="jx-row"><span class="k">Top hazards</span><span class="v">{parts}</span></div>'
        except Exception:
            pass
    alloc = _g(row, "nri_eal_alloc")
    alloc_row = (f'<div class="jx-row"><span class="k">This hex share</span>'
                 f'<span class="v"><b>${alloc:,.0f}</b> /yr (pop-allocated)</span></div>'
                 if alloc is not None else "")
    return f"""
      <div class="jx-lab jx-sec">◆ Hazard risk — FEMA NRI · tract</div>
      <div class="jx-row"><span class="k">Expected annual loss</span><span class="v"><b>${eal:,.0f}</b> /yr tract</span></div>
      {alloc_row}
      {top}
      <div class="jx-row"><span class="k">Rating</span><span class="v">{_g(row, 'nri_risk_rating') or '—'}</span></div>"""


def lodes_html(row) -> str:
    j = _g(row, "lodes_jobs")
    if j is None:
        return ""
    mix = " · ".join(f"{n} {int(v):,}" for n, v in
                     [("retail", _g(row, "lodes_jobs_retail")), ("food", _g(row, "lodes_jobs_food")),
                      ("health", _g(row, "lodes_jobs_health")), ("edu", _g(row, "lodes_jobs_edu"))]
                     if v)
    mix_row = (f'<div class="jx-row"><span class="k">Mix</span><span class="v">{mix}</span></div>'
               if mix else "")
    return f"""
      <div class="jx-lab jx-sec">◆ Jobs — Census LODES · BG-centroid</div>
      <div class="jx-row"><span class="k">Workplace jobs</span><span class="v"><b>{int(j):,}</b> (block-group lump)</span></div>
      {mix_row}"""


def lodesod_html(row) -> str:
    """Who works here — worker-origin profile from census home↔work pairs."""
    w = _g(row, "lodesod_workers")
    if w is None:
        return ""
    inc = _g(row, "lodesod_home_income")
    far = _g(row, "lodesod_pct_far")
    nov = _g(row, "lodesod_home_novehicle_pct")
    cty = _g(row, "lodesod_top_origin_county")
    shr = _g(row, "lodesod_top_origin_share")
    origin = (f'<div class="jx-row"><span class="k">Top origin county</span>'
              f'<span class="v">{cty} · {_fmt(shr, "{:.0f}%")}</span></div>' if cty else "")
    return f"""
      <div class="jx-lab jx-sec">◆ Who works here — LODES O-D</div>
      <div class="jx-row"><span class="k">Workers</span><span class="v"><b>{_fmt(w)}</b></span></div>
      <div class="jx-row"><span class="k">Live in tracts of</span><span class="v"><b>{_fmt(inc, '${:,.0f}')}</b> median income</span></div>
      <div class="jx-row"><span class="k">Commute &gt;25 km</span><span class="v">{_fmt(far, '{:.0f}%')} · {_fmt(nov, '{:.0f}%')} carless homes</span></div>
      {origin}"""


def gdelt_html(row) -> str:
    """Geocoded news attention — counts by category, one tone number."""
    n = _g(row, "gdelt_events")
    if n is None:
        return ""
    tone = _g(row, "gdelt_tone_mean")
    return f"""
      <div class="jx-lab jx-sec">◆ News attention — GDELT · place-level</div>
      <div class="jx-row"><span class="k">Geocoded events (180 d)</span><span class="v"><b>{_fmt(n)}</b> · {_fmt(_g(row, 'gdelt_days'), '{:.0f}')} days</span></div>
      <div class="jx-row"><span class="k">Protest · conflict</span><span class="v">{_fmt(_g(row, 'gdelt_protest'))} · {_fmt(_g(row, 'gdelt_conflict'))}</span></div>
      <div class="jx-row"><span class="k">Mean tone</span><span class="v">{_fmt(tone, '{:+.1f}')}</span></div>"""


def eaglei_html(row) -> str:
    """Power reliability — EAGLE-I county outage history. Headline is the
    SAIDI-like hours-dark-per-customer; raw any-customer-out hours saturate
    for big counties and are not shown."""
    h = _g(row, "eaglei_hrs_dark_per_cust_yr")
    if h is None:
        return ""
    return f"""
      <div class="jx-lab jx-sec">◆ Power reliability — EAGLE-I · county</div>
      <div class="jx-row"><span class="k">Hours dark / customer</span><span class="v"><b>{h:,.1f}</b> /yr</span></div>
      <div class="jx-row"><span class="k">Worst event</span><span class="v">{_fmt(_g(row, 'eaglei_max_out'))} customers out</span></div>"""


def mly_html(row) -> str:
    """Street furniture — Mapillary detections (coverage-biased: counts are
    visibility-weighted by how much imagery exists)."""
    tot = _g(row, "mly_features_total")
    if tot is None:
        return ""
    return f"""
      <div class="jx-lab jx-sec">◆ Streetscape — Mapillary · detections</div>
      <div class="jx-row"><span class="k">Crosswalks</span><span class="v"><b>{_fmt(_g(row, 'mly_crosswalks'))}</b></span></div>
      <div class="jx-row"><span class="k">Lights · poles</span><span class="v">{_fmt(_g(row, 'mly_streetlights'))} · {_fmt(_g(row, 'mly_poles'))}</span></div>
      <div class="jx-row"><span class="k">Cones (construction)</span><span class="v">{_fmt(_g(row, 'mly_cones'))}</span></div>
      <div class="jx-row"><span class="k">All detections</span><span class="v">{_fmt(tot)}</span></div>"""


def osmped_html(row) -> str:
    """Pedestrian/curb attributes — OSM tags + NYC planimetric width, with
    decision ratios (share of controlled crossings, share of accessible kerbs)."""
    if all(_g(row, c) is None for c in ("osm_sidewalk_len_m", "osm_cross_signalized",
                                        "osm_kerb_lowered", "osm_cross_marked",
                                        "swd_width_eff_m")):
        return ""
    rows = []
    w = _g(row, "swd_width_eff_m")
    if w is not None:
        rows.append(f'<div class="jx-row"><span class="k">Effective width</span>'
                    f'<span class="v"><b>{w:.1f} m</b> · 2·area/perimeter, NYC planimetrics</span></div>')
    ow = _g(row, "osmw_width_med_m")
    if ow is not None:
        rows.append(f'<div class="jx-row"><span class="k">Tagged width</span>'
                    f'<span class="v"><b>{ow:.1f} m</b> · OSM width=*, n={_fmt(_g(row, "osmw_width_n"), "{:.0f}")}</span></div>')
    sm = _g(row, "osmw_smooth_med")
    if sm is not None:
        lbl = ["excellent", "good", "intermediate", "bad", "very bad",
               "horrible", "very horrible", "impassable"][int(min(7, max(0, round(sm))))]
        rows.append(f'<div class="jx-row"><span class="k">Tagged smoothness</span>'
                    f'<span class="v">{lbl} · n={_fmt(_g(row, "osmw_smooth_n"), "{:.0f}")}</span></div>')
    sl = _g(row, "osm_sidewalk_len_m")
    if sl is not None:
        rows.append(f'<div class="jx-row"><span class="k">Sidewalk mapped</span>'
                    f'<span class="v"><b>{_fmt(sl)}</b> m</span></div>')
    sig, mk, um = (_g(row, "osm_cross_signalized") or 0, _g(row, "osm_cross_marked") or 0,
                   _g(row, "osm_cross_unmarked") or 0)
    if sig + mk + um > 0:
        ctrl = 100 * (sig + mk) / (sig + mk + um)
        rows.append(f'<div class="jx-row"><span class="k">Crossings</span>'
                    f'<span class="v">{sig:.0f} signal · {mk:.0f} marked · {um:.0f} unmarked → '
                    f'<b>{ctrl:.0f}%</b> controlled</span></div>')
    lo, fl_, ra = (_g(row, "osm_kerb_lowered") or 0, _g(row, "osm_kerb_flush") or 0,
                   _g(row, "osm_kerb_raised") or 0)
    if lo + fl_ + ra > 0:
        acc = 100 * (lo + fl_) / (lo + fl_ + ra)
        rows.append(f'<div class="jx-row"><span class="k">Kerbs</span>'
                    f'<span class="v">{lo:.0f} lowered · {fl_:.0f} flush · {ra:.0f} raised → '
                    f'<b>{acc:.0f}%</b> rollable</span></div>')
    t = _g(row, "osm_tactile")
    if t is not None and t > 0:
        rows.append(f'<div class="jx-row"><span class="k">Tactile paving</span>'
                    f'<span class="v">{t:.0f}</span></div>')
    return '<div class="jx-lab jx-sec">◆ Pedestrian — OSM / NYC planimetrics</div>' + "".join(rows)


def wzdx_html(row) -> str:
    """Live work zones — WZDx snapshot."""
    z = _g(row, "wzdx_zones")
    if z is None:
        return ""
    snap = str(_g(row, "wzdx_snapshot") or "")[:10]
    return f"""
      <div class="jx-lab jx-sec">◆ Work zones — WZDx · snapshot {snap}</div>
      <div class="jx-row"><span class="k">Active zones</span><span class="v"><b>{_fmt(z)}</b> · {_fmt(_g(row, 'wzdx_lane_impact'))} lane-closing</span></div>
      <div class="jx-row"><span class="k">Type</span><span class="v">{_g(row, 'wzdx_top_type') or '—'}</span></div>"""


def phys_html(row) -> str:
    """Sidewalk physics — measured walk-vibration where we walked, 3DEP slope
    everywhere in the pilot grid. Honest flags: measured vs inferred."""
    sl = _g(row, "slope_pct_med")
    rg = _g(row, "swp_rough_g_rms")
    if sl is None and rg is None:
        return ""
    rows = []
    if rg is not None:
        rows.append(f'<div class="jx-row"><span class="k">Walk vibration</span>'
                    f'<span class="v"><b>{rg:.2f} g</b> RMS · measured, '
                    f'{_fmt(_g(row, "swp_walks"), "{:.0f}")} walk(s)</span></div>')
    if sl is not None:
        rows.append(f'<div class="jx-row"><span class="k">Terrain slope</span>'
                    f'<span class="v"><b>{sl:.1f}%</b> med · {_fmt(_g(row, "slope_pct_p95"), "{:.0f}")}% p95 · inferred (3DEP 10 m)</span></div>')
        rows.append(f'<div class="jx-row"><span class="k">Elevation</span>'
                    f'<span class="v">{_fmt(_g(row, "elev_m_med"), "{:.0f}")} m · range {_fmt(_g(row, "elev_range_m"), "{:.0f}")} m</span></div>')
    return ('<div class="jx-lab jx-sec">\u25c6 Sidewalk physics \u2014 walks / 3DEP</div>' + "".join(rows))


def flows_html(row) -> str:
    if _g(row, "wm_inflow") is None and _g(row, "wm_outflow") is None:
        return ""
    return f"""
      <div class="jx-lab jx-sec">◆ Flows — O-D panel</div>
      <div class="jx-row"><span class="k">In / out</span><span class="v"><b>{_fmt(_g(row, 'wm_inflow'))}</b> · {_fmt(_g(row, 'wm_outflow'))} trips</span></div>
      <div class="jx-row"><span class="k">Destination diversity</span><span class="v">{_fmt(_g(row, 'wm_dest_diversity'), '{:.0f}')} unique</span></div>"""


def render_card(row) -> None:
    """City card — NYC gets Vision Zero + streetscape."""
    street = phys_html(row)
    sw = _g(row, "nycsw_sidewalk_length_m")
    if sw is not None:
        street = (f'<div class="jx-lab jx-sec">◆ Streetscape — NYC DOT</div>'
                  f'<div class="jx-row"><span class="k">Sidewalk</span><span class="v">{_fmt(sw)} m</span></div>')
    st.markdown(base_html(row, top=nyc_safety_html(row), street=street), unsafe_allow_html=True)


def render_us_card(row) -> None:
    """US drill card — FARS + rates on top; census, NRI, jobs, worker-origins,
    traffic, work zones, flows, news, power, streetscape, pedestrian sections
    light up as their parquets exist."""
    st.markdown(base_html(row,
                          top=fars_html(row) + rates_html(row),
                          census=acs_html(row) + nri_html(row) + lodes_html(row) + lodesod_html(row),
                          traffic=hpms_html(row) + wzdx_html(row),
                          flows=flows_html(row) + gdelt_html(row) + eaglei_html(row),
                          street=phys_html(row) + mly_html(row) + osmped_html(row)),
                unsafe_allow_html=True)


def parse_selection(event, current):
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    objs = sel.get("objects") if isinstance(sel, dict) else getattr(sel, "objects", None)
    if not objs:
        return None
    for _lid, ol in objs.items():
        if ol and isinstance(ol[0], dict):
            h = ol[0].get("h3_index")
            if h and h != current:
                return h
    return None


def hex_map(df, lat, lon, zoom, pitch, key, tooltip_html, focus=None, bearing=0.0):
    layers = [pdk.Layer(
        "H3HexagonLayer", id="atlas", data=df, get_hexagon="h3_index",
        get_fill_color="_color", get_elevation="_elev", elevation_scale=1,
        extruded=True, pickable=True, auto_highlight=True, coverage=0.9,
        # tween color/height when the shade layer changes (same mounted deck)
        transitions={"getFillColor": 450, "getElevation": 450},
    )]
    if focus is not None and len(focus):
        # persistent selection accent — stroked, raised, full cobalt
        fdf = focus.copy()
        fdf["_elev_f"] = fdf["_elev"] * 1.05 + 6.0
        layers.append(pdk.Layer(
            "H3HexagonLayer", id="atlas-focus", data=fdf, get_hexagon="h3_index",
            get_fill_color=[30, 58, 138, 235], get_elevation="_elev_f",
            elevation_scale=1, extruded=True, pickable=False, coverage=0.98,
            stroked=True, get_line_color=[246, 244, 239, 255], line_width_min_pixels=2,
        ))
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom,
                                         pitch=pitch, bearing=bearing),
        map_style="light",
        tooltip={"html": tooltip_html,
                 "style": {"backgroundColor": "#F6F4EF", "color": "#1A1815", "fontSize": "12px",
                           "padding": "10px", "borderRadius": "4px", "border": "1px solid #E0DDD4"}},
    )
    return st.pydeck_chart(deck, use_container_width=True, height=580,
                           on_select="rerun", selection_mode="single-object", key=key)



# ---- page ----
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:14px;margin:0 0 2px;">'
    '<span style="font-family:ui-monospace,Menlo,monospace;font-size:.78rem;'
    'letter-spacing:.3em;font-weight:700;">JANUS</span>'
    '<span style="font-family:ui-monospace,Menlo,monospace;font-size:.62rem;'
    'letter-spacing:.14em;text-transform:uppercase;color:#7A7F85;">Atlas · New York</span>'
    '<span style="margin-left:auto;font-family:ui-monospace,Menlo,monospace;'
    'font-size:.62rem;letter-spacing:.08em;">'
    '<a href="https://shan-kmr.github.io/geo-landing/" target="_blank" '
    'style="color:#7A7F85!important;margin-right:12px;">janus ↗</a>'
    '<a href="https://github.com/shan-kmr/curbai" target="_blank" '
    'style="color:#7A7F85!important;">code ↗</a></span></div>'
    '<div style="font-size:.92rem;color:#4A4E54;margin:0 0 10px;">'
    'Every cell of the city, decoded — raw open data, live movement, real buildings. '
    'Zoom to split the tiles; click a street cell for its full card.</div>'
    '<div style="height:1px;background:#E9EAEC;margin:0 0 14px;"></div>',
    unsafe_allow_html=True)


# ---- the tiled map: one continuous res-5 → res-9 ladder ----
from curbai.janusmap import janusmap  # noqa: E402

# real NYC footprints (Plate II) — set after the public tiles upload
BUILDINGS_URL = os.environ.get(
    "JANUS_BUILDINGS_URL",
    "https://huggingface.co/datasets/skay97/curbai-tiles/resolve/main/nyc_buildings.pmtiles")

LODF = {6: DATADIR / "us_lod" / "r6.parquet",
        7: DATADIR / "us_lod" / "r7.parquet",
        8: DATADIR / "us_lod" / "r8.parquet"}
TILE_LAYERS = {
    "Structure · none": (None, ""),
    "Population": ("kontur_population", "residents"),
    "POIs / places": ("poi_count", "POIs"),
    "Buildings": ("building_count", "buildings"),
    "Movement · visits": ("wt_visit_count", "visits"),
    "Roads": ("road_count", "road segments"),
    "Night-lights": ("nightlight_2021", "brightness"),
}

@st.cache_data(show_spinner=False)
def r5_payload(col: str | None) -> dict:
    df = pd.read_parquet(US_R5)
    # NYC focus (temporary): only regional res-5 parents ship to the client
    df = df[(df.center_lat.between(40.35, 41.10)) & (df.center_lon.between(-74.50, -73.45))]
    out = {"h3": df.h3_index.tolist(),
           "lat": df.center_lat.round(4).tolist(),
           "lon": df.center_lon.round(4).tolist(), "val": None}
    if col and col in df.columns:
        out["val"] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2).tolist()
    return out

@st.cache_data(show_spinner=False)
def lod_vmax(col: str) -> dict:
    con = duckdb.connect()
    vm = {"5": float(pd.read_parquet(US_R5, columns=[col])[col].max() or 1)}
    for res, f in LODF.items():
        vm[str(res)] = float(con.execute(
            f"SELECT max({col}) FROM read_parquet('{f.as_posix()}')").fetchone()[0] or 1)
    vm["9"] = float(con.execute(
        f"SELECT max({col}) FROM read_parquet('{US_R9_GLOB}')").fetchone()[0] or 1)
    return vm

@st.cache_data(show_spinner=False)
def fetch_chunk(res: int, parent: str, col: str | None) -> dict:
    con = duckdb.connect()
    vcol = f", {col} AS val" if col else ""
    if res == 9:
        q = (f"SELECT h3_index{vcol}, building_count, avg_floors, max_height "
             f"FROM read_parquet('{US_R9_GLOB}') WHERE res5 = ?")
    else:
        q = f"SELECT h3_index{vcol} FROM read_parquet('{LODF[res].as_posix()}') WHERE res5 = ?"
    d = con.execute(q, [parent]).df()
    out = {"h3": d.h3_index.tolist(),
           "val": d.val.fillna(0).round(2).tolist() if col else None}
    if res == 9:
        out["bc"] = d.building_count.fillna(0).astype(int).tolist()
        out["fl"] = d.avg_floors.fillna(0).round(1).tolist()
        out["mh"] = d.max_height.fillna(0).round(0).tolist()
    return out

@st.cache_data(ttl=18, show_spinner=False)
def live_buses() -> list:
    try:
        import requests as rq
        from google.transit import gtfs_realtime_pb2
        r = rq.get("https://gtfsrt.prod.obanyc.com/vehiclePositions", timeout=8)
        f = gtfs_realtime_pb2.FeedMessage(); f.ParseFromString(r.content)
        out = []
        for e in f.entity:
            v = e.vehicle
            if (v.position.latitude and 40.45 < v.position.latitude < 41.0
                    and -74.35 < v.position.longitude < -73.6):
                out.append([round(v.position.longitude, 5), round(v.position.latitude, 5),
                            int(v.position.bearing or 0), v.vehicle.id or e.id])
        return out
    except Exception:
        return []

@st.cache_data(ttl=55, show_spinner=False)
def live_bikes() -> list:
    try:
        import requests as rq
        info = rq.get("https://gbfs.citibikenyc.com/gbfs/en/station_information.json", timeout=8).json()["data"]["stations"]
        stat = rq.get("https://gbfs.citibikenyc.com/gbfs/en/station_status.json", timeout=8).json()["data"]["stations"]
        cap = {s["station_id"]: (s["lat"], s["lon"], max(1, s.get("capacity", 1))) for s in info}
        out = []
        for s_ in stat:
            c = cap.get(s_["station_id"])
            if not c:
                continue
            la, lo, capn = c
            out.append([round(lo, 5), round(la, 5),
                        round(min(1.0, s_.get("num_bikes_available", 0) / capn), 2)])
        return out
    except Exception:
        return []

layer_name = st.selectbox("Shade the tiles by", list(TILE_LAYERS), index=0, key="jm_layer")
col, unit = TILE_LAYERS[layer_name]

# chunk cache — refetch (cheap, cached) when the shade column changes
if st.session_state.get("jm_col") != col:
    st.session_state["jm_col"] = col
    st.session_state["jm_chunks"] = {
        res: {p: fetch_chunk(int(res), p, col) for p in parents}
        for res, parents in st.session_state.get("jm_parents", {}).items()}
st.session_state.setdefault("jm_parents", {})
st.session_state.setdefault("jm_chunks", {})

focus = st.session_state.get("us_focus9")
st.caption("New York · zoom to split the tiles (res 5 → 9) · click a tile to dive, "
           "click a street-level cell for its card · buildings appear up close"
           + (f" · shaded by {layer_name.lower()}" if col else ""))

left, right = st.columns([2, 1], gap="large")
with left:
    @st.fragment(run_every=20)
    def map_fragment():
        live = {"buses": live_buses(), "bikes": live_bikes(), "ts": int(time.time())}
        ev = janusmap(
            r5=r5_payload(col),
            chunks=st.session_state["jm_chunks"],
            layer={"col": col, "label": unit, "vmax": (lod_vmax(col) if col else {})},
            focus=st.session_state.get("us_focus9"),
            buildings_url=BUILDINGS_URL, live=live, height=580, key="jm_map")
        if ev and ev.get("nonce") != st.session_state.get("jm_nonce"):
            st.session_state["jm_nonce"] = ev.get("nonce")
            if ev.get("t") == "need":
                res = str(ev["res"])
                parents = st.session_state["jm_parents"].setdefault(res, set())
                chunks = st.session_state["jm_chunks"].setdefault(res, {})
                for p in ev.get("parents", []):
                    if p not in parents:
                        parents.add(p)
                        chunks[p] = fetch_chunk(int(res), p, col)
                st.rerun()
            elif ev.get("t") == "select":
                st.session_state["us_focus9"] = ev.get("h3")
                st.rerun()
    map_fragment()
with right:
    if focus:
        kids = load_us_children(h3.h3_to_parent(focus, 5))
        frow = kids[kids.h3_index == focus]
        if len(frow):
            render_us_card(frow.iloc[0])
        st.caption("Raw open data, keyed to one H3 cell. No scores.")
    else:
        df5 = load_us_overview()
        tot = {"Res-5 tiles": f"{len(df5):,}",
               "Res-9 inside": f"{int(df5.n_res9.sum()):,}",
               "POIs": f"{int(df5.poi_count.fillna(0).sum()):,}",
               "Buildings": f"{int(df5.building_count.fillna(0).sum()):,}"}
        if "fars_crashes" in df5.columns:
            tot["Fatal crashes 22–24"] = f"{int(df5.fars_crashes.fillna(0).sum()):,}"
        rows = "".join(f'<div class="jx-row"><span class="k">{k}</span>'
                       f'<span class="v"><b>{v}</b></span></div>' for k, v in tot.items())
        st.markdown(f"""
        <div class="jx-card">
          <div class="jx-cid">United States · tiled</div>
          <div class="jx-lab" style="margin-top:6px">◆ On this map</div>
          {rows}
          <div class="jx-lab jx-sec">◆ How it works</div>
          <div class="jx-txt">Zoom and the tiles split — res-5 country tiles down to
          174 m street cells, with buildings rising up close. Click a street cell
          and its full raw-data card opens here.</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Raw open data, keyed to H3. No scores.")

