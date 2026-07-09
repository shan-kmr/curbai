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
import sys
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
US = "United States"

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


# ---------- loaders ----------

@st.cache_data(show_spinner="Loading city grid…")
def load(slug: str) -> pd.DataFrame:
    df = pd.read_parquet(DATADIR / f"{slug}_base.parquet")
    if slug == "nyc" and (DATADIR / "nyc_crashes_h3.parquet").exists():
        crash = pd.read_parquet(DATADIR / "nyc_crashes_h3.parquet")
        cols = [c for c in crash.columns if c not in ("center_lat", "center_lon")]
        df = df.merge(crash[cols], on="h3_index", how="left")
    return df


@st.cache_data(show_spinner="Loading the United States…")
def load_us_overview() -> pd.DataFrame:
    df = pd.read_parquet(US_R5)
    if FARS.exists():
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
    for side in (FARS, ACS, HPMS):
        if not side.exists():
            continue
        s = con.execute(
            f"SELECT t.* FROM read_parquet('{side.as_posix()}') t "
            "JOIN kids USING (h3_index)"
        ).df()
        s = s.drop(columns=[c for c in ("center_lat", "center_lon") if c in s])
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
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


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

def base_html(row, top: str = "", census: str = "", traffic: str = "", street: str = "") -> str:
    """The shared raw-data card. Optional fragments slot in at fixed points:
    top (safety), census (after Who's here), traffic (after Roads), street
    (before Climate)."""
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

    return f"""
    <div class="jx-card">
      <div class="jx-cid">R9 · {row.h3_index[-12:]} · 0.105 km²</div>
      {top}
      <div class="jx-lab jx-sec">◆ Who's here — Kontur</div>
      <div class="jx-row"><span class="k">Population</span><span class="v"><b>{_fmt(pop)}</b> modeled residents</span></div>
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
      {street}
      <div class="jx-lab jx-sec">◆ Climate — WorldClim</div>
      <div class="jx-row"><span class="k">Temp · rain</span><span class="v">{_fmt(temp, '{:.0f}')}°C · {_fmt(precip)} mm/yr</span></div>
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
    return f"""
      <div class="jx-lab jx-sec">◆ Traffic — FHWA HPMS</div>
      <div class="jx-row"><span class="k">AADT (busiest)</span><span class="v"><b>{_fmt(a)}</b> veh/day</span></div>
      <div class="jx-row"><span class="k">Covered segments</span><span class="v">{_fmt(_g(row, 'hpms_seg_count'))}</span></div>
      {truck_row}"""


def render_card(row) -> None:
    """City card — NYC gets Vision Zero + streetscape."""
    street = ""
    sw = _g(row, "nycsw_sidewalk_length_m")
    if sw is not None:
        street = (f'<div class="jx-lab jx-sec">◆ Streetscape — NYC DOT</div>'
                  f'<div class="jx-row"><span class="k">Sidewalk</span><span class="v">{_fmt(sw)} m</span></div>')
    st.markdown(base_html(row, top=nyc_safety_html(row), street=street), unsafe_allow_html=True)


def render_us_card(row) -> None:
    """US drill card — FARS on top, census + traffic when their data exists."""
    st.markdown(base_html(row, top=fars_html(row), census=acs_html(row),
                          traffic=hpms_html(row)), unsafe_allow_html=True)


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


def hex_map(df, lat, lon, zoom, pitch, key, tooltip_html):
    deck = pdk.Deck(
        layers=[pdk.Layer(
            "H3HexagonLayer", id="atlas", data=df, get_hexagon="h3_index",
            get_fill_color="_color", get_elevation="_elev", elevation_scale=1,
            extruded=True, pickable=True, auto_highlight=True, coverage=0.9,
        )],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=pitch),
        map_style="light",
        tooltip={"html": tooltip_html,
                 "style": {"backgroundColor": "#F6F4EF", "color": "#1A1815", "fontSize": "12px",
                           "padding": "10px", "borderRadius": "4px", "border": "1px solid #E0DDD4"}},
    )
    return st.pydeck_chart(deck, use_container_width=True, height=580,
                           on_select="rerun", selection_mode="single-object", key=key)


# ---- page ----
ui.header(
    "Janus · The Hex Atlas",
    "Every hex, decoded.",
    "Click a cell — crashes, people, buildings, places, movement, the road network — "
    "all raw, all open data. The whole United States, drillable to 174 m, plus ten deep cities. "
    "<span class='sig'>Consented movement in. Defensible signal out.</span>",
)

scopes = ([US] if US_R5.exists() else []) + list(CITIES)
scope = st.sidebar.radio("Where", scopes, index=0)

if scope == US:
    r5_sel = st.session_state.get("us_r5_sel")

    if r5_sel is None:
        # ---- national overview (res-5) ----
        df5 = load_us_overview()
        opts = dict(LAYERS)
        if "fars_crashes" in df5.columns:
            opts = {"Fatal crashes 2022–24": ("fars_crashes", "{:,.0f} fatal crashes"), **LAYERS}
        layer_name = st.selectbox("Shade the map by", list(opts), index=0, key="atlas_layer_us")
        df5 = apply_layer(df5, *opts[layer_name], elev_max=45000)

        extra = (f" · {int(df5.fars_crashes.sum()):,} fatal crashes · {int(df5.fars_killed.sum()):,} killed"
                 if "fars_crashes" in df5.columns else "")
        st.caption(f"United States · {len(df5):,} res-5 cells (~252 km²){extra} · "
                   f"shaded by {layer_name.lower()} · click any cell to open it at 174 m")

        left, right = st.columns([2, 1], gap="large")
        with left:
            event = hex_map(df5, 39.5, -98.0, 3.4, 30, "us_deck",
                            "<b>{_val_str}</b><br/>click to open this cell at 174 m")
            new = parse_selection(event, None)
            if new:
                st.session_state["us_r5_sel"] = new
                st.rerun()
        with right:
            tot = {
                "Res-5 cells": f"{len(df5):,}",
                "Res-9 inside": f"{int(df5.n_res9.sum()):,}",
                "POIs": f"{int(df5.poi_count.fillna(0).sum()):,}",
                "Buildings": f"{int(df5.building_count.fillna(0).sum()):,}",
                "Road segments": f"{int(df5.road_count.fillna(0).sum()):,}",
            }
            if "fars_crashes" in df5.columns:
                tot["Fatal crashes 22–24"] = f"{int(df5.fars_crashes.fillna(0).sum()):,}"
                tot["Killed"] = f"{int(df5.fars_killed.fillna(0).sum()):,}"
            rows = "".join(f'<div class="jx-row"><span class="k">{k}</span><span class="v"><b>{v}</b></span></div>'
                           for k, v in tot.items())
            st.markdown(f"""
            <div class="jx-card">
              <div class="jx-cid">United States · res-5 overview</div>
              <div class="jx-lab" style="margin-top:6px">◆ On this map</div>
              {rows}
              <div class="jx-lab jx-sec">◆ How it works</div>
              <div class="jx-txt">Every cell is ~252 km². Click one and it opens as ~2,300
              res-9 hexes at 174 m — each with its own raw data card: FARS fatal crashes,
              census, buildings, places, movement, roads, climate.</div>
            </div>""", unsafe_allow_html=True)
            st.caption("Raw open data, keyed to H3. No scores.")

    else:
        # ---- drill view (res-9 inside one res-5 cell) ----
        if st.button("◀ Back to the United States"):
            st.session_state.pop("us_r5_sel", None)
            st.session_state.pop("us_focus", None)
            st.rerun()

        kids = load_us_children(r5_sel)
        opts = dict(LAYERS)
        if "fars_crashes" in kids.columns:
            opts = {"Fatal crashes 2022–24": ("fars_crashes", "{:,.0f} fatal crashes"), **LAYERS}
        layer_name = st.selectbox("Shade the map by", list(opts), index=0, key="atlas_layer_us_drill")
        col, valfmt = opts[layer_name]
        kids = apply_layer(kids, col, valfmt)

        top = kids.nlargest(1, col)  # empty if the column is all-NaN (e.g. no fatal crashes here)
        default_focus = ((top.iloc[0] if len(top) else kids.iloc[0]).h3_index
                         if len(kids) else None)
        if st.session_state.get("us_r5_prev") != r5_sel:
            st.session_state["us_r5_prev"] = r5_sel
            st.session_state["us_focus"] = default_focus

        clat, clon = h3.h3_to_geo(r5_sel)
        extra = (f" · {int(kids.fars_crashes.fillna(0).sum()):,} fatal crashes"
                 if "fars_crashes" in kids.columns else "")
        st.caption(f"United States · cell {r5_sel[-9:]} · {len(kids):,} res-9 hexes{extra} · "
                   f"shaded by {layer_name.lower()} · click any hex for the full card")

        left, right = st.columns([2, 1], gap="large")
        with left:
            event = hex_map(kids, clat, clon, 9.4, 45, "us_drill_deck",
                            "<b>{_val_str}</b><br/>click for the full card")
            new = parse_selection(event, st.session_state.get("us_focus"))
            if new:
                st.session_state["us_focus"] = new
                st.rerun()
        with right:
            focus = st.session_state.get("us_focus", default_focus)
            frow = kids[kids.h3_index == focus]
            if len(frow) or len(kids):
                render_us_card(frow.iloc[0] if len(frow) else kids.iloc[0])
            st.caption("Raw open data, keyed to one H3 cell. No scores.")

else:
    # ---- city view (res-9 direct) ----
    slug, clat, clon, czoom = CITIES[scope]
    df = load(slug)

    opts = dict(LAYERS)
    if "crashes" in df.columns:
        opts = {"Collisions": ("crashes", "{:,.0f} collisions"), **LAYERS}
    layer_name = st.selectbox("Shade the map by", list(opts), index=0, key="atlas_layer")
    df = apply_layer(df, *opts[layer_name])

    default_focus = df.nlargest(1, "poi_count").iloc[0].h3_index
    if st.session_state.get("atlas_city") != slug:
        st.session_state["atlas_city"] = slug
        st.session_state["atlas_focus"] = default_focus

    extra = f" · {int(df.crashes.sum()):,} collisions" if "crashes" in df.columns else ""
    st.caption(f"{scope} · {len(df):,} hexes · shaded by {layer_name.lower()}{extra} · click any cell for the full card")

    left, right = st.columns([2, 1], gap="large")
    with left:
        event = hex_map(df, clat, clon, czoom, 45, "atlas_deck",
                        "<b>{_val_str}</b><br/>click for the full card")
        new = parse_selection(event, st.session_state.get("atlas_focus"))
        if new:
            st.session_state["atlas_focus"] = new
            st.rerun()
    with right:
        focus = st.session_state.get("atlas_focus", default_focus)
        frow = df[df.h3_index == focus]
        render_card(frow.iloc[0] if len(frow) else df.iloc[0])
        st.caption("Raw open data, keyed to one H3 cell. No scores.")
