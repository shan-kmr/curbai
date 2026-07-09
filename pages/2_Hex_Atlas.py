"""
The Hex Atlas — New York. First raw layer: crashes.

Click any hex → the raw NYC Vision Zero breakdown for that cell: count, deaths,
mode (pedestrian / cyclist / motorist), peak hour + day, top contributing
factors. No score — the raw detail is the product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curbai import ui  # noqa: E402

st.set_page_config(page_title="Janus — The Hex Atlas", page_icon="🔬", layout="wide")
ui.inject()
ui.header(
    "Janus · The Hex Atlas — New York",
    "Every hex, decoded.",
    "Click a cell — crashes, who lives there, the built form — all raw, all open data. "
    "The map is shaded by collision count. <span class='sig'>Consented movement in. Defensible signal out.</span>",
)

DATA = Path(__file__).resolve().parents[1] / "data" / "nyc_crashes_h3.parquet"
BASE = Path(__file__).resolve().parents[1] / "data" / "nyc_base_h3.parquet"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner="Loading crash grid…")
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    if BASE.exists():
        df = df.merge(pd.read_parquet(BASE), on="h3_index", how="left")
    mx = np.log1p(df.crashes.max())
    df["_norm"] = np.log1p(df.crashes) / mx
    df["_color"] = df["_norm"].apply(ui.count_color)
    df["_elev"] = df["crashes"].astype(float) * 1.4
    return df


def fmt_hour(h: int) -> str:
    if h is None or h < 0:
        return "—"
    ampm = "am" if h < 12 else "pm"
    return f"{(h % 12) or 12}{ampm}"


def _g(row: pd.Series, col: str):
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _fmt(v, f: str = "{:,.0f}") -> str:
    return "—" if v is None else f.format(v)


def _dist(km) -> str:
    if km is None:
        return "—"
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.1f} km"


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


def render_card(row: pd.Series) -> None:
    hh = json.loads(row.hour_hist)
    facs = json.loads(row.top_factors)
    mx = max(hh) or 1
    bars = "".join(f'<div class="b" style="height:{max(2, int(v / mx * 100))}%"></div>' for v in hh)
    fac_html = "".join(
        f'<div class="jx-fac"><span>{k.title()}</span><span class="c">{v}</span></div>'
        for k, v in facs
    ) or '<div class="jx-fac"><span class="c">— none coded —</span></div>'

    pop = _g(row, "kontur_population") or _g(row, "population")
    nl = _g(row, "nightlight_2021")
    bc, fl, mh = _g(row, "building_count"), _g(row, "avg_floors"), _g(row, "max_height")
    poi, cat, ntr = _g(row, "poi_count"), _cat(_g(row, "top_category")), _g(row, "count_transit")
    dh, dp, dt = _g(row, "dist_hospital_km"), _g(row, "dist_park_km"), _g(row, "dist_transit_km")
    rc, rpri, rres = _g(row, "road_count"), _g(row, "road_primary"), _g(row, "road_residential")
    vis, pkh, nf, rog = (_g(row, "wt_visit_count"), _g(row, "wt_peak_hour"),
                         _g(row, "wt_night_fraction"), _g(row, "wt_radius_of_gyration_km"))
    sw = _g(row, "nycsw_sidewalk_length_m")
    temp, precip = _g(row, "annual_mean_temp"), _g(row, "annual_precipitation")
    txt = _g(row, "llmgeovec_text")
    peak_v = fmt_hour(int(pkh)) if pkh is not None else "—"

    st.markdown(f"""
    <div class="jx-card">
      <div class="jx-cid">R9 · {row.h3_index[-12:]} · New York · 0.105 km²</div>
      <div class="jx-lab" style="margin-top:6px">◆ Safety — NYPD Vision Zero</div>
      <div class="jx-big">{row.crashes:,}<small> collisions</small></div>
      <div class="jx-row"><span class="k">Killed / injured</span><span class="v"><b>{row.killed}</b> · {row.injured:,}</span></div>
      <div class="jx-row"><span class="k">Pedestrian</span><span class="v">{row.ped_inj} inj · {row.ped_kill} killed</span></div>
      <div class="jx-row"><span class="k">Peak</span><span class="v"><b>{DOW[row.peak_dow] if row.peak_dow>=0 else '—'} {fmt_hour(row.peak_hour)}</b></span></div>
      <div class="jx-hist">{bars}</div>
      <div class="jx-hticks"><span>12a</span><span>6a</span><span>12p</span><span>6p</span><span>11p</span></div>
      <div class="jx-lab jx-sec" style="margin-top:6px">Top contributing factors</div>{fac_html}

      <div class="jx-lab jx-sec">◆ Who's here — Kontur</div>
      <div class="jx-row"><span class="k">Population</span><span class="v"><b>{_fmt(pop)}</b> est. residents</span></div>
      <div class="jx-row"><span class="k">Night-lights</span><span class="v">{_fmt(nl, '{:.0f}')}</span></div>

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

      <div class="jx-lab jx-sec">◆ Movement — trajectory data</div>
      <div class="jx-row"><span class="k">Visits</span><span class="v"><b>{_fmt(vis)}</b> · peak {peak_v}</span></div>
      <div class="jx-row"><span class="k">Night · radius</span><span class="v">{_fmt(nf*100 if nf is not None else None, '{:.0f}')}% · {_fmt(rog, '{:.1f}')} km</span></div>

      <div class="jx-lab jx-sec">◆ Streetscape — NYC DOT</div>
      <div class="jx-row"><span class="k">Sidewalk</span><span class="v">{_fmt(sw)} m</span></div>

      <div class="jx-lab jx-sec">◆ Climate — WorldClim</div>
      <div class="jx-row"><span class="k">Temp · rain</span><span class="v">{_fmt(temp, '{:.0f}')}°C · {_fmt(precip)} mm/yr</span></div>

      <div class="jx-lab jx-sec">◆ The model reads</div>
      <div class="jx-txt">{(str(txt)[:200] + '…') if txt else '—'}</div>
    </div>
    """, unsafe_allow_html=True)


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


df = load()
if "atlas_focus" not in st.session_state:
    st.session_state["atlas_focus"] = df.nlargest(1, "crashes").iloc[0].h3_index

st.caption(f"{len(df):,} hexes · {int(df.crashes.sum()):,} collisions · {int(df.killed.sum()):,} killed · source: NYC Open Data h9gi-nx95")

left, right = st.columns([2, 1], gap="large")

with left:
    focus = st.session_state["atlas_focus"]
    frow = df[df.h3_index == focus]
    clat = float(frow.center_lat.iloc[0]) if len(frow) else 40.71
    clon = float(frow.center_lon.iloc[0]) if len(frow) else -73.94
    deck = pdk.Deck(
        layers=[pdk.Layer(
            "H3HexagonLayer", id="crash", data=df, get_hexagon="h3_index",
            get_fill_color="_color", get_elevation="_elev", elevation_scale=1,
            extruded=True, pickable=True, auto_highlight=True, coverage=0.9,
        )],
        initial_view_state=pdk.ViewState(latitude=40.715, longitude=-73.945, zoom=10.1, pitch=45, bearing=0),
        map_style="light",
        tooltip={"html": "<b>{crashes} collisions</b><br/>{killed} killed · click for detail",
                 "style": {"backgroundColor": "#F6F4EF", "color": "#1A1815", "fontSize": "12px",
                           "padding": "10px", "borderRadius": "4px", "border": "1px solid #E0DDD4"}},
    )
    event = st.pydeck_chart(deck, use_container_width=True, height=580,
                            on_select="rerun", selection_mode="single-object", key="atlas_deck")
    new = parse_selection(event, focus)
    if new:
        st.session_state["atlas_focus"] = new
        st.rerun()

with right:
    focus = st.session_state["atlas_focus"]
    frow = df[df.h3_index == focus]
    render_card(frow.iloc[0] if len(frow) else df.nlargest(1, "crashes").iloc[0])
    st.caption("One raw layer of many. Next: population, roads, transit, places — same cell, same tap.")
