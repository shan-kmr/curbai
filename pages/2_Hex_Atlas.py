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


def render_card(row: pd.Series) -> None:
    hh = json.loads(row.hour_hist)
    facs = json.loads(row.top_factors)
    mx = max(hh) or 1
    bars = "".join(f'<div class="b" style="height:{max(2, int(v / mx * 100))}%"></div>' for v in hh)
    fac_html = "".join(
        f'<div class="jx-fac"><span>{k.title()}</span><span class="c">{v}</span></div>'
        for k, v in facs
    ) or '<div class="jx-fac"><span class="c">— none coded —</span></div>'

    pop, dens = _g(row, "population"), _g(row, "pop_density")
    bc, fl, mh = _g(row, "building_count"), _g(row, "avg_floors"), _g(row, "max_height")
    ar, nl = _g(row, "total_building_area"), _g(row, "nightlight_2021")

    st.markdown(f"""
    <div class="jx-card">
      <div class="jx-cid">R9 · {row.h3_index[-12:]} · New York · 0.105 km²</div>
      <div class="jx-lab" style="margin-top:6px">◆ Safety — NYPD Vision Zero</div>
      <div class="jx-big">{row.crashes:,}<small> collisions on record</small></div>
      <div class="jx-row"><span class="k">People killed</span><span class="v"><b>{row.killed}</b></span></div>
      <div class="jx-row"><span class="k">People injured</span><span class="v">{row.injured:,}</span></div>
      <div class="jx-row"><span class="k">Pedestrian</span><span class="v">{row.ped_inj} inj · {row.ped_kill} killed</span></div>
      <div class="jx-row"><span class="k">Cyclist</span><span class="v">{row.cyc_inj} inj · {row.cyc_kill} killed</span></div>
      <div class="jx-row"><span class="k">Motorist</span><span class="v">{row.mot_inj:,} inj · {row.mot_kill} killed</span></div>
      <div class="jx-row"><span class="k">Peak</span><span class="v"><b>{DOW[row.peak_dow] if row.peak_dow>=0 else '—'} {fmt_hour(row.peak_hour)}</b></span></div>
      <div class="jx-lab">By hour of day</div>
      <div class="jx-hist">{bars}</div>
      <div class="jx-hticks"><span>12a</span><span>6a</span><span>12p</span><span>6p</span><span>11p</span></div>
      <div class="jx-lab">Top contributing factors</div>
      {fac_html}
      <div class="jx-lab" style="margin-top:16px">◆ Who's here — WorldPop / GHSL</div>
      <div class="jx-row"><span class="k">Population</span><span class="v"><b>{_fmt(pop)}</b> residents</span></div>
      <div class="jx-lab">◆ Built form — Overture buildings</div>
      <div class="jx-row"><span class="k">Buildings</span><span class="v"><b>{_fmt(bc)}</b> · avg {_fmt(fl, '{:.0f}')} fl</span></div>
      <div class="jx-row"><span class="k">Tallest</span><span class="v">{_fmt(mh, '{:.0f}')} m</span></div>
      <div class="jx-row"><span class="k">Footprint</span><span class="v">{_fmt(ar/1000 if ar else None)}k m²</span></div>
      <div class="jx-row"><span class="k">Night-lights (2021)</span><span class="v">{_fmt(nl, '{:.0f}')}</span></div>
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
