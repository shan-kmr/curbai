"""
CurbAI — block-level scoring for autonomous mobility.

Main page. Three tabs, one H3 grid over San Francisco:
    1. AV Rider Launch Readiness
    2. Autonomous Delivery Handoff
    3. Rides → Eats Conversion Upside

Click any hex to select it — the side panel updates with a per-component
score breakdown and its five most-similar cells elsewhere in the city.

Reads data/sf_scored.parquet (produced by scripts/build_sf.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import pydeck as pdk
import streamlit as st
from geopy.exc import GeopyError
from geopy.geocoders import Nominatim

from curbai import scoring, similarity
from curbai.loader import SCORED_PATH, feature_columns, load_sf_scored

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CurbAI — block-level scoring for autonomous mobility",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

SF_CENTER_LAT = 37.773
SF_CENTER_LON = -122.441

# ---------------------------------------------------------------------------
# Cached data + similarity
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading SF scored grid…")
def get_data() -> pd.DataFrame:
    return load_sf_scored()


@st.cache_resource(show_spinner="Building similarity index…")
def get_similarity_index(df: pd.DataFrame):
    feat_cols = [
        c
        for c in feature_columns(df)
        if c not in ("score_av", "score_delivery", "score_eats")
    ]
    return similarity.build_similarity(df, feat_cols), feat_cols


# ---------------------------------------------------------------------------
# Warm colormap (matches the beige/cocoa theme)
# ---------------------------------------------------------------------------


def score_to_color(score: float) -> list[int]:
    """Score in [0, 1] → RGBA list, warm cocoa → tan → cream ramp."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return [60, 50, 42, 120]
    t = max(0.0, min(1.0, float(score)))
    stops = [
        (0.00, (90, 60, 40)),     # dark saddle
        (0.25, (140, 95, 55)),    # medium saddle
        (0.50, (200, 150, 90)),   # tan
        (0.75, (230, 195, 130)),  # warm sand
        (1.00, (255, 225, 160)),  # gold cream
    ]
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 + 1e-9)
            rgb = [int(c0[i] + f * (c1[i] - c0[i])) for i in range(3)]
            return rgb + [215]
    return [255, 225, 160, 215]


# ---------------------------------------------------------------------------
# Address geocoder
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Looking up address…")
def geocode_sf(address: str) -> tuple[float, float] | None:
    if not address.strip():
        return None
    try:
        geocoder = Nominatim(user_agent="curbai-demo")
        full = f"{address}, San Francisco, CA"
        loc = geocoder.geocode(full, timeout=5)
        if loc is None:
            return None
        return float(loc.latitude), float(loc.longitude)
    except GeopyError:
        return None


def nearest_cell(df: pd.DataFrame, lat: float, lon: float) -> str | None:
    dlat = df["center_lat"].values - lat
    dlon = df["center_lon"].values - lon
    dist2 = dlat * dlat + dlon * dlon
    if len(dist2) == 0:
        return None
    idx = int(dist2.argmin())
    return str(df["h3_index"].iloc[idx])


# ---------------------------------------------------------------------------
# Tab config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabConfig:
    key: str
    score_col: str
    title: str
    tagline: str
    scoring_fn: Callable
    weights: dict[str, float]


TABS: list[TabConfig] = [
    TabConfig(
        key="av",
        score_col="score_av",
        title="AV Rider Launch Readiness",
        tagline="Where should the next autonomous rider pickup launch?",
        scoring_fn=scoring.av_readiness_score,
        weights=scoring.AV_WEIGHTS,
    ),
    TabConfig(
        key="delivery",
        score_col="score_delivery",
        title="Autonomous Delivery Handoff",
        tagline="Where can a delivery robot drop a package and have a human find it?",
        scoring_fn=scoring.delivery_handoff_score,
        weights=scoring.DELIVERY_WEIGHTS,
    ),
    TabConfig(
        key="eats",
        score_col="score_eats",
        title="Rides → Eats Conversion Upside",
        tagline="Which drop-off zones have the highest food-delivery cross-sell opportunity?",
        scoring_fn=scoring.eats_upside_score,
        weights=scoring.EATS_WEIGHTS,
    ),
]


# ---------------------------------------------------------------------------
# Deck renderer
# ---------------------------------------------------------------------------


def render_deck(df: pd.DataFrame, tab: TabConfig, focus_h3: str | None) -> pdk.Deck:
    df = df.copy()
    df["_color"] = df[tab.score_col].apply(score_to_color)
    df["_elevation"] = df[tab.score_col] * 400
    df["_tooltip_score"] = (df[tab.score_col] * 100).round(1)

    focus_df = df[df["h3_index"] == focus_h3] if focus_h3 else None

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        id=f"hex_{tab.key}",
        data=df,
        get_hexagon="h3_index",
        get_fill_color="_color",
        get_elevation="_elevation",
        elevation_scale=1,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        coverage=0.92,
    )

    layers: list[pdk.Layer] = [hex_layer]

    if focus_df is not None and len(focus_df) > 0:
        focus_layer = pdk.Layer(
            "H3HexagonLayer",
            id=f"focus_{tab.key}",
            data=focus_df,
            get_hexagon="h3_index",
            get_fill_color=[240, 220, 180, 170],
            get_elevation="_elevation",
            elevation_scale=1.5,
            extruded=True,
            pickable=False,
            coverage=1.0,
            stroked=True,
            line_width_min_pixels=3,
            get_line_color=[240, 220, 180, 255],
        )
        layers.append(focus_layer)

    view_center_lat = SF_CENTER_LAT
    view_center_lon = SF_CENTER_LON
    zoom = 11.4
    if focus_df is not None and len(focus_df) > 0:
        view_center_lat = float(focus_df["center_lat"].iloc[0])
        view_center_lon = float(focus_df["center_lon"].iloc[0])
        zoom = 13.1

    view = pdk.ViewState(
        latitude=view_center_lat,
        longitude=view_center_lon,
        zoom=zoom,
        pitch=42,
        bearing=0,
    )

    tooltip = {
        "html": (
            f"<b>{tab.title}</b><br/>"
            "<span style='opacity:0.8'>click to select</span><br/>"
            "Score: {_tooltip_score}<br/>"
            "H3: {h3_index}<br/>"
            "POIs: {poi_count} · Restaurants: {restaurant_count}"
        ),
        "style": {
            "backgroundColor": "#2a221d",
            "color": "#e6d5b8",
            "fontSize": "12px",
            "padding": "10px",
            "borderRadius": "4px",
            "border": "1px solid #4a3c30",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="dark",
        tooltip=tooltip,
    )


# ---------------------------------------------------------------------------
# Side panel renderers
# ---------------------------------------------------------------------------


def render_breakdown(df: pd.DataFrame, tab: TabConfig, h3_id: str) -> None:
    row = df[df["h3_index"] == h3_id]
    if len(row) == 0:
        st.warning("Cell not found.")
        return

    row_df = row.iloc[[0]]
    score_series, components = tab.scoring_fn(row_df)
    score = float(score_series.iloc[0])

    st.markdown(f"### `{h3_id[-12:]}`")
    st.caption(
        f"Center: {row['center_lat'].iloc[0]:.5f}, {row['center_lon'].iloc[0]:.5f}"
    )
    st.metric(label=tab.title, value=f"{score * 100:.1f} / 100")

    st.markdown("#### Score breakdown")
    for component_name, weight in tab.weights.items():
        value = float(components[component_name].iloc[0])
        contrib = weight * value
        st.progress(
            min(max(value, 0.0), 1.0),
            text=f"{component_name.replace('_', ' ')}  ·  w={weight:.2f}  ·  contrib={contrib:.3f}",
        )

    st.markdown("#### Raw features")
    raw_cols = [
        "poi_count",
        "unique_categories",
        "category_entropy",
        "intersection_count",
        "building_count",
        "transit_stop_count",
        "amenity_count",
        "restaurant_count",
        "restaurant_count_kring",
        "nightlife_count",
        "safety_count",
        "greenery_count",
        "shop_count",
    ]
    raw = (
        row_df[raw_cols]
        .T.rename(columns={row_df.index[0]: "value"})
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    st.dataframe(raw, hide_index=True, use_container_width=True)


def render_similar(
    df: pd.DataFrame, sim: similarity.SimilarityIndex, h3_id: str, tab: TabConfig
) -> None:
    st.markdown("#### Similar cells elsewhere in SF")
    st.caption(
        "Nearest neighbors by z-scored feature vector (FAISS L2). "
        "Cells that share this one's neighborhood character."
    )
    neighbors = sim.query(h3_id, k=5)
    if not neighbors:
        st.info("No neighbors found.")
        return
    rows = []
    for nb_h3, dist in neighbors:
        nb = df[df["h3_index"] == nb_h3]
        if len(nb) == 0:
            continue
        rows.append(
            {
                "h3": nb_h3[-8:],
                "center": f"{nb['center_lat'].iloc[0]:.4f}, {nb['center_lon'].iloc[0]:.4f}",
                "score": round(float(nb[tab.score_col].iloc[0]) * 100, 1),
                "top category": str(nb["top_category"].iloc[0]),
                "dist": round(dist, 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    if "focus_h3" not in st.session_state:
        st.session_state["focus_h3"] = None


def set_focus(h3_id: str | None) -> None:
    st.session_state["focus_h3"] = h3_id


def parse_pydeck_selection(event, current_focus: str | None) -> str | None:
    """Extract the h3_index of the clicked hex, if any. Returns None if no new selection."""
    if event is None:
        return None
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if not sel:
        return None
    objects = sel.get("objects") if isinstance(sel, dict) else getattr(sel, "objects", None)
    if not objects:
        return None
    # objects is a dict: layer_id -> list of picked object dicts
    for _layer_id, obj_list in objects.items():
        if not obj_list:
            continue
        first = obj_list[0]
        h3_hit = first.get("h3_index") if isinstance(first, dict) else None
        if h3_hit and h3_hit != current_focus:
            return h3_hit
    return None


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------


def render_tab(
    df: pd.DataFrame, sim: similarity.SimilarityIndex, tab: TabConfig
) -> None:
    st.markdown(f"### {tab.title}")
    st.caption(tab.tagline)

    left, right = st.columns([2, 1], gap="large")

    with left:
        addr_col, top_col = st.columns([3, 2])
        with addr_col:
            addr = st.text_input(
                "Search an SF address",
                key=f"addr_{tab.key}",
                placeholder="e.g. 1455 Market Street",
            )
            if addr:
                latlon = geocode_sf(addr)
                if latlon is None:
                    st.warning("Address not found.")
                else:
                    lat, lon = latlon
                    h3_id = nearest_cell(df, lat, lon)
                    if h3_id is not None:
                        set_focus(h3_id)

        with top_col:
            top_options = ["—"] + [
                f"#{i + 1}  ({r[tab.score_col]:.2f})  {r['h3_index'][-8:]}"
                for i, r in df.nlargest(10, tab.score_col).reset_index().iterrows()
            ]
            top_n = st.selectbox(
                "Jump to top-scoring cell",
                options=top_options,
                key=f"top_{tab.key}",
            )
            if top_n != "—":
                short = top_n.split()[-1]
                match = df[df["h3_index"].str.endswith(short)]
                if len(match) > 0:
                    set_focus(str(match["h3_index"].iloc[0]))

        focus_h3 = st.session_state.get("focus_h3")
        deck = render_deck(df, tab, focus_h3)

        event = st.pydeck_chart(
            deck,
            use_container_width=True,
            height=580,
            on_select="rerun",
            selection_mode="single-object",
            key=f"pydeck_{tab.key}",
        )

        new_focus = parse_pydeck_selection(event, focus_h3)
        if new_focus is not None:
            set_focus(new_focus)
            st.rerun()

        st.caption(
            f"Click any hex to select it.  "
            f"{len(df):,} H3 res-9 cells (~150 m edge).  "
            "Cocoa = low score, warm cream = high. Height encodes score."
        )

    with right:
        focus_h3 = st.session_state.get("focus_h3")
        if focus_h3 is None:
            focus_h3 = str(df.nlargest(1, tab.score_col)["h3_index"].iloc[0])
            st.caption("Showing the current top-scoring cell. Click the map to inspect others.")
        else:
            st.caption("Selected cell. Click another hex to change.")

        render_breakdown(df, tab, focus_h3)
        st.markdown("---")
        render_similar(df, sim, focus_h3, tab)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.markdown("# CurbAI")
    st.caption(
        "Block-level scoring for autonomous mobility. One H3 resolution-9 "
        "grid over San Francisco, three scoring functions per cell: where "
        "an autonomous ride service should launch, where a delivery robot "
        "can drop, and which ride drop-offs have the highest food-delivery "
        "cross-sell opportunity. See the Methodology page in the sidebar "
        "for data sources and scoring formulas."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_state()

    if not SCORED_PATH.exists():
        st.error(
            f"Missing `{SCORED_PATH.name}`. Run:\n\n"
            "```\n"
            "python scripts/bootstrap_data.py\n"
            "python scripts/fetch_osm.py\n"
            "python scripts/build_sf.py\n"
            "```"
        )
        return

    df = get_data()
    sim, _ = get_similarity_index(df)

    render_header()
    st.markdown("")

    tab_objs = st.tabs([t.title for t in TABS])
    for t_obj, tab in zip(tab_objs, TABS):
        with t_obj:
            render_tab(df, sim, tab)


if __name__ == "__main__":
    main()
