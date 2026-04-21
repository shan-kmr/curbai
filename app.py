"""
CurbIndex — geospatial intelligence for urban mobility, built on open data.

Three tabs demonstrating what a geo data platform enables:
    1. Site Intelligence       — where should a business open?
    2. Brand Location Planner  — pick a category, find the white space
    3. Neighborhood Character  — what is this block's functional identity?

Click any hex to select it — the side panel updates with per-component
score breakdown and the five most-similar cells elsewhere in the city.
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

from curbai import brand_planner, catchment, scoring, similarity, temporal
from curbai.loader import (
    SCORED_PATH,
    feature_columns,
    load_pois_sf,
    load_road_graph,
    load_sf_scored,
)

st.set_page_config(
    page_title="CurbIndex — geospatial intelligence for urban mobility",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

SF_CENTER_LAT = 37.773
SF_CENTER_LON = -122.441


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading SF scored grid…")
def get_data() -> pd.DataFrame:
    return load_sf_scored()


@st.cache_resource(show_spinner="Building similarity index…")
def get_similarity_index(df: pd.DataFrame):
    feat_cols = [
        c for c in feature_columns(df)
        if c not in ("score_site", "score_character")
    ]
    return similarity.build_similarity(df, feat_cols), feat_cols


@st.cache_data(show_spinner="Loading category data…")
def get_categories() -> list[str]:
    return brand_planner.list_categories(min_count=10)


@st.cache_data(show_spinner="Loading POIs…")
def get_pois() -> pd.DataFrame:
    return load_pois_sf()


@st.cache_resource(show_spinner="Loading walk-network…")
def get_road_graph():
    return load_road_graph()


@st.cache_data(show_spinner=False)
def compute_catchment(h3_id: str, max_minutes: int = 15) -> dict[str, float]:
    """Dijkstra walk-time from h3_id; cached per selected cell.

    Streamlit's @st.cache_data hashes the args — the road graph and cells df
    are fetched inside so the cache key stays small.
    """
    G = get_road_graph()
    df = get_data()
    return catchment.walk_time_cells(h3_id, G, df, max_minutes=max_minutes)


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


def score_to_color(score: float) -> list[int]:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return [60, 50, 42, 120]
    t = max(0.0, min(1.0, float(score)))
    stops = [
        (0.00, (90, 60, 40)),
        (0.25, (140, 95, 55)),
        (0.50, (200, 150, 90)),
        (0.75, (230, 195, 130)),
        (1.00, (255, 225, 160)),
    ]
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 + 1e-9)
            rgb = [int(c0[i] + f * (c1[i] - c0[i])) for i in range(3)]
            return rgb + [215]
    return [255, 225, 160, 215]


# ---------------------------------------------------------------------------
# Geocoder
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Looking up address…")
def geocode_sf(address: str) -> tuple[float, float] | None:
    if not address.strip():
        return None
    try:
        geocoder = Nominatim(user_agent="curbindex-demo")
        loc = geocoder.geocode(f"{address}, San Francisco, CA", timeout=5)
        return (float(loc.latitude), float(loc.longitude)) if loc else None
    except GeopyError:
        return None


def nearest_cell(df: pd.DataFrame, lat: float, lon: float) -> str | None:
    dlat = df["center_lat"].values - lat
    dlon = df["center_lon"].values - lon
    idx = int((dlat * dlat + dlon * dlon).argmin())
    return str(df["h3_index"].iloc[idx])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    if "focus_h3" not in st.session_state:
        st.session_state["focus_h3"] = None


def set_focus(h3_id: str | None) -> None:
    st.session_state["focus_h3"] = h3_id


def parse_pydeck_selection(event, current_focus: str | None) -> str | None:
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
    for _lid, obj_list in objects.items():
        if not obj_list:
            continue
        h3_hit = obj_list[0].get("h3_index") if isinstance(obj_list[0], dict) else None
        if h3_hit and h3_hit != current_focus:
            return h3_hit
    return None


# ---------------------------------------------------------------------------
# Deck builder
# ---------------------------------------------------------------------------


def _catchment_color(minutes: float) -> list[int]:
    """Translucent teal tint by walk-time bucket (closer = more opaque)."""
    if minutes <= 5:
        return [100, 210, 230, 150]
    if minutes <= 10:
        return [100, 210, 230, 90]
    if minutes <= 15:
        return [100, 210, 230, 50]
    return [0, 0, 0, 0]


def build_deck(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    focus_h3: str | None,
    catchment_walk_times: dict[str, float] | None = None,
) -> pdk.Deck:
    df = df.copy()
    df["_color"] = df[score_col].apply(score_to_color)
    df["_elev"] = df[score_col] * 400
    df["_score_pct"] = (df[score_col] * 100).round(1)

    focus_df = df[df["h3_index"] == focus_h3] if focus_h3 else None

    layers: list[pdk.Layer] = [
        pdk.Layer(
            "H3HexagonLayer",
            id="hex",
            data=df,
            get_hexagon="h3_index",
            get_fill_color="_color",
            get_elevation="_elev",
            elevation_scale=1,
            extruded=True,
            pickable=True,
            auto_highlight=True,
            coverage=0.92,
        )
    ]
    if catchment_walk_times:
        catch_rows = [
            {"h3_index": h, "_catch_color": _catchment_color(t)}
            for h, t in catchment_walk_times.items()
        ]
        catch_df = pd.DataFrame(catch_rows)
        layers.append(
            pdk.Layer(
                "H3HexagonLayer",
                id="catchment",
                data=catch_df,
                get_hexagon="h3_index",
                get_fill_color="_catch_color",
                extruded=False,
                pickable=False,
                coverage=1.0,
                stroked=False,
            )
        )
    if focus_df is not None and len(focus_df) > 0:
        layers.append(
            pdk.Layer(
                "H3HexagonLayer",
                id="focus",
                data=focus_df,
                get_hexagon="h3_index",
                get_fill_color=[240, 220, 180, 170],
                get_elevation="_elev",
                elevation_scale=1.5,
                extruded=True,
                pickable=False,
                coverage=1.0,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[240, 220, 180, 255],
            )
        )

    clat, clon, zoom = SF_CENTER_LAT, SF_CENTER_LON, 11.4
    if focus_df is not None and len(focus_df) > 0:
        clat = float(focus_df["center_lat"].iloc[0])
        clon = float(focus_df["center_lon"].iloc[0])
        zoom = 13.1

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=clat, longitude=clon, zoom=zoom, pitch=42),
        map_style="dark",
        tooltip={
            "html": f"<b>{title}</b><br/>click to select<br/>"
                    "Score: {_score_pct}<br/>H3: {h3_index}",
            "style": {
                "backgroundColor": "#2a221d",
                "color": "#e6d5b8",
                "fontSize": "12px",
                "padding": "10px",
                "borderRadius": "4px",
            },
        },
    )


# ---------------------------------------------------------------------------
# Side panels
# ---------------------------------------------------------------------------


def render_temporal_strip(row: pd.Series) -> None:
    """Compact 4-bar time-of-day strip for a single cell row."""
    missing = [c for c in temporal.temporal_columns() if c not in row.index]
    if missing:
        return
    st.markdown("#### Time-of-day activity")
    st.caption("Category-derived estimate · 0–1 scale")
    bars = pd.DataFrame(
        {
            "bucket": [temporal.bucket_display_name(b) for b in temporal.TEMPORAL_BUCKETS],
            "activity": [float(row[f"activity_{b}"]) for b in temporal.TEMPORAL_BUCKETS],
        }
    )
    st.bar_chart(bars, x="bucket", y="activity", height=180, use_container_width=True)


def render_catchment_summary(focus_h3: str, df: pd.DataFrame) -> None:
    """Show 5/10/15-min walk-time reach for the focused cell."""
    walk_times = compute_catchment(focus_h3, max_minutes=15)
    if not walk_times:
        return
    summary = catchment.catchment_summary(walk_times, df)
    st.markdown("#### Walk-time catchment")
    st.caption("Road-network Dijkstra · 4.8 km/h (80 m/min)")
    cols = st.columns(3)
    for col, minutes in zip(cols, (5, 10, 15)):
        bucket = summary[f"{minutes}_min"]
        col.metric(
            label=f"{minutes}-min walk",
            value=f"{bucket['n_cells']} cells",
            delta=f"{bucket['n_pois']:,} POIs",
            delta_color="off",
        )


def render_breakdown(
    df: pd.DataFrame,
    scoring_fn: Callable,
    weights: dict[str, float],
    score_label: str,
    h3_id: str,
) -> None:
    row = df[df["h3_index"] == h3_id]
    if len(row) == 0:
        st.warning("Cell not found.")
        return
    row_df = row.iloc[[0]]
    score_series, comps = scoring_fn(row_df)
    score = float(score_series.iloc[0])

    st.markdown(f"### `{h3_id[-12:]}`")
    st.caption(f"{row['center_lat'].iloc[0]:.5f}, {row['center_lon'].iloc[0]:.5f}")
    st.metric(label=score_label, value=f"{score * 100:.1f} / 100")

    st.markdown("#### Score breakdown")
    for name, w in weights.items():
        val = float(comps[name].iloc[0])
        st.progress(min(max(val, 0.0), 1.0), text=f"{name.replace('_', ' ')}  ·  w={w:.2f}  ·  {w * val:.3f}")

    render_temporal_strip(row_df.iloc[0])

    st.markdown("#### Raw features")
    raw_cols = [c for c in [
        "poi_count", "unique_categories", "category_entropy",
        "intersection_count", "building_count", "transit_stop_count",
        "amenity_count", "restaurant_count", "restaurant_count_kring",
        "nightlife_count", "safety_count", "greenery_count", "shop_count",
    ] if c in df.columns]
    raw = row_df[raw_cols].T.rename(columns={row_df.index[0]: "value"}).reset_index().rename(columns={"index": "feature"})
    st.dataframe(raw, hide_index=True, use_container_width=True)


def render_brand_breakdown(
    opp_df: pd.DataFrame, category: str, h3_id: str, pois_df: pd.DataFrame | None = None
) -> None:
    row = opp_df[opp_df["h3_index"] == h3_id]
    if len(row) == 0:
        st.warning("Cell not found.")
        return
    r = row.iloc[0]
    cat_display = brand_planner.category_display_name(category)

    st.markdown(f"### `{h3_id[-12:]}`")
    st.caption(f"{r['center_lat']:.5f}, {r['center_lon']:.5f}")
    st.metric(label=f"Opportunity for {cat_display}", value=f"{r['opportunity'] * 100:.1f} / 100")

    st.markdown("#### White-space analysis")
    col_a, col_b = st.columns(2)
    col_a.metric(f"{cat_display} in this cell", int(r["cat_count_cell"]))
    col_b.metric(f"{cat_display} within ~500m", int(r["cat_count_kring"]))

    st.progress(
        min(max(float(r["demand_proxy"]), 0.0), 1.0),
        text=f"Demand proxy: {r['demand_proxy']:.2f}",
    )

    if r["cat_count_kring"] == 0:
        st.success(f"No {cat_display} competitors within ~500 m — wide-open white space.")
    elif r["cat_count_cell"] == 0:
        st.info(f"No {cat_display} in this cell, but {int(r['cat_count_kring'])} nearby — moderate opportunity.")
    else:
        st.caption(f"{int(r['cat_count_cell'])} already here. Opportunity score reflects saturation.")

    if pois_df is not None:
        competitors = brand_planner.nearest_competitors(
            category, h3_id, pois_df, opp_df, k=5, max_m=1500
        )
        if competitors:
            st.markdown("#### Nearest competitors")
            for c in competitors:
                st.markdown(f"- **{c['name']}**  · ~{c['distance_m']} m {c['direction']}")
            st.caption(
                "Open-data: names + Euclidean distance. First-party upgrade: "
                "visit-share per competitor."
            )


def render_similar(
    df: pd.DataFrame, sim: similarity.SimilarityIndex, h3_id: str, score_col: str
) -> None:
    st.markdown("#### Similar cells elsewhere in SF")
    neighbors = sim.query(h3_id, k=5)
    if not neighbors:
        st.info("No neighbors found.")
        return
    rows = []
    for nb_h3, dist in neighbors:
        nb = df[df["h3_index"] == nb_h3]
        if len(nb) == 0:
            continue
        rows.append({
            "h3": nb_h3[-8:],
            "center": f"{nb['center_lat'].iloc[0]:.4f}, {nb['center_lon'].iloc[0]:.4f}",
            "score": round(float(nb[score_col].iloc[0]) * 100, 1) if score_col in nb.columns else "—",
            "top category": str(nb["top_category"].iloc[0]),
            "dist": round(dist, 2),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Address search + top-cell selector
# ---------------------------------------------------------------------------


def render_controls(df: pd.DataFrame, score_col: str, key_prefix: str) -> None:
    addr_col, top_col = st.columns([3, 2])
    with addr_col:
        addr = st.text_input("Search an SF address", key=f"addr_{key_prefix}", placeholder="e.g. 1455 Market Street")
        if addr:
            latlon = geocode_sf(addr)
            if latlon is None:
                st.warning("Address not found.")
            else:
                h3_id = nearest_cell(df, *latlon)
                if h3_id:
                    set_focus(h3_id)
    with top_col:
        if score_col in df.columns:
            top_options = ["—"] + [
                f"#{i+1}  ({r[score_col]:.2f})  {r['h3_index'][-8:]}"
                for i, r in df.nlargest(10, score_col).reset_index().iterrows()
            ]
            top_n = st.selectbox("Jump to top-scoring cell", options=top_options, key=f"top_{key_prefix}")
            if top_n != "—":
                short = top_n.split()[-1]
                match = df[df["h3_index"].str.endswith(short)]
                if len(match) > 0:
                    set_focus(str(match["h3_index"].iloc[0]))


# ---------------------------------------------------------------------------
# Static tab renderer (Site Intelligence, Neighborhood Character)
# ---------------------------------------------------------------------------


def render_static_tab(
    df: pd.DataFrame,
    sim: similarity.SimilarityIndex,
    score_col: str,
    title: str,
    tagline: str,
    scoring_fn: Callable,
    weights: dict[str, float],
    key: str,
) -> None:
    st.markdown(f"### {title}")
    st.caption(tagline)

    left, right = st.columns([2, 1], gap="large")
    with left:
        render_controls(df, score_col, key)
        focus_h3 = st.session_state.get("focus_h3")
        walk_times = compute_catchment(focus_h3) if focus_h3 else None
        deck = build_deck(df, score_col, title, focus_h3, catchment_walk_times=walk_times)
        event = st.pydeck_chart(deck, use_container_width=True, height=580, on_select="rerun", selection_mode="single-object", key=f"deck_{key}")
        new_focus = parse_pydeck_selection(event, focus_h3)
        if new_focus:
            set_focus(new_focus)
            st.rerun()
        st.caption(
            f"Click any hex to select. {len(df):,} H3 res-9 cells. Cocoa = low, cream = high. "
            "Selected cell tints a teal walk-time ring (5/10/15 min)."
        )

    with right:
        focus_h3 = st.session_state.get("focus_h3")
        if focus_h3 is None:
            focus_h3 = str(df.nlargest(1, score_col)["h3_index"].iloc[0])
            st.caption("Showing top-scoring cell. Click the map to inspect others.")
        else:
            st.caption("Selected cell. Click another hex to change.")
        render_breakdown(df, scoring_fn, weights, title, focus_h3)
        st.markdown("---")
        render_catchment_summary(focus_h3, df)
        st.markdown("---")
        render_similar(df, sim, focus_h3, score_col)


# ---------------------------------------------------------------------------
# Brand Location Planner tab
# ---------------------------------------------------------------------------


def render_brand_planner_tab(
    base_df: pd.DataFrame,
    sim: similarity.SimilarityIndex,
) -> None:
    st.markdown("### Brand Location Planner")
    st.caption("Pick a business category. See where demand is high but supply is thin.")

    categories = get_categories()
    if not categories:
        st.error("No category data found. Run scripts/build_sf.py.")
        return

    # Display-friendly names.
    cat_display_map = {c: brand_planner.category_display_name(c) for c in categories}
    display_names = [cat_display_map[c] for c in categories]

    left, right = st.columns([2, 1], gap="large")

    with left:
        cat_col, addr_col = st.columns([2, 3])
        with cat_col:
            selected_display = st.selectbox(
                "Business category",
                options=display_names,
                index=0,
                key="brand_cat",
            )
            # Reverse lookup.
            selected_cat = categories[display_names.index(selected_display)]

        with addr_col:
            addr = st.text_input("Search an SF address", key="addr_brand", placeholder="e.g. 1455 Market Street")
            if addr:
                latlon = geocode_sf(addr)
                if latlon is None:
                    st.warning("Address not found.")
                else:
                    h3_id = nearest_cell(base_df, *latlon)
                    if h3_id:
                        set_focus(h3_id)

        # Compute opportunity scores for selected category.
        opp_df = brand_planner.compute_opportunity(selected_cat, base_df)

        # Top-cell selector from opportunity scores.
        top_options = ["—"] + [
            f"#{i+1}  ({r['opportunity']:.2f})  {r['h3_index'][-8:]}"
            for i, r in opp_df.nlargest(10, "opportunity").reset_index().iterrows()
        ]
        top_n = st.selectbox("Jump to highest opportunity", options=top_options, key="top_brand")
        if top_n != "—":
            short = top_n.split()[-1]
            match = opp_df[opp_df["h3_index"].str.endswith(short)]
            if len(match) > 0:
                set_focus(str(match["h3_index"].iloc[0]))

        focus_h3 = st.session_state.get("focus_h3")
        walk_times = compute_catchment(focus_h3) if focus_h3 else None
        deck = build_deck(
            opp_df,
            "opportunity",
            f"Opportunity: {selected_display}",
            focus_h3,
            catchment_walk_times=walk_times,
        )
        event = st.pydeck_chart(deck, use_container_width=True, height=580, on_select="rerun", selection_mode="single-object", key="deck_brand")
        new_focus = parse_pydeck_selection(event, focus_h3)
        if new_focus:
            set_focus(new_focus)
            st.rerun()

        # Summary stats.
        n_zero = int((opp_df["cat_count_cell"] == 0).sum())
        n_total = int(opp_df["cat_count_cell"].sum())
        st.caption(
            f"{selected_display}: **{n_total}** locations across **{len(opp_df) - n_zero}** cells. "
            f"**{n_zero}** cells have zero — potential white space."
        )

    with right:
        focus_h3 = st.session_state.get("focus_h3")
        if focus_h3 is None:
            focus_h3 = str(opp_df.nlargest(1, "opportunity")["h3_index"].iloc[0])
            st.caption("Showing top opportunity cell. Click the map to inspect.")
        else:
            st.caption("Selected cell.")
        try:
            pois_df = get_pois()
        except FileNotFoundError:
            pois_df = None
        render_brand_breakdown(opp_df, selected_cat, focus_h3, pois_df=pois_df)
        st.markdown("---")
        render_catchment_summary(focus_h3, base_df)
        st.markdown("---")
        render_similar(base_df, sim, focus_h3, "score_site")


# ---------------------------------------------------------------------------
# Temporal Patterns tab
# ---------------------------------------------------------------------------


def render_temporal_tab(df: pd.DataFrame, sim: similarity.SimilarityIndex) -> None:
    st.markdown("### Temporal Patterns")
    st.caption(
        "When is this block alive? A category-derived activity profile for "
        "four time buckets. Open-data primitive — the first-party version is "
        "actual device-density per cell per hour."
    )

    temporal_cols = temporal.temporal_columns()
    missing = [c for c in temporal_cols if c not in df.columns]
    if missing:
        st.error(
            f"Missing temporal columns {missing}. Re-run `python scripts/build_sf.py`."
        )
        return

    left, right = st.columns([2, 1], gap="large")

    with left:
        bucket_labels = {b: temporal.bucket_display_name(b) for b in temporal.TEMPORAL_BUCKETS}
        default_bucket = "evening"
        selected_label = st.radio(
            "Time bucket",
            options=[bucket_labels[b] for b in temporal.TEMPORAL_BUCKETS],
            index=temporal.TEMPORAL_BUCKETS.index(default_bucket),
            horizontal=True,
            key="temporal_bucket",
        )
        selected_bucket = next(
            b for b, lbl in bucket_labels.items() if lbl == selected_label
        )
        score_col = f"activity_{selected_bucket}"

        render_controls(df, score_col, "temporal")
        focus_h3 = st.session_state.get("focus_h3")
        walk_times = compute_catchment(focus_h3) if focus_h3 else None
        deck = build_deck(
            df,
            score_col,
            f"Activity — {bucket_labels[selected_bucket]}",
            focus_h3,
            catchment_walk_times=walk_times,
        )
        event = st.pydeck_chart(
            deck,
            use_container_width=True,
            height=580,
            on_select="rerun",
            selection_mode="single-object",
            key="deck_temporal",
        )
        new_focus = parse_pydeck_selection(event, focus_h3)
        if new_focus:
            set_focus(new_focus)
            st.rerun()
        st.caption(
            "Morning = transit + intersections + shops. "
            "Midday = restaurants + shops + amenities. "
            "Evening = nightlife + restaurants + amenities. "
            "Late night = nightlife + restaurants."
        )

    with right:
        focus_h3 = st.session_state.get("focus_h3")
        if focus_h3 is None:
            focus_h3 = str(df.nlargest(1, score_col)["h3_index"].iloc[0])
            st.caption("Showing top-activity cell. Click the map to inspect others.")
        else:
            st.caption("Selected cell.")

        row = df[df["h3_index"] == focus_h3]
        if len(row) > 0:
            r = row.iloc[0]
            st.markdown(f"### `{focus_h3[-12:]}`")
            st.caption(f"{r['center_lat']:.5f}, {r['center_lon']:.5f}")
            st.metric(
                label=f"Activity — {bucket_labels[selected_bucket]}",
                value=f"{float(r[score_col]) * 100:.1f} / 100",
            )
            render_temporal_strip(r)

        st.markdown("---")
        render_catchment_summary(focus_h3, df)
        st.markdown("---")
        render_similar(df, sim, focus_h3, score_col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_state()

    if not SCORED_PATH.exists():
        st.error(f"Missing `{SCORED_PATH.name}`. Run the data pipeline first.")
        return

    df = get_data()
    sim, _ = get_similarity_index(df)

    st.markdown("# CurbIndex")
    st.caption(
        "Geospatial intelligence for urban mobility, built on open data. "
        "A prototype demonstrating what becomes possible when you can score "
        "every block in a city for commercial and mobility readiness — "
        "using only open-source data. See the Methodology page in the sidebar."
    )
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Site Intelligence",
        "Brand Location Planner",
        "Neighborhood Character",
        "Temporal Patterns",
    ])

    with tab1:
        render_static_tab(
            df, sim,
            score_col="score_site",
            title="Site Intelligence",
            tagline="Where should a business open? Composite score from foot traffic, accessibility, commercial vibrancy, and demographic density.",
            scoring_fn=scoring.site_intelligence_score,
            weights=scoring.SITE_WEIGHTS,
            key="site",
        )

    with tab2:
        render_brand_planner_tab(df, sim)

    with tab3:
        render_static_tab(
            df, sim,
            score_col="score_character",
            title="Neighborhood Character",
            tagline="What is this block's livability and commercial identity? Walkability, green space, safety, evening vibrancy, and mixed-use character.",
            scoring_fn=scoring.neighborhood_character_score,
            weights=scoring.CHARACTER_WEIGHTS,
            key="character",
        )

    with tab4:
        render_temporal_tab(df, sim)


if __name__ == "__main__":
    main()
