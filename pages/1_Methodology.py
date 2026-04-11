"""Methodology page — data sources, feature engineering, scoring formulas."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from curbai import scoring

st.set_page_config(
    page_title="CurbAI — Methodology",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# Methodology")
st.caption(
    "Data sources, feature engineering, scoring formulas, and limitations. "
    "Everything on this page is reproducible from the scripts in the repo."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

st.markdown("## Data sources")
st.markdown(
    """
CurbAI uses open data only. No API keys, no proprietary feeds. The full data
pipeline runs in three scripts, each emitting files into `data/raw/`:

| Script | What it does | Time |
|---|---|---|
| `scripts/bootstrap_data.py` | One-time copy of the SF subset of a sibling H3 cell grid (1,112 cells) and 51,572 Overture Maps POIs for the SF bounding box. After this runs once, CurbAI is fully standalone. | <30 s |
| `scripts/fetch_osm.py` | Pulls OpenStreetMap data for the SF bbox via OSMnx: the drive road network (9,890 nodes / 27,261 edges), 158,765 building footprints, 7,415 transit stops, and 15,571 amenities. | ~1–2 min |
| `scripts/build_sf.py` | Bins everything into H3 resolution-9 cells, computes 13 per-cell features, applies the three scoring functions, and writes `data/sf_scored.parquet`. | ~30 s |
"""
)

st.markdown("**San Francisco bounding box**")
st.code("[-122.52, 37.71, -122.36, 37.83]  # (west, south, east, north)")

st.markdown(
    """
**Granularity.** H3 resolution 9 cells average ~0.105 km² and have an edge
length of ~174 m — small enough that a single cell corresponds to a recognizable
block, large enough to smooth out noise in sparse features like transit stops.
The full SF coverage at res-9 is 1,112 cells.
"""
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Feature engineering")
st.markdown(
    """
All 13 per-cell features are computed offline in `scripts/build_sf.py`. Each
row in the table below is a column in `data/sf_scored.parquet`.
"""
)

features = pd.DataFrame(
    [
        ("poi_count", "Overture master grid", "Passthrough — POIs with centroid inside cell"),
        ("unique_categories", "Overture master grid", "Count of distinct primary categories inside the cell"),
        ("category_entropy", "Overture master grid", "Shannon entropy of category distribution — a proxy for mixed use"),
        ("intersection_count", "OSM road graph", "Road graph nodes (= intersections) binned by centroid to cell"),
        ("building_count", "OSM buildings", "Building footprint centroids binned to cell"),
        ("transit_stop_count", "OSM transit", "public_transport + highway=bus_stop + railway=station|tram_stop|subway_entrance"),
        ("amenity_count", "OSM amenities", "Total OSM amenities (any tag in our filter set) per cell"),
        ("restaurant_count", "OSM amenities", "amenity ∈ {restaurant, cafe, fast_food, food_court, ice_cream, bar, pub}"),
        ("restaurant_count_kring", "derived", "Sum of restaurant_count over the H3 k=2 ring (~550 m radius, 19 cells)"),
        ("nightlife_count", "OSM amenities", "amenity ∈ {bar, pub, cafe, ice_cream}"),
        ("safety_count", "OSM amenities", "amenity ∈ {police, hospital, clinic, fire_station}"),
        ("greenery_count", "OSM amenities", "leisure ∈ {park, garden}"),
        ("shop_count", "OSM amenities", "Any non-empty shop tag"),
    ],
    columns=["feature", "source", "computation"],
)
st.dataframe(features, hide_index=True, use_container_width=True)

st.markdown(
    """
**Normalization.** Every feature used by a scoring function is passed through
robust min-max normalization: the 5th and 95th percentiles define the [0, 1]
range, and values outside are clipped. This makes the scores resilient to a
single outlier cell (e.g. a Union Square block with 200 restaurants) and keeps
the dynamic range tight enough for a visible color ramp.
"""
)

st.code(
    """def _norm(series):
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0.0, 1.0)""",
    language="python",
)

# ---------------------------------------------------------------------------
# Scoring formulas
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Scoring formulas")

st.markdown(
    """
Each score is a weighted sum of five or six components, each normalized to
[0, 1]. Weights are held in a Python dict and can be edited directly in
`curbai/scoring.py` — the downstream parquet and UI will pick up changes on
the next `python scripts/build_sf.py` run.
"""
)


def _weight_df(weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [(k, f"{v:.2f}") for k, v in weights.items()], columns=["component", "weight"]
    )


# ---- AV ----
st.markdown("### 1. AV Rider Launch Readiness")
st.markdown("*Where should an autonomous ride service launch its next pickup zone?*")
st.dataframe(_weight_df(scoring.AV_WEIGHTS), hide_index=True, use_container_width=False)
st.markdown(
    """
- **road_simplicity** — `1 − norm(intersection_count)`.
  AVs prefer simpler grid geometry. High intersection density means complex
  turns, yield interactions, and multi-way stops.
- **pickup_curb_proxy** — `1 − 2 × |norm(building_count) − 0.5|`.
  A U-shaped preference: empty lots have nowhere for riders to wait, while
  skyscraper canyons have no legal curb space. The sweet spot is mid-density.
- **demand_density** — `(norm(poi_count) + norm(amenity_count)) / 2`.
  Two independent proxies for "people are here".
- **transit_complement** — `1 − 2 × |norm(transit_stop_count) − 0.5|`.
  Similar U shape: a transit desert has walk-up demand but no mode choice, while
  a saturated transit corridor competes with the service.
- **ambient_population** — `norm(category_entropy)`.
  Mixed-use neighborhoods (residential + retail + office) have riders across
  the day, not just during commute windows.
"""
)

# ---- Delivery ----
st.markdown("### 2. Autonomous Delivery Handoff")
st.markdown("*Where can a delivery robot drop a package and have a human find it?*")
st.dataframe(_weight_df(scoring.DELIVERY_WEIGHTS), hide_index=True, use_container_width=False)
st.markdown(
    """
- **curb_accessible** — `1 − 2 × |norm(building_count) − 0.5|`.
  Can the robot physically pull up, without being blocked by a warehouse-style
  loading dock or a skyscraper curb cut.
- **low_canyon** — `1 − norm(building_count)`.
  Line-of-sight proxy. Canyons break line-of-sight to the recipient.
- **pedestrian_density** — `(norm(amenity_count) + norm(intersection_count)) / 2`.
  Walkable neighborhoods have legible handoff flows.
- **safety_proxy** — `norm(safety_count)`.
  Nearby police, hospital, clinic, or fire station = ambient foot traffic and
  lower perceived risk during the handoff window.
- **recovery_zones** — `norm(greenery_count)`.
  Parks and greenery are good retry-wait zones during failed handoffs.
- **daytime_light** — `norm(poi_count)`.
  Dense POIs are a proxy for "there is activity here"; dead zones score low.
"""
)

# ---- Eats ----
st.markdown("### 3. Rides → Eats Conversion Upside")
st.markdown("*Which ride drop-off zones have the highest food-delivery cross-sell opportunity?*")
st.dataframe(_weight_df(scoring.EATS_WEIGHTS), hide_index=True, use_container_width=False)
st.markdown(
    """
- **restaurant_supply** — `norm(restaurant_count_kring)`.
  Walkable restaurant density in a 19-cell (~550 m) H3 neighborhood — the
  supply side of the conversion.
- **drop_gravity** — `norm(poi_count)`.
  A proxy for ride drop-off volume, which we cannot observe directly.
- **category_diversity** — `(norm(unique_categories) + norm(category_entropy)) / 2`.
  Diverse neighborhoods have more "I want X cuisine" triggers.
- **evening_activity** — `norm(nightlife_count)`.
  Bars, pubs, cafes — the temporal signature of "just dropped off, still
  hungry".
- **underserved_home** — `1 − norm(restaurant_count)`.
  Cells where the local restaurant supply *at the cell itself* is thin score
  higher, because residents there have a higher lift from ordering delivery
  than walking downstairs.
"""
)

st.code(
    """# To tune the weights, edit curbai/scoring.py and re-run:
python scripts/build_sf.py
# The Streamlit app will pick up the new data/sf_scored.parquet on next load.""",
    language="bash",
)

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 'Similar cells elsewhere in SF'")
st.markdown(
    """
The similar-cells panel in the main app uses a FAISS `IndexFlatL2` over a
z-scored matrix of all numeric features (excluding the three scores). For
every cell the side panel shows, we look up its five nearest neighbors in
that feature space.

This is a deliberately simple starting point. It answers the question "which
other SF cells have the same neighborhood character as this one?" — not "which
other cells have the same population, elevation, and climate?". Swap the
feature matrix for real learned cell embeddings and the same pipeline works
without any code changes.
"""
)

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Limitations and next steps")
st.markdown(
    """
**Known gaps in v1.**

- **Demand is a proxy.** `poi_count` and `amenity_count` stand in for actual
  pickup/drop-off and delivery demand, which would come from a ride-hailing
  platform's own telemetry. Swapping in real trip-end density would change
  the ranking meaningfully in purely residential areas.
- **Feature normalization is city-local.** Scores are defined relative to the
  distribution of *this* city's features. A "high" score in SF does not directly
  compare to a "high" score in NYC. For cross-city comparison we would either
  re-fit percentiles on a pooled city set or use globally-fit learned
  embeddings.
- **No temporal layer yet.** The Eats score approximates evening activity via
  `nightlife_count`, but the right signal is time-of-day ride-drop density
  crossed with restaurant open hours — that needs hourly data.
- **Safety is a proxy.** `safety_count` (police/hospital/clinic/fire) is a
  terrible proxy for street-level safety. For a production product this
  should be replaced with city-specific open incident data (e.g. SFPD CAD
  data via DataSF).
- **Similarity is first-order.** The FAISS index is over raw z-scored
  features; a learned foundation-model embedding would capture interactions
  that a linear z-score cannot.

**Where this grows next.**

- Add NYC, LA, Austin, Phoenix. All of the above are one config-line
  changes given the current pipeline.
- Replace the POI-density demand proxy with real ride-drop telemetry if the
  data were available.
- Backtest the scores against a held-out ride-drop dataset, then use the
  backtest to learn the weights instead of setting them by hand.
- Hot-swap the similarity index for learned cell embeddings from a
  multimodal geo foundation model.
"""
)

# ---------------------------------------------------------------------------
# Reproduce
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Reproduce from scratch")
st.code(
    """# Clone the repo and set up a fresh venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# One-time SF data bootstrap (reads a sibling project once, then independent)
python scripts/bootstrap_data.py

# Open data fetch for SF (OpenStreetMap via OSMnx)
python scripts/fetch_osm.py

# Feature engineering + scoring precompute
python scripts/build_sf.py

# Run the app
streamlit run app.py""",
    language="bash",
)
