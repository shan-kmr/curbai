"""Methodology — data sources, scoring formulas, and the first-party data upgrade path."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from curbai import scoring

st.set_page_config(
    page_title="CurbIndex — Methodology",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("# Methodology")
st.caption(
    "Data sources, feature engineering, scoring formulas, and what first-party "
    "location data would unlock on top of this open-data baseline."
)

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## What you're looking at")
st.markdown("""
One H3 resolution-9 grid over San Francisco (~174 m edge, 1,112 cells), three
tabs demonstrating three commercial applications of block-level geospatial
intelligence — all built on open data, all reproducible, all running live.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Data sources")
st.markdown("""
CurbIndex uses open data only. No API keys, no proprietary feeds.

| Script | What it does | Time |
|---|---|---|
| `scripts/bootstrap_data.py` | One-time copy of the SF subset of an H3 cell grid (1,112 cells) and 51,572 Overture Maps POIs for the SF bbox. After this runs once, CurbIndex is fully standalone. | <30 s |
| `scripts/fetch_osm.py` | Pulls OpenStreetMap data for SF via OSMnx: drive road network (9,890 nodes / 27,261 edges), 158,765 building footprints, 7,415 transit stops, 15,571 amenities. | ~2 min |
| `scripts/build_sf.py` | Bins everything into H3 cells, computes 13 per-cell features, produces per-category POI counts (1,238 categories), and applies the scoring functions. | ~30 s |

**San Francisco bbox:** `[-122.52, 37.71, -122.36, 37.83]`
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Feature engineering")

features = pd.DataFrame([
    ("poi_count", "Overture", "POIs with centroid inside cell"),
    ("unique_categories", "Overture", "Distinct primary categories in cell"),
    ("category_entropy", "Overture", "Shannon entropy of category distribution — proxy for mixed use"),
    ("intersection_count", "OSM roads", "Road graph nodes (intersections) binned to cell"),
    ("building_count", "OSM buildings", "Building footprint centroids binned to cell"),
    ("transit_stop_count", "OSM transit", "Bus stops, subway entrances, train stations"),
    ("amenity_count", "OSM amenities", "Total amenities per cell"),
    ("restaurant_count", "OSM amenities", "restaurant, cafe, fast_food, food_court, bar, pub, ice_cream"),
    ("restaurant_count_kring", "derived", "Sum of restaurant_count over H3 k=2 ring (~550 m)"),
    ("nightlife_count", "OSM amenities", "bar, pub, cafe, ice_cream"),
    ("safety_count", "OSM amenities", "police, hospital, clinic, fire_station"),
    ("greenery_count", "OSM amenities", "park, garden"),
    ("shop_count", "OSM amenities", "Any non-empty shop tag"),
], columns=["feature", "source", "computation"])
st.dataframe(features, hide_index=True, use_container_width=True)

st.markdown("""
**Normalization.** Features are passed through robust min-max: 5th and 95th
percentiles define [0, 1], values outside are clipped.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Scoring formulas")


def _weight_df(w: dict) -> pd.DataFrame:
    return pd.DataFrame([(k, f"{v:.2f}") for k, v in w.items()], columns=["component", "weight"])


st.markdown("### 1. Site Intelligence")
st.markdown("*Where should a business open?*")
st.dataframe(_weight_df(scoring.SITE_WEIGHTS), hide_index=True, use_container_width=False)
st.markdown("""
- **foot_traffic** — `(norm(poi_count) + norm(amenity_count)) / 2`. Two independent proxies for "people are here."
- **accessibility** — `(norm(transit_stop_count) + norm(intersection_count)) / 2`. How reachable is the block?
- **commercial_vibrancy** — `(norm(category_entropy) + norm(nightlife_count)) / 2`. Is this an active, diverse commercial zone?
- **demographic_density** — `norm(building_count)`. Residential + commercial buildings = people who live and work here.
- **retail_ecosystem** — `(norm(shop_count) + norm(restaurant_count)) / 2`. Existing retail anchors nearby.
""")

st.markdown("### 2. Brand Location Planner")
st.markdown("*Pick a category. See the white space.*")
st.markdown("""
This tab is **dynamic** — the score recomputes when you change the category dropdown. The formula:

```
demand = norm(poi_count) × 0.4 + norm(amenity_count) × 0.3 + norm(transit_stops) × 0.3
saturation = cat_count_in_kring / max(cat_count_in_kring)
opportunity = demand × (1 - saturation)
```

High opportunity = high demand + low same-category supply in the neighborhood. The k-ring sums
across H3 k=2 neighbors (~550 m radius, 19 cells) so the scoring captures walkable competition,
not just in-cell.

**1,238 categories** are available in the dropdown, covering the full Overture Maps taxonomy
from the 51,572 SF POIs.
""")

st.markdown("### 3. Neighborhood Character")
st.markdown("*What is this block's functional identity?*")
st.dataframe(_weight_df(scoring.CHARACTER_WEIGHTS), hide_index=True, use_container_width=False)
st.markdown("""
- **amenity_walkability** — `(norm(amenity_count) + norm(shop_count)) / 2`. Can you walk to daily needs?
- **green_access** — `norm(greenery_count)`. Parks and gardens within the cell.
- **safety_perception** — `norm(safety_count) × 0.6 + norm(nightlife_count) × 0.4`. Physical safety infrastructure + "lit at night" proxy.
- **evening_vibrancy** — `(norm(nightlife_count) + norm(restaurant_count)) / 2`. Is this block alive after 6pm?
- **connectivity** — `(norm(transit_stop_count) + norm(intersection_count)) / 2`. Transit + road graph density.
- **mixed_use** — `norm(category_entropy)`. Residential × commercial × service = a living neighborhood.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## What first-party location data would unlock")
st.markdown("""
Every score in CurbIndex uses **open-data proxies** for signals that a platform
with first-party device-level location data could measure directly. The table
below maps each proxy to its first-party equivalent.

The architecture stays identical — you swap the data layer, not the scoring
logic. That upgrade path is the platform value.
""")

upgrade = pd.DataFrame([
    ("Site Intelligence", "foot_traffic", "poi_count + amenity_count", "Actual hourly foot-traffic volume per cell from device signals"),
    ("Site Intelligence", "accessibility", "transit_stop_count + intersection_count", "Real mode-of-transport distribution — % car vs walk vs transit per cell"),
    ("Site Intelligence", "commercial_vibrancy", "category_entropy + nightlife_count", "Real-time user density at evening hours, filtered to commercial intent"),
    ("Site Intelligence", "demographic_density", "building_count", "Age, income-proxy, and interest-graph composition of actual visitors"),
    ("Brand Planner", "demand_proxy", "poi + amenity + transit composite", "Actual visit volume to the target category per cell per day"),
    ("Brand Planner", "saturation", "Same-category POI count in k-ring", "Market-share distribution among competitors from visit counts"),
    ("Brand Planner", "(not yet available)", "—", "Repeat-visit rate: what fraction of visitors come back within 30 days"),
    ("Neighborhood Character", "safety_perception", "safety_count + nightlife", "Behavioral signal: do people actually walk through this area after dark?"),
    ("Neighborhood Character", "evening_vibrancy", "nightlife + restaurant count", "Hour-by-hour user density curve — is this block alive at 9pm on a Thursday?"),
    ("Neighborhood Character", "amenity_walkability", "amenity + shop count", "Which amenities residents actually USE (visit dwell > 5 min) vs just exist on the map"),
], columns=["tab", "component", "open-data proxy (current)", "first-party upgrade"])
st.dataframe(upgrade, hide_index=True, use_container_width=True)

st.markdown("""
**The gap between the two columns IS the platform value.** Open data tells you
what's *there*. First-party location data tells you what people *do with* what's
there. The scoring architecture doesn't change — only the input fidelity.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Trending and loyalty signals (not yet in v0)")
st.markdown("""
Two signal families are commercially valuable but hard to proxy with open data:

**Trending / growth.** Is this neighborhood getting hotter or cooling off?
- Open proxy: VIIRS nightlight satellite data (annual snapshots, 2014–2021) — cells getting brighter over time = economic growth. Not yet integrated but the data pipeline supports it.
- First-party version: user-density growth rate per cell. "+15% MAU in this block vs last quarter."

**Loyalty / repeat visitation.** Do people come back?
- Open proxy: Yelp review velocity (reviews/month at businesses geocoded into H3 cells) — available in the sibling GeoInsights project but not yet wired into CurbIndex.
- First-party version: actual repeat-visit rate from device fingerprints. "68% of visitors to this block return within 30 days."

Both are plug-in-ready: write a feature extraction function, drop a new parquet into `data/`, and every tab's scoring function picks it up.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Similarity")
st.markdown("""
The "similar cells" panel in the main app uses a FAISS `IndexFlatL2` over a
z-scored matrix of all numeric features. For every selected cell, it returns
the five nearest neighbors in feature space — cells elsewhere in SF that share
the same neighborhood character.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Limitations")
st.markdown("""
- **Demand is a proxy.** POI count and amenity density stand in for actual foot traffic and commercial demand. Swapping in real visitation data would change rankings in purely residential areas.
- **Normalization is city-local.** A "high" score in SF does not compare directly to a "high" score in NYC. Cross-city comparison requires re-fitting on a pooled city set or using globally-fit embeddings.
- **No temporal layer yet.** Evening vibrancy approximates time-of-day patterns via nightlife count, but hour-level demand curves require longitudinal data.
- **Safety is a proxy.** Police/hospital count is a poor proxy for street-level safety. Production use would require open incident data (e.g. SFPD CAD data via DataSF).
- **Similarity is first-order.** Linear z-score, not a learned embedding. A multimodal foundation model would capture feature interactions.
""")

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Reproduce from scratch")
st.code("""python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/bootstrap_data.py   # one-time data bootstrap
python scripts/fetch_osm.py        # OpenStreetMap fetch for SF
python scripts/build_sf.py         # feature engineering + scoring

streamlit run app.py""", language="bash")
