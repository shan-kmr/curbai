---
title: CurbIndex
emoji: 🔭
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Click any block — see what open data reveals about place
license: mit
---

# CurbIndex

**Geospatial intelligence, built entirely on open data.** Click any block on the
map and it decodes: a composite score, the components behind it, the most-similar
blocks elsewhere, and a walk-time catchment — no proprietary data, no device
tracking. This is the first city (San Francisco) of the Janus visualisation layer.

## What it shows

One H3 res-9 grid (1,112 cells, ~150 m edge), four lenses:

1. **Site Intelligence** — where should a business open? Foot-traffic, accessibility, and activity per cell.
2. **Brand Location Planner** — pick a category; find where demand is high but supply is thin (white-space + nearest competitors, with bearings).
3. **Neighborhood Character** — walkability, green, safety, evening life, mixed-use.
4. **Temporal Patterns** — when is this block alive? POI-inferred morning / midday / evening / late-night.

Click any hex to select it — the side panel shows the per-component score
breakdown and the five most-similar cells elsewhere in the city (FAISS
nearest-neighbor over a z-scored feature matrix). Search any SF address to zoom.
See the **Methodology** page in the sidebar for full data-source and
scoring-formula details.

## Design

Paper (`#F6F4EF`) and cobalt (`#1E3A8A`) — the Janus house style.

## Data sources

All open, all no-auth, all cached locally:

- **H3 cell grid + Overture POIs** (filtered to the SF bbox).
- **Roads, buildings, transit stops, amenities** — OpenStreetMap via OSMnx.
- **San Francisco bbox** `[-122.52, 37.71, -122.36, 37.83]` (west, south, east, north).

## Reproduce locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/bootstrap_data.py   # one-time data bootstrap
python scripts/fetch_osm.py        # fetch OSM for SF (~1-2 min)
python scripts/build_sf.py         # features + scores (~30 sec)

streamlit run app.py
```

## Roadmap

Toward the Janus visualisation layer: globalise the grid (beyond SF), a
validated risk surface, a SHAP driver panel ("why this cell"), the
postcode-is-blind resolution toggle, and cities + insurance lenses.

## License

MIT.
